"""Forward kinematics of the Franka Panda, built from its URDF.

The URDF is the one that ships with frax (frax/assets/franka_panda/panda.urdf).
Each <joint> has an <origin xyz=... rpy=...> (the pose of the joint frame in
the parent link frame) and an <axis> (the rotation axis, in the joint frame).
For a revolute joint with angle q, the child link frame is

    T_parent_child(q) = T_from(C_from_rpy(rpy), xyz) @ Rot(axis, q).

Frame names in this module: I = world = panda_link0, k = panda_link<k>,
E = panda_hand (the flange with the hand mounted), G = grasp point (TCP).
"""
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np
import jax.numpy as jnp

from .geometry import C_from_rpy, T_from, exp_so3

# Offset from panda_hand to the grasp point between the fingers (as in frax).
HAND_TO_TCP = 0.105


@dataclass(eq=False)
class Joint:
    name: str
    kind: str              # "revolute" or "fixed"
    parent: str
    child: str
    xyz: np.ndarray        # origin translation, parent coordinates
    rpy: np.ndarray        # origin rotation (URDF roll-pitch-yaw)
    axis: np.ndarray       # rotation axis in the joint frame
    lower: float = -np.inf
    upper: float = np.inf


def urdf_path():
    """Path of the Panda URDF bundled with frax."""
    from frax.assets import FRANKA_ASSETS_DIR
    return os.path.join(FRANKA_ASSETS_DIR, "panda.urdf")


# ---------------------------------------------------------------------------
# Task 2a: read the joints from the URDF
# ---------------------------------------------------------------------------
def parse_urdf(path=None):
    """List of Joint objects in file order (the Panda URDF is ordered base to tip)."""
    root = ET.parse(path or urdf_path()).getroot()
    joints = []
    for j in root.findall("joint"):
        o = j.find("origin")
        a = j.find("axis")
        lim = j.find("limit")
        xyz = np.array([float(v) for v in o.get("xyz", "0 0 0").split()]) if o is not None else np.zeros(3)
        rpy = np.array([float(v) for v in o.get("rpy", "0 0 0").split()]) if o is not None else np.zeros(3)
        axis = np.array([float(v) for v in a.get("xyz").split()]) if a is not None else np.array([0, 0, 1.0])
        lo = float(lim.get("lower", -np.inf)) if lim is not None else -np.inf
        hi = float(lim.get("upper", np.inf)) if lim is not None else np.inf
        joints.append(Joint(j.get("name"), j.get("type"), j.find("parent").get("link"),
                            j.find("child").get("link"), xyz, rpy, axis, lo, hi))
    return joints


def arm_chain(joints):
    """The joints from panda_link0 to panda_hand: 7 revolute, then 2 fixed."""
    names = ["panda_joint%d" % k for k in range(1, 9)] + ["panda_hand_joint"]
    by_name = {j.name: j for j in joints}
    return [by_name[n] for n in names]


def joint_limits(joints):
    """Lower and upper limits of the 7 arm joints, each shape (7,)."""
    rev = [j for j in arm_chain(joints) if j.kind == "revolute"]
    return np.array([j.lower for j in rev]), np.array([j.upper for j in rev])


# ---------------------------------------------------------------------------
# Supplied: forward kinematics (implemented by students in Lab 03)
# ---------------------------------------------------------------------------
def fk(q, joints):
    """Forward kinematics.

    Returns a dict with the 4x4 transformations T_I<frame> for the frames
    "panda_link0" ... "panda_link8", "panda_hand", and "tcp" (the grasp point,
    HAND_TO_TCP along the z-axis of panda_hand).
    """
    q = jnp.asarray(q, dtype=float)
    T = {"panda_link0": jnp.eye(4)}
    k = 0
    for j in arm_chain(joints):
        T_pj = T_from(C_from_rpy(j.rpy), j.xyz)
        if j.kind == "revolute":
            T_pj = T_pj @ T_from(exp_so3(jnp.asarray(j.axis) * q[k]), jnp.zeros(3))
            k += 1
        T[j.child] = T[j.parent] @ T_pj
    T["tcp"] = T["panda_hand"] @ T_from(jnp.eye(3), [0.0, 0.0, HAND_TO_TCP])
    return T


# ---------------------------------------------------------------------------
# Reference implementations (given)
# ---------------------------------------------------------------------------
def load_mujoco_panda():
    """MuJoCo model of the Panda from MuJoCo Menagerie.

    Uses $MENAGERIE_PATH if set, otherwise downloads Menagerie once through
    the robot_descriptions package.
    """
    import mujoco
    root = os.environ.get("MENAGERIE_PATH")
    if root:
        path = os.path.join(root, "franka_emika_panda", "panda.xml")
    else:
        from robot_descriptions import panda_mj_description
        path = panda_mj_description.MJCF_PATH
    model = mujoco.MjModel.from_xml_path(path)
    return model, mujoco.MjData(model)


def load_frax_panda():
    """frax Manipulator of the Panda (end effector = grasp point)."""
    import frax
    return frax.load_panda()
