"""Point-based velocities of the Panda flange, computed in several ways.

Twists in this module follow the course convention: (v, omega), world
coordinates, v is the velocity of the point at the origin of the named frame.
The legacy twist_* names return w_B, not the world-origin spatial twist V_s.
Use velocity_representations to convert explicitly to spatial and body twists.
"""
import numpy as np
import jax.numpy as jnp

from .geometry import log_so3, T_from, T_inv, apply_point
from .panda import fk


# ---------------------------------------------------------------------------
# Task 5a (Lecture Part II): angular velocity by finite differences
# ---------------------------------------------------------------------------
def omega_finite_difference(q, qdot, joints, frame="panda_hand", h=1e-6):
    """I_omega of the frame from two poses a short time h apart.

    Use C(t + h) = exp([omega h]_x) C(t), so omega = log(C(t + h) C(t)^T) / h.
    """
    C0 = fk(q, joints)[frame][:3, :3]
    C1 = fk(q + h * qdot, joints)[frame][:3, :3]
    return log_so3(C1 @ C0.T) / h


# ---------------------------------------------------------------------------
# Task 5b (Lecture Part III): twist from the joint axes
# ---------------------------------------------------------------------------
def joint_axes(q, joints):
    """World axes n_k and points J_k on the axes of the 7 revolute joints.

    Returns (n, p), both shape (7, 3). In the Panda URDF every joint turns
    about the z-axis of its child link frame, so n_k is the third column of
    C_I,link_k and p_k is the origin of link_k.
    """
    T = fk(q, joints)
    n = jnp.array([T["panda_link%d" % k][:3, 2] for k in range(1, 8)])
    p = jnp.array([T["panda_link%d" % k][:3, 3] for k in range(1, 8)])
    return n, p


def twist_from_axes(q, qdot, joints, frame="panda_hand"):
    """(v, omega) of the frame: omega = sum qdot_k n_k, v = sum qdot_k n_k x (r - p_k)."""
    raise NotImplementedError("see the docstring and the assignment text")


# ---------------------------------------------------------------------------
# Task 5c (Lecture Part III): shifting the reference point & library adapters
# ---------------------------------------------------------------------------
def shift_twist(twist, r_BQ):
    """Twist at Q from the twist at B: v_Q = v_B + omega x r_BQ, omega unchanged."""
    raise NotImplementedError("see the docstring and the assignment text")


def twist_mujoco(model, data, q, qdot, body="hand"):
    """(v, omega) of a MuJoCo body frame, world coordinates.

    mj_objectVelocity returns (omega, v), angular part first. mjOBJ_XBODY
    refers to the body frame at data.xpos; the last argument 0 selects world axes.
    """
    import mujoco
    data.qpos[:7] = q
    data.qvel[:7] = qdot
    mujoco.mj_forward(model, data)
    res = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY, model.body(body).id, res, 0)
    return np.r_[res[3:], res[:3]]


def twist_frax(robot, q, qdot):
    """(v, omega) of the frax end effector (the grasp point): J(q) @ qdot.

    frax's ee_jacobian has the linear rows first, [Jv; Jw], like the course.
    """
    import jax.numpy as jnp
    return robot.ee_jacobian(jnp.asarray(q)) @ jnp.asarray(qdot)


# ---------------------------------------------------------------------------
# Task 5d (Lecture Part IV): spatial and body twists
# ---------------------------------------------------------------------------
def velocity_representations(T_IB, w_B):
    """Return (V_s, V_b), both linear-first arrays of shape (6,).

    T_IB maps body to world coordinates. w_B = (v_B, omega) uses world
    axes and the velocity of body origin B. Change the reference point for
    V_s: v_s = v_B - omega x r. Change axes for V_b: (C.T v_B, C.T omega).
    Check the two choices separately; neither changes the physical motion.
    """
    raise NotImplementedError("see the docstring and the assignment text")


# ---------------------------------------------------------------------------
# Pepper Bonus Task A: instantaneous screw axis & moving-camera transport theorem
# ---------------------------------------------------------------------------
def screw_axis_from_twist(V_s):
    """Recover (n, a, h, theta_dot) from a spatial twist V_s = (v_s, omega) with ||omega|| > 0.

    Returns:
      n: unit axis direction omega / ||omega||, shape (3,)
      a: point on the screw axis closest to the world origin (a . omega == 0), shape (3,)
      h: pitch (omega . v_s) / ||omega||^2, scalar
      theta_dot: angular speed ||omega||, scalar
    """
    raise NotImplementedError("see the docstring and the assignment text")


def camera_reading_rate(q, qdot, T_EK, I_r_IP, I_v_P, joints):
    """Time derivative d/dt (K_r_KP) of a moving cup P in a moving camera K.

    Uses the transport theorem in camera coordinates:
        d/dt (K_r_KP) = K_v_P - K_v_K - K_omega_IK x K_r_KP,
    where T_IK = T_IE @ T_EK and w_K is shifted from the twist w_E of the hand frame E.
    """
    raise NotImplementedError("see the docstring and the assignment text")
