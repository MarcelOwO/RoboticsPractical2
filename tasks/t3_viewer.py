"""Task 3: an interactive frame viewer for the Panda, built on your own FK.

Run:   uv run python tasks/t3_viewer.py      then open http://localhost:8080

Seven sliders set the joint angles. The viewer draws every link frame, a
skeleton through the frame origins, a camera K mounted on the gripper, and
a cup P that stands still on the table. The panel shows the cup in camera
coordinates, K_r_KP, and whether the camera can see it.
"""
import os
import sys
import time

import numpy as np
import viser

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from im560 import geometry as g  # noqa: E402
from im560 import panda  # noqa: E402

# The camera sits 5 cm beside the hand and looks along the hand's z-axis.
T_EK = g.T_from(np.eye(3), [0.05, 0.0, 0.0])
I_r_IP = np.array([0.55, 0.0, 0.1])     # the cup, fixed in the world
HALF_FOV = np.deg2rad(30.0)             # half opening angle of the camera


def wxyz(T):
    """viser wants (w, x, y, z), the same order as the course and MuJoCo."""
    return np.asarray(g.quat_from_C(T[:3, :3]))


def camera_reading(T_IE):
    """Cup in camera coordinates, and whether it lies inside the viewing cone."""
    raise NotImplementedError("see the docstring and the assignment text")
    # Supplied visibility check: the camera looks along +z.
    angle = np.arccos(np.clip(K_r_KP[2] / np.linalg.norm(K_r_KP), -1.0, 1.0))
    return K_r_KP, bool(K_r_KP[2] > 0 and angle < HALF_FOV)


def main():
    joints = panda.parse_urdf()
    lo, hi = panda.joint_limits(joints)
    q0 = np.array([0.0, -np.pi / 6, 0.0, -3 * np.pi / 4, 0.0, 5 * np.pi / 9, np.pi / 4])

    server = viser.ViserServer()
    server.initial_camera.position = (1.3, -1.1, 0.9)
    server.initial_camera.look_at = (0.3, 0.0, 0.35)
    server.scene.add_grid("/table", width=1.5, height=1.5, cell_size=0.1)
    server.scene.add_box("/cup", color=(60, 110, 200), dimensions=(0.07, 0.07, 0.1),
                         position=I_r_IP - [0, 0, 0.05])
    names = ["panda_link%d" % k for k in range(8)] + ["panda_hand", "tcp"]
    frames = {n: server.scene.add_frame("/" + n, axes_length=0.08, axes_radius=0.004) for n in names}
    cam = server.scene.add_frame("/camera", axes_length=0.06, axes_radius=0.006,
                                 origin_color=(200, 120, 0))
    skeleton = server.scene.add_line_segments("/skeleton", np.zeros((1, 2, 3)), (90, 90, 90),
                                              thickness=0.006)

    sliders = [server.gui.add_slider(f"q{k + 1}", float(lo[k]), float(hi[k]), 0.01, float(q0[k]))
               for k in range(7)]
    info = server.gui.add_markdown("")

    def update(_=None):
        q = np.array([s.value for s in sliders])
        T = panda.fk(q, joints)
        for n in names:
            frames[n].position, frames[n].wxyz = np.asarray(T[n][:3, 3]), wxyz(T[n])
        T_IK = T["panda_hand"] @ T_EK
        cam.position, cam.wxyz = np.asarray(T_IK[:3, 3]), wxyz(T_IK)
        pts = np.array([T[n][:3, 3] for n in names])
        skeleton.points = np.stack([pts[:-1], pts[1:]], axis=1)
        K_r_KP, visible = camera_reading(T["panda_hand"])
        info.content = (f"**Cup in camera frame** K_r_KP = ({K_r_KP[0]:+.3f}, {K_r_KP[1]:+.3f}, "
                        f"{K_r_KP[2]:+.3f}) m  \n**Visible:** {'yes' if visible else 'no'}  \n"
                        "**TCP** I_r = ({:+.3f}, {:+.3f}, {:+.3f}) m".format(*T["tcp"][:3, 3]))

    for s in sliders:
        s.on_update(update)
    update()
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
