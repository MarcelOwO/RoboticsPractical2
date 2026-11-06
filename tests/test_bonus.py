"""Pepper bonus tasks: instantaneous screw axis, moving-camera transport theorem, and AX = XB."""
import jax
import jax.numpy as jnp
import numpy as np

from im560 import geometry as g
from im560 import panda
from im560 import velocity as v


def test_screw_axis_from_twist_b8_and_single_joint():
    # Written exercise B8(b): A8 twist V_s = (1, 0, 0.1, 0, 0, 1)
    n, a, h, theta_dot = v.screw_axis_from_twist(np.array([1.0, 0.0, 0.1, 0.0, 0.0, 1.0]))
    assert np.allclose(n, [0.0, 0.0, 1.0])
    assert np.allclose(a, [0.0, 1.0, 0.0])
    assert np.isclose(h, 0.1)
    assert np.isclose(theta_dot, 1.0)

    # Written exercise B8(c): A6 twist V_s = (0.4, 0, 0, 0, 0, 1) -> pure rotation about (0, 0.4, 0)
    n2, a2, h2, _ = v.screw_axis_from_twist(np.array([0.4, 0.0, 0.0, 0.0, 0.0, 1.0]))
    assert np.allclose(n2, [0.0, 0.0, 1.0])
    assert np.allclose(a2, [0.0, 0.4, 0.0])
    assert np.isclose(h2, 0.0)

    # Single Panda joint moving: ISA has zero pitch and lies on that joint's physical axis
    joints = panda.parse_urdf()
    q = jnp.array([0.2, -0.5, 0.1, -2.0, 0.3, 1.5, -0.4])
    T_IE = panda.fk(q, joints)["panda_hand"]
    axes_n, axes_p = v.joint_axes(q, joints)
    for k in range(7):
        qdot = jnp.zeros(7).at[k].set(1.3)
        w_E = v.twist_from_axes(q, qdot, joints)
        V_s, _ = v.velocity_representations(T_IE, w_E)
        n_k, a_k, h_k, rate_k = v.screw_axis_from_twist(V_s)
        assert np.isclose(h_k, 0.0, atol=1e-12)
        assert np.isclose(rate_k, 1.3, atol=1e-12)
        assert np.allclose(n_k, axes_n[k], atol=1e-12)
        # Closest point a_k lies on the line through axes_p[k] along n_k
        assert np.allclose(np.cross(n_k, a_k - axes_p[k]), 0.0, atol=1e-12)


def test_camera_reading_rate_matches_jax_jvp():
    joints = panda.parse_urdf()
    q = jnp.array([0.1, -0.4, 0.2, -2.1, -0.3, 1.6, 0.5])
    qdot = jnp.array([0.2, -0.1, 0.3, 0.15, -0.25, 0.1, -0.2])
    T_EK = g.T_from(g.Cz(0.1) @ g.Cx(-0.05), [0.05, -0.02, 0.03])
    I_r_IP = jnp.array([0.55, 0.10, 0.15])
    I_v_P = jnp.array([-0.12, 0.08, 0.04])

    rate_transport = v.camera_reading_rate(q, qdot, T_EK, I_r_IP, I_v_P, joints)

    def cam_coords(q_var, p_var):
        T_IK = panda.fk(q_var, joints)["panda_hand"] @ T_EK
        return g.apply_point(g.T_inv(T_IK), p_var)

    _, rate_jvp = jax.jvp(cam_coords, (q, I_r_IP), (qdot, I_v_P))
    assert np.allclose(rate_transport, rate_jvp, atol=1e-12)


def test_calibrate_hand_eye_ax_xb():
    rng = np.random.default_rng(563)
    # True unknown flange-to-camera transform X = T_EK and world-to-target pose T_IP
    X_true = g.T_from(g.Cz(0.25) @ g.Cy(-0.15) @ g.Cx(0.10), [0.05, -0.03, 0.07])
    T_IP = g.T_from(g.Cz(-0.4) @ g.Cx(0.2), [0.60, 0.05, 0.12])

    T_IE_list, T_KP_list = [], []
    for _ in range(6):
        phi = rng.normal(size=3)
        phi = 0.6 * phi / np.linalg.norm(phi)
        r = rng.uniform([0.3, -0.3, 0.3], [0.6, 0.3, 0.7])
        T_IE = g.T_from(g.exp_so3(phi), r)
        T_KP = g.T_inv(T_IE @ X_true) @ T_IP
        T_IE_list.append(T_IE)
        T_KP_list.append(T_KP)

    X_est = g.calibrate_hand_eye(np.stack(T_IE_list), np.stack(T_KP_list))
    assert g.is_rotation(X_est[:3, :3])
    assert np.allclose(X_est, X_true, atol=1e-10)
