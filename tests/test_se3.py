"""Task 5 (Lecture Part IV): SE(3) exponential/logarithm, interpolation, and wheel odometry."""
import numpy as np
import pytest

from im560 import geometry as g

rng = np.random.default_rng(562)


def test_exp_se3_pure_translation_and_small_angle():
    u = np.array([0.4, -0.2, 0.7])
    T_trans = g.exp_se3(np.r_[u, 0.0, 0.0, 0.0])
    assert np.allclose(T_trans[:3, :3], np.eye(3))
    assert np.allclose(T_trans[:3, 3], u)
    # Near-zero angle uses Taylor branch smoothly
    phi_small = np.array([1e-7, -2e-7, 1.5e-7])
    xi = np.r_[u, phi_small]
    T_small = g.exp_se3(xi)
    assert np.allclose(g.log_se3(T_small), xi, atol=1e-12)


def test_exp_se3_screw_a8():
    # Written exercise A8: axis at a = (0, 1, 0), n = e_z, h = 0.1 m/rad, theta = pi/2
    theta = np.pi / 2
    s_v = np.array([1.0, 0.0, 0.1])
    xi = theta * np.r_[s_v, 0.0, 0.0, 1.0]
    T = g.exp_se3(xi)
    assert np.allclose(T[:3, :3], g.Cz(theta))
    assert np.allclose(T[:3, 3], [1.0, 1.0, 0.1 * theta])
    assert np.allclose(g.log_se3(T), xi)


@pytest.mark.parametrize("max_angle", [1e-6, 1e-2, 1.2, 2.8])
def test_log_se3_inverts_exp_se3(max_angle):
    for _ in range(15):
        u = rng.normal(size=3)
        n = rng.normal(size=3)
        phi = rng.uniform(0.0, max_angle) * n / np.linalg.norm(n)
        xi = np.r_[u, phi]
        T = g.exp_se3(xi)
        assert np.allclose(g.log_se3(T), xi, atol=1e-8)
        assert np.allclose(g.exp_se3(g.log_se3(T)), T, atol=1e-8)


def test_pose_interpolation_b9():
    # Written exercise B9: midpoint between I and T1 of A8
    theta = np.pi / 2
    T0 = np.eye(4)
    T1 = g.exp_se3(theta * np.array([1.0, 0.0, 0.1, 0.0, 0.0, 1.0]))
    T_dec = g.pose_interp_decoupled(T0, T1, 0.5)
    T_scr = g.pose_interp_screw(T0, T1, 0.5)
    assert np.allclose(T_dec[:3, :3], g.Cz(np.pi / 4))
    assert np.allclose(T_scr[:3, :3], g.Cz(np.pi / 4))
    assert np.allclose(T_dec[:3, 3], [0.5, 0.5, 0.05 * np.pi / 2])
    assert np.allclose(T_scr[:3, 3], [np.sin(np.pi / 4), 1.0 - np.cos(np.pi / 4), 0.05 * np.pi / 2])
    assert np.isclose(np.linalg.norm(T_dec[:3, 3] - T_scr[:3, 3]), np.hypot(0.5 - np.sqrt(0.5), 0.5 - (1 - np.sqrt(0.5))))


def test_differential_drive_odometry_c3():
    # Written exercise C3: v = 1 m/s forward, omega_z = 0.5 rad/s, circle of radius R = 2 m about (0, 2, 0)
    V_b = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.5])
    T0 = np.eye(4)
    # 1 step of 1.0 s
    T_euler_1 = g.integrate_body_twist(T0, V_b[None], 1.0, method="euler")[-1]
    T_exact_1 = g.integrate_body_twist(T0, V_b[None], 1.0, method="exact")[-1]
    r_true = np.array([2.0 * np.sin(0.5), 2.0 * (1.0 - np.cos(0.5)), 0.0])
    assert np.allclose(T_euler_1[:3, 3], [1.0, 0.0, 0.0])
    assert np.allclose(T_exact_1[:3, 3], r_true)
    # 10 steps of 0.1 s: exact stays on the analytical circle at every step
    traj_exact = g.integrate_body_twist(T0, np.tile(V_b, (10, 1)), 0.1, method="exact")
    traj_euler = g.integrate_body_twist(T0, np.tile(V_b, (10, 1)), 0.1, method="euler")
    assert np.allclose(traj_exact[-1, :3, 3], r_true)
    radii = np.linalg.norm(traj_exact[:, :3, 3] - np.array([0.0, 2.0, 0.0]), axis=1)
    assert np.allclose(radii, 2.0)
    assert 0.02 < np.linalg.norm(traj_euler[-1, :3, 3] - r_true) < 0.03
