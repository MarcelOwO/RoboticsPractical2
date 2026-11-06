"""Task 1. MuJoCo's own math functions (mju_*) serve as the reference."""
import mujoco
import numpy as np
import pytest

from im560 import geometry as g

rng = np.random.default_rng(560)


def random_phi(max_angle=np.pi):
    n = rng.normal(size=3)
    return rng.uniform(0.0, max_angle) * n / np.linalg.norm(n)


def mj_quat_from_phi(phi):
    q = np.zeros(4)
    t = np.linalg.norm(phi)
    mujoco.mju_axisAngle2Quat(q, phi / t, t)
    return q


def mj_mat_from_quat(q):
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, q)
    return m.reshape(3, 3)


# --- 1a ---------------------------------------------------------------------
def test_skew_is_cross_product():
    for _ in range(20):
        a, b = rng.normal(size=3), rng.normal(size=3)
        assert np.allclose(g.skew(a) @ b, np.cross(a, b))
        assert np.allclose(g.vee(g.skew(a)), a)


@pytest.mark.parametrize("C, axis", [(g.Cx, 0), (g.Cy, 1), (g.Cz, 2)])
def test_elementary_rotations(C, axis):
    a = 0.7
    e = np.eye(3)[axis]
    assert np.allclose(C(a), mj_mat_from_quat(mj_quat_from_phi(a * e)))


def test_urdf_rpy_is_fixed_axes():
    # turn about fixed x by 90 deg, then about fixed z by 90 deg
    C = g.C_from_rpy([np.pi / 2, 0.0, np.pi / 2])
    assert np.allclose(C @ np.array([0, 1, 0]), [0, 0, 1])       # y -> z (x-turn), z stays under the z-turn
    assert np.allclose(C @ np.array([1, 0, 0]), [0, 1, 0])       # x stays under the x-turn, then x -> y


def test_is_rotation():
    assert g.is_rotation(g.Cz(0.3) @ g.Cx(-1.1))
    assert not g.is_rotation(np.diag([1.0, 1.0, -1.0]))     # a reflection
    assert not g.is_rotation(1.01 * np.eye(3))              # a stretch
    assert not g.is_rotation(np.eye(3) + g.skew([0, 0, 0.1]))  # one naive integration step


# --- 1b ---------------------------------------------------------------------
def test_exp_matches_mujoco():
    for _ in range(20):
        phi = random_phi()
        assert np.allclose(g.exp_so3(phi), mj_mat_from_quat(mj_quat_from_phi(phi)))


@pytest.mark.parametrize("max_angle", [1e-8, 1e-3, 1.0, np.pi - 1e-3, np.pi - 1e-7])
def test_log_inverts_exp(max_angle):
    for _ in range(20):
        phi = random_phi(max_angle)
        C = g.exp_so3(phi)
        assert np.allclose(g.exp_so3(g.log_so3(C)), C, atol=1e-7)
        assert np.linalg.norm(g.log_so3(C)) <= np.pi + 1e-9


# --- 1c ---------------------------------------------------------------------
def test_quaternion_product_matches_mujoco():
    for _ in range(20):
        p, q = mj_quat_from_phi(random_phi()), mj_quat_from_phi(random_phi())
        res = np.zeros(4)
        mujoco.mju_mulQuat(res, p, q)
        assert np.allclose(g.quat_mul(p, q), res)


def test_quaternion_and_matrix_agree():
    for _ in range(20):
        p, q = mj_quat_from_phi(random_phi()), mj_quat_from_phi(random_phi())
        assert np.allclose(g.C_from_quat(p), mj_mat_from_quat(p))
        # composition: xi_AC = xi_AB (x) xi_BC  <->  C_AC = C_AB C_BC
        assert np.allclose(g.C_from_quat(g.quat_mul(p, q)), g.C_from_quat(p) @ g.C_from_quat(q))
        # the inverse is the conjugate
        assert np.allclose(g.quat_mul(p, g.quat_conj(p)), [1, 0, 0, 0])


def test_quat_from_C_up_to_sign():
    for _ in range(20):
        q = mj_quat_from_phi(random_phi())
        qq = g.quat_from_C(mj_mat_from_quat(q))
        assert np.allclose(qq, q) or np.allclose(qq, -q)
        assert qq[0] >= 0


# --- 1d ---------------------------------------------------------------------
def test_transformations():
    C_AB, A_r_AB = g.Cz(np.pi / 2), np.array([1.0, 2.0, 0.0])
    T_AB = g.T_from(C_AB, A_r_AB)
    assert np.allclose(g.apply_point(T_AB, [1.0, 0, 0]), [1, 3, 0])       # Lecture 03 written assignment, A2
    assert np.allclose(g.apply_direction(T_AB, [1.0, 0, 0]), [0, 1, 0])
    T_BA = g.T_inv(T_AB)
    assert np.allclose(T_BA[:3, 3], [-2, 1, 0])
    assert np.allclose(T_AB @ T_BA, np.eye(4))
