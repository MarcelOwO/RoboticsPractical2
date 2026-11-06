"""Tasks 2 and 5: forward kinematics and velocities against MuJoCo and frax.

Run one task at a time, e.g.  uv run pytest -k "fk"  or  uv run pytest -k "velocity or reference".
"""
import mujoco
import numpy as np
import pytest

from im560 import panda, velocity

rng = np.random.default_rng(7)
CASES = range(5)


@pytest.fixture(scope="module")
def joints():
    return panda.parse_urdf()


def sample(joints, i):
    lo, hi = panda.joint_limits(joints)
    return np.random.default_rng(100 + i).uniform(lo, hi)


@pytest.fixture(scope="module")
def mj():
    return panda.load_mujoco_panda()


@pytest.fixture(scope="module")
def robot():
    return panda.load_frax_panda()


# --- Task 2 -----------------------------------------------------------------
def test_parse_urdf(joints):
    chain = panda.arm_chain(joints)
    lo, hi = panda.joint_limits(joints)
    assert [j.kind for j in chain] == ["revolute"] * 7 + ["fixed"] * 2
    assert np.allclose(chain[0].xyz, [0, 0, 0.333])
    assert np.isclose(hi[3], -0.0698)


def test_fk_zero_pose(joints):
    T = panda.fk(np.zeros(7), joints)
    # all joints at zero: the flange is 0.088 m in front of the axis of joint 1
    assert np.allclose(T["panda_hand"][:3, 3], [0.088, 0, 0.333 + 0.316 + 0.384 - 0.107])


@pytest.mark.parametrize("i", CASES)
def test_fk_matches_mujoco(mj, joints, i):
    q = sample(joints, i)
    model, data = mj
    data.qpos[:7] = q
    mujoco.mj_forward(model, data)
    T = panda.fk(q, joints)
    for name_mj, name_urdf in [("link%d" % k, "panda_link%d" % k) for k in range(1, 8)] + [("hand", "panda_hand")]:
        b = model.body(name_mj).id
        assert np.allclose(T[name_urdf][:3, 3], data.xpos[b], atol=1e-9)
        assert np.allclose(T[name_urdf][:3, :3], data.xmat[b].reshape(3, 3), atol=1e-6)


@pytest.mark.parametrize("i", CASES)
def test_fk_matches_frax(robot, joints, i):
    q = sample(joints, i)
    import jax.numpy as jnp
    T = panda.fk(q, joints)
    assert np.allclose(np.asarray(robot.ee_transform(jnp.asarray(q))), T["tcp"], atol=1e-9)


# --- Task 5 -----------------------------------------------------------------
@pytest.mark.parametrize("i", CASES)
def test_angular_velocity_three_ways(mj, joints, i):
    q = sample(joints, i)
    qdot = rng.normal(size=7)
    w_fd = velocity.omega_finite_difference(q, qdot, joints)
    tw = velocity.twist_from_axes(q, qdot, joints)
    tw_mj = velocity.twist_mujoco(*mj, q, qdot)
    assert np.allclose(w_fd, tw[3:], atol=1e-4)
    assert np.allclose(tw, tw_mj, atol=1e-9)


@pytest.mark.parametrize("i", CASES)
def test_reference_point(robot, joints, i):
    q = sample(joints, i)
    qdot = rng.normal(size=7)
    T = panda.fk(q, joints)
    tw_hand = velocity.twist_from_axes(q, qdot, joints, "panda_hand")
    tw_tcp = velocity.twist_frax(robot, q, qdot)
    r = T["tcp"][:3, 3] - T["panda_hand"][:3, 3]
    assert np.allclose(tw_hand[3:], tw_tcp[3:], atol=1e-9)              # same omega
    assert not np.allclose(tw_hand[:3], tw_tcp[:3], atol=1e-3)         # different v
    assert np.allclose(velocity.shift_twist(tw_hand, r), tw_tcp, atol=1e-9)
