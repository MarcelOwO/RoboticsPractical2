"""Numerical kernels must support tracing, batching and derivatives at identity."""
import runpy
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from im560 import geometry as g, panda, velocity


def test_rotation_derivatives_at_identity_and_half_turn_values():
    zero = jnp.zeros(3)
    for f in [g.exp_so3, g.quat_from_phi, lambda p: g.log_so3(g.exp_so3(p))]:
        assert np.all(np.isfinite(jax.jacfwd(f)(zero)))
        assert np.all(np.isfinite(jax.jacrev(f)(zero)))
    assert np.allclose(jax.jacfwd(lambda p: g.log_so3(g.exp_so3(p)))(zero), np.eye(3))
    phis = jnp.array([[0., 0., 0.], [1e-10, 0., 0.], [.2, -.3, .4], [np.pi, 0., 0.], [0., -np.pi, 0.]])
    matrices = jax.jit(jax.vmap(g.exp_so3))(phis)
    round_trip = jax.jit(jax.vmap(lambda C: g.exp_so3(g.log_so3(C))))(matrices)
    assert np.allclose(matrices, round_trip, atol=1e-10)
    assert np.all(jax.jit(jax.vmap(g.is_rotation))(matrices))


def test_fk_derivative_matches_point_velocity():
    joints = panda.parse_urdf()
    q = jnp.array([.2, -.3, .1, -1.8, .2, 1.4, .3])
    rates = jnp.array([.1, -.2, .3, .1, -.4, .2, .1])
    pose = jax.jit(lambda q: panda.fk(q, joints)['panda_hand'])
    w = jax.jit(lambda q, dq: velocity.twist_from_axes(q, dq, joints))(q, rates)
    Jp = jax.jacfwd(lambda q: pose(q)[:3, 3])(q)
    assert np.allclose(Jp @ rates, w[:3], atol=1e-10)
    Cdot = jax.jvp(lambda q: pose(q)[:3, :3], (q,), (rates,))[1]
    assert np.allclose(g.vee(Cdot @ pose(q)[:3, :3].T), w[3:], atol=1e-10)
    Vs, Vb = jax.jit(velocity.velocity_representations)(pose(q), w)
    local = jnp.array([.1, .2, -.1])
    C, r = pose(q)[:3, :3], pose(q)[:3, 3]
    assert np.allclose(Vs[:3]+jnp.cross(Vs[3:], r+C @ local), C @ (Vb[:3]+jnp.cross(Vb[3:], local)))


def test_scan_constant_body_rate_and_gradient():
    scope = runpy.run_path(str(Path(__file__).parents[1]/'tasks/t5_integration.py'))
    integrate = scope['integrate']
    C0 = g.Cx(.4)
    omega = jnp.tile(jnp.array([0., 0., 1.]), (8, 1))
    expected = jax.vmap(lambda k: C0 @ g.Cz(k*.01))(jnp.arange(8))
    for method in ['exponential', 'quaternion']:
        assert np.allclose(integrate(method, .01, C0, omega), expected)
    derivative = jax.grad(lambda w: integrate('exponential', .01, C0, w)[-1].sum())(jnp.zeros_like(omega))
    assert np.all(np.isfinite(derivative))
