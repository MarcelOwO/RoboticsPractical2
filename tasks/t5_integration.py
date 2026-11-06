"""Task 4 (Lecture Part II): track the orientation of the hand from a simulated gyroscope.

Run:   uv run python tasks/t5_integration.py

The arm follows a smooth joint trajectory q(t). A gyroscope on the hand
measures B_omega_IB at a fixed rate. Starting from the true initial
orientation, we integrate the measurements with three methods and compare
with the true orientation from forward kinematics:

    naive        C <- C (I + dt [omega]_x)
    exponential  C <- C exp([omega dt]_x)
    quaternion   xi <- normalize(xi (x) xi(omega dt))

The error is the angle of C_true^T C_estimate, in degrees.
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from im560 import geometry as g  # noqa: E402
from im560 import panda, velocity  # noqa: E402

import jax
import jax.numpy as jnp
from functools import partial

joints = panda.parse_urdf()
Q0 = np.array([0.0, -np.pi / 6, 0.0, -3 * np.pi / 4, 0.0, 5 * np.pi / 9, np.pi / 4])
AMP = np.array([0.8, 0.4, 0.6, 0.5, 1.0, 0.6, 1.5])
FREQ = np.array([0.11, 0.17, 0.13, 0.19, 0.23, 0.29, 0.31])     # Hz


def trajectory(t):
    """Joint angles and rates of a smooth test motion."""
    w = 2 * np.pi * FREQ
    return jnp.asarray(Q0) + jnp.asarray(AMP) * jnp.sin(w*t), jnp.asarray(AMP*w) * jnp.cos(w*t)


def simulate(dt, T_end, bias=(0., 0., 0.), noise=0.0, seed=0):
    """Sample true poses and midpoint body rates; dt and duration set array shapes."""
    ts = jnp.arange(0., T_end, dt)
    pose = jax.jit(lambda t: panda.fk(trajectory(t)[0], joints)["panda_hand"][:3, :3])
    def body_rate(t):
        q, qdot = trajectory(t)
        C = panda.fk(q, joints)["panda_hand"][:3, :3]
        return C.T @ (qdot @ velocity.joint_axes(q, joints)[0])
    C_true = jax.jit(jax.vmap(pose))(ts)
    omega = jax.jit(jax.vmap(body_rate))(ts + dt/2)
    omega = omega + jnp.asarray(bias) + noise*jax.random.normal(jax.random.PRNGKey(seed), omega.shape)
    return ts, C_true, omega


@partial(jax.jit, static_argnames=("method",))
def integrate(method, dt, C0, omega):
    """Scan gyro samples, recording each orientation before its update.

    method is static when compiling. Implement only the exponential step.
    """
    if method not in ("naive", "exponential", "quaternion"):
        raise ValueError("unknown integration method")
    def step(state, w):
        C, xi = state
        output = g.C_from_quat(xi) if method == "quaternion" else C
        if method == "naive":
            C = C @ (jnp.eye(3) + dt*g.skew(w))
        elif method == "exponential":
            raise NotImplementedError("see the docstring and the assignment text")
        else:
            xi = g.quat_mul(xi, g.quat_from_phi(w*dt))
            xi = xi / jnp.linalg.norm(xi)
        return (C, xi), output
    initial = (jnp.asarray(C0, dtype=float), g.quat_from_C(C0))
    return jax.lax.scan(step, initial, jnp.asarray(omega))[1]


def error_deg(C_true, C_est):
    """Projected orientation error; raw matrix orthogonality is checked separately."""
    C_true, C_est = jnp.asarray(C_true), jnp.asarray(C_est)
    U, _, Vt = jnp.linalg.svd(C_est, full_matrices=False)
    correction = jnp.broadcast_to(jnp.eye(3), U.shape).at[:, 2, 2].set(jnp.linalg.det(U @ Vt))
    rotations = U @ correction @ Vt
    return jax.vmap(lambda A, B: jnp.rad2deg(jnp.linalg.norm(g.log_so3(A.T @ B))))(C_true, rotations)


def main():
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, dt in zip(axs, (0.01, 0.001)):
        ts, C_true, omega = simulate(dt, 20.0)
        for method, col in (("naive", "tab:red"), ("exponential", "tab:blue"), ("quaternion", "tab:green")):
            estimates = integrate(method, dt, C_true[0], omega)
            e = error_deg(C_true, estimates)
            drift = jnp.linalg.norm(jnp.swapaxes(estimates, -1, -2) @ estimates - jnp.eye(3), axis=(1, 2))
            print(f"{method}: max orthogonality error = {drift.max():.2e}")
            ax.semilogy(ts[1:], e[1:], color=col, label=method, lw=2, ls="--" if method == "quaternion" else "-")
            print(f"dt = {dt:5.3f} s, {method:12s}: error after 20 s = {e[-1]:.2e} deg")
        ax.set_title(f"dt = {dt * 1000:.0f} ms, exact gyroscope")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("orientation error [deg]")
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig("t5_integration.png", dpi=120)

    # A real gyroscope: bias 0.1 deg/s about x, white noise 0.01 rad/s per sample, 1 kHz
    ts, C_true, omega = simulate(0.001, 20.0, bias=np.deg2rad([0.1, 0.0, 0.0]), noise=0.01)
    e = error_deg(C_true, integrate("exponential", 0.001, C_true[0], omega))
    print(f"real gyroscope, exponential update: error after 20 s = {e[-1]:.2f} deg")


if __name__ == "__main__":
    main()
