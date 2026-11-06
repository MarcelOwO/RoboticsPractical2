"""Rotations and rigid transformations in ETH notation.

Conventions (see the notation reference):
    A_r = C_AB @ B_r                 rotation matrix C_AB maps B-coordinates to A-coordinates
    xi_AB = (w, x, y, z)             unit quaternion, scalar first, Hamilton product
    T_AB = [[C_AB, A_r_AB], [0, 1]]  homogeneous transformation
    phi_AB = theta * n               rotation vector, C_AB = exp([phi_AB]_x)

All functions take and return JAX arrays of float64.
"""
import jax
import jax.numpy as jnp

EPS = 1e-9


# ---------------------------------------------------------------------------
# Supplied from Lecture 02: skew matrices and elementary rotations
# ---------------------------------------------------------------------------
def skew(a):
    """[a]_x, the matrix with skew(a) @ b == jnp.cross(a, b)."""
    return jnp.array([[0.0, -a[2], a[1]],
                     [a[2], 0.0, -a[0]],
                     [-a[1], a[0], 0.0]])


def vee(S):
    """Inverse of skew: the vector a with skew(a) == S (S skew-symmetric)."""
    return jnp.array([S[2, 1], S[0, 2], S[1, 0]])


def Cx(a):
    """Elementary rotation about x by the angle a (rad)."""
    c, s = jnp.cos(a), jnp.sin(a)
    return jnp.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def Cy(a):
    """Elementary rotation about y by the angle a (rad)."""
    c, s = jnp.cos(a), jnp.sin(a)
    return jnp.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def Cz(a):
    """Elementary rotation about z by the angle a (rad)."""
    c, s = jnp.cos(a), jnp.sin(a)
    return jnp.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def C_from_rpy(rpy):
    """Rotation matrix of a URDF <origin rpy="roll pitch yaw">.

    URDF turns about the FIXED axes x, then y, then z, which is the same matrix
    as Cz(yaw) @ Cy(pitch) @ Cx(roll).
    """
    roll, pitch, yaw = rpy
    return Cz(yaw) @ Cy(pitch) @ Cx(roll)


def is_rotation(C, tol=1e-9):
    """JAX boolean: orthogonality and determinant +1, with absolute tolerance."""
    C = jnp.asarray(C, dtype=float)
    if C.shape != (3, 3):
        return jnp.asarray(False)
    return jnp.allclose(C.T @ C, jnp.eye(3), atol=tol, rtol=0) & (jnp.abs(jnp.linalg.det(C)-1) < tol)


# ---------------------------------------------------------------------------
# Supplied from Lecture 02: exponential and logarithm on SO(3)
# ---------------------------------------------------------------------------
def exp_so3(phi):
    """Rodrigues with a smooth small-angle branch, including derivatives at zero."""
    phi = jnp.asarray(phi, dtype=float)
    t2 = phi @ phi
    small = t2 < 1e-8
    t = jnp.sqrt(jnp.where(small, 1.0, t2))
    a = jnp.where(small, 1-t2/6+t2*t2/120, jnp.sin(t)/t)
    b = jnp.where(small, .5-t2/24+t2*t2/720, (1-jnp.cos(t))/(t*t))
    K = skew(phi)
    return jnp.eye(3) + a*K + b*(K @ K)


def log_so3(C):
    """Principal rotation vector; smooth at identity, sign ambiguous at pi.

    The principal logarithm has no unique derivative at a half-turn.
    Safe denominators also keep inactive branches finite under vmap.
    """
    C = jnp.asarray(C, dtype=float)
    w = .5 * vee(C-C.T)
    s2 = w @ w
    c = (jnp.trace(C)-1)/2
    s = jnp.sqrt(jnp.maximum(s2, 1e-30))
    theta = jnp.arctan2(s, c)
    def near_pi(_):
        B = (C+C.T+2*jnp.eye(3))/4
        k = jnp.argmax(jnp.diag(B))
        n = B[:, k] / jnp.sqrt(jnp.maximum(B[k, k], 1e-30))
        n = n / jnp.maximum(jnp.linalg.norm(n), 1e-15)
        n = jnp.where(n @ w < 0, -n, n)
        return theta*n
    def regular(_):
        return jnp.where((s2 < 1e-12) & (c > 0),
                         (1+s2/6+3*s2*s2/40)*w, (theta/s)*w)
    return jax.lax.cond(c < -1+1e-8, near_pi, regular, operand=None)


# ---------------------------------------------------------------------------
# Supplied from Lecture 02: unit quaternions (w, x, y, z), Hamilton convention
# ---------------------------------------------------------------------------
def quat_mul(p, q):
    """Hamilton product p (x) q."""
    p0, pv = p[0], jnp.asarray(p[1:], float)
    q0, qv = q[0], jnp.asarray(q[1:], float)
    return jnp.r_[p0 * q0 - pv @ qv, p0 * qv + q0 * pv + jnp.cross(pv, qv)]


def quat_conj(q):
    """Conjugate (w, -x, -y, -z); the inverse of a unit quaternion."""
    return jnp.r_[q[0], -jnp.asarray(q[1:], float)]


def quat_from_phi(phi):
    """Scalar-first unit quaternion; finite derivatives at zero rotation."""
    phi = jnp.asarray(phi, dtype=float)
    t2 = phi @ phi
    small = t2 < 1e-8
    t = jnp.sqrt(jnp.where(small, 1.0, t2))
    real = jnp.where(small, 1-t2/8+t2*t2/384, jnp.cos(t/2))
    scale = jnp.where(small, .5-t2/48+t2*t2/3840, jnp.sin(t/2)/t)
    return jnp.concatenate((real[None], scale*phi))


def C_from_quat(q):
    """Rotation matrix of a unit quaternion: (2 q0^2 - 1) I + 2 q0 [qv]_x + 2 qv qv^T."""
    q0, qv = q[0], jnp.asarray(q[1:], float)
    return (2 * q0**2 - 1) * jnp.eye(3) + 2 * q0 * skew(qv) + 2 * jnp.outer(qv, qv)


def quat_from_C(C):
    """Unit quaternion of a rotation matrix, with w >= 0 (one of the two choices +-q)."""
    return quat_from_phi(log_so3(C))


# ---------------------------------------------------------------------------
# Task 1: homogeneous transformations
# ---------------------------------------------------------------------------
def T_from(C_AB, A_r_AB):
    """T_AB from the rotation C_AB and the position A_r_AB of the origin of B."""
    raise NotImplementedError("see the docstring and the assignment text")


def T_inv(T_AB):
    """T_BA, computed with C^T and -C^T r (no general matrix inverse)."""
    raise NotImplementedError("see the docstring and the assignment text")


def apply_point(T_AB, B_r_BP):
    """A_r_AP from B_r_BP (homogeneous coordinate 1)."""
    raise NotImplementedError("see the docstring and the assignment text")


def apply_direction(T_AB, B_u):
    """A_u from B_u (homogeneous coordinate 0: only the rotation acts)."""
    raise NotImplementedError("see the docstring and the assignment text")


# ---------------------------------------------------------------------------
# Task 5 (Lecture Part IV): SE(3) exponential/logarithm, interpolation, odometry
# ---------------------------------------------------------------------------
def exp_se3(xi):
    """SE(3) exponential exp(xi^) for a linear-first 6-vector xi = (u, phi).

    With t = ||phi|| and K = [phi]_x, the translation is V(phi) @ u where
        V(phi) = I + ((1 - cos t) / t^2) K + ((t - sin t) / t^3) K^2,
    using a smooth Taylor branch for t^2 < 1e-8 so pure translations (phi = 0)
    and near-straight arcs have finite values and derivatives.
    """
    raise NotImplementedError("see the docstring and the assignment text")


def log_se3(T):
    """SE(3) logarithm: linear-first 6-vector xi = (u, phi) with exp_se3(xi) == T.

    Uses phi = log_so3(C) and u = V(phi)^{-1} r, where for t = ||phi|| < pi and K = [phi]_x:
        V(phi)^{-1} = I - 0.5 K + (1 / t^2 - (1 + cos t) / (2 t sin t)) K^2,
    with the Taylor series 1/12 + t^2/720 + t^4/30240 for t^2 < 1e-8.
    """
    raise NotImplementedError("see the docstring and the assignment text")


def pose_interp_decoupled(T0, T1, s):
    """Interpolate T0 -> T1 at s in [0, 1] by straight-line origin + SO(3) geodesic."""
    raise NotImplementedError("see the docstring and the assignment text")


def pose_interp_screw(T0, T1, s):
    """Interpolate T0 -> T1 at s in [0, 1] along one constant screw: T0 @ exp_se3(s * log_se3(T0^{-1} T1))."""
    raise NotImplementedError("see the docstring and the assignment text")


def integrate_body_twist(T0, V_b_samples, dt, method="exact"):
    """Integrate body-twist samples V_b[k] = (v_b, omega_b) over steps dt, returning all poses (N + 1, 4, 4).

    method == "euler": r_{k+1} = r_k + (C_k @ v_b) * dt, C_{k+1} = C_k @ exp_so3(omega_b * dt)
    method == "exact": T_{k+1} = T_k @ exp_se3(V_b * dt)
    """
    if method not in ("euler", "exact"):
        raise ValueError("unknown odometry method")
    raise NotImplementedError("see the docstring and the assignment text")


# ---------------------------------------------------------------------------
# Pepper Bonus Task B: Hand-eye calibration AX = XB
# ---------------------------------------------------------------------------
def calibrate_hand_eye(T_IE_list, T_KP_list):
    """Solve T_IE^(i) X T_KP^(i) = const for the flange-to-camera transform X = T_EK.

    Uses all pairs (i < j) with A_ij = (T_IE^(i))^{-1} T_IE^(j) and
    B_ij = T_KP^(i) (T_KP^(j))^{-1}, so A_ij X = X B_ij:
      1. Rotation: log_so3(C_A) = C_X log_so3(C_B), solved on SO(3) via SVD of sum alpha_k beta_k^T.
      2. Translation: (C_A - I) r_X = C_X r_B - r_A, solved via least squares.
    """
    raise NotImplementedError("see the docstring and the assignment text")
