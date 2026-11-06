"""Task 5d: point versus axes, with no robot model or download."""
import numpy as np
from im560.velocity import velocity_representations


def test_offset_rotation():
    T = np.eye(4)
    T[:3, 3] = [1, 0, 0]
    # Axis through (0, 0.7, 0), angular rate 0.6 rad/s.
    w = np.array([0.42, 0.6, 0, 0, 0, 0.6])
    spatial, body = velocity_representations(T, w)
    assert np.allclose(spatial, [0.42, 0, 0, 0, 0, 0.6])
    assert np.allclose(body, w)


def test_same_material_point_velocity_from_both_twists():
    T = np.array([[0, 0, 1, 1], [0, 1, 0, -2], [-1, 0, 0, 0.5], [0, 0, 0, 1.]])
    w = np.array([0.2, -0.4, 0.1, 0.3, 0.1, -0.5])
    spatial, body = velocity_representations(T, w)
    C, r = T[:3, :3], T[:3, 3]
    for q in [np.zeros(3), np.array([0.3, -0.1, 0.2]), -C.T @ r]:
        p = r + C @ q
        expected = w[:3] + np.cross(w[3:], p - r)
        assert np.allclose(spatial[:3] + np.cross(spatial[3:], p), expected)
        assert np.allclose(C @ (body[:3] + np.cross(body[3:], q)), expected)
