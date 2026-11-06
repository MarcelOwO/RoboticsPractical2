"""The supplied diagnostic separates angular error from matrix stretch."""
from pathlib import Path
import runpy
import numpy as np


def test_orientation_error_ignores_stretch_but_detects_rotation():
    # Use explicit matrices so this supplied diagnostic can be checked before
    # students implement their rotation constructors.
    scope = runpy.run_path(str(Path(__file__).parents[1] / "tasks/t5_integration.py"))
    angle = 0.2
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    truth = np.array([np.eye(3), np.eye(3)])
    estimates = np.array([np.diag([1.2, 1.1, 1.05]), R @ np.diag([1.2, 1.1, 1.05])])
    assert np.allclose(scope["error_deg"](truth, estimates), [0, np.rad2deg(angle)])
