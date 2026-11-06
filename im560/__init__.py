"""Lecture 03: transformations and velocities of the Franka Panda."""
import os
import warnings

# float64 everywhere: frax and the comparisons in the tests need it.
os.environ.setdefault("JAX_ENABLE_X64", "1")

# frax prints CPU tuning advice on import; it does not matter for this assignment.
warnings.filterwarnings("ignore", message=r"\[frax\]")

from jax import config
config.update("jax_enable_x64", True)
