# Lab 02: Programming — Transformations and Velocities

The assignment is Part II of `new/sheets/02_transforms.qmd` (also rendered as HTML).
Students work in `new/sheets/dist/02`; the source directory
`new/sheets/02` contains instructor solutions.
From the repository root, build the starter with:

```sh
python3 new/sheets/make_starter.py 02
```

Target: about 5 hours, excluding setup.

Inside the starter, run `uv sync`, then work in the order of Lecture 03 Parts I–V:

1. Task 1 (Lecture Part I): implement point/direction transformations (`T_from`, `T_inv`, `apply_point`, `apply_direction`) and run `uv run pytest tests/test_geometry.py`.
   All rotation utilities, the URDF parser, and forward kinematics (`im560/panda.py`) are supplied.
2. Task 2 (Lecture Part I): implement `camera_reading` in `tasks/t3_viewer.py`; viewer updates are supplied. Run `uv run python tasks/t3_viewer.py` and verify the camera measurement.
3. Task 3 (Lecture Part II): implement the exponential step inside `integrate` in `tasks/t5_integration.py`; all other updates and sensor samples are supplied. Run `uv run python tasks/t5_integration.py` and compare orientation error and loss of orthogonality separately.
4. Task 4 (Lecture Parts II, III & IV): implement `twist_from_axes`, `shift_twist`, and `velocity_representations` in `im560/velocity.py`. Run `uv run pytest tests/test_panda.py -k "velocity or reference"` and `uv run pytest tests/test_velocity_conventions.py`.
5. Task 5 (Lecture Parts IV & V): implement `exp_se3`, `log_se3`, `pose_interp_decoupled`, `pose_interp_screw`, and `integrate_body_twist` in `im560/geometry.py`. Run `uv run pytest tests/test_se3.py`.
6. Optional `:pepper:` Bonus Tasks A & B: implement `screw_axis_from_twist` and `camera_reading_rate` in `im560/velocity.py` and `calibrate_hand_eye` in `im560/geometry.py`. Run `uv run pytest tests/test_bonus.py`.

For the complete required test run, use `uv run pytest`.
The numerical core uses JAX; the supplied scan loop handles integration.
Use `jax.numpy as jnp` for numerical work. Arrays are immutable: use `.at[...].set(...)` to construct or update blocks. Float64 is enabled by `im560`. Use `jax.lax` for loops or branches that depend on traced values; parsing, plotting, viewer callbacks, and MuJoCo interfaces may use Python/NumPy. Convert with `np.asarray` only at those interfaces. Parse joint metadata once, outside JIT, and close over it, for example `jax.jit(lambda q: fk(q, joints))`.

Tests for later tasks can fail until those tasks are implemented.
MuJoCo's Panda model is downloaded once through `robot_descriptions`.
For a local checkout, set `MENAGERIE_PATH=/path/to/mujoco_menagerie`.

Required report: Q2, Q5, Q7, Q8, Q10, Q11, one screenshot, and the supplied integration plot. Other questions and `:pepper:` bonus tasks are optional.
