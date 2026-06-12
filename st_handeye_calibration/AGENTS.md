# Repository Guidelines

## Project Structure & Module Organization

This directory contains a Python hand-eye calibration implementation. Core package code lives in `st_handeye/`: camera models, board configuration, I/O, optimization, evaluation, and the high-level calibrator. The main CLI entry point is `calibrate.py`. Pose and frame diagnostics are in `analyze_poses.py` and `analyze_frame_error.py`. Sample inputs and generated visualizations live under `data/`; avoid committing new bulky image outputs unless they are needed as fixtures. `_backup_old/` is historical reference code, not the active implementation.

## Build, Test, and Development Commands

Create an isolated environment before installing dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run calibration on the bundled sample data:

```bash
python calibrate.py data data/poses.csv -c data/camera_params.yaml --visualize
```

Use diagnostics when validating pose quality or reprojection behavior:

```bash
python analyze_poses.py
python analyze_frame_error.py
```

There is currently no package build step or formal test command.

## Coding Style & Naming Conventions

Use Python 3 with 4-space indentation. Keep modules small and domain-focused, following the existing names in `st_handeye/` such as `camera.py`, `optimizer.py`, and `calibrator.py`. Use `snake_case` for functions, variables, and file names; use `PascalCase` for classes such as `HandEyeCalibrator`, `BoardConfig`, and `OptimizationParams`. Prefer NumPy and SciPy primitives for matrix and transform operations instead of ad hoc list math. Keep CLI options explicit and descriptive.

## Testing Guidelines

No automated test suite is present. For changes, validate against `data/` with `calibrate.py` and inspect the generated YAML plus optional reprojection images in `data/reprojection/`. When adding tests, place them under `tests/`, name files `test_*.py`, and prefer small deterministic fixtures over generated calibration artifacts. Cover board detection, pose loading, transform math, and optimizer output invariants.

## Commit & Pull Request Guidelines

Recent history uses conventional-style commit prefixes, especially `feat:` and `fix:`. Keep messages concise and scoped, for example `fix: correct pose error weighting` or `feat: add detection export option`. Pull requests should include a short problem statement, implementation summary, commands run, and any calibration metrics or before/after reprojection evidence. Link related issues when available and mention changes to data formats, camera parameters, or CLI flags.

## Security & Configuration Tips

Do not commit private camera calibration files, robot logs, or patient/production imagery. Keep local outputs such as `calibration_result*.yaml`, detection images, and reprojection images out of commits unless intentionally adding reference artifacts.
