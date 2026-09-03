# Paper reproduction wrappers

This directory collects portable entry points for the experiments reported in
the manuscript. They call the source snapshot in `src/igra_gen` and require
users to provide all data and model paths explicitly.

## Files

- `run_2020_evaluation.py`: annual R, R+A, R+S, and R+A+S conditioning runs.
- `run_2020_holdout.py`: 24-case aircraft or surface-station holdout run.
- `config/selected_interface_2019.json`: selected likelihood parameters and
  interface settings.
- `manifests/evaluation_timesteps_2020.json`: the 723 annual evaluation
  indices.
- `manifests/holdout_timesteps_2020.json`: the 24 seasonally distributed
  holdout indices.

The original runs used 16 ensemble members, 50 EDM denoising steps, and base
seed 17. The annual wrappers accept `--ensemble`, `--steps`, and `--seed` so a
small smoke test can be run before launching the full calculation.

