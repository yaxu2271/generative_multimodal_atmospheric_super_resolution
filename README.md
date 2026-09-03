# Generative Atmospheric Super-Resolution with Heterogeneous Observations

Research code supporting the manuscript **Generative Atmospheric
Super-Resolution across Heterogeneous Observing Systems through Composable
Interfaces**.

The repository implements diffusion posterior sampling with a fixed
13-variable atmospheric diffusion prior and three observation sources:

- **R**: IGRA radiosonde profiles
- **A**: NOAA MADIS aircraft reports
- **S**: NOAA MADIS METAR surface-station reports

The observation interfaces specify which measurements are retained, how they
are mapped to the gridded state, how residuals are counted, and how each
source contributes to the likelihood. The paper develops and calibrates these
interfaces with 2019 observations and evaluates the selected configuration at
723 analysis times in 2020.

## Repository status

This is a public research snapshot prepared from source commit
`5875fe981a77a00a3d1392d24f84b8285cf48a41`, the code revision recorded for
the paper experiments. The public wrappers in `reproduction/` expose data and
checkpoint locations as command-line arguments; no Purdue filesystem layout
is required by those wrappers.

The repository does **not** include ERA5 fields, NOAA observations, processed
observation products, trained model weights, or generated posterior samples.
See [DATA.md](DATA.md) for the expected inputs and their public providers.

## Main entry points

- `src/igra_gen/run_aircraft_13var_persistent.py`: persistent posterior
  sampler and R/A/S observation operators
- `scripts/preprocess_madis_aircraft_13var_npz.py`: MADIS aircraft processing
- `scripts/preprocess_madis_metar_13var_npz.py`: MADIS METAR processing
- `scripts/download_igra_noaa_por.py` and
  `scripts/build_igra_from_noaa_por.py`: IGRA acquisition and construction
- `scripts/run_independent_year_2019_*.py`: 2019 interface development and
  likelihood-parameter selection
- `reproduction/run_2020_evaluation.py`: R-only, R+A, R+S, and R+A+S annual
  evaluation wrapper
- `reproduction/run_2020_holdout.py`: aircraft and surface-station held-out
  evaluation wrapper

## Environment

The experiments used Python 3.11 and PyTorch on NVIDIA A100 GPUs. Create an
environment and install the Python dependencies with:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
```

The trained atmospheric prior is required for posterior sampling but is not
distributed in this repository. The Hydra configuration supplied with that
checkpoint and the ERA5 normalization files must be provided to the public
evaluation wrappers.

## Reproducing the 2020 conditioning comparisons

The selected observation-interface settings are recorded in
`reproduction/config/selected_interface_2019.json`, and the 723 evaluation
indices are in `reproduction/manifests/evaluation_timesteps_2020.json`.

For example, an R+A+S run is launched as follows:

```bash
python reproduction/run_2020_evaluation.py \
  --configuration R+A+S \
  --checkpoint /path/to/checkpoint.pt \
  --hydra-config /path/to/checkpoint_hydra_config.yaml \
  --era5-root /path/to/era5_1.40625deg \
  --igra-pkl /path/to/igra_2020.pkl \
  --aircraft-root /path/to/processed_madis_aircraft_2020 \
  --surface-root /path/to/processed_madis_metar_2020 \
  --output-root /path/to/output
```

Use `--configuration R`, `R+A`, or `R+S` for the matched comparisons. Run
`python reproduction/run_2020_evaluation.py --help` for the complete interface.

## Scope and provenance

This snapshot preserves the research implementation used for the paper. The
paper workflow is identified explicitly above and in
`reproduction/README.md`. Historical development manifests may contain the
original Purdue filesystem paths as provenance strings. They do not contain
the referenced data.

Third-party code embedded in individual source files retains its original
copyright and license notices. No project-wide license has yet been assigned;
all rights not covered by those notices are reserved by the authors.

## Authors

Yang Xu, Dibyajyoti Chakraborty, Haiwen Guan, Sen Wang, and Romit Maulik.
