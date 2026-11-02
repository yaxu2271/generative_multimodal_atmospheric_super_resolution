# Paper reproduction wrappers

This directory collects portable entry points for the experiments reported in
the manuscript. They call the source snapshot in `src/igra_gen` and require
users to provide all data and model paths explicitly.

## Files

- `run_2020_evaluation.py`: annual R, R+A, R+S, and R+A+S posterior-sampling
  runs, plus the fixed-parameter Prior, A, S, and A+S removal sensitivity.
- `run_2020_holdout.py`: 24-case aircraft or surface-station held-out-
  conditioning posterior-sampling run.
- `create_2020_holdout_split.py`: reproducible 80/20 cell-based split for the
  24 held-out-evaluation analysis times.
- `validate_holdout_split.py`: checks the aircraft split against the source
  reports, including its CONUS domain, partition counts, and cell disjointness.
- `build_holdout_cell_targets.py`: cell-mean evaluation targets constructed
  from the observations excluded by the two splits.
- `config/atmospheric_prior_13var.yaml`: resolved architecture and dataset
  configuration for the fixed 13-variable atmospheric prior.
- `config/selected_interface_2019.json`: selected likelihood parameters and
  interface settings.
- `manifests/evaluation_timesteps_2020.json`: the 723 annual evaluation
  indices.
- `manifests/holdout_timesteps_2020.json`: the 24 seasonally distributed
  holdout indices.

The original runs used 16 ensemble members, 50 EDM denoising steps, and base
seed 17. The annual wrappers accept `--ensemble`, `--steps`, and `--seed` so a
small smoke test can be run before launching the full calculation.

The selected aircraft interface uses MADIS `dataSource` codes 0, 1, and 5.
Both wrappers read these codes from `selected_interface_2019.json` and pass
them explicitly to the sampler.

## Running the annual evaluation

For example, launch an R+A+S posterior-sampling run with:

```bash
python reproduction/run_2020_evaluation.py \
  --configuration R+A+S \
  --checkpoint /path/to/checkpoint.pt \
  --era5-root /path/to/era5_1.40625deg \
  --igra-pkl /path/to/igra_2020.pkl \
  --aircraft-root /path/to/processed_madis_aircraft_2020 \
  --surface-root /path/to/processed_madis_metar_2020 \
  --output-root /path/to/output
```

Use `--configuration R`, `R+A`, or `R+S` for the matched comparisons. Run
`python reproduction/run_2020_evaluation.py --help` for the complete command
interface. The wrapper uses `config/atmospheric_prior_13var.yaml` by default;
`--hydra-config` can override it when another compatible resolved
configuration is required.

For the 24-case no-R removal sensitivity, pass
`reproduction/manifests/holdout_timesteps_2020.json` through `--timesteps` and
run the same wrapper with `--configuration Prior`, `A`, `S`, or `A+S`. The A
and S likelihood parameters remain the values selected with R present; these
runs are therefore removal ablations, not separately optimized no-R methods.

## Creating the held-out splits

Create separate aircraft and surface roots before running the corresponding
held-out posterior-sampling comparison:

```bash
python reproduction/create_2020_holdout_split.py \
  --source aircraft \
  --input-root /path/to/processed_madis_aircraft_2020 \
  --output-root /path/to/aircraft_holdout_root \
  --era5-root /path/to/era5_1.40625deg

python reproduction/create_2020_holdout_split.py \
  --source surface \
  --input-root /path/to/processed_madis_metar_2020 \
  --output-root /path/to/surface_holdout_root \
  --era5-root /path/to/era5_1.40625deg
```

Both splits first retain reports whose native coordinates lie within the
paper's CONUS domain. For each analysis time, the surface split excludes a
random 20% of the union of surface-observed ERA5 cells. The aircraft split
independently excludes a random 20% of observed ERA5 cells for each of the six
aircraft-targeted variables. Both use base seed 17 and write the retained
targets to `obs/`, the excluded targets to `heldout_obs/`, and an auditable
split manifest and CSV summary alongside them.

The aircraft split can be checked independently before sampling:

```bash
python reproduction/validate_holdout_split.py \
  --input-root /path/to/processed_madis_aircraft_2020 \
  --split-root /path/to/aircraft_holdout_root \
  --era5-root /path/to/era5_1.40625deg \
  --output-json /path/to/aircraft_holdout_root/validation.json
```

## Running the held-out conditioning experiments

Pass the split output root for the selected source; the wrapper reads retained
observations from its `obs/` subdirectory. The other source continues to use
its complete processed root:

```bash
python reproduction/run_2020_holdout.py \
  --holdout aircraft \
  --checkpoint /path/to/checkpoint.pt \
  --era5-root /path/to/era5_1.40625deg \
  --igra-pkl /path/to/igra_2020.pkl \
  --aircraft-root /path/to/aircraft_holdout_root \
  --surface-root /path/to/processed_madis_metar_2020 \
  --output-root /path/to/aircraft_holdout_samples

python reproduction/run_2020_holdout.py \
  --holdout surface \
  --checkpoint /path/to/checkpoint.pt \
  --era5-root /path/to/era5_1.40625deg \
  --igra-pkl /path/to/igra_2020.pkl \
  --aircraft-root /path/to/processed_madis_aircraft_2020 \
  --surface-root /path/to/surface_holdout_root \
  --output-root /path/to/surface_holdout_samples
```

Build the common cell-mean target table used by the held-out analysis with:

```bash
python reproduction/build_holdout_cell_targets.py \
  --aircraft-excluded-root /path/to/aircraft_holdout_root/heldout_obs \
  --surface-excluded-root /path/to/surface_holdout_root/heldout_obs \
  --spatial-domain CONUS \
  --output-csv /path/to/holdout_cell_targets.csv
```

The sampling wrappers generate posterior samples and per-run metadata. They do not
perform the downstream metric aggregation, bootstrap analysis, or figure and
table generation reported in the manuscript. The corresponding downstream
scripts and compact reported summaries are documented in
[`../analysis/README.md`](../analysis/README.md).
