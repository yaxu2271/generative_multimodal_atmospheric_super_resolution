# Analysis artifacts

This directory connects the manuscript's reported evidence to the analysis
code and compact outputs used to produce it. It is an analysis snapshot, not a
redistribution of the ERA5 fields, NOAA observations, diffusion checkpoint, or
posterior sample arrays described in `../DATA.md`.

## Contents

- `scripts/`: final annual, held-out-observation, probabilistic, dispersion,
  composition, spatial, residual-kriging, no-R sensitivity, and plotting
  analyses.
- `summary_tables/`: compact CSV outputs underlying the principal numerical
  claims in the manuscript.
- `manuscript_tables/`: the LaTeX tables included by the manuscript.

## Claim-to-artifact map

| Manuscript evidence | Analysis script | Compact output |
|---|---|---|
| Annual area-weighted R+A+S RMSE changes, per-variable effects, and 3-, 7-, 14-, and 21-day moving-block sensitivity | `analyze_annual_rmse.py`, `bootstrap_annual_rmse.py` | `area_weighted_rmse_summary.csv`, `area_weighted_rmse_14day_intervals.csv`, `area_weighted_rmse_block_sensitivity.csv` |
| 2019 area-weighted parameter-selection audit | `audit_area_weighted_2019_selection.py` | `area_weighted_2019_selection_summary.json` |
| R/A/S composition | `analyze_composition.py` | `area_weighted_rmse_summary.csv` |
| Errors at observations excluded from conditioning | `analyze_heldout_observations.py` | `heldout_family_summary.csv`, `heldout_variable_summary.csv` |
| Area-weighted CRPS and fair-CRPS analyses | `analyze_probabilistic_metrics.py`, `bootstrap_crps.py` | `area_weighted_crps_summary.csv`, `area_weighted_crps_block_sensitivity.csv` |
| Area-weighted spread-skill, coverage, and rank histograms | `analyze_spread_skill_and_rank.py` | `area_weighted_spread_skill_coverage_by_group.csv`, `area_weighted_spread_skill_coverage_by_variable.csv`, `area_weighted_rank_histogram_by_group.csv`, `area_weighted_rank_histogram_by_variable.csv` |
| Spatial, spectral, observation-geometry, field-example, and guidance diagnostics | the corresponding `plot_*` or `analyze_spatial_*` scripts | final manuscript figures are generated from external analysis inputs |
| Fixed-background residual-kriging reference | `run_residual_kriging_baseline.py` | `residual_kriging_selected_hyperparameters.json`, `residual_kriging_summary.json`, `residual_kriging_by_variable.csv` |
| Fixed-parameter no-R removal sensitivity | `analyze_no_r_sensitivity.py` | `no_r_sensitivity_24case.csv` |

## Rerun boundary

The numerical summaries are small enough to distribute directly. Recomputing
them from scratch requires the external ERA5 files, observation products,
normalization files, checkpoint, and posterior samples. Analysis inputs are
supplied through the environment variables named in each script. Numerical
analyses use subdirectories of `analysis/generated/` by default; standalone
plotting scripts write to `analysis/figures/` or `analysis/tables/` unless an
output variable is provided.

For example, the annual R+A+S analysis expects directories containing the
ERA5 fields, R-only samples, and R+A+S samples, together with the evaluation
manifest:

```bash
ERA5_ROOT=/path/to/era5 \
R_ONLY_SAMPLES_ROOT=/path/to/r_only/samples \
RAS_SAMPLES_ROOT=/path/to/ras/samples \
EVALUATION_TIMESTEP_MANIFEST=reproduction/manifests/evaluation_timesteps_2020.json \
python analysis/scripts/analyze_annual_rmse.py
```

Each standalone script names any additional required environment variable in
its startup error. Scripts with a command-line interface, such as
`analyze_composition.py` and `plot_observation_geometry.py`, list their inputs
under `--help`.

The statistical inference follows the revised manuscript: gridded scores use
cosine-latitude area weights, and paired analysis-time RMSE or CRPS changes are
summarized with moving-block bootstrap intervals.
The area-weighted RMSE and CRPS summary files retain explicitly labeled
equal-gridpoint columns only as a parity audit against the earlier analysis;
the revised manuscript reports the area-weighted columns.

The residual-kriging script uses the same prepared R/A/S observation products
as the diffusion experiments. Configure `KRIGING_RUN_ROOT`, `ERA5_ROOT`,
`IGRA_2019_PKL`, `AIRCRAFT_2019_ROOT`, `SURFACE_2019_ROOT`, `IGRA_2020_PKL`,
`AIRCRAFT_2020_ROOT`, `SURFACE_2020_ROOT`, `EVALUATION_TIMESTEP_MANIFEST`, and
`AREA_WEIGHTED_DIFFUSION_METRICS_CSV`, then run its four stages in order:

```bash
python analysis/scripts/run_residual_kriging_baseline.py background
python analysis/scripts/run_residual_kriging_baseline.py select --workers 8
python analysis/scripts/run_residual_kriging_baseline.py evaluate --chunk-id 0 --chunk-size 30
python analysis/scripts/run_residual_kriging_baseline.py aggregate
```

The evaluation stage is chunkable; all chunks must finish before aggregation.
Run the script with `--help` for the complete staged interface. The no-R
analysis accepts all input and output locations as command-line arguments; run
it with `--help` for details.
