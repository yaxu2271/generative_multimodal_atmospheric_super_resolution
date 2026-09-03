# Prespecified 2019 Selection Rule

This file freezes the ranking and tie-handling logic before any 2019 posterior
ranking is inspected. The expanded protocol list is
`candidate_manifest_2019_v1.json`.

## Primary score

For candidate protocol `P`, compute each variable and timestamp relative to the
fixed 2019 IGRA-only baseline before averaging:

```text
Delta(P) = mean_{month,date,variable}
           100 * (RMSE(P) - RMSE(IGRA)) / RMSE(IGRA)
```

The primary rank is the smallest strict-CONUS all13 `Delta(P)`. This is a
unitless mean of like-with-like relative errors; RMSE values with different
physical units are never directly averaged.

## Paired tie test

- Resampling unit: calendar month, retaining the paired 1st/15th cases.
- Replicates: 10,000.
- Seed: 1701.
- Two candidates are statistically tied when the 95% paired-bootstrap interval
  of their score difference contains zero.

When tied on all13, use the modality-target score: constrained6 for aircraft
and surface3 for METAR. Joint candidates remain ranked by all13 while both
target groups are reported as guardrails.

## Deterministic residual ties

1. Source policy: retain broader audited coverage first: `all_qc`, then
   `exclude_tamdar`, then `legacy_keep015`.
2. Pressure window: prefer `around5` over `around25`, because it has smaller
   vertical representativeness mismatch.
3. Operator: prefer explicit resolution-matched multiplicity control in the
   order `V4`, `V2`, `V1`, `V4c`; V4c is last because distance weighting adds
   complexity without a separately calibrated length scale.
4. Numerical calibration: choose the smallest log-distance to the
   prespecified grid center; if still tied, prefer weaker guidance in the order
   lower gamma, lower lambda, larger std.
5. Joint interaction: choose the smallest log-distance to `(1x,1x)`, then the
   lower total lambda multiplier.

No 2020 result may be consulted to rank candidates or break a tie. A discovered
code/data defect invalidates the affected stage and must be documented before a
replacement manifest is frozen.
