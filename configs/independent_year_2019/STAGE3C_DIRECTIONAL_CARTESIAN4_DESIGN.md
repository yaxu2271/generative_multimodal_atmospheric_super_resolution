# Stage III-C Directional 4 x 4 x 4 Calibration

Stage III-C replaces the canceled one-axis Stage III-B guard array with a
complete directional Cartesian calibration grid for each added modality.
Only the 24 prespecified 2019 development cases may be used for ranking.

Each Stage III `3 x 3 x 3` grid receives one additional level per parameter.
The direction is frozen from the completed Stage III 2019 response surfaces:

- Aircraft lambda extends upward because the high-side Stage III marginal
  response is better than the low-side response.
- Aircraft std extends downward because the selected candidate lies on the
  lower Stage III boundary.
- Aircraft gamma extends upward because the high-side Stage III marginal
  response is better than the low-side response.
- Surface-station lambda extends downward because the low-side Stage III
  marginal response is better than the high-side response.
- Surface-station std extends downward because the selected candidate lies on
  the lower Stage III boundary.
- Surface-station gamma extends upward because the selected candidate lies on
  the upper Stage III boundary.

The resulting grids are:

- Aircraft lambda: `0.05, 0.1, 0.2, 0.4`
- Aircraft std: `1.25e-4, 2.5e-4, 5e-4, 1e-3`
- Aircraft gamma: `2e-6, 5e-6, 1e-5, 2e-5`
- Surface-station lambda: `0.05, 0.1, 0.2, 0.4`
- Surface-station std: `1.25e-4, 2.5e-4, 5e-4, 1e-3`
- Surface-station gamma: `5e-6, 1e-5, 2e-5, 4e-5`

Each full cube contains 64 combinations. The 27 combinations completed in
Stage III are reused, leaving 37 new full-quality protocols per modality and
74 total. Canceled Stage III-B tasks are not treated as reusable results.

After Stage III-C, each modality is selected using the frozen 2019 ranking,
paired-month uncertainty, target-variable guardrail, and stability checks.
No 2020 result may be consulted.
