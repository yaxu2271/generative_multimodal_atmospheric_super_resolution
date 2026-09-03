# IGRA Parity Gate v2: spatially tolerant station matching

The original v1 audit matched station coordinates after rounding latitude and
longitude to four decimals. It failed with 63.1% reference coverage, while the
matched values retained near-machine-precision agreement. A diagnostic then
showed that most unmatched reference records have a current NOAA IGRA v2.2
counterpart within a few kilometers, consistent with retrospective station
coordinate revisions rather than a variable-transform defect.

The v1 failure is retained as an audit artifact. It is not overwritten or
relabelled as a pass.

## Corrected matching definition

Version 2 uses deterministic greedy one-to-one geodesic matching independently
for each timestamp and variable. Candidate edges must be no farther than 25 km.
This tolerance is substantially smaller than the 1.40625-degree atmospheric
state grid and is used only to establish preprocessing parity; the rebuilt
2019 pickle retains the current official NOAA coordinates.

The threshold is not changed after candidate-posterior inspection: no 2019 GPU
ranking is allowed before this gate passes.

## Required checks

1. The rebuilt pickle has the DJ nested layout and all 13 channels.
2. The 2020 pickle has 1,464 six-hourly slots; the 2019 pickle has 1,460.
3. All 24 prespecified 00 UTC dates contain finite observations.
4. One-to-one spatially matched 2020 reference coverage is at least 95% within
   25 km.
5. At least 99% of matched normalized values agree within `1e-6`.
6. The median normalized absolute difference is at most `1e-6` and the 95th
   percentile is at most `1e-5`.
7. Tight-distance coverage at 0.02, 0.1, 0.5, 1, 2, 5 and 10 km is reported,
   so the 25 km tolerance cannot conceal the actual displacement distribution.

If this versioned gate fails, GPU calibration remains blocked.
