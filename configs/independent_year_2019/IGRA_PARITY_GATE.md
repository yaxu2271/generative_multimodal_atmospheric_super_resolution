# IGRA Parity Gate

This gate is frozen before the complete 994-row station-list rebuild is
inspected. The one malformed station-list footer row is excluded by requiring
an 11-character alphanumeric IGRA station identifier, leaving 993 valid period-
of-record archives active in 2019 or 2020.

The NOAA IGRA v2.2 rebuild may differ from the historical archive used to make
DJ's 2020 pickle because IGRA is retrospectively quality controlled. Therefore,
the gate tests data structure, coordinate overlap, transformations, and
numerical agreement on shared records rather than requiring byte identity.

## Required checks

1. The rebuilt pickle has the DJ nested layout and all 13 channels.
2. The 2020 pickle has 1,464 six-hourly slots; the 2019 pickle has 1,460.
3. All 24 prespecified 00 UTC dates contain finite observations.
4. Coordinate-matched 2020 reference-pair coverage is at least 95%.
5. At least 99% of coordinate-matched normalized values agree within `1e-6`.
6. The median normalized absolute difference is at most `1e-6` and the 95th
   percentile is at most `1e-5`.
7. Transformations match the historical notebook: temperature to kelvin,
   geopotential height multiplied by 9.8, meteorological wind components, and
   specific humidity from temperature/dewpoint depression.

If any numerical threshold fails, GPU calibration remains blocked until the
difference is traced to either a parser defect or a documented IGRA archive
revision.
