# Stage III-B Outer-Guard Design

This stage audits whether the completed 2019 numerical-calibration grids were
wide enough. It is specified before examining any Stage III-B output and uses
only the 24 prespecified 2019 development cases.

For each modality, one parameter is moved outside the completed Stage III grid
while the other two remain fixed at the provisional 2019 Stage III selection.
Both lower and upper guards are evaluated for each of `lambda`, `std`, and
`gamma`, giving six aircraft and six surface-station protocols.

The complete protocol settings are frozen in
`stage3b_outer_guard_manifest_v1.json`. If no guard materially improves the
prespecified 2019 selection criterion, the current Stage III selection is
retained. If a guard materially improves it, a local 2019-only expansion around
that guard is required before numerical parameters can be frozen.

Stage III-B does not authorize a joint interaction sweep, a 2020 evaluation, or
any manuscript claim that 2020 was historically unseen.
