# Early stopping and checkpoint selection

All settings live in `policy.yaml`. Evaluation cadence, smoothing horizon, comparison lag and confirmation duration are measured in **epochs**. The last three must be exact multiples of cadence.

At each scheduled audit, form Q/D/P aggregates, then trailing medians over the configured horizon, endpoints included. A complete window is required. Enter maturity when Q reaches `quality_enter_ratio` and D reaches `defect_enter_normalized`. Remain mature while Q and D meet their lower retention thresholds. Progress is absent when P is at most `progress_max`. A missing P cannot establish lack of progress.

Count elapsed consecutive epochs with maturity and no progress. Reset on a failed condition or missing cadence. Stop at `confirmation_epochs`; do not count observations as epochs. The general module evaluates one pooled dataset bank and has no orientation groups.

Defaults: 2-epoch audits, lag 50, smoothing horizon 20 (11 observations), confirmation 100 (51 observations), Q enter 1.00/retain 0.97, D enter 0.50/retain 0.40, P maximum 0.55. There is no prescribed stopping epoch and no weighted sum in the stopping rule.

Checkpoint selection prioritizes mature observations, then the smaller unsmoothed distance |Q-1|+|D-1|; ties retain the earlier selection. Q and D each have coefficient 1. P does not rank checkpoints. This selection is a configurable-code convention distinct from stopping, not proof of an absolute best model. The selected checkpoint is copied into `best/checkpoint.bin` and documented in `best/selection.json`.

Checkpoint ranking settings are in `selection.yaml`: `prefer_mature`, `quality_distance_weight`, `presence_distance_weight`. Defaults are true, 1 and 1. These do not change the stopping gates.
