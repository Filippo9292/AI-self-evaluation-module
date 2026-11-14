# Protocol and sampling

`protocol.yaml` contains shared `state` and three named questions: quality (five-level score), presence (noul) and relative (A/B choice). Placeholders are `{candidate}`, `{previous}`, `{positive_references}` and `{negative_references}`. The two reference placeholders resolve to five image assets each. The complete default texts and criteria are in `configs/protocol.example.yaml`.

The default quality text and five realness criteria preserve the historical realness rubric; the presence and comparison texts preserve the historical visible-defect and neutral similarity questions. Contexts remove dataset and orientation names. For a task without defects, edit the presence question, criteria and group descriptions consistently. Rebuild the baseline after any prompt change.

Baseline sample size per bank is min(N, real_max_images, max(real_min_images, ceil(real_sqrt_multiplier * sqrt(N)))). Every sampled real image is evaluated `baseline_visits` times; average those visits per image before computing anchor statistics. Generated images use `visits_per_image`. Each case/visit has a deterministic seeded reference draw, reused across epochs, with candidate-hash exclusion. Real banks are hash-deduplicated and disjoint; generated duplicate files remain separate cases.

Normalization: Q uses the median generated quality divided by the configured real-target quantile. D uses the mean generated presence minus median real-control presence, divided by the difference between median target and control presence. Baselines with nonpositive quality reference or nonpositive presence separation fail rather than fabricating scores.

For current and previous outputs of the same case, average P(A=current) in forward order and P(B=current) in reverse order. Then average over cases. The lag is `comparison_lag_epochs`. Epoch labels and stop targets are absent from API content.
