# Output files

* `identity.json`: public protocol/settings and immutable experiment identity, excluding credentials.
* `baseline/baseline.json`: target/control per-image scores and normalization anchors.
* `requests.jsonl`: completed responses keyed by request-content hash; no API keys or image payloads.
* `attempts.jsonl`: durable HTTP-attempt identifiers, including uncertain failures.
* `usage.json`: recorded request counts and token usage.
* `training/epoch_*/`: frozen input manifest, saved panel, references, scores CSV/JSON and observation/decision.
* `training/epoch_summary.csv`, `training/boxplot_statistics.csv`: aggregates, decisions and quartiles.
* `training/boxplots.png`, `boxplots.pdf`, `report.html`: configured training graphics.
* `training/curves.png`, `curves.pdf`: controller curves and confirmation counter, always generated with training reports in the selected formats and also embedded in HTML.
* `inference/<id>/`: per-image scores, references, summary and real-versus-generated boxplots.
* `best/checkpoint.bin`, `best/selection.json`: selected checkpoint and selection metadata.

`reporting.yaml` controls `boxplot_every_epochs` (default 50), `include_first_epoch`, `formats`, `dpi` and `render_every_observations`. Controller curves are always generated alongside the boxplots. Evaluation cadence is separately controlled in `policy.yaml`. A box represents one epoch's images, not the temporal smoothing window. Box edges are Q1/Q3, the center is the median, whiskers use 1.5 IQR and external points are outliers. Means are shown for D/P because their controller aggregates use means.

`report` uses completed files only and never starts API requests. Modifying reporting settings does not invalidate scores. Review paths and source image identifiers before sharing output folders.
