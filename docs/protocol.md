# Protocol and statistics

## Evaluation unit

A visit evaluates one candidate against five distinct real references. Three visits with separately drawn reference panels are the default. Candidate hash exclusion is applied to real and generated candidates, including exact copies of real files. An eligible pool must contain at least five references after exclusion. The baseline therefore requires at least six unique real files.

Real folders are pooled; the software does not infer class labels or acquisition groups. Supply a homogeneous target-domain pool, or run separate configurations/output directories for separate groups. No orientation-specific behavior is built in. A held-out test set should remain outside a pool used for stopping decisions.

SHA-256 deduplication detects identical bytes, not visually identical re-encodings or related physical specimens. Independence of samples and correct target-class membership remain dataset responsibilities.

## State and questions

The editable YAML includes a `state` object and one named `score` question with `instructions` and five `criteria`. The default question and rubric reproduce the earlier consistency experiment; its context omits acquisition angle and FDM terminology. The wording still refers to defects/background. Edit all relevant fields for another domain and rebuild the baseline.

The renderer replaces `{candidate_image}` and `{reference_1}` to `{reference_5}`. The provider adapter changes the transport, not rubric meanings. One candidate is sent per API request to avoid introducing cross-candidate context. The result must contain a score in [0,4], probabilities for levels 0 through 4, and a compatible probability-weighted score. Provider confidence is optional and is not substituted for the score.

## Sampling

For N unique eligible real images, the default candidate count is:

```text
n_real = min(N, 256, max(32, ceil(4 * sqrt(N))))
```

For M generated panel images at a checkpoint:

```text
n_checkpoint = min(M, n_real, 64)
```

These are bounded engineering heuristics, not statistical sample-size guarantees. The entire inference directory is evaluated, without the checkpoint cap.

Pseudo-random draws use a stored seed and deterministic stage/image/visit keys. Each visit samples five distinct real references without replacement within the panel. Panels are sampled again for each visit and checkpoint; overlapping references across visits are allowed. Restarting the same planned request reproduces its original panel. Each checkpoint uses stable generation slots, but new reference draws.

Three visits are averaged per image. An incomplete image is not used in a completed distribution; the whole checkpoint must complete before its policy state advances. Per-image standard deviation is descriptive variation across the three visits. Because panels change, this variation combines contextual sensitivity and service variation. It does not isolate API stochasticity. This first release does not automatically run an additional fixed-panel repeatability experiment.

## Preprocessing

Files are decoded, EXIF orientation is applied, and RGB PNG bytes are prepared. The general default preserves native dimensions up to a maximum side of 384 pixels, keeping six inline images within conservative request-size limits. Larger images are reduced while keeping aspect ratio. The JEV preset additionally sets a minimum side of 280 pixels to meet its documented input contract; smaller images are enlarged with the configured interpolation (nearest by default). This enlargement creates no new detail. If the aspect ratio cannot satisfy both limits, preparation fails explicitly rather than silently distorting or cropping. An explicit `[width, height]` resize and interpolation can be configured. The same transform is used for real and generated images. No photometric enhancement is applied. Both original-file and prepared-byte hashes are recorded. Internal provider preprocessing is not observable through this package.

Settings precedence is general defaults, provider preprocessing defaults, optional advanced YAML, then main-YAML `settings` overrides. A dataset-specific exact 128-to-384 nearest-neighbor replica can be selected later with `resize: [384, 384]`; this is not hard-coded into the generic module.

## Box plots and distance

All summaries use one mean score per distinct real candidate or generated file. Quantiles use NumPy's default linear interpolation. Box: Q25 and Q75; central line: median; whiskers: extreme observed values within 1.5 IQR of the box; outliers are saved and plotted. Whiskers are not confidence intervals.

The stopping distance is:

```text
D = (abs(Q25_generated - Q25_real)
   + abs(Q50_generated - Q50_real)
   + abs(Q75_generated - Q75_real)) / 12
```

It lies in [0,1] for scores in [0,4]. Smaller is closer. It summarizes the center of the score distribution, not complete image-distribution similarity. Matching quartiles can hide different tails or diversity. Q10, Q90, extrema and outliers remain in the output, but do not automatically control the policy.

The real-score distribution is a reference, not an optimum or upper bound. Under the requested closest-reference policy, scores above the real range may also increase the distance. This is intentional and must be assessed for the application. A broad/narrow box alone does not prove generation diversity or mode collapse.

## Internal resampling diagnostic

Baseline construction bootstraps the recorded real per-image means into panels of the configured checkpoint size and reports median/P95 distances to the full baseline. It makes no extra API calls, does not create independent observations and is not a population equivalence test. Its suggested noise scale does not automatically replace the fixed YAML thresholds. A different/smaller checkpoint sample size may have different variability.

Qualification against clearly incompatible/poor examples is recommended for a real experiment. Real-to-real agreement alone cannot establish useful discrimination. No downstream utility or defect classification result is inferred from this consistency score.
