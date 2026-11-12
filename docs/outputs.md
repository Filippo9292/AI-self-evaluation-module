# Output schema and recovery

```text
output_dir/
  identity.json                 # evaluation protocol + dataset fingerprints
  resolved_config.json          # allow-listed public configuration, no API key
  resolved_model.json           # requested and first reported backend identifier
  real_manifest.json            # pooled real files and excluded exact duplicates
  requests.jsonl                # started/received/valid/error events
  usage.json                    # cumulative request and reported-token totals
  baseline/
    manifest.json
    status.json
    scores.csv
    scores.json
    summary.json
    resampling_diagnostic.json
    real_boxplot.png / .svg
  inference/<id>/
    manifest.json
    status.json
    scores.csv / .json
    summary.json
    comparison.json
    comparison.png / .svg
    report.html
  training/
    policy.json
    generation_panel.json
    state.json
    history.csv / .json
    distance_curve.png / .svg
    checkpoint_boxes_001.png / .svg
    report.html
    epoch_000001/
      checkpoint_source.json
      manifest.json
      status.json
      scores.csv / .json
      summary.json
      observation.json          # authoritative committed policy event
      failure.json              # present if a past attempt failed
  checkpoints/epoch_000001/<original_checkpoint_filename>
```

## Scores

`scores.csv` includes `image_id`, `file_path`, original `image_hash`, `score_mean`, descriptive `score_std_across_visits`, `visits_completed`, provider/model, `protocol_id`, `evaluation_status` and visit `call_ids`. Each completed row has all configured visits. During interruption the table may contain only completed images; `status.json` remains incomplete, so do not treat a partial table as a final distribution. Every generated file receives a row on successful completion.

`requests.jsonl` retains each successful visit's raw provider response (before semantic validation), probabilities, optional confidence, reported usage, reference order and hashes, preprocessed hashes, payload digest and timing. No image base64 or authorization headers are persisted. Inputs remain in the caller's image folders and must be archived alongside the run to reproduce it. Paths identify local files and may contain private project names; review manifests before sharing.

## Resume

Rerun the same command/configuration to reuse valid visits. A completed baseline and unchanged inference manifest are reused. A pending request may already have been billed even if its answer was not received. After reviewing the journal, explicitly enable `settings.api.retry_failed_requests: true` to permit another attempt. Restarts never infer missing answers or invent zero scores. Lower-level HTTP events can exceed completed visits because of retries or invalid responses; `usage.json` reports this distinction.

Call limits count all attempts in this output directory, including mock attempts, failures and rate-limit retries. Mock usage is explicitly marked synthetic and has zero billed/network tokens. No price conversion or invoice total is inferred.

An epoch commits only after complete scoring and successful best-checkpoint preservation. State/history/plots derive from committed observations; `report` regenerates figures without calls and requires a completed baseline. A historical `failure.json` can remain after successful recovery; the final `status.json` and `observation.json` establish completion.

Do not run concurrent commands against one output directory. An abandoned `.run.lock` requires manual PID review. An incomplete JSONL tail is reported as corruption rather than silently dropped; preserve the original and repair it before resuming.
