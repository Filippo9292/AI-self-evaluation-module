# AI Self Evaluation Module

Evaluate generated images against real examples, and use checkpoint image evaluations to stop a generative-model training run.

This Python package provides **AI Self Evaluation** for an inference folder and **AI Early Stopping** for saved checkpoints. It is independent of CycleGAN, a particular dataset, image orientation, or a single decision-model vendor. The trainer supplies a checkpoint file and a fixed generation panel; the package supplies evaluation, records, plots, a retained best checkpoint, and a stopping decision.

**Status: research prototype, version 0.1.0.** The offline workflow is implemented and tested. Live provider adapters are implemented from documented contracts but have not been qualified against paid services in this repository. The included mock demonstration is synthetic and provides no evidence of a judge's visual accuracy or of training improvement. No dataset, model weights, API keys, or thesis-specific results are bundled.

## What it does

- Builds a real-to-real baseline using randomly drawn candidates and five real reference images per visit.
- Scores each image three times by default, with fresh random reference panels, and averages the visits before constructing a box plot.
- Assigns a continuous **0-4 visual-consistency score** to every image in an inference folder.
- Evaluates a bounded panel generated from one checkpoint at each scheduled observation.
- Stops after a confirmed approach to the real-score distribution, or a plateau in that approach.
- Retains a copy of each new best checkpoint, and returns the best observed checkpoint rather than the final one.
- Saves per-image scores, visit records, quartiles, decisions, token usage and PNG/SVG/HTML reports.
- Resumes completed requests from its local journal, without recharging for cached completed visits.

The score is consistency with the chosen references and rubric. It is not a probability that an image is correct, a defect detector, a diversity measure, or a claim of downstream usefulness.

## Install

Requires Python 3.10 or newer. From a downloaded or cloned repository:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install .
```

For local development use `python -m pip install -e .`. You may put the repository inside your model directory, or install it elsewhere. Copying a directory alone does not attach it to an arbitrary training loop: use the small integration described below.

## Try the offline demo first

```bash
python -m ai_self_evaluation demo --output demo-workspace
```

This creates small synthetic images, evaluates inference, simulates successive generator snapshots, and exercises early stopping using a deliberately simple brightness-based mock evaluator. It makes **zero network calls**. Every plot/report identifies the mock data.

Open:

```text
demo-workspace/results/inference/inference/report.html
demo-workspace/results/training/report.html
```

The demo refuses to overwrite a non-empty output directory. Choose another directory to repeat it. It simulates checkpoint outputs; it does not train a GAN.

## Configure a real evaluation

Copy `configs/evaluation.example.yaml` to a private `evaluation.local.yaml`. All relative paths in that file, including optional configuration files, resolve relative to the file itself, not to the shell's working directory.

```yaml
provider: jev
api_key: "YOUR_API_KEY"
real_image_dirs: ["dataset/train/target", "dataset/validation/target"]
generated_image_dir: "generated"
output_dir: "outputs/my-experiment"
```

The real directories are recursively scanned and pooled. Include only the real class/domain you want the generator to reproduce. Exact duplicate real files are excluded using SHA-256. Generated files are all retained: each filename receives its own score row even when contents are identical. PNG, JPEG, WebP, BMP and TIFF are accepted and converted to RGB PNG for the API.

The private YAML may contain the API key directly. `evaluation.local.yaml` and `*.local.yaml` are ignored by Git. Output files never serialize the key; keep the private source YAML out of screenshots or uploads. Alternatively, leave `api_key` empty and set the environment variable named in the provider preset. `.env.example` documents names; automatic `.env` loading is not enabled.

Use at least six unique real images. An image is excluded from its own reference panel by its original file hash. This does not establish independence between physical specimens, near-duplicates, or acquisitions. Keep a final test set out of the real-image pool when it is intended as an independent final evaluation.

## State, questions and rubric

The default protocol is in `configs/protocol.example.yaml` and in the installed package. It preserves the original single-image consistency question:

> How visually consistent is candidate C with the appearance of defects and background in the known-real references R1-R5?

It defines `state`, one `score` question, contextual instructions, five reference placeholders, and five ordered criteria. It introduces no FDM, CycleGAN or orientation-specific terms. The inherited wording **does mention defects and background**: for a different task the user may edit it, but must build a new baseline.

Point `protocol_file` at your copy, or put the full protocol under `evaluation:` in the main YAML. Placeholders are `{candidate_image}` and `{reference_1}` through `{reference_5}`. The provider adapter resolves these into the appropriate image references. The package requires one five-level score question in this release; it does not silently turn an arbitrary prompt into a numerical quality measure.

See [protocol and statistics](docs/protocol.md) for sampling, scoring and the distance definition.

## Plan and build the real baseline

```bash
python -m ai_self_evaluation plan --config evaluation.local.yaml --max-epochs 300
python -m ai_self_evaluation baseline --config evaluation.local.yaml
```

`plan` is read-only, requires no key, and makes no API calls. It estimates baseline, inference and scheduled checkpoint calls, excluding retries and reuse. The baseline is reused by subsequent commands with the same output directory. Both evaluation commands build it automatically if necessary.

For a small live qualification, use a separate configuration/output directory and a small real folder. Inspect the responses, actual model identifier, image capability, score distributions and image judgments before a larger run. A valid response schema does not verify visual correctness.

## Evaluate inference images

```bash
ai-self-evaluation --config evaluation.local.yaml --images generated --id final
```

Equivalent:

```bash
python -m ai_self_evaluation evaluate --config evaluation.local.yaml --images generated --id final
```

Omit `--images` to use `generated_image_dir` from the YAML. Explicit CLI paths resolve from the shell's working directory. Every image gets three visits by default, a mean score and an across-visit standard deviation. There is no automatic filtering or deletion. Use a new `--id` for another image set; changing an existing set under the same ID is rejected.

Results: `outputs/my-experiment/inference/final/`. Open `report.html` for the real/generated box plots and score table. `scores.csv` and `scores.json` contain per-file scores; `comparison.json` contains distribution summaries. PNG and SVG figures can be reused in reports.

## Connect early stopping to your trainer

```python
from ai_self_evaluation import TrainingMonitor

monitor = TrainingMonitor("evaluation.local.yaml")
for epoch in range(1, max_epochs + 1):
    trainer.train_epoch(epoch)
    if not monitor.due(epoch):
        continue
    checkpoint = trainer.save_checkpoint(epoch)
    images_dir = trainer.generate_fixed_panel(checkpoint, epoch)
    decision = monitor.observe(epoch, images_dir, checkpoint)
    if decision["should_stop"]:
        trainer.load_checkpoint(decision["best_checkpoint"])
        break
```

The `trainer` methods are integration points, not package APIs. No particular framework is imported. See the complete [integration example](examples/training_integration.py) and [integration contract](docs/integration.md), including how to restore the best at the ordinary epoch budget.

The panel must come from **one saved checkpoint**, use fixed input/seed identities and stable filenames/count, and run without updating weights or normalization statistics. The package verifies filename/count stability, but cannot infer whether the trainer used the correct weights or input seeds. It does not capture fake images from evolving minibatch states.

Checkpoint integration accepts a **single existing checkpoint file** in this release. Save model/optimizer/scheduler/RNG state together if resuming is required. The package copies each new best into its own output folder before committing the decision; it never deletes source checkpoints. A multi-file trainer can serialize one archive before observation. Keep evaluation outputs outside the input-image folders.

For script-based integration:

```bash
ai-early-stopping --config evaluation.local.yaml --epoch 30 --images monitoring/epoch_030 --checkpoint checkpoints/epoch_030.pt
```

The command prints JSON. The trainer must read `should_stop`, stop normally, and load `best_checkpoint`. Add `--stop-exit-code` to receive exit code **10** on a stopping decision; normal operation returns 0, errors return 2. Copying the module or executing this command does not forcibly kill a separate training process.

## Settings and stopping rules

Packaged defaults appear in `configs/advanced.example.yaml`. Copy it and set `advanced_file`, or override selected values under `settings:` in the main YAML.

| Setting | Default |
| --- | ---: |
| Reference images per visit | 5 |
| Visits per image | 3 |
| Maximum real candidates | 256 |
| Maximum checkpoint candidates | 64 |
| Observation interval | 1 epoch |
| Minimum epoch for stopping | 30 |
| Plateau patience | 30 valid monitored epochs |
| Additional target window | 10 valid monitored epochs |
| Target confirmations | 3 consecutive in-tolerance observations |
| Minimum progress in box distance | 0.005 |
| Target tolerance in box distance | 0.025 |
| Target reset tolerance | 0.04 |

These are **engineering defaults**, not validated universal thresholds. The target window suspends plateau stopping while it is being confirmed; marked deterioration resets it. An expired unconfirmed window returns to ordinary plateau logic. The absolute best updates at every smaller distance; patience resets only after sufficient accumulated improvement. Ties keep the earlier checkpoint. Missing evaluations do not consume patience or trigger stopping.

The internal real-score bootstrap diagnostic is saved for interpretation. It does not silently change any threshold. See [policy details](docs/early-stopping.md).

## Provider presets

```bash
python -m ai_self_evaluation providers
```

| YAML name | Default model | Implementation status |
| --- | --- | --- |
| `jev` | `jev-1.14.0` | Adapter implemented; live qualification pending |
| `liquid` | `d1-ultra` | Adapter implemented; live qualification pending |
| `drex` | `drex-max-v1.1` | Adapter implemented; live qualification pending |
| `perplexity` | `pplx-decider-v1.2` | Experimental; live qualification pending |
| `mercury` | `mercury-decide` | Experimental; live qualification pending |
| `baseten` | Elephant default slug | Experimental; live qualification pending |
| `celeris` | `celeris-1.1-decision` | Experimental; six-image capability also needs qualification |
| `mock` | `mock-brightness-v1` | Offline synthetic demo only |

Provider YAML files are in `src/ai_self_evaluation/presets/providers/`. Use `provider_file` to select a private edited copy (for example another compatible Baseten slug). Changing the provider/model or preprocessing requires a new baseline. An unchanged service model identifier does not prove unchanged proprietary weights. No fallback to a different model is performed.

[Provider documentation and qualification status](docs/providers.md) lists sources, transport differences and exclusions. In particular, a four-image endpoint cannot run the six-image protocol without changing the experiment; no collage workaround is used.

## Persistence, errors and reproducibility

The output directory binds the real-file manifest, protocol, provider configuration, preprocessing, random seed and sampling settings. Changing these is rejected: select a new output directory. Early-stopping settings are frozen separately after the first training observation.

`requests.jsonl` stores image identifiers/hashes, reference order, prepared-image hashes, payload digests, raw redacted responses, per-level probabilities, latency and token usage. It omits image base64 and authorization headers. A started request is recorded before dispatch. Completed valid visits are cached; uncertain transport outcomes require review and explicit `api.retry_failed_requests: true` before resubmission. This avoids silently assuming an unreceived answer was not billed.

Only HTTP 429 is retried automatically (bounded exponential backoff). Other HTTP errors and timeouts stop evaluation. A failure interrupts the observation without a policy update. A configured `api.max_calls` is a cumulative attempt ceiling in the output directory, including retries; the default is `null`. There are no interactive budget prompts. Change the YAML ceiling before resuming if desired.

`usage.json` contains actual attempts and provider-reported token totals, not an invoice. No universal price is assumed. Writes for committed decisions/configuration are atomic, and a run lock prevents concurrent writers. After a forcibly killed process, inspect the lock PID and remove `.run.lock` only if that process has ended. An interrupted journal line must be repaired explicitly; it is never silently discarded.

```bash
python -m ai_self_evaluation report --config evaluation.local.yaml
```

Regenerates reports from completed records without new API requests. See [output schema](docs/outputs.md).

## Test and publish

```bash
python -m unittest discover -s tests -v
```

Tests cover policy semantics, target confirmation, failures without patience updates, checkpoint preservation, resume/idempotency, manifest/protocol drift, reference exclusion, score validation, transport construction, call ceilings and credential omission. GitHub Actions is configured for Windows/Linux and Python 3.10/3.12; hosted CI results are pending until publication.

The repository uses the MIT license. Before a public release, fill in author/repository/DOI metadata in `CITATION.cff`, run live qualification for whichever providers will be advertised as verified, and pin a release tag/commit for the thesis. Private configuration, generated runs and checkpoints are excluded from Git. See [contributing](CONTRIBUTING.md).
