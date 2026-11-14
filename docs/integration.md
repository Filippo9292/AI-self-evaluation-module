# Training and inference integration

Install the package and configure original-real target/control banks. The model's training dataset may contain offline augmentation; never point the real banks at that folder. The software cannot infer original status from pixels.

The trainer calls `TrainingMonitor.due(epoch)` and, when due, saves a checkpoint and generates a bounded panel. Call `observe(epoch, panel_directory, checkpoint_file)`. Stable relative filenames must identify the same source inputs or latent seeds through training. The monitor selects a deterministic bounded subset, freezes the filenames and copies images into the observation output. It compares with the stored panel at the configured earlier epoch.

The returned decision includes `should_stop`, smoothed signals and `best`. The trainer decides how to interrupt its loop. Resume the last completed observation with the same images/checkpoint without paying again. Changed completed inputs are rejected. A partial stage resumes completed cached requests; transport failures are not retried automatically.

Inference calls `Engine(Config(path)).evaluate(images_directory, identifier)` or the CLI `evaluate`. It evaluates all input files, without a temporal comparison, and writes per-image quality/presence scores and distribution plots. Use a safe stable identifier. CLI operations lock their output directories; the monitor locks its own operations.
