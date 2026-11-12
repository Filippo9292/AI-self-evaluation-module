# Trainer integration contract

The package intentionally imports no PyTorch, TensorFlow, GAN implementation, dataset loader or orientation logic. It reads images and a checkpoint file produced by the host trainer.

## Four responsibilities of the trainer

1. Finish a scheduled epoch and save a stable checkpoint file before evaluating it.
2. Generate an observation panel from those exact weights, with fixed latent seeds or source inputs, stable filenames and a stable image count.
3. Call `TrainingMonitor.observe(epoch, images_dir, checkpoint)` and act on the returned decision.
4. Restore `best_checkpoint` after early stopping or at the ordinary maximum epoch budget.

Use an evaluation/inference mode, disable gradient tracking, and avoid updates to batch-normalization statistics during panel generation. Restore the previous training mode afterward. A complete resumable checkpoint may need generator/discriminator states, optimizers, schedulers, epoch, random-number states and mixed-precision state. The module treats checkpoint bytes as opaque; use the format appropriate to your framework.

This release copies a single checkpoint file. If your model uses multiple files, archive them into one file before passing it to the monitor, and unpack when restoring. No source file is deleted. Every newly best checkpoint remains in the output tree, so provision disk space or archive runs afterward. Automatic checkpoint pruning is not implemented.

## Minimal Python pattern

See [examples/training_integration.py](../examples/training_integration.py). `TrainingMonitor.due(epoch)` checks the configured observation interval. `observe` returns:

```json
{
  "evaluated": true,
  "last_epoch": 45,
  "should_stop": false,
  "reason": "continue",
  "best_epoch": 40,
  "best_distance": 0.03,
  "best_checkpoint": "/absolute/output/checkpoints/epoch_000040/checkpoint.pt"
}
```

Additional state fields are returned for audit. Epochs are one-based positive integers. `observe` on an unscheduled epoch returns `evaluated: false` and does not inspect images or call an API. The caller must stop when a prior decision says stop.

Generation panel filenames are validated across observations, including unselected available slots. The package cannot prove that the same filenames correspond to the same latent inputs, or that images were generated from the passed checkpoint. These are trainer responsibilities. Checkpoint panels are sampled deterministically by slot to the configured cap, with fresh reference panels at each visit.

## Command-line pattern

Run the model-specific training step/panel generation first, then call:

```bash
ai-early-stopping --config evaluation.local.yaml --epoch 50 --images generated/epoch_050 --checkpoint checkpoints/epoch_050.pt --stop-exit-code
```

Read stdout JSON and return code: 0 = completed/continue (or unscheduled), 10 = stop requested when this flag is enabled, 2 = error. No signals are sent to your trainer. The original training command cannot be modified automatically by dropping a folder into its directory.

For multi-GPU or distributed training, call the observer from one coordinator only and broadcast the returned decision. Synchronize workers before snapshot/panel generation. The run lock prevents multiple writers to the same output directory; it is not a distributed process barrier.

## Future model-specific example

A CycleGAN adapter can later choose the intended translation direction and maintain one configuration/output tree per acquisition group. Nothing in the current generic module assumes two generators, adversarial losses, defective/healthy folder naming, or two orientations. There is no real CycleGAN validation claim in this release.
