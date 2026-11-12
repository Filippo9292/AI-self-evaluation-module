"""Copy the integration pattern into your own trainer; no framework dependency here."""
from ai_self_evaluation import TrainingMonitor


def train_with_monitor(config_path, trainer, max_epochs):
    """The trainer supplies train_epoch/save_checkpoint/generate_panel/load_checkpoint.

    generate_panel must evaluate ONE saved generator state, with fixed input/seed
    identities and filenames, without updating weights or training statistics.
    save_checkpoint must include all state needed to resume training.
    """
    monitor = TrainingMonitor(config_path)
    for epoch in range(1, max_epochs + 1):
        trainer.train_epoch(epoch)
        if monitor.due(epoch):
            checkpoint = trainer.save_checkpoint(epoch)
            images = trainer.generate_panel(checkpoint=checkpoint, epoch=epoch)
            decision = monitor.observe(epoch, images, checkpoint)
            if decision["should_stop"]:
                trainer.load_checkpoint(decision["best_checkpoint"])
                return decision
    # Budget exhausted: use the best completed observation, even without an early stop.
    decision = monitor.best()
    if decision:
        trainer.load_checkpoint(decision["best_checkpoint"])
    return decision
