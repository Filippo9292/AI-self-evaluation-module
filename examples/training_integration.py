from ai_self_evaluation import TrainingMonitor

def train_with_evaluation(config, epochs, train_one_epoch, save_checkpoint, generate_fixed_panel):
    monitor = TrainingMonitor(config)
    for epoch in epochs:
        train_one_epoch()
        if monitor.due(epoch):
            decision = monitor.observe(epoch, generate_fixed_panel(epoch), save_checkpoint(epoch))
            if decision["should_stop"]:
                return decision
    return None
