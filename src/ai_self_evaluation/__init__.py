"""Image self-evaluation and external checkpoint monitoring."""

__version__ = "0.1.0"


def __getattr__(name):
    if name == "TrainingMonitor":
        from .monitor import TrainingMonitor
        return TrainingMonitor
    raise AttributeError(name)
