"""Reference-based evaluation and checkpoint monitoring."""
__version__ = "0.2.0"
def __getattr__(name):
    if name == "TrainingMonitor":
        from .monitor import TrainingMonitor
        return TrainingMonitor
    raise AttributeError(name)
