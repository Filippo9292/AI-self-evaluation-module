"""Offline end-to-end fixture. No GAN training and no decision service is used."""
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from .errors import EvaluationError
from .monitor import TrainingMonitor
from .util import write_json


def create_images(folder, brightness, count, seed):
    folder.mkdir(parents=True, exist_ok=True)
    random = np.random.default_rng(seed)
    for i in range(count):
        pixels = np.clip(random.normal(brightness + (i % 5) * 2, 10, (48, 48, 3)), 0, 255).astype("uint8")
        Image.fromarray(pixels).save(folder / f"image_{i:03d}.png")


def run_demo(output):
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise EvaluationError("Demo output must be empty; choose a new directory to preserve earlier results.")
    output.mkdir(parents=True, exist_ok=True)
    create_images(output / "real", 180, 16, 1)
    create_images(output / "generated", 160, 12, 2)
    config = {"provider": "mock", "api_key": "", "real_image_dirs": ["real"], "generated_image_dir": "generated", "output_dir": "results",
              "settings": {"early_stopping": {"min_epochs": 2, "plateau_patience_epochs": 3,
                            "post_target_epochs": 2, "target_confirmations": 2, "resampling_trials": 50},
                           "sampling": {"checkpoint_max_images": 8}}}
    path = output / "evaluation.local.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    # Import locally to avoid a CLI import cycle.
    from .cli import main
    if main(["evaluate", "--config", str(path)]):
        raise EvaluationError("Offline inference demonstration failed.")
    monitor = TrainingMonitor(path)
    decision = None
    for epoch, brightness in enumerate([40, 90, 140, 175, 180, 180, 180, 180], 1):
        images = output / "generated_checkpoints" / f"epoch_{epoch:03d}"
        # Identical noise seed/slots across checkpoints simulates a fixed generation panel.
        create_images(images, brightness, 8, 42)
        checkpoint = output / "source_checkpoints" / f"epoch_{epoch:03d}.json"
        write_json(checkpoint, {"synthetic_demo": True, "epoch": epoch, "brightness": brightness})
        decision = monitor.observe(epoch, images, checkpoint)
        if decision["should_stop"]:
            break
    result = {"synthetic_demonstration_only": True, "api_calls": 0,
              "inference_report": str(output / "results/inference/inference/report.html"),
              "training_report": str(output / "results/training/report.html"), "decision": decision}
    write_json(output / "demo_result.json", result)
    return result
