import argparse
import json
import re
import sys
from pathlib import Path

from .config import Config, PRESETS, load_yaml
from .engine import Engine
from .errors import EvaluationError
from .images import real_sample_size, scan
from .monitor import TrainingMonitor
from .reporting import baseline_report, inference_report
from .statistics import box_distance
from .util import RunLock, write_json


def parser():
    root = argparse.ArgumentParser(description="Reference-based image scoring and checkpoint early stopping.")
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("providers", help="List packaged provider presets and verification status")
    for name in ("plan", "baseline", "evaluate", "monitor", "report"):
        cmd = commands.add_parser(name)
        cmd.add_argument("--config", required=True, type=Path)
        if name == "plan":
            cmd.add_argument("--max-epochs", type=int)
        if name == "evaluate":
            cmd.add_argument("--images", type=Path, help="Overrides generated_image_dir")
            cmd.add_argument("--id", default="inference", help="Stable name for this inference set")
        if name == "monitor":
            cmd.add_argument("--images", required=True, type=Path)
            cmd.add_argument("--epoch", required=True, type=int)
            cmd.add_argument("--checkpoint", required=True, type=Path)
            cmd.add_argument("--stop-exit-code", action="store_true", help="Return exit code 10 on a stopping decision")
    demo = commands.add_parser("demo", help="Generate and run a synthetic offline software demonstration")
    demo.add_argument("--output", type=Path, default=Path("demo-workspace"))
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "providers":
            print(json.dumps([load_yaml(p) for p in sorted((PRESETS / "providers").glob("*.yaml"))], indent=2))
            return 0
        if args.command == "demo":
            from .demo import run_demo
            print(json.dumps(run_demo(args.output), indent=2))
            return 0
        config = Config(args.config)
        if args.command == "plan":
            real, duplicates = scan(config.real_dirs, deduplicate=True)
            if len(real) < 6:
                raise EvaluationError("At least six unique real images are required.")
            s = config.settings["sampling"]
            n = real_sample_size(len(real), s)
            result = {"provider": config.provider["name"], "model": config.provider["model"], "protocol_id": config.protocol_id,
                      "unique_real_images": len(real), "duplicate_files_excluded": len(duplicates), "baseline_images": n,
                      "visits_per_image": s["visits_per_image"], "baseline_planned_calls": n * s["visits_per_image"],
                      "checkpoint_max_planned_calls": min(n, s["checkpoint_max_images"]) * s["visits_per_image"],
                      "network_requests_made": 0, "note": "Counts exclude rate-limit retries; completed cached requests are reused."}
            if config.generated_dir and config.generated_dir.is_dir():
                generated, _ = scan([config.generated_dir])
                result["inference_images"] = len(generated)
                result["inference_planned_calls"] = len(generated) * s["visits_per_image"]
            if args.max_epochs is not None:
                if args.max_epochs < 1:
                    raise EvaluationError("max-epochs must be positive.")
                observations = args.max_epochs // config.settings["early_stopping"]["every_n_epochs"]
                result["maximum_observations"] = observations
                result["training_max_planned_calls"] = observations * result["checkpoint_max_planned_calls"]
            print(json.dumps(result, indent=2))
            return 0
        if args.command == "monitor":
            result = TrainingMonitor(args.config).observe(args.epoch, args.images, args.checkpoint)
            print(json.dumps(result, indent=2))
            return 10 if result["should_stop"] and args.stop_exit_code else 0
        with RunLock(config.output_dir):
            engine = Engine(config)
            if args.command == "report":
                from .util import read_json
                status = engine.root / "baseline/status.json"
                if not status.exists() or not read_json(status).get("complete"):
                    raise EvaluationError("Report regeneration needs a completed baseline; it never initiates API calls.")
            rows, real = engine.baseline()
            baseline_report(engine.root, real, config.provider["adapter"] == "mock")
            if args.command == "evaluate":
                if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", args.id):
                    raise EvaluationError("Inference ID must be 1-80 letters, digits, underscores or hyphens.")
                images_dir = args.images or config.generated_dir
                if not images_dir:
                    raise EvaluationError("Provide --images or generated_image_dir in the YAML.")
                if config.output_dir.is_relative_to(Path(images_dir).resolve()):
                    raise EvaluationError("Evaluation outputs must be outside the input-image directory.")
                images, _ = scan([images_dir])  # Keep duplicate generated files: every file gets a row.
                stage = f"inference/{args.id}"
                generated_rows, generated = engine.score_images(images, stage)
                distance = box_distance(generated, real)
                inference_report(engine.root / stage, real, generated, generated_rows, engine.usage(), distance)
                result = {"images": len(images), "distance": distance, "report": str(engine.root / stage / "report.html")}
            elif args.command == "report":
                # Regenerate completed training and inference reports without making calls.
                from .reporting import training_report
                from .util import read_json
                observations = [read_json(p) for p in sorted((engine.root / "training").glob("epoch_*/observation.json"))]
                if observations:
                    training_report(engine.root, real, observations, config.settings, config.provider["adapter"] == "mock")
                for folder in (engine.root / "inference").glob("*"):
                    if (folder / "status.json").exists() and read_json(folder / "status.json").get("complete"):
                        inferred = read_json(folder / "summary.json")
                        inference_report(folder, real, inferred, read_json(folder / "scores.json"), engine.usage(), box_distance(inferred, real))
                result = {"output_dir": str(engine.root), "reports_regenerated": True}
            else:
                result = {"real_images_evaluated": len(rows), "summary": real, "output_dir": str(engine.root)}
            write_json(engine.root / "usage.json", engine.usage())
            print(json.dumps(result, indent=2))
        return 0
    except (EvaluationError, OSError, ValueError) as exc:
        # Do not print tracebacks or a private configuration/payload.
        print(f"Evaluation stopped: {exc}", file=sys.stderr)
        return 2


def self_evaluation():
    return main(["evaluate", *sys.argv[1:]])


def early_stopping():
    return main(["monitor", *sys.argv[1:]])
