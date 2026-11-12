import shutil
from pathlib import Path

from .config import Config
from .engine import Engine
from .errors import EvaluationError
from .images import sample_candidates, scan
from .policy import advance
from .reporting import baseline_report, training_report
from .statistics import box_distance
from .util import RunLock, digest, file_digest, now, read_json, write_json


class TrainingMonitor:
    """Framework-independent checkpoint observer; the trainer obeys should_stop."""
    def __init__(self, config_path, provider=None):
        self.config = Config(config_path)
        self.provider = provider

    def due(self, epoch):
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1:
            raise EvaluationError("epoch must be a positive integer.")
        return epoch % self.config.settings["early_stopping"]["every_n_epochs"] == 0

    def observe(self, epoch, images_dir, checkpoint):
        if not self.due(epoch):
            return {"evaluated": False, "should_stop": False, "reason": "not_scheduled"}
        with RunLock(self.config.output_dir):
            return self._observe(epoch, images_dir, checkpoint)

    def _observe(self, epoch, images_dir, checkpoint):
        checkpoint = Path(checkpoint).resolve()
        if not checkpoint.is_file():
            raise EvaluationError("checkpoint must be an existing file (save the complete trainer state in it).")
        checkpoint_hash = file_digest(checkpoint)
        engine = Engine(self.config, self.provider)
        real_rows, real_summary = engine.baseline()
        baseline_report(engine.root, real_summary, self.config.provider["adapter"] == "mock")
        folder = engine.root / "training"
        folder.mkdir(exist_ok=True)
        policy_id = digest(self.config.settings["early_stopping"])
        policy_file = folder / "policy.json"
        if policy_file.exists() and read_json(policy_file)["policy_id"] != policy_id:
            raise EvaluationError("Early-stopping settings changed during a run; choose a new output directory.")
        write_json(policy_file, {"policy_id": policy_id, "settings": self.config.settings["early_stopping"]})
        images, _ = scan([images_dir])
        n = min(len(images), len(real_rows), self.config.settings["sampling"]["checkpoint_max_images"])
        # The same lexical slot selection across epochs; the trainer fixes input/seed identities.
        selected = sample_candidates(images, n, self.config.settings["sampling"]["seed"], "monitor-slots")
        selected = sorted(selected, key=lambda image: image.path.name)
        panel = {"available_image_names": [str(p.path.relative_to(Path(images_dir).resolve())) for p in images],
                 "selected_image_names": [str(p.path.relative_to(Path(images_dir).resolve())) for p in selected]}
        panel_file = folder / "generation_panel.json"
        if panel_file.exists() and read_json(panel_file) != panel:
            raise EvaluationError("Generation panel filenames/count changed. Generate the same input/seed slots at every checkpoint.")
        write_json(panel_file, panel)
        identity = {"checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_hash,
                    "images": [p.manifest() for p in selected]}
        stage = f"training/epoch_{epoch:06d}"
        stage_path = engine.root / stage
        stage_path.mkdir(parents=True, exist_ok=True)
        checkpoint_file = stage_path / "checkpoint_source.json"
        if checkpoint_file.exists() and read_json(checkpoint_file) != identity:
            raise EvaluationError("Checkpoint or images changed for an already planned epoch.")
        write_json(checkpoint_file, identity)
        decisions = sorted(folder.glob("epoch_*/observation.json"))
        observations = [read_json(p) for p in decisions]
        existing = next((o for o in observations if o["epoch"] == epoch), None)
        if existing:
            training_report(engine.root, real_summary, observations, self.config.settings, self.config.provider["adapter"] == "mock")
            return {"evaluated": True, **existing["decision"]}
        if observations and epoch <= observations[-1]["epoch"]:
            raise EvaluationError("Cannot insert an observation before the latest committed epoch.")
        if observations and observations[-1]["decision"]["should_stop"]:
            raise EvaluationError("Run has stopped; use a new output directory for a new run.")
        try:
            rows, summary = engine.score_images(selected, stage)
        except EvaluationError as exc:
            write_json(stage_path / "failure.json", {"utc": now(), "error": str(exc), "policy_advanced": False})
            raise
        distance = box_distance(summary, real_summary)
        previous = observations[-1]["decision"] if observations else None
        decision = advance(previous, epoch, distance, self.config.settings["early_stopping"])
        if decision["improved_best"]:
            target = engine.root / "checkpoints" / f"epoch_{epoch:06d}" / checkpoint.name
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".copying")
            shutil.copy2(checkpoint, temporary)
            if file_digest(temporary) != checkpoint_hash:
                raise EvaluationError("Checkpoint changed while copying; no policy decision was committed.")
            temporary.replace(target)
            decision["best_checkpoint"] = str(target)
        elif previous:
            decision["best_checkpoint"] = previous["best_checkpoint"]
        decision["evaluated"] = True
        observation = {"epoch": epoch, "summary": summary, "distance": distance,
                       "checkpoint": identity, "decision": decision, "usage": engine.usage()}
        # This atomic file is the commit point. Reports/state can be regenerated from it.
        write_json(stage_path / "observation.json", observation)
        observations.append(observation)
        write_json(folder / "state.json", decision)
        training_report(engine.root, real_summary, observations, self.config.settings, self.config.provider["adapter"] == "mock")
        write_json(engine.root / "usage.json", engine.usage())
        return decision

    def best(self):
        """Return the latest committed decision, including its retained best checkpoint."""
        folder = self.config.output_dir / "training"
        observations = sorted(folder.glob("epoch_*/observation.json"))
        return read_json(observations[-1])["decision"] if observations else None
