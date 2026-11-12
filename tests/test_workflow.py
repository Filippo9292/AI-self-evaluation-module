import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import requests
import yaml
from PIL import Image

from ai_self_evaluation.cli import main
from ai_self_evaluation.config import Config, PRESETS, load_yaml
from ai_self_evaluation.engine import Engine
from ai_self_evaluation.errors import EvaluationError
from ai_self_evaluation.images import prepared_image, real_sample_size, sample_references, scan
from ai_self_evaluation.monitor import TrainingMonitor
from ai_self_evaluation.providers import Provider, build_payload, validate_response
from ai_self_evaluation.util import RunLock, read_json, records


class Workspace(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.real = self.root / "real"
        self.real.mkdir()
        for i in range(8):
            Image.new("RGB", (12, 12), (160 + i, 170, 180)).save(self.real / f"{i}.png")
        self.generated = self.root / "generated"
        self.generated.mkdir()
        for i in range(3):
            Image.new("RGB", (12, 12), (100 + i, 170, 180)).save(self.generated / f"{i}.png")
        self.raw = {"provider": "mock", "api_key": "PRIVATE-TEST-KEY", "real_image_dirs": ["real"],
                    "generated_image_dir": "generated", "output_dir": "results",
                    "settings": {"early_stopping": {"min_epochs": 1, "plateau_patience_epochs": 2, "resampling_trials": 10}}}
        self.path = self.root / "evaluation.local.yaml"
        self.save_config()

    def save_config(self):
        self.path.write_text(yaml.safe_dump(self.raw), encoding="utf-8")

    def checkpoint(self, epoch):
        path = self.root / f"checkpoint_{epoch}.json"
        path.write_text(json.dumps({"epoch": epoch}), encoding="utf-8")
        return path


class WorkflowTests(Workspace):
    def test_complete_inference_resume_and_no_secret_export(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["evaluate", "--config", str(self.path)]), 0)
        root = self.root / "results"
        before = len(records(root / "requests.jsonl"))
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["evaluate", "--config", str(self.path)]), 0)
        self.assertEqual(before, len(records(root / "requests.jsonl")))
        rows = read_json(root / "inference/inference/scores.json")
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(row["visits_completed"] == 3 for row in rows))
        self.assertTrue((root / "inference/inference/report.html").exists())
        self.assertTrue((root / "inference/inference/comparison.svg").exists())
        for file in root.rglob("*"):
            if file.is_file() and file.suffix in (".json", ".jsonl", ".csv", ".html"):
                self.assertNotIn("PRIVATE-TEST-KEY", file.read_text(encoding="utf-8"))

    def test_baseline_change_is_rejected(self):
        Engine(Config(self.path)).baseline()
        Image.new("RGB", (12, 12), "red").save(self.real / "0.png")
        with self.assertRaises(EvaluationError):
            Engine(Config(self.path))

    def test_protocol_change_requires_new_baseline(self):
        Engine(Config(self.path)).baseline()
        protocol = load_yaml(PRESETS / "protocol.yaml")
        protocol["questions"]["visual_consistency"]["criteria"][0] = "Different criterion"
        self.raw["evaluation"] = protocol
        self.save_config()
        with self.assertRaises(EvaluationError):
            Engine(Config(self.path))

    def test_reference_exclusion_and_deduplication(self):
        (self.real / "copy.png").write_bytes((self.real / "0.png").read_bytes())
        images, duplicates = scan([self.real], deduplicate=True)
        self.assertEqual(len(images), 8)
        self.assertEqual(len(duplicates), 1)
        for visit in range(10):
            refs = sample_references(images, images[0], 42, visit)
            self.assertEqual(len({p.sha256 for p in refs}), 5)
            self.assertNotIn(images[0].sha256, {p.sha256 for p in refs})

    def test_duplicate_generated_files_each_get_a_score(self):
        (self.generated / "copy.png").write_bytes((self.generated / "0.png").read_bytes())
        engine = Engine(Config(self.path))
        images, _ = scan([self.generated])
        rows, _ = engine.score_images(images, "inference/copies")
        self.assertEqual(len(rows), 4)

    def test_checkpoint_retention_and_idempotency(self):
        monitor = TrainingMonitor(self.path)
        checkpoint = self.checkpoint(1)
        first = monitor.observe(1, self.generated, checkpoint)
        self.assertTrue(Path(first["best_checkpoint"]).is_file())
        self.assertEqual(Path(first["best_checkpoint"]).read_bytes(), checkpoint.read_bytes())
        second = monitor.observe(1, self.generated, checkpoint)
        self.assertEqual(first, second)
        monitor.observe(2, self.generated, self.checkpoint(2))
        stop = monitor.observe(3, self.generated, self.checkpoint(3))
        self.assertTrue(stop["should_stop"])
        # Fresh reference panels can move the mock score even for identical candidates.
        self.assertEqual(Path(stop["best_checkpoint"]).read_bytes(),
                         (self.root / f"checkpoint_{stop['best_epoch']}.json").read_bytes())
        retained = Path(stop["best_checkpoint"]).read_bytes()
        (self.root / f"checkpoint_{stop['best_epoch']}.json").write_text("overwritten by trainer")
        self.assertEqual(Path(stop["best_checkpoint"]).read_bytes(), retained)

    def test_failure_does_not_commit_policy_and_resume_reuses_visits(self):
        config = Config(self.path)
        Engine(config).baseline()
        base = Provider(config)

        class Flaky:
            calls = 0
            def evaluate(self, body, images):
                self.calls += 1
                if self.calls == 3:
                    raise requests.Timeout("uncertain")
                return base.evaluate(body, images)

        with self.assertRaises(EvaluationError):
            TrainingMonitor(self.path, Flaky()).observe(1, self.generated, self.checkpoint(1))
        self.assertFalse((config.output_dir / "training/state.json").exists())
        self.raw["settings"]["api"] = {"retry_failed_requests": True}
        self.save_config()
        result = TrainingMonitor(self.path).observe(1, self.generated, self.root / "checkpoint_1.json")
        self.assertEqual(result["best_epoch"], 1)
        history = records(config.output_dir / "requests.jsonl")
        monitor_valid = [r for r in history if r["status"] == "valid" and r["stage"].startswith("training/")]
        self.assertEqual(len(monitor_valid), 9)

    def test_configured_cap_does_not_create_stopping_decision(self):
        self.raw["settings"]["api"] = {"max_calls": 1}
        self.save_config()
        with self.assertRaises(EvaluationError):
            TrainingMonitor(self.path).observe(1, self.generated, self.checkpoint(1))
        self.assertFalse((self.root / "results/training/state.json").exists())

    def test_plan_is_read_only(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(["plan", "--config", str(self.path), "--max-epochs", "100"]), 0)
        self.assertEqual(json.loads(output.getvalue())["network_requests_made"], 0)
        self.assertFalse((self.root / "results").exists())

    def test_lock(self):
        with RunLock(self.root):
            with self.assertRaises(EvaluationError):
                with RunLock(self.root):
                    pass

    def test_panel_drift_does_not_advance_policy(self):
        monitor = TrainingMonitor(self.path)
        monitor.observe(1, self.generated, self.checkpoint(1))
        Image.new("RGB", (12, 12), "green").save(self.generated / "extra.png")
        with self.assertRaises(EvaluationError):
            monitor.observe(2, self.generated, self.checkpoint(2))
        self.assertEqual(monitor.best()["last_epoch"], 1)

    def test_report_on_missing_baseline_makes_no_calls(self):
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(["report", "--config", str(self.path)]), 2)
        self.assertFalse((self.root / "results/requests.jsonl").exists())


class ProtocolTests(Workspace):
    def test_provider_minimum_preprocessing_is_explicit(self):
        self.raw["provider"] = "jev"
        self.save_config()
        config = Config(self.path)
        image = scan([self.real])[0][0]
        data = prepared_image(image, config.settings["preprocessing"])
        with Image.open(io.BytesIO(data)) as transformed:
            self.assertEqual(transformed.size, (280, 280))

    def test_examples_match_installed_defaults(self):
        repo = Path(__file__).resolve().parents[1]
        for name in ("advanced", "protocol"):
            self.assertEqual(load_yaml(repo / "configs" / f"{name}.example.yaml"),
                             load_yaml(PRESETS / f"{name}.yaml"))

    def test_sample_sizes(self):
        settings = Config(self.path).settings["sampling"]
        self.assertEqual([real_sample_size(n, settings) for n in (20, 50, 100, 1000, 10000)], [20, 32, 40, 127, 256])

    def test_all_adapters_keep_six_images_and_same_question(self):
        protocol = Config(self.path).protocol
        images = [b"image"] * 6
        for name in ("jev", "liquid", "drex", "perplexity", "mercury", "baseten", "celeris"):
            with self.subTest(provider=name):
                provider = load_yaml(PRESETS / "providers" / f"{name}.yaml")
                body = build_payload(provider, protocol, images)
                question = body["questions"]["visual_consistency"]
                self.assertEqual(question["criteria"], protocol["questions"]["visual_consistency"]["criteria"])
                self.assertNotIn("{reference_", json.dumps(body))
                if name == "jev":
                    self.assertEqual(len(body["assets"]), 6)
                elif name in ("liquid", "drex"):
                    self.assertEqual(len(body["images"]), 6)
                elif name == "celeris":
                    self.assertEqual(len(body["x_celeris"]["images"]), 6)
                else:
                    parts = body["state"]["messages"][0]["content"] if name == "baseten" else body["state"]
                    self.assertEqual(sum(p["type"] == "image_url" for p in parts), 6)

    def test_response_validation_rejects_inconsistent_score(self):
        body = {"questions": {"q": {"type": "score"}}}
        response = {"model": "test", "answers": {"q": {"score": 4., "probabilities": {str(i): float(i == 0) for i in range(5)}}}}
        with self.assertRaises(EvaluationError):
            validate_response(body, response)

    def test_too_few_real_images(self):
        for p in list(self.real.glob("*.png"))[5:]:
            p.unlink()
        with self.assertRaises(EvaluationError):
            Engine(Config(self.path))

    def test_invalid_configuration(self):
        self.raw["settings"]["sampling"] = {"visits_per_image": 1}
        self.save_config()
        with self.assertRaises(EvaluationError):
            Config(self.path)


if __name__ == "__main__":
    unittest.main()
