import json
import statistics
import time
from pathlib import Path

import requests

from .errors import EvaluationError
from .images import prepared_image, real_sample_size, sample_candidates, sample_references, scan
from .providers import Provider, build_payload, validate_response
from .statistics import describe, reference_resampling
from .util import append_json, digest, now, read_json, records, write_csv, write_json


class Engine:
    """Persistent evaluator. Caller holds the output-directory RunLock."""
    def __init__(self, config, provider=None):
        self.config = config
        self.root = config.output_dir
        self.root.mkdir(parents=True, exist_ok=True)
        self.real, self.duplicates = scan(config.real_dirs, deduplicate=True)
        if len(self.real) < 6:
            raise EvaluationError("At least six unique real images are required.")
        self.dataset_id = digest([p.manifest() for p in self.real])
        identity = {"protocol_id": config.protocol_id, "dataset_id": self.dataset_id}
        identity_path = self.root / "identity.json"
        if identity_path.exists() and read_json(identity_path) != identity:
            raise EvaluationError("Dataset or evaluation protocol changed. Choose a new output_dir to build a new baseline.")
        write_json(identity_path, identity)
        write_json(self.root / "resolved_config.json", config.public())
        write_json(self.root / "real_manifest.json", {"images": [p.manifest() for p in self.real], "excluded_duplicates": self.duplicates})
        self.backend = provider or Provider(config)
        self.journal = self.root / "requests.jsonl"
        history = records(self.journal)
        self.attempts = sum(row["status"] == "started" for row in history)
        self.last = {}
        self.cache = {}
        for row in history:
            self.last[row["call_id"]] = row
            if row["status"] == "valid":
                self.cache[row["call_id"]] = row
        self.image_cache = {}
        self.resolved_model = None
        model_path = self.root / "resolved_model.json"
        if model_path.exists():
            self.resolved_model = read_json(model_path)["model"]

    def _safe(self, value):
        # Redact even if a remote error unexpectedly echoes the supplied key.
        encoded = json.dumps(value, ensure_ascii=True, allow_nan=False)
        if self.config.api_key:
            encoded = encoded.replace(json.dumps(self.config.api_key)[1:-1], "[REDACTED]")
        return json.loads(encoded)

    def _record(self, row):
        row = self._safe({"utc": now(), **row})
        append_json(self.journal, row)
        self.last[row["call_id"]] = row

    def _image(self, image):
        if image.sha256 not in self.image_cache:
            if len(self.image_cache) >= 512:
                self.image_cache.pop(next(iter(self.image_cache)))
            self.image_cache[image.sha256] = prepared_image(image, self.config.settings["preprocessing"])
        return self.image_cache[image.sha256]

    def call(self, candidate, references, stage, visit):
        images = [self._image(im) for im in [candidate, *references]]
        body = build_payload(self.config.provider, self.config.protocol, images)
        payload_id = digest(body)
        call_id = digest([self.config.protocol_id, stage, str(candidate.path), candidate.sha256, visit, payload_id])
        if call_id in self.cache:
            return self.cache[call_id]
        previous = self.last.get(call_id)
        if previous and not self.config.settings["api"]["retry_failed_requests"]:
            raise EvaluationError("A previous request is incomplete or failed. Review requests.jsonl; set api.retry_failed_requests: true to explicitly allow another attempt.")
        if self.config.provider["adapter"] != "mock" and (not self.config.api_key or self.config.api_key == "PASTE_YOUR_API_KEY_HERE"):
            raise EvaluationError("Set api_key in your private YAML or the provider's environment variable.")
        raw_size = len(json.dumps(body).encode())
        if raw_size > self.config.settings["api"]["max_request_bytes"]:
            raise EvaluationError("Encoded request exceeds max_request_bytes. Reduce image size in a NEW baseline configuration.")
        meta = {"call_id": call_id, "stage": stage, "candidate": str(candidate.path), "candidate_sha256": candidate.sha256,
                "references": [p.manifest() for p in references], "visit": visit,
                "payload_sha256": payload_id, "request_bytes": raw_size,
                "prepared_sha256": [digest(im) for im in images]}
        for attempt in range(self.config.settings["api"]["rate_limit_retries"] + 1):
            maximum = self.config.settings["api"]["max_calls"]
            if maximum is not None and self.attempts >= maximum:
                raise EvaluationError("Configured cumulative max_calls reached; no stopping decision was made.")
            self.attempts += 1
            self._record({**meta, "status": "started", "attempt_number": self.attempts})
            start = time.perf_counter()
            try:
                status, response = self.backend.evaluate(body, images)
            except requests.RequestException as exc:
                self._record({**meta, "status": "uncertain", "error_type": type(exc).__name__})
                raise EvaluationError("Transport failed; server completion is unknown. Review the journal before explicitly retrying.") from None
            except EvaluationError:
                self._record({**meta, "status": "failed", "error_type": "configuration"})
                raise
            # Persist the raw response before validating its semantics.
            self._record({**meta, "status": "received", "http_status": status, "latency_seconds": time.perf_counter() - start, "response": response})
            if status == 429 and attempt < self.config.settings["api"]["rate_limit_retries"]:
                time.sleep(min(2 ** attempt, 8))
                continue
            if status != 200:
                self._record({**meta, "status": "failed", "http_status": status})
                raise EvaluationError(f"Provider returned HTTP {status}; inspect the redacted request journal.")
            try:
                parsed = validate_response(body, response)
                model = response["model"]
                if self.resolved_model is not None and model != self.resolved_model:
                    raise EvaluationError("Provider model identifier changed during this run. Use a new baseline/output directory.")
            except EvaluationError:
                self._record({**meta, "status": "invalid_response"})
                raise
            self.resolved_model = model
            write_json(self.root / "resolved_model.json", {"requested_model": self.config.provider["model"], "model": model})
            row = self._safe({**meta, "status": "valid", **parsed, "model": model, "usage": response.get("usage", {})})
            self._record(row)
            self.cache[call_id] = row
            return row
        raise EvaluationError("Request did not complete.")

    def score_images(self, images, stage):
        settings = self.config.settings["sampling"]
        rows = []
        directory = self.root / stage
        directory.mkdir(parents=True, exist_ok=True)
        manifest = {"images": [p.manifest() for p in images], "protocol_id": self.config.protocol_id}
        if (directory / "manifest.json").exists() and read_json(directory / "manifest.json") != manifest:
            raise EvaluationError(f"Image set changed for {stage}. Use a new inference ID or output directory.")
        write_json(directory / "manifest.json", manifest)
        write_json(directory / "status.json", {"complete": False, "expected_images": len(images)})
        for index, image in enumerate(images):
            visits = []
            for visit in range(settings["visits_per_image"]):
                # Fresh, reproducible draws for every checkpoint/image/visit.
                key = [stage, index, visit] if stage.startswith("training/") else [stage, image.sha256, visit]
                refs = sample_references(self.real, image, settings["seed"], key)
                visits.append(self.call(image, refs, stage, visit))
            scores = [v["score"] for v in visits]
            rows.append({"image_id": digest(str(image.path))[:16], "file_path": str(image.path), "image_hash": image.sha256,
                         "score_mean": statistics.mean(scores), "score_std_across_visits": statistics.stdev(scores),
                         "visits_completed": len(visits), "provider": self.config.provider["name"],
                         "model": self.resolved_model, "protocol_id": self.config.protocol_id,
                         "evaluation_status": "complete", "call_ids": [v["call_id"] for v in visits]})
            write_csv(directory / "scores.csv", rows)
            write_json(directory / "scores.json", rows)
        summary = describe([row["score_mean"] for row in rows])
        write_json(directory / "summary.json", summary)
        write_json(directory / "status.json", {"complete": True, "expected_images": len(images), "completed_images": len(rows)})
        return rows, summary

    def baseline(self):
        folder = self.root / "baseline"
        status = folder / "status.json"
        if status.exists() and read_json(status).get("complete"):
            return read_json(folder / "scores.json"), read_json(folder / "summary.json")
        settings = self.config.settings["sampling"]
        selected = sample_candidates(self.real, real_sample_size(len(self.real), settings), settings["seed"], "real-candidates")
        rows, summary = self.score_images(selected, "baseline")
        monitor_n = min(len(rows), settings["checkpoint_max_images"])
        diagnostic = reference_resampling([row["score_mean"] for row in rows], monitor_n,
                                         self.config.settings["early_stopping"]["resampling_trials"], settings["seed"])
        write_json(folder / "resampling_diagnostic.json", diagnostic)
        return rows, summary

    def usage(self):
        history = records(self.journal)
        received = [r for r in history if r["status"] == "received"]
        return {"attempts": sum(r["status"] == "started" for r in history), "valid_visits": len(self.cache),
                "input_tokens_reported": sum(r.get("response", {}).get("usage", {}).get("input_tokens", 0) or 0 for r in received),
                "output_tokens_reported": sum(r.get("response", {}).get("usage", {}).get("output_tokens", 0) or 0 for r in received),
                "synthetic": self.config.provider["adapter"] == "mock"}
