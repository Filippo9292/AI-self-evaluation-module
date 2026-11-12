import copy
import math
import os
from pathlib import Path

import yaml

from .errors import EvaluationError
from .util import digest

PRESETS = Path(__file__).parent / "presets"


def load_yaml(path):
    try:
        value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise EvaluationError(f"Cannot read YAML: {path}") from exc
    if not isinstance(value, dict):
        raise EvaluationError(f"YAML must contain a mapping: {path}")
    return value


def merge(base, update):
    if not isinstance(update, dict):
        raise EvaluationError("Configuration overrides must be YAML mappings.")
    out = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


class Config:
    def __init__(self, path):
        self.path = Path(path).resolve()
        raw = load_yaml(self.path)

        def absolute(value):
            p = Path(value).expanduser()
            return p.resolve() if p.is_absolute() else (self.path.parent / p).resolve()

        name = raw.get("provider", "jev")
        if not isinstance(name, str) or name not in {p.stem for p in (PRESETS / "providers").glob("*.yaml")}:
            raise EvaluationError("Unknown provider. Use the providers command to list built-in presets.")
        self.provider = load_yaml(absolute(raw["provider_file"]) if raw.get("provider_file") else PRESETS / "providers" / f"{name}.yaml")
        if not all(isinstance(self.provider.get(key), str) and self.provider[key] for key in ("name", "adapter", "endpoint", "model")):
            raise EvaluationError("Provider configuration needs name, adapter, endpoint and model strings.")
        self.settings = load_yaml(PRESETS / "advanced.yaml")
        if self.provider.get("preprocessing"):
            self.settings["preprocessing"] = merge(self.settings["preprocessing"], self.provider["preprocessing"])
        if raw.get("advanced_file"):
            self.settings = merge(self.settings, load_yaml(absolute(raw["advanced_file"])))
        self.settings = merge(self.settings, raw.get("settings", {}))
        self.protocol = load_yaml(absolute(raw["protocol_file"]) if raw.get("protocol_file") else PRESETS / "protocol.yaml")
        if "evaluation" in raw:
            self.protocol = raw["evaluation"]
        if not isinstance(self.protocol, dict):
            raise EvaluationError("evaluation must contain state and questions as a YAML mapping.")
        dirs = raw.get("real_image_dirs")
        if not isinstance(dirs, list) or not dirs or any(not isinstance(p, str) for p in dirs):
            raise EvaluationError("real_image_dirs must be a non-empty YAML list of directory paths.")
        self.real_dirs = [absolute(p) for p in dirs]
        self.generated_dir = absolute(raw["generated_image_dir"]) if raw.get("generated_image_dir") else None
        self.output_dir = absolute(raw.get("output_dir", "outputs/evaluation"))
        if any(self.output_dir.is_relative_to(directory) for directory in self.real_dirs):
            raise EvaluationError("output_dir must be outside the real-image directories.")
        if self.generated_dir and self.output_dir.is_relative_to(self.generated_dir):
            raise EvaluationError("output_dir must be outside generated_image_dir.")
        # Private YAML is allowed. Environment variables are an optional fallback.
        self.api_key = raw.get("api_key") or os.environ.get(self.provider.get("api_key_env", ""), "")
        if not isinstance(self.api_key, str):
            raise EvaluationError("api_key must be a string.")
        self._validate()
        # The generated-data path, output path and operating policy do not define the judge.
        self.protocol_id = digest({"version": 1, "provider": self.provider,
                                   "protocol": self.protocol,
                                   "sampling": self.settings["sampling"],
                                   "preprocessing": self.settings["preprocessing"]})

    def _validate(self):
        s = self.settings["sampling"]
        for name in ("seed", "references_per_visit", "visits_per_image", "real_min_images", "real_max_images", "checkpoint_max_images"):
            value = s.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < (0 if name == "seed" else 1):
                raise EvaluationError(f"sampling.{name} must be a positive integer (seed may be zero).")
        if s["references_per_visit"] != 5:
            raise EvaluationError("This protocol requires exactly five references per visit.")
        if s["visits_per_image"] < 2:
            raise EvaluationError("At least two visits per image are required; default: three.")
        if s["real_min_images"] > s["real_max_images"]:
            raise EvaluationError("real_min_images cannot exceed real_max_images.")
        if not isinstance(s["real_sqrt_multiplier"], (int, float)) or not math.isfinite(s["real_sqrt_multiplier"]) or s["real_sqrt_multiplier"] <= 0:
            raise EvaluationError("real_sqrt_multiplier must be finite and positive.")
        if self.provider.get("max_images", 0) < 6:
            raise EvaluationError("Provider must support at least six images in one request.")
        if self.provider.get("adapter") != "mock" and not self.provider.get("endpoint", "").startswith("https://"):
            raise EvaluationError("Live provider endpoint must use HTTPS.")
        q = self.protocol.get("questions", {})
        if len(q) != 1 or not isinstance(self.protocol.get("state"), (dict, list, str)):
            raise EvaluationError("Define state and exactly one score question.")
        question = next(iter(q.values()))
        if question.get("type") != "score" or not isinstance(question.get("criteria"), list) or len(question["criteria"]) != 5:
            raise EvaluationError("The score question must define exactly five ordered criteria (0-4).")
        if any(not isinstance(v, str) or not v.strip() for v in question["criteria"]):
            raise EvaluationError("Score criteria must be non-empty strings.")
        if not question.get("instructions"):
            raise EvaluationError("Question instructions are required.")
        serialized = str(self.protocol)
        for token in ["{candidate_image}"] + [f"{{reference_{i}}}" for i in range(1, 6)]:
            if token not in serialized:
                raise EvaluationError(f"Missing protocol placeholder: {token}")
        p = self.settings["early_stopping"]
        for name in ("every_n_epochs", "min_epochs", "plateau_patience_epochs", "post_target_epochs", "target_confirmations", "resampling_trials"):
            if isinstance(p[name], bool) or not isinstance(p[name], int) or p[name] < 1:
                raise EvaluationError(f"early_stopping.{name} must be a positive integer.")
        for name in ("min_delta", "target_tolerance", "target_reset_tolerance"):
            if not isinstance(p[name], (int, float)) or not math.isfinite(p[name]) or not 0 < p[name] <= 1:
                raise EvaluationError(f"early_stopping.{name} must lie in (0, 1].")
        if p["target_reset_tolerance"] < p["target_tolerance"]:
            raise EvaluationError("target_reset_tolerance must be >= target_tolerance.")
        api = self.settings["api"]
        for key in ("timeout_seconds", "max_request_bytes"):
            if not isinstance(api[key], (int, float)) or not math.isfinite(api[key]) or api[key] <= 0:
                raise EvaluationError(f"api.{key} must be finite and positive.")
        if not isinstance(api["rate_limit_retries"], int) or not 0 <= api["rate_limit_retries"] <= 10:
            raise EvaluationError("rate_limit_retries must be an integer between 0 and 10.")
        if api["max_calls"] is not None and (isinstance(api["max_calls"], bool) or not isinstance(api["max_calls"], int) or api["max_calls"] < 1):
            raise EvaluationError("max_calls must be null or a positive integer.")
        pre = self.settings["preprocessing"]
        if pre["interpolation"] not in ("nearest", "bilinear", "bicubic", "lanczos"):
            raise EvaluationError("Unknown interpolation method.")
        if not isinstance(pre["max_side"], int) or pre["max_side"] < 1:
            raise EvaluationError("preprocessing.max_side must be a positive integer.")
        if pre.get("min_side") is not None and (isinstance(pre["min_side"], bool) or not isinstance(pre["min_side"], int) or not 1 <= pre["min_side"] <= pre["max_side"]):
            raise EvaluationError("preprocessing.min_side must be null or a positive integer <= max_side.")
        if pre.get("resize") is not None and (not isinstance(pre["resize"], list) or len(pre["resize"]) != 2 or any(not isinstance(v, int) or v < 1 for v in pre["resize"])):
            raise EvaluationError("preprocessing.resize must be null or [width, height].")
        if not isinstance(self.settings["reporting"]["boxes_per_page"], int) or self.settings["reporting"]["boxes_per_page"] < 1:
            raise EvaluationError("boxes_per_page must be positive.")

    def public(self):
        # Explicit allow-list: never serialize the private YAML or credentials.
        provider = {k: v for k, v in self.provider.items() if k not in ("api_key", "headers")}
        return {"protocol_id": self.protocol_id, "provider": provider,
                "evaluation": self.protocol, "settings": self.settings,
                "real_image_dirs": [str(p) for p in self.real_dirs]}
