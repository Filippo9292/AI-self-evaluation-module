"""One semantic request, explicit image transports. No automatic provider fallback."""
import base64
import io
import json
import math
import time

import numpy as np
import requests
from PIL import Image

from .errors import EvaluationError
from .util import digest


def replace(value, labels):
    if isinstance(value, str):
        for key, label in labels.items():
            value = value.replace(key, label)
        return value
    if isinstance(value, list):
        return [replace(v, labels) for v in value]
    if isinstance(value, dict):
        return {k: replace(v, labels) for k, v in value.items()}
    return value


def build_payload(provider, protocol, images):
    adapter = provider["adapter"]
    tokens = ["{candidate_image}"] + [f"{{reference_{i}}}" for i in range(1, 6)]
    names = ["candidate_C"] + [f"reference_R{i}" for i in range(1, 6)]
    encoded = [base64.b64encode(im).decode("ascii") for im in images]
    data_urls = ["data:image/png;base64," + item for item in encoded]
    if adapter == "image_array":
        names = [f"Image {i}" for i in range(1, 7)]
    rendered = replace(protocol, dict(zip(tokens, names)))
    body = {"model": provider["model"], **rendered}
    if adapter == "typesafe_assets":
        body["assets"] = {name: {"type": "image", "format": "png", "data_base64": data} for name, data in zip(names, encoded)}
    elif adapter == "image_array":
        body["images"] = data_urls
    elif adapter in ("image_parts", "baseten_messages"):
        parts = [{"type": "text", "text": json.dumps(rendered["state"], ensure_ascii=True)}]
        for name, url in zip(names, data_urls):
            parts.extend([{"type": "text", "text": name}, {"type": "image_url", "image_url": {"url": url}}])
        body["state"] = {"messages": [{"role": "user", "content": parts}]} if adapter == "baseten_messages" else parts
    elif adapter == "celeris_images":
        body["x_celeris"] = {"images": dict(zip(names, data_urls))}
    elif adapter != "mock":
        raise EvaluationError(f"Unknown provider adapter: {adapter}")
    return body


def validate_response(body, result):
    if not isinstance(result, dict) or not isinstance(result.get("model"), str) or not result["model"]:
        raise EvaluationError("Provider response has no model identifier.")
    answers = result.get("answers")
    if not isinstance(answers, dict) or set(answers) != set(body["questions"]):
        raise EvaluationError("Response question identifiers do not match the request.")
    answer = next(iter(answers.values()))
    if not isinstance(answer, dict):
        raise EvaluationError("Provider answer must be an object.")
    if answer.get("type", "score") != "score":
        raise EvaluationError("Provider returned a non-score answer.")
    score = answer.get("score")
    if isinstance(score, bool) or not isinstance(score, (float, int)) or not math.isfinite(score) or not 0 <= score <= 4:
        raise EvaluationError("Invalid score: expected a finite number from 0 to 4.")
    probabilities = answer.get("probabilities")
    if not isinstance(probabilities, dict) or set(probabilities) != {str(i) for i in range(5)}:
        raise EvaluationError("Missing or unexpected rubric probabilities (expected 0-4).")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1 for v in probabilities.values()):
        raise EvaluationError("Invalid rubric probability.")
    if abs(sum(probabilities.values()) - 1) > .03:
        raise EvaluationError("Rubric probabilities do not sum to one.")
    if abs(score - sum(int(k) * v for k, v in probabilities.items())) > .07:
        raise EvaluationError("Score differs from the probability-weighted rubric level.")
    confidence = answer.get("confidence")
    if confidence is not None and (isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or not 0 <= confidence <= 1):
        raise EvaluationError("Invalid provider confidence.")
    return {"score": float(score), "probabilities": probabilities, "confidence": confidence}


class Provider:
    def __init__(self, config):
        self.config = config
        self.session = requests.Session()

    def evaluate(self, body, images):
        if self.config.provider["adapter"] == "mock":
            # Deliberately simple synthetic fixture. This is NOT a visual quality model.
            means = []
            for raw in images:
                with Image.open(io.BytesIO(raw)) as image:
                    means.append(float(np.asarray(image, dtype=float).mean()))
            score = max(0., min(4., 3.2 - abs(means[0] - float(np.mean(means[1:]))) / 60.))
            low = min(3, int(score))
            probabilities = {str(i): 0. for i in range(5)}
            probabilities[str(low)] = low + 1 - score
            probabilities[str(low + 1)] = score - low
            qid = next(iter(body["questions"]))
            return 200, {"model": self.config.provider["model"], "answers": {qid: {"type": "score", "score": score, "probabilities": probabilities}},
                         "usage": {"input_tokens": 0, "output_tokens": 0}, "synthetic": True}
        if not self.config.api_key or self.config.api_key == "PASTE_YOUR_API_KEY_HERE":
            raise EvaluationError("Set api_key in your private YAML or the provider's environment variable.")
        headers = {"Authorization": "Bearer " + self.config.api_key, "Content-Type": "application/json",
                   **self.config.provider.get("headers", {})}
        response = self.session.post(self.config.provider["endpoint"], headers=headers, json=body,
                                     timeout=self.config.settings["api"]["timeout_seconds"])
        try:
            result = response.json()
        except ValueError:
            result = {"error": "Non-JSON provider response", "http_status": response.status_code}
        return response.status_code, result
