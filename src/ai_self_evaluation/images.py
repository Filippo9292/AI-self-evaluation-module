import io
import math
import random
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps

from .errors import EvaluationError
from .util import digest

EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    sha256: str
    width: int
    height: int

    def manifest(self):
        return {"path": str(self.path), "sha256": self.sha256, "width": self.width, "height": self.height}


def scan(directories, deduplicate=False):
    paths = set()
    for directory in directories:
        directory = Path(directory).resolve()
        if not directory.is_dir():
            raise EvaluationError(f"Image directory does not exist: {directory}")
        paths.update(p.resolve() for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)
    images, duplicates, seen = [], [], {}
    for path in sorted(paths):
        try:
            raw = path.read_bytes()
            with Image.open(io.BytesIO(raw)) as image:
                image.load()
                width, height = image.size
        except (OSError, ValueError) as exc:
            raise EvaluationError(f"Unreadable image: {path}") from exc
        record = ImageRecord(path, digest(raw), width, height)
        if deduplicate and record.sha256 in seen:
            duplicates.append({"path": str(path), "duplicate_of": seen[record.sha256]})
            continue
        images.append(record)
        seen[record.sha256] = str(path)
    if not images:
        raise EvaluationError("No supported image files found.")
    return images, duplicates


def prepared_image(record, settings):
    raw = record.path.read_bytes()
    if digest(raw) != record.sha256:
        raise EvaluationError(f"Image changed after planning: {record.path}")
    with Image.open(io.BytesIO(raw)) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        method = getattr(Image.Resampling, settings["interpolation"].upper())
        if settings.get("resize"):
            image = image.resize(tuple(settings["resize"]), method)
        elif max(image.size) > settings["max_side"]:
            image.thumbnail((settings["max_side"], settings["max_side"]), method)
        minimum = settings.get("min_side")
        if minimum and min(image.size) < minimum:
            factor = minimum / min(image.size)
            size = tuple(math.ceil(side * factor) for side in image.size)
            if max(size) > settings["max_side"]:
                raise EvaluationError("Image aspect ratio cannot satisfy min_side/max_side together. Configure an explicit resize or a suitable shared preprocessing limit.")
            image = image.resize(size, method)
        stream = io.BytesIO()
        image.save(stream, format="PNG")
    return stream.getvalue()


def real_sample_size(n, settings):
    return min(n, settings["real_max_images"], max(settings["real_min_images"], math.ceil(settings["real_sqrt_multiplier"] * math.sqrt(n))))


def rng(seed, key):
    return random.Random(int(digest([seed, key])[:16], 16))


def sample_candidates(images, count, seed, key):
    return rng(seed, key).sample(images, min(count, len(images)))


def sample_references(pool, candidate, seed, key):
    eligible = [p for p in pool if p.sha256 != candidate.sha256]
    if len(eligible) < 5:
        raise EvaluationError("Five distinct real references are required after excluding the candidate hash.")
    return rng(seed, key).sample(eligible, 5)
