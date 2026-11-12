import numpy as np


def describe(scores):
    values = np.asarray(scores, dtype=float)
    q10, q25, median, q75, q90 = np.quantile(values, [.1, .25, .5, .75, .9])
    iqr = q75 - q25
    within = values[(values >= q25 - 1.5 * iqr) & (values <= q75 + 1.5 * iqr)]
    return {"n": len(values), "mean": float(values.mean()), "std": float(values.std(ddof=1)) if len(values) > 1 else 0.,
            "q10": float(q10), "q25": float(q25), "median": float(median), "q75": float(q75), "q90": float(q90),
            "whisker_low": float(within.min()), "whisker_high": float(within.max()),
            "min": float(values.min()), "max": float(values.max()),
            "outliers": values[(values < q25 - 1.5 * iqr) | (values > q75 + 1.5 * iqr)].tolist()}


def box_distance(left, right):
    return sum(abs(left[k] - right[k]) for k in ("q25", "median", "q75")) / 12.


def reference_resampling(scores, sample_size, trials, seed):
    """Internal resampling diagnostic, not a population equivalence test."""
    random = np.random.default_rng(seed)
    baseline = describe(scores)
    distances = [box_distance(describe(random.choice(scores, size=sample_size, replace=True)), baseline) for _ in range(trials)]
    return {"method": "bootstrap_of_recorded_per_image_means", "sample_size": sample_size,
            "trials": trials, "distance_p50": float(np.quantile(distances, .5)),
            "distance_p95": float(np.quantile(distances, .95)),
            "interpretation": "Internal sampling variability only; not automatic calibration or proof of equivalence."}
