"""Regenerable research figures. Box whiskers are Tukey whiskers, not confidence intervals."""
import html
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .util import read_json, write_csv, write_json


def box(summary, label):
    return {"label": label, "med": summary["median"], "q1": summary["q25"], "q3": summary["q75"],
            "whislo": summary["whisker_low"], "whishi": summary["whisker_high"], "fliers": summary["outliers"]}


def save_figure(figure, stem):
    figure.savefig(str(stem) + ".png", dpi=160, bbox_inches="tight")
    figure.savefig(str(stem) + ".svg", bbox_inches="tight")
    plt.close(figure)


def boxes(summaries, labels, stem, title, synthetic=False):
    figure, axis = plt.subplots(figsize=(max(6.4, len(labels) * .6), 4.8))
    artists = axis.bxp([box(s, label) for s, label in zip(summaries, labels)], showfliers=True, patch_artist=True)
    for index, artist in enumerate(artists["boxes"]):
        artist.set_facecolor("#b8c3cb" if index == 0 else "#5c9ba2")
    axis.set(ylim=(-.1, 4.1), ylabel="Visual consistency score (0-4)", title=title)
    axis.grid(axis="y", alpha=.2)
    if len(labels) > 8:
        axis.tick_params(axis="x", labelrotation=60)
    figure.text(.02, .005, "SYNTHETIC MOCK - not model-quality evidence" if synthetic else "One value per image: mean across visits. Whiskers: Tukey 1.5 IQR.", fontsize=8)
    figure.tight_layout(rect=(0, .04, 1, 1))
    save_figure(figure, stem)


def baseline_report(root, summary, synthetic):
    folder = root / "baseline"
    boxes([summary], [f"Real (n={summary['n']})"], folder / "real_boxplot", "Real-to-real reference", synthetic)


def inference_report(folder, real, generated, rows, usage, distance):
    synthetic = usage["synthetic"]
    boxes([real, generated], [f"Real (n={real['n']})", f"Generated (n={generated['n']})"], folder / "comparison", "Inference evaluation", synthetic)
    write_json(folder / "comparison.json", {"real": real, "generated": generated, "box_distance": distance, "usage": usage})
    cells = "".join("<tr><td>" + html.escape(str(row["file_path"])) + "</td><td>" + f"{row['score_mean']:.4f}" + "</td><td>" + f"{row['score_std_across_visits']:.4f}" + "</td></tr>" for row in rows)
    warning = "<p class='notice'>SYNTHETIC MOCK DATA: this is a software demonstration, not an image-quality experiment.</p>" if synthetic else ""
    content = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AI Self Evaluation - inference</title><style>body{{font:16px system-ui;margin:32px auto;padding:0 20px;max-width:1100px;color:#20333b}}img{{max-width:100%}}table{{border-collapse:collapse;width:100%}}td,th{{padding:9px;text-align:left;border-bottom:1px solid #ddd;overflow-wrap:anywhere}}.notice{{background:#fff0c2;padding:16px}}code{{background:#eef3f5}}</style>
<h1>Inference evaluation</h1>{warning}<p>Box distance to the real reference: <b>{distance:.5f}</b>. Scores describe consistency with the supplied references; they are not probabilities of correctness or downstream utility.</p>
<img src="comparison.png" alt="Real and generated score box plots"><p>Each box uses per-image means. Whiskers are Tukey 1.5 IQR, not confidence intervals. Both sets use the same evaluation protocol.</p>
<p>Machine-readable results: <a href="scores.csv">scores.csv</a>, <a href="comparison.json">comparison.json</a>.</p>
<table><thead><tr><th>Image</th><th>Mean score</th><th>SD across visits</th></tr></thead><tbody>{cells}</tbody></table></html>"""
    (folder / "report.html").write_text(content, encoding="utf-8")


def training_report(root, real, observations, settings, synthetic):
    folder = root / "training"
    folder.mkdir(exist_ok=True)
    summaries, labels = [real], ["Real"]
    rows = []
    for observation in observations:
        summaries.append(observation["summary"])
        labels.append(str(observation["epoch"]))
        rows.append({"epoch": observation["epoch"], "distance": observation["distance"], **observation["summary"],
                     "best_epoch": observation["decision"]["best_epoch"], "stalled_epochs": observation["decision"]["stalled_epochs"],
                     "should_stop": observation["decision"]["should_stop"], "reason": observation["decision"]["reason"]})
    page_size = settings["reporting"]["boxes_per_page"]
    pages = []
    for page, start in enumerate(range(1, len(summaries), page_size), 1):
        name = f"checkpoint_boxes_{page:03d}"
        boxes([real] + summaries[start:start + page_size], ["Real"] + labels[start:start + page_size], folder / name,
              "Checkpoint scores - x axis: epoch", synthetic)
        pages.append(name)
    figure, axis = plt.subplots(figsize=(9, 4.5))
    axis.plot([r["epoch"] for r in rows], [r["distance"] for r in rows], marker="o", color="#317b85", label="Checkpoint distance")
    axis.axhline(settings["early_stopping"]["target_tolerance"], color="#b47827", linestyle="--", label="Target tolerance")
    best = observations[-1]["decision"]["best_epoch"]
    axis.axvline(best, color="#44545b", linestyle=":", label=f"Best: epoch {best}")
    axis.set(xlabel="Epoch", ylabel="Box distance (smaller is closer)", title="SYNTHETIC MOCK trajectory" if synthetic else "Distance to real reference")
    axis.legend()
    axis.grid(alpha=.2)
    figure.tight_layout()
    save_figure(figure, folder / "distance_curve")
    write_csv(folder / "history.csv", rows)
    write_json(folder / "history.json", observations)
    images = "".join(f'<img src="{name}.png" alt="Checkpoint box plots page {i}">' for i, name in enumerate(pages, 1))
    decision = observations[-1]["decision"]
    (folder / "report.html").write_text(f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Training evaluation</title>
<style>body{{font:16px system-ui;max-width:1200px;margin:32px auto;padding:20px;color:#20333b}}img{{max-width:100%}}pre{{white-space:pre-wrap}}</style>
<h1>Checkpoint monitoring</h1><p>{"SYNTHETIC MOCK - software demonstration only." if synthetic else "Reference-based monitoring; quality and diversity require independent validation."}</p>
<p>Best epoch: {best}. Latest decision: {html.escape(decision['reason'])}. Stop: {decision['should_stop']}.</p>
<img src="distance_curve.png" alt="Distance trajectory">{images}<p>All observations: <a href="history.csv">CSV</a> / <a href="history.json">JSON</a>. Individual scores are in each epoch folder. A missing observation does not advance patience.</p></html>''', encoding="utf-8")
