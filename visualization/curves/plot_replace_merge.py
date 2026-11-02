"""Plot complete independent Replace and cumulative Merge evaluations."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from plot_merge import read_candidates, plt, style_panel, make_figure

METRICS = ("model_utility", "forget_quality")


def read_metrics(path):
    summary = json.loads(path.read_text())
    values = {key: float(summary[key]) for key in METRICS}
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError(f"Non-finite MU/FQ in {path}")
    return values


def read_results(replace_dir, merge_dir):
    merge = read_candidates(merge_dir)
    ranking = json.loads((merge_dir / "ranking.json").read_text())
    count = ranking["num_layers"]
    candidates = ranking["candidates"]
    if (len(candidates) != count
            or any(type(row["layer"]) is not int for row in candidates)
            or sorted(row["layer"] for row in candidates) != list(range(count))):
        raise ValueError("Ranking must contain every Replace candidate exactly once")
    indexed = {row["layer"]: row for row in candidates}
    replace = []
    for layer in range(count):
        path = replace_dir / f"layer-{layer}" / "TOFU_SUMMARY.json"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != indexed[layer]["summary_sha256"]:
            raise ValueError(f"Layer {layer}: Replace result differs from the Merge ranking source")
        replace.append(dict(layer=layer, **read_metrics(path)))
    for row in merge:
        if not all(math.isfinite(row[key]) for key in METRICS):
            raise ValueError(f"Non-finite Merge MU/FQ for k={row['k']}")
    return replace, merge


def plot_results(replace_dir, merge_dir, output_dir, baseline=None, original=None):
    replace, merge = read_results(replace_dir, merge_dir)
    heldout = bool(json.loads((merge_dir / "ranking.json").read_text()).get("evaluation"))
    if heldout and original:
        raise ValueError("A single reference summary cannot represent both validation and test panels")
    baseline_values = read_metrics(baseline) if baseline else None
    original_values = read_metrics(original) if original else None
    merge_points = ([dict(k=0, **baseline_values)] if baseline_values else []) + merge
    figure, axes = make_figure()
    try:
        for axis, kind, points, coordinate in (
            (axes[0], "Replace", replace, "layer"),
            (axes[1], "Merge", merge_points, "k"),
        ):
            right = axis.twinx()
            for target, metric, label, color in (
                (axis, "model_utility", "Model Utility", "#75449a"),
                (right, "forget_quality", "Forget Quality", "#b86b2b"),
            ):
                target.plot([row[coordinate] for row in points],
                            [row[metric] for row in points], "o-", color=color,
                            markersize=4, label=label)
                if original_values:
                    target.axhline(original_values[metric], color=color,
                                   linestyle="--", alpha=.7, label=f"Original {label}")
                target.set_ylabel(label, color=color)
                target.tick_params(axis="y", colors=color)
                target.ticklabel_format(axis="y", style="plain", useOffset=False)
            axis.set_xlabel("Physical layer index" if kind == "Replace" else "Number of merged layers (k)")
            handles, labels = axis.get_legend_handles_labels()
            other_handles, other_labels = right.get_legend_handles_labels()
            style_panel(axis, ("Replace (validation)" if kind == "Replace" else "Merge (test)") if heldout else
                        ("Independent Replace" if kind == "Replace" else "Cumulative Merge"),
                        handles + other_handles, labels + other_labels)
            right.set_box_aspect(1)
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for extension in ("png", "pdf"):
            path = output_dir / f"replace_merge.{extension}"
            figure.savefig(path, dpi=180)
            paths.append(path)
    finally:
        plt.close(figure)
    path = output_dir / "replace_merge.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["kind", "layer", "k", *METRICS])
        writer.writeheader()
        for kind, points, coordinate in (("replace", replace, "layer"), ("merge", merge_points, "k")):
            for row in points:
                writer.writerow(dict(kind=kind, **{coordinate: row[coordinate]},
                                     **{key: row[key] for key in METRICS}))
        if original_values:
            writer.writerow(dict(kind="original", **original_values))
    return [*paths, path]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replace-dir", type=Path, required=True)
    parser.add_argument("--merge-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, help="Default: <merge-dir>/figures")
    parser.add_argument("--baseline-summary", type=Path, help="Full-FT TOFU_SUMMARY.json; adds measured k=0")
    parser.add_argument("--original-summary", type=Path, help="Unlearn TOFU_SUMMARY.json; adds reference lines")
    args = parser.parse_args()
    try:
        paths = plot_results(args.replace_dir, args.merge_dir,
                             args.output_dir or args.merge_dir / "figures",
                             args.baseline_summary, args.original_summary)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Plot error: {error}\n")
    for path in paths:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
