"""Plot the complete Merge MU/FQ sweep without selecting an operating point."""
import argparse
import csv
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from select_k import read_candidates


def style_panel(axis, title, handles, labels):
    """Use a square, fully framed panel with its legend above the data."""
    axis.set_box_aspect(1)
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)
    axis.set_title(title, pad=30 if len(labels) <= 2 else 44)
    axis.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, 1.005),
                ncol=2, frameon=False, fontsize=10, borderaxespad=0,
                handlelength=1.8, columnspacing=1.2, handletextpad=.5)
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis.grid(alpha=.2)


def make_figure():
    """Keep both exports identical in panel size and compact outer spacing."""
    figure = plt.figure(figsize=(12.4, 6))
    axes = [figure.add_axes([left / 12.4, .55 / 6, 4.7 / 12.4, 4.7 / 6])
            for left in (.8, 6.9)]
    return figure, axes


def plot_merge(results_dir):
    """Read measured k=1..L results and save two panels plus their source values."""
    rows = read_candidates(results_dir)
    if not all(math.isfinite(row[key]) for row in rows
               for key in ("model_utility", "forget_quality")):
        raise ValueError("Cannot plot a complete sweep with non-finite MU/FQ")
    ks = [row["k"] for row in rows]
    figure, axes = make_figure()
    for axis, metric, label, color in (
        (axes[0], "model_utility", "Model Utility", "#75449a"),
        (axes[1], "forget_quality", "Forget Quality", "#b86b2b"),
    ):
        axis.plot(ks, [row[metric] for row in rows], marker="o", markersize=4,
                  linewidth=1.8, color=color, label=label)
        axis.set_xlabel("Number of merged layers (k)")
        axis.set_ylabel(label)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.ticklabel_format(axis="y", style="plain", useOffset=False)
        handles, labels = axis.get_legend_handles_labels()
        style_panel(axis, "Cumulative Merge (test)" if (results_dir / "evaluation_protocol.json").is_file() else "Cumulative Merge", handles, labels)
    paths = []
    try:
        for extension in ("png", "pdf"):
            path = results_dir / f"merge_curves.{extension}"
            figure.savefig(path, dpi=180)
            paths.append(path)
    finally:
        plt.close(figure)
    data_path = results_dir / "merge_curves.csv"
    with data_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["k", "model_utility", "forget_quality"])
        writer.writeheader()
        writer.writerows({key: row[key] for key in writer.fieldnames} for row in rows)
    return [*paths, data_path]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        paths = plot_merge(args.results_dir)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Plot error: {error}\n")
    for path in paths:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
