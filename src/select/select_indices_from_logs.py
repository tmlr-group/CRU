"""Select CRU layers from per-layer replacement evaluation summaries."""
import argparse
import json
from pathlib import Path


def read_summary(path: Path) -> tuple[float, float] | None:
    try:
        with path.open() as handle:
            payload = json.load(handle)
        return float(payload["model_utility"]), float(payload["forget_quality"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=4)
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be positive")

    rows = []
    for candidate in sorted(args.results_dir.glob("layer-*")):
        try:
            layer = int(candidate.name.removeprefix("layer-"))
        except ValueError:
            continue
        values = read_summary(candidate / "TOFU_SUMMARY.json")
        if values is not None:
            rows.append((layer, *values))
    if not rows:
        raise SystemExit(f"No TOFU_SUMMARY.json files found under {args.results_dir}")

    utility_rank = {layer: rank for rank, (layer, _, _) in enumerate(sorted(rows, key=lambda r: (-r[1], r[0])), 1)}
    quality_rank = {layer: rank for rank, (layer, _, _) in enumerate(sorted(rows, key=lambda r: (-r[2], r[0])), 1)}
    ranked = sorted(rows, key=lambda r: (utility_rank[r[0]] + quality_rank[r[0]], r[0]))
    selected = [layer for layer, _, _ in ranked[:args.top_k]]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "# CRU selection; generated from replacement evaluation results\n"
        + "selected_layers: " + ",".join(map(str, selected)) + "\n"
        + "# layer,model_utility,forget_quality,rank_sum\n"
        + "\n".join(
            f"{layer},{utility:.10g},{quality:.10g},{utility_rank[layer] + quality_rank[layer]}"
            for layer, utility, quality in ranked
        ) + "\n"
    )
    print(f"Wrote {len(selected)} selected layers to {args.output}: {selected}")


if __name__ == "__main__":
    main()
