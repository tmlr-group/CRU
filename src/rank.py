"""Rank all Transformer layers using their measured MU and signed FQ."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def model_layer_count(base_model: Path) -> int:
    config = json.loads((base_model / "config.json").read_text())
    count = config["num_hidden_layers"]
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError("Full-FT config must define a positive num_hidden_layers")
    return count


def load_ranking(path: Path, base_model: Path) -> dict:
    payload = json.loads(path.read_text())
    count = model_layer_count(base_model)
    order = payload["ranking"]
    if (
        payload.get("protocol") != "fullft_base_v2"
        or payload.get("num_layers") != count
        or not isinstance(order, list)
        or any(type(layer) is not int for layer in order)
        or sorted(order) != list(range(count))
    ):
        raise ValueError(f"{path}: expected a complete ranking of layers 0..{count - 1}")
    if Path(payload["base_model"]).resolve() != base_model.resolve():
        raise ValueError(f"{path}: ranking was generated for a different Full-FT base")
    digest = hashlib.sha256((base_model / "config.json").read_bytes()).hexdigest()
    if payload.get("base_config_sha256") != digest:
        raise ValueError(f"{path}: Full-FT config changed after ranking")
    return payload


def rank_layers(results_dir: Path, base_model: Path, output: Path) -> dict:
    count = model_layer_count(base_model)
    rows = []
    # A missing or invalid candidate must not silently shorten the ranking.
    for layer in range(count):
        path = results_dir / f"layer-{layer}" / "TOFU_SUMMARY.json"
        try:
            raw = path.read_bytes()
            summary = json.loads(raw)
            utility = float(summary["model_utility"])
            quality = float(summary["forget_quality"])
            if not math.isfinite(utility) or not math.isfinite(quality):
                raise ValueError("MU and FQ must be finite")
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise ValueError(f"Cannot rank layer {layer}: {path}: {error}") from error
        rows.append(dict(layer=layer, model_utility=utility, forget_quality=quality,
                         summary=str(path.resolve()), summary_sha256=hashlib.sha256(raw).hexdigest()))

    utility_rank = {row["layer"]: rank for rank, row in enumerate(
        sorted(rows, key=lambda row: (-row["model_utility"], row["layer"])), 1)}
    quality_rank = {row["layer"]: rank for rank, row in enumerate(
        sorted(rows, key=lambda row: (-row["forget_quality"], row["layer"])), 1)}
    for row in rows:
        row.update(utility_rank=utility_rank[row["layer"]], quality_rank=quality_rank[row["layer"]])
        row["rank_sum"] = row["utility_rank"] + row["quality_rank"]
    rows.sort(key=lambda row: (row["rank_sum"], row["layer"]))
    payload = dict(protocol="fullft_base_v2", base_model=str(base_model.resolve()),
                   base_config_sha256=hashlib.sha256((base_model / "config.json").read_bytes()).hexdigest(),
                   num_layers=count, ranking=[row["layer"] for row in rows], candidates=rows)
    evaluation_file = results_dir / "evaluation_protocol.json"
    if evaluation_file.is_file():
        evaluation = json.loads(evaluation_file.read_text())
        if evaluation.get("role") != "validation":
            raise ValueError("Layer ranking must use validation results, not test results")
        payload["evaluation"] = evaluation
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    rank = commands.add_parser("rank", help="Save the complete ordering; do not choose a prefix")
    rank.add_argument("--results-dir", type=Path, required=True)
    rank.add_argument("--base-model", type=Path, required=True)
    rank.add_argument("--output", type=Path, required=True)
    count = commands.add_parser("count", help="Validate a saved ranking and print its layer count")
    count.add_argument("--ranking", type=Path, required=True)
    count.add_argument("--base-model", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "count":
            print(load_ranking(args.ranking, args.base_model)["num_layers"])
        else:
            payload = rank_layers(args.results_dir, args.base_model, args.output)
            print(f"Wrote complete layer ranking to {args.output}: {payload['ranking']}")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Ranking error: {error}\n")


if __name__ == "__main__":
    main()
