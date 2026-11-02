"""Select the highest-FQ ranked prefix satisfying a user-specified MU floor."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def select_k(candidates, min_mu):
    """On equal FQ prefer smaller k; never relax an infeasible MU requirement."""
    if not math.isfinite(min_mu):
        raise ValueError("The MU threshold must be finite")
    for row in candidates:
        if not all(math.isfinite(row[key]) for key in ("model_utility", "forget_quality")):
            raise ValueError(f"Non-finite metrics for k={row['k']}")
    eligible = [row for row in candidates if row["model_utility"] >= min_mu]
    return min(eligible, key=lambda row: (-row["forget_quality"], row["k"])) if eligible else None


def read_candidates(sweep_dir):
    ranking_bytes = (sweep_dir / "ranking.json").read_bytes()
    ranking = json.loads(ranking_bytes)
    ranking_hash = hashlib.sha256(ranking_bytes).hexdigest()
    order = ranking["ranking"]
    count = ranking["num_layers"]
    if (type(count) is not int or count < 1 or not isinstance(order, list)
            or any(type(layer) is not int for layer in order)
            or sorted(order) != list(range(count))):
        raise ValueError("Expected a complete layer ranking")
    candidates = []
    for k in range(1, count + 1):
        eval_dir = sweep_dir / f"k-{k}" / "eval"
        summary = json.loads((eval_dir / "TOFU_SUMMARY.json").read_text())
        provenance = json.loads((eval_dir / "CRU_MERGE.json").read_text())
        if (provenance["k"] != k or provenance["selected_layers"] != order[:k]
                or provenance.get("ranking_sha256") != ranking_hash
                or provenance.get("base_model") != ranking.get("base_model")
                or provenance.get("base_config_sha256") != ranking.get("base_config_sha256")):
            raise ValueError(f"k={k}: result does not match the fixed ranking")
        candidates.append(dict(
            k=k,
            model_utility=float(summary["model_utility"]),
            forget_quality=float(summary["forget_quality"]),
            selected_layers=order[:k],
            summary=str((eval_dir / "TOFU_SUMMARY.json").resolve()),
        ))
    return candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True, help="Complete validation Merge sweep")
    parser.add_argument("--min-mu", type=float, required=True, help="Minimum acceptable validation MU")
    parser.add_argument("--output", type=Path, help="Default: <results-dir>/selection.json")
    args = parser.parse_args()
    try:
        protocol_file = args.results_dir / "evaluation_protocol.json"
        if protocol_file.is_file() and json.loads(protocol_file.read_text()).get("role") == "test":
            raise ValueError("Do not select k using held-out test results; provide a separate validation sweep")
        candidates = read_candidates(args.results_dir)
        selected = select_k(candidates, args.min_mu)
        result = dict(
            rule="max_fq_above_mu_floor",
            min_mu=args.min_mu,
            tie_break="smaller_k",
            status="selected" if selected else "no_feasible_candidate",
            selected=selected,
            candidates=candidates,
        )
        output = args.output or args.results_dir / "selection.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Selection error: {error}\n")
    if selected:
        print(f"Selected k={selected['k']}: MU={selected['model_utility']:.6g}, "
              f"FQ={selected['forget_quality']:.6g}. Saved to {output}")
    else:
        print(f"No candidate meets MU >= {args.min_mu}. Saved to {output}; no k selected.")


if __name__ == "__main__":
    main()
