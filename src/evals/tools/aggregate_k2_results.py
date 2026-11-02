from pathlib import Path as _CRUPath
import sys as _cru_sys
import os as _cru_os
_cru_root = _CRUPath(__file__).resolve().parents[3]
if str(_cru_root / "src") not in _cru_sys.path:
    _cru_sys.path.insert(0, str(_cru_root / "src"))
from cru_paths import CRU_ROOT
if __name__ == "__main__":
    _cru_os.chdir(CRU_ROOT)

"""
Aggregate exhaustive k=2 results and compare greedy top-2 vs. optimal pair.

For each (block, method), reads all 120 TOFU_SUMMARY.json files from
saves_new/k2_blkB_TAG/pair_PAIRCODE/TOFU_SUMMARY.json, computes rank-sum
scores (same criterion as CRU), finds the optimal pair, and compares it
against the greedy top-2 from the existing select files.

Usage:
    python src/evals/tools/aggregate_k2_results.py
"""

import json
import os
from itertools import combinations
from scipy.stats import rankdata

SAVE_ROOT = "saves_new"
SELECT_ROOT = "select"
MODEL = "1B"
NUM_LAYERS = 16

# All C(16,2) = 120 pairs, in the same order as the array job mapping
ALL_PAIRS = [(i, j) for i in range(NUM_LAYERS) for j in range(i + 1, NUM_LAYERS)]

def pair_to_code(i, j):
    return i * 100 + j

def load_summary(path):
    with open(path) as f:
        d = json.load(f)
    return d["model_utility"], d["forget_quality"]

def load_greedy_top2(block, tag):
    """Load top-2 layers from the existing CRU select file."""
    path = os.path.join(SELECT_ROOT, f"{MODEL}-{tag}-b{block}-new.txt")
    with open(path) as f:
        lines = f.readlines()
    selected = list(map(int, lines[2].split(": ")[1].strip().split(",")))
    top2 = tuple(sorted(selected[:2]))
    return top2

def rank_sum_score(mus, fqs):
    """Lower rank-sum = better (higher MU rank + higher FQ rank = smaller rank indices)."""
    # rankdata assigns rank 1 to smallest; we want rank 1 for highest MU and highest FQ
    # so we negate before ranking
    mu_ranks = rankdata([-m for m in mus])   # rank 1 = highest MU
    fq_ranks = rankdata([-f for f in fqs])   # rank 1 = highest FQ (least negative)
    return [mu_ranks[i] + fq_ranks[i] for i in range(len(mus))]

def main():
    print(f"{'Block':>5} {'Method':>6} | {'Greedy top-2':>14} {'Greedy MU':>10} {'Greedy FQ':>10} | "
          f"{'Optimal pair':>14} {'Optimal MU':>10} {'Optimal FQ':>10} | {'Gap (rank-sum)':>14}")
    print("-" * 110)

    for block in range(4):
        for tag in ["GA", "NPO"]:
            results = {}  # pair -> (MU, FQ)
            missing = []

            for (i, j) in ALL_PAIRS:
                code = pair_to_code(i, j)
                summary_path = os.path.join(
                    SAVE_ROOT, f"k2_blk{block}_{tag}", f"pair_{code}", "TOFU_SUMMARY.json"
                )
                if not os.path.exists(summary_path):
                    missing.append((i, j))
                    continue
                mu, fq = load_summary(summary_path)
                results[(i, j)] = (mu, fq)

            if missing:
                print(f"Block {block} {tag}: missing {len(missing)}/120 results — skipping")
                continue

            pairs = list(results.keys())
            mus = [results[p][0] for p in pairs]
            fqs = [results[p][1] for p in pairs]
            scores = rank_sum_score(mus, fqs)

            # Best pair (lowest rank-sum score)
            best_idx = scores.index(min(scores))
            best_pair = pairs[best_idx]
            best_mu, best_fq = results[best_pair]
            best_score = scores[best_idx]

            # Greedy top-2
            try:
                greedy_pair = load_greedy_top2(block, tag)
            except FileNotFoundError:
                greedy_pair = None

            if greedy_pair and greedy_pair in results:
                greedy_mu, greedy_fq = results[greedy_pair]
                greedy_idx = pairs.index(greedy_pair)
                greedy_score = scores[greedy_idx]
                gap = greedy_score - best_score
            else:
                greedy_mu = greedy_fq = greedy_score = gap = float("nan")

            print(
                f"{block:>5} {tag:>6} | "
                f"{str(greedy_pair):>14} {greedy_mu:>10.4f} {greedy_fq:>10.4f} | "
                f"{str(best_pair):>14} {best_mu:>10.4f} {best_fq:>10.4f} | "
                f"{gap:>14.1f}"
            )

            # Also save full results table
            out_path = os.path.join(SAVE_ROOT, f"k2_blk{block}_{tag}", "k2_summary.json")
            out = {
                "all_pairs": {str(p): {"MU": results[p][0], "FQ": results[p][1], "rank_sum": scores[pairs.index(p)]}
                              for p in pairs},
                "best_pair": list(best_pair),
                "best_MU": best_mu,
                "best_FQ": best_fq,
                "best_rank_sum": best_score,
                "greedy_top2": list(greedy_pair) if greedy_pair else None,
                "greedy_MU": greedy_mu,
                "greedy_FQ": greedy_fq,
                "greedy_rank_sum": greedy_score,
                "rank_sum_gap": gap,
            }
            with open(out_path, "w") as f:
                json.dump(out, f, indent=2)
            print(f"  → saved to {out_path}")

if __name__ == "__main__":
    main()
