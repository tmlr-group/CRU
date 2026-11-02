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
Aggregate exhaustive k=2 results and compare greedy top-2 vs. true optimal pair.

Reads saves_o/k2_{GA,NPO}/pair_*/TOFU_SUMMARY.json, computes rank-sum scores,
finds the optimal pair, and compares against the greedy top-2 from select files.

Usage:
    python src/evals/tools/aggregate_k2.py
"""

import json
import os
from scipy.stats import rankdata

SAVE_ROOT = "saves_o"
SELECT_ROOT = "select"
NUM_LAYERS = 16
ALL_PAIRS = [(i, j) for i in range(NUM_LAYERS) for j in range(i + 1, NUM_LAYERS)]

def pair_to_code(i, j):
    return i * 100 + j

def load_summary(path):
    with open(path) as f:
        d = json.load(f)
    return d["model_utility"], d["forget_quality"]

def load_greedy_top2(tag):
    path = os.path.join(SELECT_ROOT, f"1B-{tag}-new.txt")
    with open(path) as f:
        lines = f.readlines()
    selected = list(map(int, lines[2].split(": ")[1].strip().split(",")))
    return tuple(sorted(selected[:2]))

def rank_sum_scores(mus, fqs):
    mu_ranks = rankdata([-m for m in mus])
    fq_ranks = rankdata([-f for f in fqs])
    return [mu_ranks[i] + fq_ranks[i] for i in range(len(mus))]

def main():
    header = (f"{'Method':>6} | {'Greedy top-2':>14} {'Greedy MU':>10} {'Greedy FQ':>10} "
              f"{'Greedy rank-sum':>15} | {'Optimal pair':>14} {'Optimal MU':>10} "
              f"{'Optimal FQ':>10} {'Optimal rank-sum':>16} | {'Gap':>5}")
    print(header)
    print("-" * len(header))

    for tag in ["GA", "NPO"]:
        results = {}
        missing = []

        for (i, j) in ALL_PAIRS:
            code = pair_to_code(i, j)
            path = os.path.join(SAVE_ROOT, f"k2_{tag}", f"pair_{code}", "TOFU_SUMMARY.json")
            if not os.path.exists(path):
                missing.append((i, j))
                continue
            mu, fq = load_summary(path)
            results[(i, j)] = (mu, fq)

        if missing:
            print(f"{tag:>6}: missing {len(missing)}/120 — run more jobs first")
            continue

        pairs = list(results.keys())
        mus = [results[p][0] for p in pairs]
        fqs = [results[p][1] for p in pairs]
        scores = rank_sum_scores(mus, fqs)

        best_idx = scores.index(min(scores))
        best_pair = pairs[best_idx]
        best_mu, best_fq = results[best_pair]
        best_score = scores[best_idx]

        greedy_pair = load_greedy_top2(tag)
        if greedy_pair in results:
            g_mu, g_fq = results[greedy_pair]
            g_score = scores[pairs.index(greedy_pair)]
        else:
            g_mu = g_fq = g_score = float("nan")

        gap = g_score - best_score

        print(f"{tag:>6} | {str(greedy_pair):>14} {g_mu:>10.4f} {g_fq:>10.4f} {g_score:>15.1f} | "
              f"{str(best_pair):>14} {best_mu:>10.4f} {best_fq:>10.4f} {best_score:>16.1f} | {gap:>5.1f}")

        out = {
            "greedy_top2": list(greedy_pair), "greedy_MU": g_mu, "greedy_FQ": g_fq,
            "greedy_rank_sum": float(g_score),
            "optimal_pair": list(best_pair), "optimal_MU": best_mu, "optimal_FQ": best_fq,
            "optimal_rank_sum": float(best_score), "rank_sum_gap": float(gap),
            "all_pairs": {str(p): {"MU": results[p][0], "FQ": results[p][1],
                                   "rank_sum": float(scores[pairs.index(p)])} for p in pairs},
        }
        out_path = os.path.join(SAVE_ROOT, f"k2_{tag}", "summary.json")
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"  → {out_path}")

if __name__ == "__main__":
    main()
