from pathlib import Path as _CRUPath
import sys as _cru_sys
_cru_root = _CRUPath(__file__).resolve().parents[2]
if str(_cru_root) not in _cru_sys.path:
    _cru_sys.path.insert(0, str(_cru_root))

"""
Layer selection for block cross-validation experiments.

Reads per-layer TOFU_SUMMARY.json from:
  saves_new/blk{k}_{method}/evals-equal-new-{layer}/TOFU_SUMMARY.json

Writes selection files to:
  select/1B-GA-b{k}-new.txt
  select/1B-NPO-b{k}-new.txt

File format (3 lines):
  model_utility: <indices sorted by model_utility desc>
  forget_quality: <indices sorted by forget_quality desc>
  selected_indices: <combined-rank sorted indices>

This optional utility reads a legacy result layout. Its selection files are not
consumed by the default five-stage pipeline or the current model loader.
"""

import os
import json
import math

BASE = "saves_new"
SELECT_DIR = "select"
NUM_LAYERS = 16
BLOCKS = [0, 1, 2, 3]
METHODS = ["GA", "NPO"]  # short names; dirs use these directly

os.makedirs(SELECT_DIR, exist_ok=True)

for block in BLOCKS:
    for method in METHODS:
        eval_dir = os.path.join(BASE, f"blk{block}_{method}")
        mu_list = []
        fq_list = []

        for layer in range(NUM_LAYERS):
            json_path = os.path.join(eval_dir, f"evals-equal-new-{layer}", "TOFU_SUMMARY.json")
            with open(json_path) as f:
                info = json.load(f)
            for metric, values in (("model_utility", mu_list), ("forget_quality", fq_list)):
                value = info.get(metric)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError(f"{json_path}: {metric} must be a finite number")
                values.append((layer, value))

        # Sort descending (higher = better), drop missing
        mu_valid = [(i, v) for i, v in mu_list if v is not None]
        fq_valid = [(i, v) for i, v in fq_list if v is not None]

        mu_sorted = [i for i, _ in sorted(mu_valid, key=lambda x: (-x[1], x[0]))]
        fq_sorted = [i for i, _ in sorted(fq_valid, key=lambda x: (-x[1], x[0]))]

        # Combined rank: sum of rank positions
        mu_rank = {i: r for r, i in enumerate(mu_sorted)}
        fq_rank = {i: r for r, i in enumerate(fq_sorted)}
        common = set(mu_rank) & set(fq_rank)

        combined = sorted([(mu_rank[i] + fq_rank[i], i) for i in common])
        selected = [i for _, i in combined]

        out_path = os.path.join(SELECT_DIR, f"1B-{method}-b{block}-new.txt")
        with open(out_path, "w") as f:
            f.write(f"model_utility: {','.join(map(str, mu_sorted))}\n")
            f.write(f"forget_quality: {','.join(map(str, fq_sorted))}\n")
            f.write(f"selected_indices: {','.join(map(str, selected))}\n")

        print(f"Block {block} {method}: mu={mu_sorted} fq={fq_sorted} selected={selected}")
        print(f"  -> {out_path}")

print("\nDone.")
