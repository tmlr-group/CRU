from pathlib import Path as _CRUPath
import sys as _cru_sys
_cru_root = _CRUPath(__file__).resolve().parents[3]
if str(_cru_root / "src") not in _cru_sys.path:
    _cru_sys.path.insert(0, str(_cru_root / "src"))
from cru_paths import CRU_INPUT_ROOT, CRU_ROOT
if __name__ == '__main__':
    import os as _cru_os
    _cru_os.chdir(CRU_ROOT)

import os
import json
from typing import Dict, List, Optional, Tuple


BASE_DIR = str(CRU_INPUT_ROOT / 'saves/unlearn/tofu_Llama-3.2-1B-Instruct_forget05_NPO')
EVALS_VALUES = list(range(16))
METRICS = {
    "mu": "model_utility",
    "fq": "forget_quality",
}


def read_json(path: str) -> Optional[Dict]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def find_baseline_dir(base_dir: str, prefix: str) -> Optional[str]:
    """
    Find a (wrt/new/plain) baseline directory whose name starts with `prefix`.
    Preference:
      1) exact match
      2) largest numeric suffix
      3) lexicographically first match
    """
    exact = os.path.join(base_dir, prefix)
    if os.path.isdir(exact) and os.path.exists(os.path.join(exact, "TOFU_SUMMARY.json")):
        return exact

    candidates: List[str] = []
    try:
        for name in os.listdir(base_dir):
            if not name.startswith(prefix):
                continue
            full = os.path.join(base_dir, name)
            if os.path.isdir(full) and os.path.exists(os.path.join(full, "TOFU_SUMMARY.json")):
                candidates.append(full)
    except FileNotFoundError:
        return None

    if not candidates:
        return None

    def numeric_suffix(path: str) -> Optional[int]:
        tail = os.path.basename(path).split("-")[-1]
        return int(tail) if tail.isdigit() else None

    numeric: List[Tuple[int, str]] = []
    non_numeric: List[str] = []
    for c in candidates:
        n = numeric_suffix(c)
        if n is not None:
            numeric.append((n, c))
        else:
            non_numeric.append(c)

    if numeric:
        numeric.sort(key=lambda x: x[0], reverse=True)
        return numeric[0][1]

    non_numeric.sort()
    return non_numeric[0]


def fmt4(x: Optional[float]) -> str:
    if x is None:
        return "-"
    try:
        return f"{float(x):.4f}"
    except Exception:
        return "-"


def collect_select_metric_values(base_dir: str, metric_key: str, idxs: List[int]) -> List[Optional[float]]:
    values: List[Optional[float]] = []
    for idx in idxs:
        sel_dir = os.path.join(base_dir, f"evals-select-new-{idx}")
        summary_path = os.path.join(sel_dir, "TOFU_SUMMARY.json")
        info = read_json(summary_path)
        values.append(None if info is None else info.get(metric_key))
    return values


def main():
    # Baseline-new (fallback to baseline), and baseline-wrt
    baseline_new_dir = find_baseline_dir(BASE_DIR, "evals-baseline-new")
    if baseline_new_dir is None:
        baseline_new_dir = find_baseline_dir(BASE_DIR, "evals-baseline")
    baseline_wrt_dir = find_baseline_dir(BASE_DIR, "evals-baseline-wrt")

    baseline_new = read_json(os.path.join(baseline_new_dir, "TOFU_SUMMARY.json")) if baseline_new_dir else None
    baseline_wrt = read_json(os.path.join(baseline_wrt_dir, "TOFU_SUMMARY.json")) if baseline_wrt_dir else None

    # Collect select values per idx
    mu_select_vals = collect_select_metric_values(BASE_DIR, METRICS["mu"], EVALS_VALUES)
    fq_select_vals = collect_select_metric_values(BASE_DIR, METRICS["fq"], EVALS_VALUES)

    # Prepare rows for two tables (mu and fq)
    # Each table: 3 rows (baseline, wrt, select) x 16 columns (idx 0..15), without row labels.
    mu_baseline_row = [fmt4(baseline_new.get(METRICS["mu"]) if baseline_new else None)] * len(EVALS_VALUES)
    mu_wrt_row = [fmt4(baseline_wrt.get(METRICS["mu"]) if baseline_wrt else None)] * len(EVALS_VALUES)
    mu_select_row = [fmt4(v) for v in mu_select_vals]

    fq_baseline_row = [fmt4(baseline_new.get(METRICS["fq"]) if baseline_new else None)] * len(EVALS_VALUES)
    fq_wrt_row = [fmt4(baseline_wrt.get(METRICS["fq"]) if baseline_wrt else None)] * len(EVALS_VALUES)
    fq_select_row = [fmt4(v) for v in fq_select_vals]

    # Markdown header (16 columns): indices 0..15
    header = "|" + "|".join(str(i) for i in EVALS_VALUES) + "|"
    sep = "|" + "|".join(["---"] * len(EVALS_VALUES)) + "|"

    # Print MU table
    print("Metric: model_utility (rows: baseline, wrt, select)")
    print(header)
    print(sep)
    print("|" + "|".join(mu_baseline_row) + "|")
    print("|" + "|".join(mu_wrt_row) + "|")
    print("|" + "|".join(mu_select_row) + "|")
    print()

    # Print FQ table
    print("Metric: forget_quality (rows: baseline, wrt, select)")
    print(header)
    print(sep)
    print("|" + "|".join(fq_baseline_row) + "|")
    print("|" + "|".join(fq_wrt_row) + "|")
    print("|" + "|".join(fq_select_row) + "|")


if __name__ == "__main__":
    main()






