from pathlib import Path as _CRUPath
import sys as _cru_sys
_cru_root = _CRUPath(__file__).resolve().parents[2]
if str(_cru_root / "src") not in _cru_sys.path:
    _cru_sys.path.insert(0, str(_cru_root / "src"))
from cru_paths import CRU_INPUT_ROOT, CRU_ROOT
if __name__ == '__main__':
    import os as _cru_os
    _cru_os.chdir(CRU_ROOT)

import os
import json
from typing import Dict, List, Tuple


MODEL_CONFIGS = {
    "1B-GA": {
        "base_dir": str(CRU_INPUT_ROOT / 'saves/unlearn/tofu_Llama-3.2-1B-Instruct_forget05_GradAscent'),
        "evals_values": list(range(16)),
    },
    "1B-NPO": {
        "base_dir": str(CRU_INPUT_ROOT / 'saves/unlearn/tofu_Llama-3.2-1B-Instruct_forget05_NPO'),
        "evals_values": list(range(16)),
    },
    # "1B-SimNPO": {
    #     "base_dir": "../open-unlearning/saves/unlearn/tofu_Llama-3.2-1B-Instruct_forget05_SimNPO",
    #     "evals_values": list(range(16)),
    # },
    # "1B-AltPO": {
    #     "base_dir": "../open-unlearning/saves/unlearn/tofu_Llama-3.2-1B-Instruct_forget05_AltPO",
    #     "evals_values": list(range(16)),
    # },
    # "1B-SatImp": {
    #     "base_dir": "../open-unlearning/saves/unlearn/tofu_Llama-3.2-1B-Instruct_forget05_SatImp",
    #     "evals_values": list(range(16)),
    # },
    # "1B-RMU": {
    #     "base_dir": "../open-unlearning/saves/unlearn/tofu_Llama-3.2-1B-Instruct_forget05_RMU",
    #     "evals_values": list(range(16)),
    # },
    "3B-GA": {
        "base_dir": str(CRU_INPUT_ROOT / 'saves/unlearn/tofu_Llama-3.2-3B-Instruct_forget05_GradAscent'),
        "evals_values": list(range(28)),
    },
    "3B-NPO": {
        "base_dir": str(CRU_INPUT_ROOT / 'saves/unlearn/tofu_Llama-3.2-3B-Instruct_forget05_NPO'),
        "evals_values": list(range(28)),
    },
    # "3B-SimNPO": {
    #     "base_dir": "../open-unlearning/saves/unlearn/tofu_Llama-3.2-3B-Instruct_forget05_SimNPO",
    #     "evals_values": list(range(28)),
    # },
    # "3B-AltPO": {
    #     "base_dir": "../open-unlearning/saves/unlearn/tofu_Llama-3.2-3B-Instruct_forget05_AltPO",
    #     "evals_values": list(range(28)),
    # },
    # "3B-SatImp": {
    #     "base_dir": "../open-unlearning/saves/unlearn/tofu_Llama-3.2-3B-Instruct_forget05_SatImp",
    #     "evals_values": list(range(28)),
    # },
    "7B-GA": {
        "base_dir": str(CRU_INPUT_ROOT / 'saves/unlearn/tofu_Llama-2-7b-chat-hf_forget05_GradAscent'),
        "evals_values": list(range(32)),
    },
    # "7B-SimNPO": {
    #     "base_dir": "../open-unlearning/saves/unlearn/tofu_Llama-2-7b-chat-hf_forget05_SimNPO",
    #     "evals_values": list(range(32)),
    # },
    # "7B-AltPO": {
    #     "base_dir": "../open-unlearning/saves/unlearn/tofu_Llama-2-7b-chat-hf_forget05_AltPO",
    #     "evals_values": list(range(32)),
    # },
    # "7B-SatImp": {
    #     "base_dir": "../open-unlearning/saves/unlearn/tofu_Llama-2-7b-chat-hf_forget05_SatImp",
    #     "evals_values": list(range(32)),
    # },
    "7B-NPO": {
        "base_dir": str(CRU_INPUT_ROOT / 'saves/unlearn/tofu_Llama-2-7b-chat-hf_forget05_NPO'),
        "evals_values": list(range(32)),
    },
    "Phi-GA": {
        "base_dir": str(CRU_INPUT_ROOT / 'saves/unlearn/tofu_Phi-3.5-mini-instruct_forget05_GradAscent'),
        "evals_values": list(range(32)),
    },
    "Phi-NPO": {
        "base_dir": str(CRU_INPUT_ROOT / 'saves/unlearn/tofu_Phi-3.5-mini-instruct_forget05_NPO'),
        "evals_values": list(range(32)),
    },
    # "Phi-SimNPO": {
    #     "base_dir": "../open-unlearning/saves/unlearn/tofu_Phi-3.5-mini-instruct_forget05_SimNPO",
    #     "evals_values": list(range(32)),
    # },
    # "Phi-AltPO": {
    #     "base_dir": "../open-unlearning/saves/unlearn/tofu_Phi-3.5-mini-instruct_forget05_AltPO",
    #     "evals_values": list(range(32)),
    # },
    # "Phi-SatImp": {
    #     "base_dir": "../open-unlearning/saves/unlearn/tofu_Phi-3.5-mini-instruct_forget05_SatImp",
    #     "evals_values": list(range(32)),
    # },
}


POSITIVE_METRICS = [
    "forget_quality",
    "model_utility",
    "extraction_strength_retain",
    "extraction_strength_retain_perturb",
]

NEGATIVE_METRICS = [
    "extraction_strength",
    "extraction_strength_perturb",
]


def _load_metric_value(json_path: str, metric_name: str):
    if not os.path.exists(json_path):
        return None
    try:
        with open(json_path, "r") as f:
            info = json.load(f)
        return info.get(metric_name, None)
    except Exception:
        return None


def _candidate_summary_paths(base_dir: str, idx: int) -> List[str]:
    # Prefer the "-more" directory if it exists; fall back to the original naming.
    return [
        os.path.join(base_dir, f"evals-equal-new-more-{idx}", "TOFU_SUMMARY.json"),
        os.path.join(base_dir, f"evals-equal-new-{idx}", "TOFU_SUMMARY.json"),
    ]


def get_sorted_indices_and_ranks(
    model_name: str,
    metric_name: str,
    higher_is_better: bool,
) -> Tuple[List[int], Dict[int, int]]:
    config = MODEL_CONFIGS[model_name]
    base_dir = config["base_dir"]
    evals_values = config["evals_values"]

    data: List[Tuple[int, float]] = []
    for idx in evals_values:
        value = None
        for path in _candidate_summary_paths(base_dir, idx):
            value = _load_metric_value(path, metric_name)
            if value is not None:
                break
        if value is not None:
            data.append((idx, value))

    if not data:
        return [], {}

    if higher_is_better:
        data.sort(key=lambda x: (-x[1], x[0]))
    else:
        data.sort(key=lambda x: (x[1], x[0]))

    sorted_indices = [idx for idx, _ in data]
    ranks = {idx: rank for rank, idx in enumerate(sorted_indices)}
    return sorted_indices, ranks


def main():
    # You can change this selection as needed.
    selected_models = [
        "7B-NPO",
        "1B-NPO",
        "3B-NPO",
        "Phi-NPO",
        "1B-GA",
        "3B-GA",
        "7B-GA",
        "Phi-GA"
    ]

    os.makedirs("select", exist_ok=True)

    for model in selected_models:
        # Compute ranks for all metrics
        metric_sorted_lists: Dict[str, List[int]] = {}
        metric_ranks: Dict[str, Dict[int, int]] = {}

        for m in POSITIVE_METRICS:
            sorted_list, ranks = get_sorted_indices_and_ranks(model, m, higher_is_better=True)
            metric_sorted_lists[m] = sorted_list
            metric_ranks[m] = ranks

        for m in NEGATIVE_METRICS:
            sorted_list, ranks = get_sorted_indices_and_ranks(model, m, higher_is_better=False)
            metric_sorted_lists[m] = sorted_list
            metric_ranks[m] = ranks

        # Intersect indices that have all six metrics
        if not metric_ranks:
            common_indices = set()
        else:
            common_indices = set.intersection(*(set(r.keys()) for r in metric_ranks.values()))

        # total_score = sum(positive ranks) - sum(negative ranks)
        # Note: ranks are 0-based. Lower total_score is better (ascending).
        combined: List[Tuple[float, int]] = []
        for idx in sorted(common_indices):
            pos_sum = sum(metric_ranks[m][idx] for m in POSITIVE_METRICS)
            neg_sum = sum(metric_ranks[m][idx] for m in NEGATIVE_METRICS)
            total_score = float(pos_sum - neg_sum)
            combined.append((total_score, idx))

        combined.sort(key=lambda x: (x[0], x[1]))  # lower score first, tiebreak by smaller idx
        selected_indices = [idx for _, idx in combined]

        # Write output file
        out_path = os.path.join("select", f"{model}-many.txt")
        with open(out_path, "w") as f:
            f.write(f"forget_quality: {','.join(map(str, metric_sorted_lists.get('forget_quality', [])))}\n")
            f.write(f"model_utility: {','.join(map(str, metric_sorted_lists.get('model_utility', [])))}\n")
            f.write(f"extraction_strength_retain: {','.join(map(str, metric_sorted_lists.get('extraction_strength_retain', [])))}\n")
            f.write(f"extraction_strength_retain_perturb: {','.join(map(str, metric_sorted_lists.get('extraction_strength_retain_perturb', [])))}\n")
            f.write(f"extraction_strength: {','.join(map(str, metric_sorted_lists.get('extraction_strength', [])))}\n")
            f.write(f"extraction_strength_perturb: {','.join(map(str, metric_sorted_lists.get('extraction_strength_perturb', [])))}\n")
            f.write(f"selected_indices: {','.join(map(str, selected_indices))}\n")

        # Echo brief confirmation to stdout
        print(f"[{model}] wrote {out_path}")
        print(f"  selected_indices (asc by total_score): {selected_indices}")


if __name__ == "__main__":
    main()


