"""Unified checkpoint-index selection for TOFU and MUSE evaluations."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from cru_paths import CRU_INPUT_ROOT, CRU_ROOT

TOFU_MODELS = {
    f"{size}-{method}": (f"tofu_Llama-3.2-{size}-Instruct_forget05_{'GradAscent' if method == 'GA' else 'NPO'}", count)
    for size, count in (("1B", 16), ("3B", 28)) for method in ("GA", "NPO")
}
TOFU_MODELS.update({
    f"7B-{method}": (f"tofu_Llama-2-7b-chat-hf_forget05_{'GradAscent' if method == 'GA' else 'NPO'}", 32)
    for method in ("GA", "NPO")
})
MUSE_MODELS = {
    f"{corpus}-{method}": (f"muse_Llama-2-7b-hf_{'News' if corpus == 'NEWS' else 'Books'}_{'GradAscent' if method == 'GA' else 'NPO' + ('_no' if corpus == 'BOOKS' and method == 'NPO' else '')}", 32)
    for corpus in ("NEWS", "BOOKS") for method in ("GA", "NPO")
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("partition", "validation", "layer", "muse"), required=True,
                        help="partition/validation rank fusion; layer top-half intersection; MUSE rank difference")
    parser.add_argument("--models", help="comma-separated model IDs; defaults depend on the selected mode")
    parser.add_argument("--parts", help="comma-separated module names (partition mode)")
    parser.add_argument("--splits", help="comma-separated validation splits (default: 01,02,03,04)")
    parser.add_argument("--base-dir", help="override the model checkpoint directory (single model only)")
    parser.add_argument("--output-dir", default="select")
    parser.add_argument("--top-k", type=int, help="limit selected indices after ranking")
    return parser


def _model_config(dataset: str, model: str, override: str | None):
    mapping = MUSE_MODELS if dataset == "muse" else TOFU_MODELS
    if model not in mapping:
        raise ValueError(f"Unknown {dataset} model {model!r}; choices: {', '.join(mapping)}")
    folder, count = mapping[model]
    return Path(override) if override else CRU_INPUT_ROOT / "saves" / "unlearn" / folder, count


def _read_ranked(base: Path, idxs: range, eval_dir: str, summary: str, metric: str):
    values = []
    for idx in idxs:
        path = base / eval_dir.format(idx=idx) / summary
        with path.open() as f:
            value = json.load(f).get(metric)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{path}: {metric} must be a finite number, got {value!r}")
        values.append((idx, value))
    values.sort(key=lambda item: (-item[1], item[0]))
    return [idx for idx, _ in values]


def _write_result(path: Path, first_name: str, first: list[int], second_name: str,
                  second: list[int], selected: list[int]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"{first_name}: {','.join(map(str, first))}\n"
        f"{second_name}: {','.join(map(str, second))}\n"
        f"selected_indices: {','.join(map(str, selected))}\n"
    )
    print(f"{path}: selected_indices={selected}")


def run(args):
    defaults = {"partition": "1B-NPO", "validation": "1B-NPO",
                "layer": "1B-GA,3B-GA,1B-NPO,3B-NPO", "muse": "BOOKS-GA,BOOKS-NPO"}
    models = [model.strip() for model in (args.models or defaults[args.mode]).split(",")]
    if not all(models) or len(set(models)) != len(models):
        raise ValueError("--models must contain distinct, nonempty model IDs")
    if args.base_dir and len(models) != 1:
        raise ValueError("--base-dir requires exactly one model")
    if args.top_k is not None and args.top_k < 1:
        raise ValueError("--top-k must be positive")
    args.models = ",".join(models)
    if args.mode == "partition":
        models = (args.models or "1B-NPO").split(",")
        parts = (args.parts or "self_attn,mlp,input_layernorm,post_attention_layernorm").split(",")
        for model in models:
            base, count = _model_config("tofu", model, args.base_dir)
            for part in parts:
                utility = _read_ranked(base, range(count), f"evals-{part}-{{idx}}", "TOFU_SUMMARY.json", "model_utility")
                quality = _read_ranked(base, range(count), f"evals-{part}-{{idx}}", "TOFU_SUMMARY.json", "forget_quality")
                ur, qr = {v:i for i,v in enumerate(utility)}, {v:i for i,v in enumerate(quality)}
                common = set(ur) & set(qr)
                selected = sorted(common, key=lambda i: (ur[i] + qr[i], i))
                if args.top_k is not None: selected = selected[:args.top_k]
                _write_result(Path(args.output_dir) / f"{model}-{part}.txt", "model_utility", utility, "forget_quality", quality, selected)
        return

    if args.mode == "validation":
        models = (args.models or "1B-NPO").split(",")
        splits = (args.splits or "01,02,03,04").split(",")
        for model in models:
            base, count = _model_config("tofu", model, args.base_dir)
            for split in splits:
                pattern = f"evals--val-{split}-{{idx}}"
                utility = _read_ranked(base, range(count), pattern, "TOFU_SUMMARY.json", "model_utility")
                quality = _read_ranked(base, range(count), pattern, "TOFU_SUMMARY.json", "forget_quality")
                ur, qr = {v:i for i,v in enumerate(utility)}, {v:i for i,v in enumerate(quality)}
                common = set(ur) & set(qr)
                selected = sorted(common, key=lambda i: (ur[i] + qr[i], i))
                if args.top_k is not None: selected = selected[:args.top_k]
                _write_result(Path(args.output_dir) / f"{model}-{split}-new.txt", "model_utility", utility, "forget_quality", quality, selected)
        return

    dataset = "muse" if args.mode == "muse" else "tofu"
    default_models = "BOOKS-GA,BOOKS-NPO" if dataset == "muse" else "1B-GA,3B-GA,1B-NPO,3B-NPO"
    models = (args.models or default_models).split(",")
    for model in models:
        base, count = _model_config(dataset, model, args.base_dir)
        evaldir = "evals-equal-new-{idx}" if dataset == "muse" else "evals-equal-{idx}"
        summary = "MUSE_SUMMARY.json" if dataset == "muse" else "TOFU_SUMMARY.json"
        first_metric, second_metric = (("retain_knowmem_ROUGE", "forget_knowmem_ROUGE") if dataset == "muse"
                                       else ("model_utility", "forget_quality"))
        first = _read_ranked(base, range(count), evaldir, summary, first_metric)
        second = _read_ranked(base, range(count), evaldir, summary, second_metric)
        if dataset == "muse":
            fr, sr = {v:i for i,v in enumerate(first)}, {v:i for i,v in enumerate(second)}
            common = set(fr) & set(sr)
            selected = sorted(common, key=lambda i: (fr[i] - sr[i], i))
            out = Path(args.output_dir) / f"{model}-new.txt"
            names = ("retain_knowmem", "forget_knowmem")
        else:
            first_set, second_set = set(first[:len(first)//2]), set(second[:len(second)//2])
            first = sorted(first_set)
            second = sorted(second_set)
            selected = sorted(first_set & second_set)
            out = Path(args.output_dir) / f"{model}.txt"
            names = ("model_utility", "forget_quality")
        if args.top_k is not None: selected = selected[:args.top_k]
        _write_result(out, names[0], first, names[1], second, selected)


def main():
    args = build_parser().parse_args()
    os.chdir(CRU_ROOT)
    run(args)

if __name__ == "__main__":
    main()
