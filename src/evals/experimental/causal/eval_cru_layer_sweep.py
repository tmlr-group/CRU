from pathlib import Path as _CRUPath
import sys as _cru_sys
import os as _cru_os
_cru_root = _CRUPath(__file__).resolve().parents[4]
if str(_cru_root / "src") not in _cru_sys.path:
    _cru_sys.path.insert(0, str(_cru_root / "src"))
from cru_paths import CRU_ROOT, CRU_INPUT_ROOT
if __name__ == "__main__":
    _cru_os.chdir(CRU_ROOT)

"""
Per-layer CRU sweep × relearn step K matrix.
For each unl@K checkpoint, replace ONE layer at a time (k=1 sweep) AND optionally
larger windows (k=3, 5, 7 centered) — measure forget + retain Q_A_Prob.

Output: causal_results/cru_layer_sweep_{size}.json
  {f"{size}_{method}_K{step}_L{layer}_k{k}": {forget, retain, layers, ...}}
"""
from pathlib import Path as _CRUPath
import sys as _cru_sys
_cru_root = _CRUPath(__file__).resolve().parents[4]
if str(_cru_root / "src") not in _cru_sys.path:
    _cru_sys.path.insert(0, str(_cru_root / "src"))
from cru_paths import CRU_INPUT_ROOT, CRU_ROOT
if __name__ == '__main__':
    import os as _cru_os
    _cru_os.chdir(CRU_ROOT)

import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from torch.nn.functional import log_softmax
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out_path", required=True)
    p.add_argument("--ckpt_root", required=True, help="path to the relearn checkpoint root")
    p.add_argument("--n_questions", type=int, default=200)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--filter", required=True, help="e.g. 1B_GradAscent")
    p.add_argument("--k_values", default="1_3_5_7", help="underscore-separated window sizes")
    return p.parse_args()


def get_dtype(s):
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[s]


def build_inputs(tok, q, a, max_len):
    full = q.strip() + " " + a
    enc_full = tok(full, return_tensors="pt", truncation=True, max_length=max_len, add_special_tokens=True)
    enc_q = tok(q.strip() + " ", return_tensors="pt", truncation=True, max_length=max_len, add_special_tokens=True)
    if enc_q.input_ids.shape[1] >= enc_full.input_ids.shape[1] - 1:
        return None
    return enc_full.input_ids, enc_full.attention_mask, enc_q.input_ids.shape[1]


@torch.no_grad()
def eval_q_a_prob(model, examples, device):
    model.eval()
    lps = []
    for ids, mask, asn in examples:
        ids = ids.to(device); mask = mask.to(device)
        logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
        logp = log_softmax(logits.float(), dim=-1)
        target = ids[:, asn:]
        pred = logp[:, asn - 1:-1, :]
        if pred.shape[1] != target.shape[1]:
            n = min(pred.shape[1], target.shape[1])
            pred = pred[:, -n:, :]; target = target[:, -n:]
        lps.append(pred.gather(-1, target.unsqueeze(-1)).squeeze(-1).mean().item())
    return float(np.exp(np.mean(lps)))


def cru_patch(model, sd_orig, layers, device):
    name_to_param = dict(model.named_parameters())
    for L in layers:
        prefix = f"model.layers.{L}."
        for name in name_to_param:
            if name.startswith(prefix) and name in sd_orig:
                name_to_param[name].data.copy_(sd_orig[name].to(name_to_param[name].dtype).to(device))


def cru_unpatch(model, sd_unl, layers, device):
    """Restore unl weights for the patched layers."""
    name_to_param = dict(model.named_parameters())
    for L in layers:
        prefix = f"model.layers.{L}."
        for name in name_to_param:
            if name.startswith(prefix) and name in sd_unl:
                name_to_param[name].data.copy_(sd_unl[name].to(name_to_param[name].dtype).to(device))


# config -> (orig_path, n_layers)
CONFIG_META = {
    "1B_GradAscent": ("saves/finetune/tofu_Llama-3.2-1B-Instruct_full", 16),
    "1B_NPO": ("saves/finetune/tofu_Llama-3.2-1B-Instruct_full", 16),
    "3B_GradAscent": ("saves/finetune/tofu_Llama-3.2-3B-Instruct_full", 28),
    "3B_NPO": ("saves/finetune/tofu_Llama-3.2-3B-Instruct_full", 28),
}


def main():
    args = parse_args()
    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)

    config_key = args.filter
    if config_key not in CONFIG_META:
        raise ValueError(f"unknown filter {config_key}; one of {list(CONFIG_META)}")
    size, method = config_key.split("_", 1)
    orig_rel, n_layers = CONFIG_META[config_key]
    sep = "_" if "_" in args.k_values else ","
    k_values = [int(x) for x in args.k_values.split(sep)]

    dtype = get_dtype(args.dtype)

    results = {}
    if Path(args.out_path).exists():
        results = json.load(open(args.out_path))
        print(f"[resume] {len(results)} entries", flush=True)

    print(f"[start] config={config_key} n_layers={n_layers} k_values={k_values}", flush=True)
    orig_path = Path(str(CRU_INPUT_ROOT)) / orig_rel
    print(f"[load_orig] {orig_path}", flush=True)
    sd_orig = AutoModelForCausalLM.from_pretrained(
        str(orig_path), torch_dtype=torch.float32, local_files_only=True
    ).state_dict()

    tok = None
    forget_examples = retain_examples = None

    t0 = time.time()
    for step in [0, 1, 5, 10, 20]:
        ckpt = Path(args.ckpt_root) / f"relearn_{size}_{method}/unl/relearn_{size}_{method}_unl_step{step}"
        if not ckpt.exists():
            print(f"[skip] {ckpt} not found", flush=True)
            continue

        if tok is None:
            tok = AutoTokenizer.from_pretrained(str(ckpt), local_files_only=True)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token

            print("[data] forget05_perturbed + retain_perturbed", flush=True)
            forget_ds = load_dataset("locuslab/TOFU", "forget05_perturbed")["train"]
            retain_ds = load_dataset("locuslab/TOFU", "retain_perturbed")["train"]
            forget_examples = []
            for i in range(min(args.n_questions, len(forget_ds))):
                ex = forget_ds[i]
                q, a = ex.get("question"), ex.get("answer")
                if q and a:
                    b = build_inputs(tok, q, a, args.max_len)
                    if b is not None:
                        forget_examples.append(b)
            retain_examples = []
            for i in range(min(args.n_questions, len(retain_ds))):
                ex = retain_ds[i]
                q, a = ex.get("question"), ex.get("answer")
                if q and a:
                    b = build_inputs(tok, q, a, args.max_len)
                    if b is not None:
                        retain_examples.append(b)
            print(f"[data] {len(forget_examples)} forget, {len(retain_examples)} retain", flush=True)

        # Load unl@K once and save its state dict for unpatch
        print(f"[load_ckpt K={step}] {ckpt}", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            str(ckpt), torch_dtype=dtype, local_files_only=True
        ).to(args.device)
        sd_unl = {n: p.detach().clone() for n, p in model.named_parameters()}

        # baseline (no patch) — eval as L=-1 k=0
        tag_base = f"{config_key}_K{step}_baseline"
        if tag_base not in results:
            f = eval_q_a_prob(model, forget_examples, args.device)
            r = eval_q_a_prob(model, retain_examples, args.device)
            results[tag_base] = {"forget": f, "retain": r, "layers": [], "k": 0,
                                 "step": step, "size": size, "method": method}
            print(f"  [base K={step}] forget={f:.4f} retain={r:.4f}  t={time.time()-t0:.0f}s", flush=True)
            with open(args.out_path, "w") as fout:
                json.dump(results, fout, indent=2)

        # k=1: sweep all layers
        # k>1: sliding windows
        for k in k_values:
            for start in range(0, n_layers - k + 1):
                layers = list(range(start, start + k))
                tag = f"{config_key}_K{step}_L{start}_k{k}"
                if tag in results:
                    continue
                cru_patch(model, sd_orig, layers, args.device)
                f = eval_q_a_prob(model, forget_examples, args.device)
                r = eval_q_a_prob(model, retain_examples, args.device)
                cru_unpatch(model, sd_unl, layers, args.device)
                results[tag] = {"forget": f, "retain": r, "layers": layers, "k": k,
                                "step": step, "size": size, "method": method}
                if start % 4 == 0 or start == n_layers - k:
                    print(f"  [K={step} L={start} k={k}] forget={f:.4f} retain={r:.4f}  t={time.time()-t0:.0f}s",
                          flush=True)
                with open(args.out_path, "w") as fout:
                    json.dump(results, fout, indent=2)

        del model, sd_unl
        torch.cuda.empty_cache()

    print(f"[done] {len(results)} entries", flush=True)


if __name__ == "__main__":
    main()
