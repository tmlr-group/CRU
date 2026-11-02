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
Apply CRU (Component-wise Replacement Unlearning): replace specified middle layers
of unlearned model with corresponding layers from original (full-finetune) model,
then evaluate TOFU MU components.

Used for seed-variance significance test of CRU's improvement (vs MU_unl).
For 1B (16 layers): replace L4-L8 (5 middle layers).
For 3B (28 layers): replace L11-L15 (5 middle layers).

Output: cru_mu_{tag}_{split}.json
"""
import argparse, json, os, time
from pathlib import Path
import numpy as np
import torch
from torch.nn.functional import log_softmax
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--orig", required=True, help="full-finetune (orig) model dir")
    p.add_argument("--unlearn", required=True, help="unlearned checkpoint dir")
    p.add_argument("--layers", required=True, help="comma-separated layer indices to replace, e.g. 4,5,6,7,8")
    p.add_argument("--out_path", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--n_questions", type=int, default=200)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--n_perts", type=int, default=5)
    p.add_argument("--split", default="retain_perturbed")
    return p.parse_args()


def get_dtype(s): return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[s]


def build_inputs(tok, q, a, max_len):
    full = q.strip() + " " + a
    enc_full = tok(full, return_tensors="pt", truncation=True, max_length=max_len, add_special_tokens=True)
    enc_q = tok(q.strip() + " ", return_tensors="pt", truncation=True, max_length=max_len, add_special_tokens=True)
    if enc_q.input_ids.shape[1] >= enc_full.input_ids.shape[1] - 1:
        return None
    return enc_full.input_ids, enc_full.attention_mask, enc_q.input_ids.shape[1]


@torch.no_grad()
def get_answer_logp(model, ids, mask, ans_start):
    logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
    logp = log_softmax(logits.float(), dim=-1)
    target = ids[:, ans_start:]
    pred = logp[:, ans_start - 1:-1, :]
    if pred.shape[1] != target.shape[1]:
        n = min(pred.shape[1], target.shape[1])
        pred = pred[:, -n:, :]; target = target[:, -n:]
    return pred.gather(-1, target.unsqueeze(-1)).squeeze(-1).mean().item()


def main():
    args = parse_args()
    layers = [int(x) for x in args.layers.split(",")]
    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
    print(f"[start] tag={args.tag} split={args.split} layers={layers}", flush=True)

    dtype = get_dtype(args.dtype)
    tok = AutoTokenizer.from_pretrained(args.unlearn, local_files_only=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    # load unl model (will be modified in-place by CRU)
    print(f"[load] unl from {args.unlearn}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.unlearn, torch_dtype=dtype, local_files_only=True).to(args.device).eval()

    # load orig state dict (CPU, fp32) for the replacement layers
    print(f"[load] orig state dict from {args.orig}", flush=True)
    sd_orig = AutoModelForCausalLM.from_pretrained(args.orig, torch_dtype=torch.float32, local_files_only=True).state_dict()

    name_to_param = dict(model.named_parameters())
    n_replaced = 0
    for L in layers:
        prefix = f"model.layers.{L}."
        for name in name_to_param:
            if name.startswith(prefix):
                if name in sd_orig:
                    name_to_param[name].data.copy_(sd_orig[name].to(name_to_param[name].dtype).to(args.device))
                    n_replaced += 1
    print(f"[ok] replaced {n_replaced} params across layers {layers}", flush=True)
    del sd_orig
    torch.cuda.empty_cache()

    print(f"[data] {args.split}", flush=True)
    ds = load_dataset("locuslab/TOFU", args.split)["train"]
    n_q = min(args.n_questions, len(ds))

    examples = []
    for i in range(n_q):
        ex = ds[i]
        q, a = ex.get("question"), ex.get("answer")
        pa = ex.get("paraphrased_answer", a)
        perts = ex.get("perturbed_answer", [a])
        if isinstance(perts, str): perts = [perts]
        if not q or not a: continue
        b_a = build_inputs(tok, q, a, args.max_len)
        if b_a is None: continue
        b_pa = build_inputs(tok, q, pa, args.max_len)
        b_perts = []
        for pert in perts[:args.n_perts]:
            bp = build_inputs(tok, q, pert, args.max_len)
            if bp is not None: b_perts.append(bp)
        if b_pa is None or not b_perts: continue
        def to_dev(b):
            ids, mask, asn = b
            return ids.to(args.device), mask.to(args.device), asn
        examples.append({"a": to_dev(b_a), "pa": to_dev(b_pa), "perts": [to_dev(bp) for bp in b_perts]})
    print(f"[ok] {len(examples)} valid examples", flush=True)

    t0 = time.time()
    lp_a, lp_pa, lp_pert_per_q = [], [], []
    for i, e in enumerate(examples):
        ids, mask, asn = e["a"]; lp_a.append(get_answer_logp(model, ids, mask, asn))
        ids, mask, asn = e["pa"]; lp_pa.append(get_answer_logp(model, ids, mask, asn))
        lps = []
        for ids, mask, asn in e["perts"]: lps.append(get_answer_logp(model, ids, mask, asn))
        lp_pert_per_q.append(np.mean(lps))
        if (i+1) % 50 == 0:
            print(f"  [{i+1}/{len(examples)}] t={time.time()-t0:.0f}s", flush=True)

    m_a = float(np.mean(lp_a)); m_pa = float(np.mean(lp_pa)); m_pert = float(np.mean(lp_pert_per_q))
    result = {
        "tag": args.tag, "split": args.split,
        "orig": args.orig, "unlearn": args.unlearn,
        "layers_replaced": layers, "n_examples": len(examples),
        "Q_A_logp": m_a, "PARA_logp": m_pa, "PERT_logp": m_pert,
        "Q_A_Prob": float(np.exp(m_a)), "PARA_Prob": float(np.exp(m_pa)),
        "PERT_Prob": float(np.exp(m_pert)),
        "Truth_Ratio_log": m_pert - m_pa,
        "per_q_lp_a": lp_a, "per_q_lp_pa": lp_pa, "per_q_lp_pert": lp_pert_per_q,
    }
    with open(args.out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[done] saved {args.out_path}  Q_A_Prob={result['Q_A_Prob']:.4f}  TR_log={result['Truth_Ratio_log']:+.3f}", flush=True)


if __name__ == "__main__":
    main()
