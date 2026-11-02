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
Random-source activation patching with TOFU MU/FQ-style scoring.

For each (target_L, seed):
  - For each TOFU retain_perturbed question:
      - sample random source layer L' (uniform from [0,N) excluding target_L)
      - run unl forward with hook at target_L replacing output with cached_orig[L'+1]
      - record: log p(answer | question), log p(perturbed_answer | question)
  - aggregate over questions:
      retain_QA_Prob = mean exp(mean_logp_answer)
      retain_Truth_Ratio = mean (geom_mean(perturbed_probs) / paraphrased_prob)
  - record per-(target_L, seed) MU proxy + raw log-prob proxy

Output: random_source_{tag}_L{L}_s{seed}.json
"""
import argparse, json, os, time, random
from pathlib import Path
import numpy as np
import torch
from torch.nn.functional import log_softmax
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--orig", required=True)
    p.add_argument("--unlearn", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--n_seeds", type=int, default=10)
    p.add_argument("--n_questions", type=int, default=200)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--max_len", type=int, default=512)
    return p.parse_args()


def get_dtype(s): return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[s]


def load_model(path, device, dtype):
    print(f"  [load] {path}", flush=True)
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype, local_files_only=True).to(device).eval()
    return m, tok


def build_inputs(tok, q, a, max_len):
    full = q.strip() + " " + a
    enc_full = tok(full, return_tensors="pt", truncation=True, max_length=max_len, add_special_tokens=True)
    enc_q = tok(q.strip() + " ", return_tensors="pt", truncation=True, max_length=max_len, add_special_tokens=True)
    if enc_q.input_ids.shape[1] >= enc_full.input_ids.shape[1] - 1:
        return None
    return enc_full.input_ids, enc_full.attention_mask, enc_q.input_ids.shape[1]


@torch.no_grad()
def get_answer_logp(model, ids, mask, ans_start):
    """log-prob of answer tokens, mean over answer length."""
    logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
    logp = log_softmax(logits.float(), dim=-1)
    target = ids[:, ans_start:]
    pred = logp[:, ans_start - 1:-1, :]
    if pred.shape[1] != target.shape[1]:
        n = min(pred.shape[1], target.shape[1])
        pred = pred[:, -n:, :]; target = target[:, -n:]
    return pred.gather(-1, target.unsqueeze(-1)).squeeze(-1).mean().item()


@torch.no_grad()
def cached_orig_states(model, ids, mask):
    out = model(input_ids=ids, attention_mask=mask, output_hidden_states=True, use_cache=False)
    return out.hidden_states


def patched_logp(unl_model, ids, mask, ans_start, target_L, target_state):
    """Forward unl_model with hook at target_L replacing output with target_state. Return log-prob of answer."""
    def hook(mod, inp, out):
        new = target_state.to(out[0].dtype if isinstance(out, tuple) else out.dtype)
        new = new.to(out[0].device if isinstance(out, tuple) else out.device)
        T = (out[0] if isinstance(out, tuple) else out).shape[1]
        new = new[:, :T, :]
        if isinstance(out, tuple):
            return (new,) + out[1:]
        return new

    handle = unl_model.model.layers[target_L].register_forward_hook(hook)
    try:
        with torch.no_grad():
            logits = unl_model(input_ids=ids, attention_mask=mask, use_cache=False).logits
        logp = log_softmax(logits.float(), dim=-1)
        target = ids[:, ans_start:]
        pred = logp[:, ans_start - 1:-1, :]
        if pred.shape[1] != target.shape[1]:
            n = min(pred.shape[1], target.shape[1])
            pred = pred[:, -n:, :]; target = target[:, -n:]
        return pred.gather(-1, target.unsqueeze(-1)).squeeze(-1).mean().item()
    finally:
        handle.remove()


def main():
    args = parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    print(f"[start] {args.tag}", flush=True)

    m_orig, tok = load_model(args.orig, args.device, get_dtype(args.dtype))
    m_unl, _ = load_model(args.unlearn, args.device, get_dtype(args.dtype))
    n_layers = m_orig.config.num_hidden_layers
    print(f"[ok] n_layers={n_layers}", flush=True)

    # Load TOFU retain_perturbed
    print("[data] retain_perturbed", flush=True)
    ds = load_dataset("locuslab/TOFU", "retain_perturbed")["train"]
    n_q = min(args.n_questions, len(ds))

    # Pre-compute baselines + cache for all questions
    print("[precompute] cache + baselines", flush=True)
    examples = []
    for i in range(n_q):
        ex = ds[i]
        q = ex.get("question"); a = ex.get("answer")
        if not q or not a: continue
        b = build_inputs(tok, q, a, args.max_len)
        if b is None: continue
        ids, mask, ans_start = b
        ids = ids.to(args.device); mask = mask.to(args.device)

        orig_hs = cached_orig_states(m_orig, ids, mask)
        lp_o_a = get_answer_logp(m_orig, ids, mask, ans_start)
        lp_u_a = get_answer_logp(m_unl, ids, mask, ans_start)

        examples.append({
            "ids": ids, "mask": mask, "ans_start": ans_start,
            "orig_hs": tuple(h.cpu() for h in orig_hs),
            "lp_o_a": lp_o_a, "lp_u_a": lp_u_a,
        })

        if (i+1) % 50 == 0:
            print(f"  [precompute] {i+1}/{n_q}", flush=True)

    print(f"[precompute done] {len(examples)} valid examples", flush=True)

    # baselines aggregated
    mean_lp_o_a = np.mean([e["lp_o_a"] for e in examples])
    mean_lp_u_a = np.mean([e["lp_u_a"] for e in examples])
    print(f"[baseline] orig mean logp(a)={mean_lp_o_a:.3f}, unl={mean_lp_u_a:.3f}", flush=True)

    # main loop: per (target_L, seed)
    results = {}
    t0 = time.time()
    total_iters = n_layers * args.n_seeds
    iter_n = 0

    for target_L in range(n_layers):
        results[target_L] = {}
        for seed in range(args.n_seeds):
            rng = random.Random(seed * 1000 + target_L)

            # for each example, sample L' uniform from [0, n_layers) \ {target_L}
            # only patch the answer input (cached_orig_hs has answer-input seq_len)
            patched_lp_a_list = []
            for e in examples:
                L_prime = rng.randrange(n_layers - 1)
                if L_prime >= target_L:
                    L_prime += 1
                target_state = e["orig_hs"][L_prime + 1].to(args.device)
                lp_a = patched_logp(m_unl, e["ids"], e["mask"], e["ans_start"],
                                     target_L, target_state)
                patched_lp_a_list.append(lp_a)

            mean_lp_a = float(np.mean(patched_lp_a_list))
            mu_proxy = float(np.exp(mean_lp_a))

            results[target_L][seed] = {
                "mean_logp_answer": mean_lp_a,
                "mu_proxy": mu_proxy,
            }

            iter_n += 1
            t = time.time() - t0
            if iter_n % 5 == 0 or iter_n == total_iters:
                eta = t / iter_n * (total_iters - iter_n)
                print(f"  [iter {iter_n}/{total_iters}] L={target_L} seed={seed} "
                      f"logp_a={mean_lp_a:.3f} mu_proxy={mu_proxy:.3f} t={t:.0f}s ETA={eta:.0f}s",
                      flush=True)

    summary = {
        "tag": args.tag,
        "n_layers": n_layers,
        "n_seeds": args.n_seeds,
        "n_examples": len(examples),
        "baseline_orig_logp_a": float(mean_lp_o_a),
        "baseline_unl_logp_a": float(mean_lp_u_a),
        "results": {str(L): {str(s): v for s, v in d.items()} for L, d in results.items()},
        "orig_path": args.orig,
        "unlearn_path": args.unlearn,
    }
    out_path = Path(args.out_dir) / f"random_source_{args.tag}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[done] saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
