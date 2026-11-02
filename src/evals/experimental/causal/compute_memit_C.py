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
Pre-compute MEMIT covariance matrix C_l for each edit layer.

C_l = E_x[k_x k_x^T] where k_x = MLP-input post-gate*up activation at layer l
(i.e., the d_intermediate vector that W_down maps to d_model).

Reference distribution: TOFU retain95 split (the data we KEEP, not forget).
Output: torch tensor saved as .pt file per layer.

Usage:
  python compute_memit_C.py --orig <model_path> --layers 4,5,6,7,8 --out_dir <dir> --n_samples 1000
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--orig", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--layers", required=True, help="comma-separated layer indices")
    p.add_argument("--ref_split", default="retain95")
    p.add_argument("--n_samples", type=int, default=1000)
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    return p.parse_args()


def get_dtype(s):
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[s]


@torch.no_grad()
def collect_C_at_layer(model, tok, ds, layer, n_samples, max_len, device):
    """Collect post-gate*up activations at given layer; compute Σ k k^T."""
    mlp = model.model.layers[layer].mlp
    d_inter = mlp.down_proj.in_features

    # accumulator: float32 to avoid drift
    C_sum = torch.zeros(d_inter, d_inter, dtype=torch.float32, device=device)
    total_tokens = 0

    captured = {}
    def hook(module, inp, out):
        # inp[0] is the input to down_proj = post gate*up activation, shape (B, T, d_inter)
        captured["k"] = inp[0].detach()

    handle = mlp.down_proj.register_forward_hook(hook)
    try:
        for i in range(min(n_samples, len(ds))):
            ex = ds[i]
            q, a = ex.get("question", ""), ex.get("answer", "")
            text = (q.strip() + " " + a).strip()
            if not text:
                continue
            enc = tok(text, return_tensors="pt", truncation=True, max_length=max_len).to(device)
            _ = model(**enc, use_cache=False)
            k = captured["k"][0]  # (T, d_inter)
            # Cast to fp32 for accumulation
            k_fp = k.float()
            C_sum += k_fp.t() @ k_fp
            total_tokens += k_fp.shape[0]
            if (i + 1) % 100 == 0:
                print(f"  layer {layer}: {i+1}/{n_samples} samples, total_tokens={total_tokens}", flush=True)
    finally:
        handle.remove()

    C = C_sum / max(total_tokens, 1)
    return C, total_tokens


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    layers = [int(x) for x in args.layers.split(",")]

    print(f"[memit_C] orig={args.orig} layers={layers} ref={args.ref_split} n={args.n_samples}", flush=True)
    dtype = get_dtype(args.dtype)
    tok = AutoTokenizer.from_pretrained(args.orig, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.orig, torch_dtype=dtype, local_files_only=True).to(args.device).eval()
    print("[ok] model loaded", flush=True)

    print(f"[data] {args.ref_split}", flush=True)
    ds = load_dataset("locuslab/TOFU", args.ref_split)["train"]
    print(f"  {len(ds)} examples", flush=True)

    for L in layers:
        out_path = out_dir / f"C_L{L}.pt"
        if out_path.exists():
            print(f"[skip layer {L}] already at {out_path}", flush=True)
            continue
        t0 = time.time()
        C, n_tok = collect_C_at_layer(model, tok, ds, L, args.n_samples, args.max_len, args.device)
        torch.save({"C": C.cpu(), "n_tokens": n_tok, "layer": L}, out_path)
        print(f"[saved] layer {L} -> {out_path}  shape={tuple(C.shape)}  n_tok={n_tok}  t={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
