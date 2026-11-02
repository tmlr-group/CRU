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
Apply CRU patch to each unl@K relearn checkpoint, then eval forget+retain Q_A_Prob.
This is the correct CRU-as-diagnostic experiment: adversary relearns, then we
apply CRU to test if middle layers still encode original knowledge.

Output: causal_results/cru_on_relearn.json
  {tag: {step: {forget: float, retain: float, layers_replaced: [...]}}}
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
    p.add_argument("--filter", default=None)
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
    n = 0
    for L in layers:
        prefix = f"model.layers.{L}."
        for name in name_to_param:
            if name.startswith(prefix) and name in sd_orig:
                name_to_param[name].data.copy_(sd_orig[name].to(name_to_param[name].dtype).to(device))
                n += 1
    return n


# (size, method) -> (orig_path, layers)
CONFIG_META = {
    ("1B", "GradAscent"): ("saves/finetune/tofu_Llama-3.2-1B-Instruct_full", [4, 5, 6, 7, 8]),
    ("1B", "NPO"): ("saves/finetune/tofu_Llama-3.2-1B-Instruct_full", [4, 5, 6, 7, 8]),
    ("3B", "GradAscent"): ("saves/finetune/tofu_Llama-3.2-3B-Instruct_full", [11, 12, 13, 14, 15]),
    ("3B", "NPO"): ("saves/finetune/tofu_Llama-3.2-3B-Instruct_full", [11, 12, 13, 14, 15]),
}


def main():
    args = parse_args()
    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)

    dtype = get_dtype(args.dtype)

    results = {}
    if Path(args.out_path).exists():
        results = json.load(open(args.out_path))
        print(f"[resume] {len(results)} entries existing", flush=True)

    # tokenizer + datasets (load once)
    tok = None
    forget_examples = None
    retain_examples = None

    t0 = time.time()
    n_done = 0
    total = sum(1 for (size, method) in CONFIG_META if (args.filter is None or args.filter in size))
    total *= 5  # 5 steps each

    for (size, method), (orig_rel, layers) in CONFIG_META.items():
        if args.filter and args.filter not in size:
            continue

        orig_path = Path(str(CRU_INPUT_ROOT)) / orig_rel
        # load orig state dict once per config
        print(f"[load_orig] {orig_path}", flush=True)
        sd_orig = AutoModelForCausalLM.from_pretrained(
            str(orig_path), torch_dtype=torch.float32, local_files_only=True
        ).state_dict()

        for step in [0, 1, 5, 10, 20]:
            tag = f"{size}_{method}_step{step}"
            if tag in results:
                n_done += 1
                continue
            ckpt = Path(args.ckpt_root) / f"relearn_{size}_{method}/unl/relearn_{size}_{method}_unl_step{step}"
            if not ckpt.exists():
                print(f"[skip] {ckpt} not found", flush=True)
                continue

            try:
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
                        if not q or not a:
                            continue
                        b = build_inputs(tok, q, a, args.max_len)
                        if b is not None:
                            forget_examples.append(b)
                    retain_examples = []
                    for i in range(min(args.n_questions, len(retain_ds))):
                        ex = retain_ds[i]
                        q, a = ex.get("question"), ex.get("answer")
                        if not q or not a:
                            continue
                        b = build_inputs(tok, q, a, args.max_len)
                        if b is not None:
                            retain_examples.append(b)
                    print(f"[data] {len(forget_examples)} forget, {len(retain_examples)} retain", flush=True)

                model = AutoModelForCausalLM.from_pretrained(
                    str(ckpt), torch_dtype=dtype, local_files_only=True
                ).to(args.device)
                n_replaced = cru_patch(model, sd_orig, layers, args.device)
                forget_p = eval_q_a_prob(model, forget_examples, args.device)
                retain_p = eval_q_a_prob(model, retain_examples, args.device)
                results[tag] = {
                    "forget": forget_p, "retain": retain_p,
                    "layers_replaced": layers, "n_params_replaced": n_replaced,
                    "step": step, "size": size, "method": method,
                }
                n_done += 1
                print(f"[{n_done}] {tag}  forget={forget_p:.4f}  retain={retain_p:.4f}  t={time.time()-t0:.0f}s",
                      flush=True)
                with open(args.out_path, "w") as f:
                    json.dump(results, f, indent=2)
                del model
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"[err] {tag}: {e}", flush=True)
                results[tag] = {"error": str(e)}

        del sd_orig
        torch.cuda.empty_cache()

    print(f"[done] {len(results)} entries", flush=True)


if __name__ == "__main__":
    main()
