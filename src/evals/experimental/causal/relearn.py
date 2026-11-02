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
Relearn experiment: how fast does forgotten knowledge come back when we fine-tune
the unl model (or CRU(unl)) on the forget set?

Hypothesis: unl recovers in 1-5 steps (fragile). CRU(unl) and orig should be
relatively stable (orig because it never forgot; CRU because the layer
replacement restored real circuits).
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from torch.nn.functional import log_softmax
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--start_model", required=True, help="starting checkpoint dir")
    p.add_argument("--orig", default=None,
                   help="if provided, do CRU patch from this orig before relearn")
    p.add_argument("--cru_layers", default=None,
                   help="underscore-encoded layers, e.g. 5_6_7_8_9_10 for CRU patch")
    p.add_argument("--out_path", required=True)
    p.add_argument("--ckpt_dir", default=None,
                   help="if set, save model checkpoints at each eval step here")
    p.add_argument("--tag", required=True)
    p.add_argument("--n_questions", type=int, default=200)
    p.add_argument("--eval_steps", default="0,1,5,10,20",
                   help="comma-separated training steps at which to eval")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def get_dtype(s):
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[s]


def build_qa_inputs(tok, q, a, max_len):
    full = q.strip() + " " + a
    enc_full = tok(full, return_tensors="pt", truncation=True, max_length=max_len, add_special_tokens=True)
    enc_q = tok(q.strip() + " ", return_tensors="pt", truncation=True, max_length=max_len, add_special_tokens=True)
    if enc_q.input_ids.shape[1] >= enc_full.input_ids.shape[1] - 1:
        return None
    return enc_full.input_ids, enc_full.attention_mask, enc_q.input_ids.shape[1]


@torch.no_grad()
def eval_q_a_prob(model, tok, eval_examples, device):
    """Same metric as apply_cru_eval.py: mean exp(logp(answer | question))."""
    model.eval()
    lps = []
    for ids, mask, asn in eval_examples:
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


def cru_patch(model, orig_path, layers, dtype, device):
    """Replace specified layers of model with orig's parameters."""
    print(f"[cru] loading orig from {orig_path}", flush=True)
    sd_orig = AutoModelForCausalLM.from_pretrained(
        orig_path, torch_dtype=torch.float32, local_files_only=True
    ).state_dict()
    name_to_param = dict(model.named_parameters())
    n = 0
    for L in layers:
        prefix = f"model.layers.{L}."
        for name in name_to_param:
            if name.startswith(prefix) and name in sd_orig:
                name_to_param[name].data.copy_(sd_orig[name].to(name_to_param[name].dtype).to(device))
                n += 1
    print(f"[cru] replaced {n} params across layers {layers}", flush=True)
    del sd_orig
    torch.cuda.empty_cache()


def build_train_batch(tok, examples, device, max_len):
    """examples: list of (q, a). Build a padded batch with labels masked on prompt tokens."""
    pad_id = tok.pad_token_id
    ids_list, mask_list, label_list = [], [], []
    max_full = 0
    pre = []
    for q, a in examples:
        enc_full = tok(q.strip() + " " + a, return_tensors="pt", truncation=True,
                       max_length=max_len, add_special_tokens=True)
        enc_q = tok(q.strip() + " ", return_tensors="pt", truncation=True,
                    max_length=max_len, add_special_tokens=True)
        ans_start = enc_q.input_ids.shape[1]
        if ans_start >= enc_full.input_ids.shape[1] - 1:
            continue
        full_ids = enc_full.input_ids[0]
        labels = full_ids.clone()
        labels[:ans_start] = -100
        pre.append((full_ids, enc_full.attention_mask[0], labels))
        max_full = max(max_full, full_ids.size(0))
    for full_ids, attn, labels in pre:
        pad_len = max_full - full_ids.size(0)
        if pad_len > 0:
            full_ids = torch.cat([full_ids, torch.full((pad_len,), pad_id, dtype=full_ids.dtype)])
            attn = torch.cat([attn, torch.zeros(pad_len, dtype=attn.dtype)])
            labels = torch.cat([labels, torch.full((pad_len,), -100, dtype=labels.dtype)])
        ids_list.append(full_ids); mask_list.append(attn); label_list.append(labels)
    return (torch.stack(ids_list).to(device),
            torch.stack(mask_list).to(device),
            torch.stack(label_list).to(device))


def main():
    args = parse_args()
    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
    eval_steps = [int(x) for x in args.eval_steps.split(",")]
    max_step = max(eval_steps)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"[start] tag={args.tag} start={args.start_model}", flush=True)
    print(f"[start] eval_steps={eval_steps} max={max_step}", flush=True)

    dtype = get_dtype(args.dtype)
    tok = AutoTokenizer.from_pretrained(args.start_model, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.start_model, torch_dtype=dtype, local_files_only=True
    ).to(args.device)

    if args.orig and args.cru_layers:
        layers = [int(x) for x in args.cru_layers.split("_")]
        cru_patch(model, args.orig, layers, dtype, args.device)

    print("[data] forget05_perturbed (eval) + forget05 (train)", flush=True)
    eval_ds = load_dataset("locuslab/TOFU", "forget05_perturbed")["train"]
    train_ds = load_dataset("locuslab/TOFU", "forget05")["train"]

    eval_examples = []
    for i in range(min(args.n_questions, len(eval_ds))):
        ex = eval_ds[i]
        q, a = ex.get("question"), ex.get("answer")
        if not q or not a:
            continue
        b = build_qa_inputs(tok, q, a, args.max_len)
        if b is not None:
            eval_examples.append(b)
    print(f"[data] {len(eval_examples)} eval examples", flush=True)

    train_pairs = []
    for i in range(len(train_ds)):
        ex = train_ds[i]
        q, a = ex.get("question"), ex.get("answer")
        if q and a:
            train_pairs.append((q, a))
    print(f"[data] {len(train_pairs)} train pairs", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed)

    log = {}
    if 0 in eval_steps:
        qap = eval_q_a_prob(model, tok, eval_examples, args.device)
        log[0] = qap
        print(f"[step 0] forget Q_A_Prob = {qap:.4f}", flush=True)
        if args.ckpt_dir:
            cp = Path(args.ckpt_dir) / f"{args.tag}_step0"
            cp.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(cp)
            tok.save_pretrained(cp)

    t0 = time.time()
    model.train()
    for step in range(1, max_step + 1):
        idx = rng.choice(len(train_pairs), args.batch_size, replace=False)
        batch = [train_pairs[i] for i in idx]
        ids, mask, labels = build_train_batch(tok, batch, args.device, args.max_len)
        optimizer.zero_grad()
        out = model(input_ids=ids, attention_mask=mask, labels=labels, use_cache=False)
        out.loss.backward()
        optimizer.step()

        if step in eval_steps:
            qap = eval_q_a_prob(model, tok, eval_examples, args.device)
            log[step] = qap
            print(f"[step {step}] loss={out.loss.item():.4f} forget Q_A_Prob = {qap:.4f}  t={time.time()-t0:.0f}s", flush=True)
            if args.ckpt_dir:
                cp = Path(args.ckpt_dir) / f"{args.tag}_step{step}"
                cp.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(cp)
                tok.save_pretrained(cp)
            model.train()

    result = {
        "tag": args.tag,
        "start_model": args.start_model,
        "orig": args.orig,
        "cru_layers": args.cru_layers,
        "eval_steps": eval_steps,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "n_eval_examples": len(eval_examples),
        "n_train_pairs": len(train_pairs),
        "step_to_q_a_prob": {str(k): v for k, v in log.items()},
    }
    with open(args.out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[done] saved {args.out_path}", flush=True)


if __name__ == "__main__":
    main()
