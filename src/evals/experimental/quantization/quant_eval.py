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
Quantization robustness evaluation for CRU.

Evaluates forget_quality (truth ratio KS-test) and model_utility under
bf16 / int8 / int4 quantization for:
  - original (no unlearning)
  - GA_full / NPO_full (full gradient-based unlearning)
  - CRU equal-K  (single layer swap, K = 0..15)
  - CRU select-K (top-K layer swap, K = 3..8)

Usage:
    python evals/quantization/quant_eval.py --method GA --mode equal  --quant int8
    python evals/quantization/quant_eval.py --method GA --mode select --quant int4
    python evals/quantization/quant_eval.py --method GA --mode baseline --quant int8
"""
from pathlib import Path as _CRUPath
import sys as _cru_sys
_cru_root = _CRUPath(__file__).resolve().parents[4]
if str(_cru_root / "src") not in _cru_sys.path:
    _cru_sys.path.insert(0, str(_cru_root / "src"))
from cru_paths import CRU_INPUT_ROOT, CRU_ROOT, resolve_input_path
if __name__ == '__main__':
    import os as _cru_os
    _cru_os.chdir(CRU_ROOT)

import tempfile
import os

import sys, os, re, json, argparse, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from scipy.stats import ks_2samp
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset

HF_CACHE = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
HF_HOME = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
WORKDIR   = str(CRU_INPUT_ROOT)

ORIGINAL_PATH  = resolve_input_path('saves/finetune/tofu_Llama-3.2-1B-Instruct_full')
GA_UNLEARN     = resolve_input_path('saves/unlearn/tofu_Llama-3.2-1B-Instruct_forget05_GradAscent')
NPO_UNLEARN    = resolve_input_path('saves/unlearn/tofu_Llama-3.2-1B-Instruct_forget05_NPO')
RETAIN_LOGS    = resolve_input_path('saves/eval/tofu_Llama-3.2-1B-Instruct_retain95/TOFU_EVAL.json')
RETAIN_SPLIT   = 'retain95'

# Selected layers (intersection of top-half MU and top-half FQ)
SELECTED = {
    'GA':  [0, 1, 2, 3],
    'NPO': [0, 1, 2, 15],
}

os.environ['HF_HOME'] = HF_HOME
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'


# ── Dataset loading ────────────────────────────────────────────────────────────

def load_tofu_splits():
    from datasets import load_from_disk
    forget   = load_from_disk(f'{WORKDIR}/datasets/forget05_v')['train']  if os.path.exists(f'{WORKDIR}/datasets/forget05_v') else None
    # Fall back to HF hub splits via utils
    return _load_hf_splits()

def _load_hf_splits():
    """Load forget05 and its perturbed version from TOFU HF dataset."""
    from src.data.utils import get_dataset
    # Use the same split names as the eval pipeline
    forget_ds   = _load_split('forget05')
    pert_ds     = _load_split('forget05_perturbed')
    return forget_ds, pert_ds

def _load_split(name):
    from datasets import load_dataset as _ld
    return _ld('locuslab/TOFU', name, split='train', cache_dir=HF_CACHE)


# ── Model loading ──────────────────────────────────────────────────────────────

def get_bnb_config(quant):
    if quant == 'int8':
        return BitsAndBytesConfig(load_in_8bit=True)
    elif quant == 'int4':
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',
        )
    return None  # bf16

def load_base_model(path, quant='bf16'):
    bnb = get_bnb_config(quant)
    kwargs = dict(device_map='auto', cache_dir=HF_CACHE)
    if bnb:
        kwargs['quantization_config'] = bnb
    else:
        kwargs['torch_dtype'] = torch.bfloat16
    return AutoModelForCausalLM.from_pretrained(path, **kwargs)

def load_tokenizer(path):
    tok = AutoTokenizer.from_pretrained(path, cache_dir=HF_CACHE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok

def swap_layers(model, unlearn_model, layer_indices):
    """Copy layer_indices layers from unlearn_model into model in-place."""
    for name, param in model.named_parameters():
        m = re.search(r'model\.layers\.(\d+)\.', name)
        if m and int(m.group(1)) in layer_indices:
            src = dict(unlearn_model.named_parameters())[name]
            param.data.copy_(src.data.to(param.device))

def build_model(original_path, unlearn_path, layer_indices, quant, tmp_dir):
    """Build swapped model: load original+unlearn in bf16, swap, save, reload with quant."""
    if not layer_indices:
        # A zero-layer CRU baseline is the original model, without unlearning.
        return load_base_model(original_path, quant)

    if quant == 'bf16':
        model = AutoModelForCausalLM.from_pretrained(
            original_path, torch_dtype=torch.bfloat16, device_map='auto', cache_dir=HF_CACHE)
        if unlearn_path:
            unlearn = AutoModelForCausalLM.from_pretrained(
                unlearn_path, torch_dtype=torch.bfloat16, device_map='auto', cache_dir=HF_CACHE)
            swap_layers(model, unlearn, set(layer_indices))
            del unlearn
        return model

    # For quantized: swap in bf16, save to tmp, reload with quant
    print(f"  Building swapped model (layers {layer_indices}) → saving to {tmp_dir}")
    model = AutoModelForCausalLM.from_pretrained(
        original_path, torch_dtype=torch.bfloat16, device_map='cpu', cache_dir=HF_CACHE)
    if unlearn_path:
        unlearn = AutoModelForCausalLM.from_pretrained(
            unlearn_path, torch_dtype=torch.bfloat16, device_map='cpu', cache_dir=HF_CACHE)
        swap_layers(model, unlearn, set(layer_indices))
        del unlearn
    os.makedirs(tmp_dir, exist_ok=True)
    model.save_pretrained(tmp_dir)
    del model
    torch.cuda.empty_cache()
    return load_base_model(tmp_dir, quant)


# ── Evaluation ─────────────────────────────────────────────────────────────────

def get_token_prob(model, tokenizer, question, answer, device, max_length=512):
    """Return sum of log-probs of answer tokens given question."""
    prompt = f"### Question: {question}\n### Answer: {answer}"
    enc = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=max_length).to(device)
    q_enc = tokenizer(f"### Question: {question}\n### Answer:", return_tensors='pt',
                      truncation=True, max_length=max_length)
    q_len = q_enc['input_ids'].shape[1]
    with torch.no_grad():
        out = model(**enc)
        logits = out.logits[0]  # (seq, vocab)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    ids = enc['input_ids'][0]
    ans_log_prob = 0.0
    for i in range(q_len, len(ids)):
        ans_log_prob += log_probs[i - 1, ids[i]].item()
    return ans_log_prob

def compute_truth_ratio(model, tokenizer, forget_ds, pert_ds, batch_size=4):
    """Compute per-sample truth ratio = P(correct) / P(perturbed)."""
    device = next(model.parameters()).device
    ratios = []
    for i in tqdm(range(len(forget_ds)), desc='  truth_ratio', leave=False):
        q  = forget_ds[i]['question']
        a  = forget_ds[i]['answer']
        pa = pert_ds[i]['answer']
        lp_correct = get_token_prob(model, tokenizer, q, a, device)
        lp_wrong   = get_token_prob(model, tokenizer, q, pa, device)
        # truth_ratio: log(P_correct) - log(P_wrong)
        ratios.append(lp_correct - lp_wrong)
    return ratios

def compute_fq(model_ratios, retain_ratios):
    """KS-test p-value (forget_quality)."""
    stat, pvalue = ks_2samp(model_ratios, retain_ratios)
    return math.log(pvalue) if pvalue > 0 else -float('inf')

def compute_mu(model, tokenizer, retain_ds, batch_size=4):
    """Model utility: accuracy on retain QA (probability of correct answer)."""
    device = next(model.parameters()).device
    correct = 0
    for i in tqdm(range(len(retain_ds)), desc='  model_utility', leave=False):
        q = retain_ds[i]['question']
        a = retain_ds[i]['answer']
        lp = get_token_prob(model, tokenizer, q, a, device)
        correct += (lp > -50)  # heuristic threshold; real MU uses relative comparison
    return correct / len(retain_ds)

def load_retain_truth_ratios():
    """Load retain model truth ratios from existing eval logs."""
    with open(RETAIN_LOGS) as f:
        data = json.load(f)
    # Extract forget_truth_ratio from retain model logs
    tr = data.get('forget_truth_ratio', {}).get('value_by_index', {})
    return [v['score'] for v in tr.values()]


# ── Main ───────────────────────────────────────────────────────────────────────

def run_eval(model, tokenizer, forget_ds, pert_ds, retain_ratios):
    ratios = compute_truth_ratio(model, tokenizer, forget_ds, pert_ds)
    fq = compute_fq(ratios, retain_ratios)
    return {'forget_quality': fq, 'truth_ratios': ratios}

def save_result(result, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"  Saved → {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', required=True, choices=['GA', 'NPO'])
    parser.add_argument('--mode',   required=True, choices=['baseline', 'equal', 'select'])
    parser.add_argument('--quant',  required=True, choices=['bf16', 'int8', 'int4'])
    parser.add_argument('--k_start', type=int, default=0)
    parser.add_argument('--k_end',   type=int, default=15)
    args = parser.parse_args()

    unlearn_path = GA_UNLEARN if args.method == 'GA' else NPO_UNLEARN
    out_base = f'saves_q/{args.method}/{args.mode}/{args.quant}'
    tmp_base = os.path.join(tempfile.gettempdir(), f'quant_eval_tmp_{args.method}_{args.quant}')

    print(f"Loading datasets...")
    forget_ds = _load_split('forget05')
    pert_ds   = _load_split('forget05_perturbed')
    print(f"  forget: {len(forget_ds)}, pert: {len(pert_ds)}")

    print(f"Loading retain truth ratios from {RETAIN_LOGS}")
    retain_ratios = load_retain_truth_ratios()
    print(f"  retain ratios: {len(retain_ratios)} samples")

    tokenizer = load_tokenizer(ORIGINAL_PATH)

    if args.mode == 'baseline':
        configs = [
            ('original', ORIGINAL_PATH, []),
            ('GA_full',  GA_UNLEARN,    None),   # None = load unlearn directly, no swap
            ('NPO_full', NPO_UNLEARN,   None),
        ]
        for tag, path, layers in configs:
            out_path = f'{out_base}/{tag}/result.json'
            if os.path.exists(out_path):
                print(f"  [skip] {out_path}"); continue
            print(f"\n[{tag}] quant={args.quant}")
            if layers is None:
                model = load_base_model(path, args.quant)
            else:
                model = build_model(ORIGINAL_PATH, None, [], args.quant, f'{tmp_base}_{tag}')
            result = run_eval(model, tokenizer, forget_ds, pert_ds, retain_ratios)
            result['tag'] = tag; result['quant'] = args.quant
            save_result(result, out_path)
            del model; torch.cuda.empty_cache()

    elif args.mode == 'equal':
        for k in range(args.k_start, args.k_end + 1):
            out_path = f'{out_base}/layer_{k}/result.json'
            if os.path.exists(out_path):
                print(f"  [skip] layer {k}"); continue
            print(f"\n[equal layer={k}] method={args.method} quant={args.quant}")
            model = build_model(ORIGINAL_PATH, unlearn_path, [k], args.quant,
                                f'{tmp_base}_layer{k}')
            result = run_eval(model, tokenizer, forget_ds, pert_ds, retain_ratios)
            result.update({'layer': k, 'method': args.method, 'quant': args.quant})
            save_result(result, out_path)
            del model; torch.cuda.empty_cache()
            import shutil
            shutil.rmtree(f'{tmp_base}_layer{k}', ignore_errors=True)

    elif args.mode == 'select':
        selected = SELECTED[args.method]
        for k in range(args.k_start, min(args.k_end + 1, len(selected) + 1)):
            out_path = f'{out_base}/K{k}/result.json'
            if os.path.exists(out_path):
                print(f"  [skip] K{k}"); continue
            layers = selected[:k]
            print(f"\n[select K={k} layers={layers}] method={args.method} quant={args.quant}")
            model = build_model(ORIGINAL_PATH, unlearn_path, layers, args.quant,
                                f'{tmp_base}_K{k}')
            result = run_eval(model, tokenizer, forget_ds, pert_ds, retain_ratios)
            result.update({'K': k, 'layers': layers, 'method': args.method, 'quant': args.quant})
            save_result(result, out_path)
            del model; torch.cuda.empty_cache()
            import shutil
            shutil.rmtree(f'{tmp_base}_K{k}', ignore_errors=True)

if __name__ == '__main__':
    main()
