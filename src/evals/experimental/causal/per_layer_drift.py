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
Compute per-layer Frobenius weight drift from unlearning.

For each unlearned model, load it + the corresponding original (full-finetuned)
model, compute ||W_unlearn^L - W_orig^L||_F for every transformer layer L.

This directly answers Reviewer 4Zho's "important vs damaged" question:
  - If drift is U-shaped (middle drifts most) → paper's U-shape on MU could be
    "patching the layer that drifted most" artifact, not "fragility"
  - If drift is flat → unlearning hits all layers similarly, U-shape on MU
    must come from something else (architectural sensitivity, knowledge structure)

Output: drift_per_layer_{tag}.json + 4-panel plot
"""
import argparse, json, os
from pathlib import Path
import torch
from safetensors.torch import safe_open

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--orig_dir", required=True)
    p.add_argument("--unl_dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tag", required=True)
    return p.parse_args()


def collect_layer_param_names(state_dict_keys, n_layers):
    """Group parameter names by transformer layer index."""
    by_layer = {L: [] for L in range(n_layers)}
    other = []
    for k in state_dict_keys:
        if "model.layers." in k:
            try:
                L = int(k.split("model.layers.")[1].split(".")[0])
                if L < n_layers:
                    by_layer[L].append(k)
            except (ValueError, IndexError):
                other.append(k)
        else:
            other.append(k)
    return by_layer, other


def load_state_dict(model_dir):
    """Load all model weights from safetensors files in directory."""
    sd = {}
    files = sorted([f for f in os.listdir(model_dir) if f.endswith(".safetensors")])
    print(f"  loading {len(files)} safetensors from {model_dir}", flush=True)
    for f in files:
        path = os.path.join(model_dir, f)
        with safe_open(path, framework="pt") as st:
            for key in st.keys():
                sd[key] = st.get_tensor(key)
    print(f"  loaded {len(sd)} params", flush=True)
    return sd


def main():
    args = parse_args()
    print(f"[start] {args.tag}", flush=True)

    print("[load] orig", flush=True)
    sd_orig = load_state_dict(args.orig_dir)
    print("[load] unl", flush=True)
    sd_unl = load_state_dict(args.unl_dir)

    # detect num layers
    layer_indices = set()
    for k in sd_orig:
        if "model.layers." in k:
            try:
                L = int(k.split("model.layers.")[1].split(".")[0])
                layer_indices.add(L)
            except (ValueError, IndexError):
                pass
    n_layers = max(layer_indices) + 1
    print(f"[ok] n_layers = {n_layers}", flush=True)

    by_layer_orig, _ = collect_layer_param_names(sd_orig.keys(), n_layers)

    drift_total_fro = {}
    drift_per_component = {L: {} for L in range(n_layers)}

    for L in range(n_layers):
        names = by_layer_orig[L]
        layer_drift_sq = 0.0
        for name in names:
            if name not in sd_unl:
                print(f"  [warn] missing in unl: {name}", flush=True)
                continue
            w_o = sd_orig[name].float()
            w_u = sd_unl[name].float()
            if w_o.shape != w_u.shape:
                print(f"  [warn] shape mismatch {name}", flush=True)
                continue
            d = (w_u - w_o).pow(2).sum().item()
            layer_drift_sq += d

            # also break down by component type
            comp = "other"
            for marker in ["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj",
                           "input_layernorm", "post_attention_layernorm"]:
                if marker in name:
                    comp = marker
                    break
            drift_per_component[L].setdefault(comp, 0.0)
            drift_per_component[L][comp] += d

        drift_total_fro[L] = layer_drift_sq ** 0.5
        # convert per-component squared sums to Frobenius
        for c in drift_per_component[L]:
            drift_per_component[L][c] = drift_per_component[L][c] ** 0.5
        print(f"  L{L:2d}  ||dw||_F = {drift_total_fro[L]:.4f}", flush=True)

    result = {
        "tag": args.tag,
        "n_layers": n_layers,
        "drift_per_layer": drift_total_fro,
        "drift_per_layer_per_component": drift_per_component,
        "orig_dir": args.orig_dir,
        "unl_dir": args.unl_dir,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[done] saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
