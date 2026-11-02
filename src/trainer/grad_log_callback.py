"""
Per-layer gradient norm logger.

Activated by env var LOG_GRAD_NORMS=1.

Records per-step:
- forget gradient norms from FORGET_BUFFER (populated in *_GradLog trainer
  subclasses via torch.autograd.grad). Falls back to param.grad if the buffer
  is empty (kept for backward-compat with the older callback contract).
- retain gradient norms + G-effect quantities (per layer / per component) when
  the trainer has accumulated retain grads via the *_GradLog trainer subclass.
- optional weight L2 norms, weight delta from initial, per-step grad cosine.

Output: <output_dir>/grad_log/grad_norms_rank<r>.jsonl  (one json per optimizer step)
"""

import json
import math
import os
import re
import logging
from collections import defaultdict
from typing import Dict, Optional, Tuple

import torch
from transformers import TrainerCallback


logger = logging.getLogger(__name__)


# Module-level shared retain-grad buffer. Both compute_loss
# (in *_GradLog trainers) and the callback access this directly so the buffer
# does not need to live on the model object — that path is unreliable when HF
# wraps the model with DeepSpeed/Accelerate engines.
RETAIN_BUFFER: Dict[str, object] = {"grads": {}, "loss": 0.0, "count": 0}

# Same idea for forget grads. We capture forget grads via torch.autograd.grad
# inside the trainer's compute_loss (with retain_graph=True so the trainer's
# accelerator.backward() can still consume the graph). This bypasses the
# "param.grad is None after DeepSpeed reduces" issue — autograd.grad fills its
# own returned tensors instead of writing into param.grad.
FORGET_BUFFER: Dict[str, object] = {"grads": {}, "loss": 0.0, "count": 0}


def reset_retain_buffer():
    RETAIN_BUFFER["grads"] = {}
    RETAIN_BUFFER["loss"] = 0.0
    RETAIN_BUFFER["count"] = 0


def reset_forget_buffer():
    FORGET_BUFFER["grads"] = {}
    FORGET_BUFFER["loss"] = 0.0
    FORGET_BUFFER["count"] = 0


_LAYER_PAT = re.compile(r"model\.layers\.(\d+)\.")

_COMPONENT_PATTERNS = [
    ("q_proj",                 re.compile(r"self_attn\.q_proj")),
    ("k_proj",                 re.compile(r"self_attn\.k_proj")),
    ("v_proj",                 re.compile(r"self_attn\.v_proj")),
    ("o_proj",                 re.compile(r"self_attn\.o_proj")),
    ("gate_proj",              re.compile(r"mlp\.gate_proj")),
    ("up_proj",                re.compile(r"mlp\.up_proj")),
    ("down_proj",              re.compile(r"mlp\.down_proj")),
    ("input_layernorm",        re.compile(r"input_layernorm")),
    ("post_attention_layernorm", re.compile(r"post_attention_layernorm")),
]


def _classify(name: str) -> Tuple[Optional[int], Optional[str], Optional[str]]:
    m = _LAYER_PAT.search(name)
    if m:
        idx = int(m.group(1))
        for comp_name, comp_pat in _COMPONENT_PATTERNS:
            if comp_pat.search(name):
                return idx, comp_name, None
        return idx, "other", None
    if "embed_tokens" in name:
        return None, None, "embed_tokens"
    if "lm_head" in name:
        return None, None, "lm_head"
    if name.endswith("model.norm.weight") or name.endswith("model.norm.bias"):
        return None, None, "final_norm"
    return None, None, name


def _unwrap(model):
    while hasattr(model, "module"):
        model = model.module
    return model


def _norms_from_named_grads(named_grads):
    layer_sq = defaultdict(float)
    comp_sq = defaultdict(float)
    other_sq = defaultdict(float)
    per_param = {}
    total_sq = 0.0
    for name, g in named_grads:
        if g is None:
            continue
        sq = float(g.detach().to(torch.float32).pow(2).sum().item())
        per_param[name] = math.sqrt(sq)
        total_sq += sq
        idx, comp, other = _classify(name)
        if idx is not None:
            layer_sq[idx] += sq
            if comp is not None:
                comp_sq[(idx, comp)] += sq
        else:
            other_sq[other] += sq
    return {
        "total": math.sqrt(total_sq),
        "per_layer": {str(k): math.sqrt(v) for k, v in sorted(layer_sq.items())},
        "per_component": {f"L{i}.{c}": math.sqrt(v) for (i, c), v in comp_sq.items()},
        "other": {k: math.sqrt(v) for k, v in other_sq.items()},
        "per_param": per_param,
    }


def _inner_products_named(named_pairs):
    layer_ip = defaultdict(float)
    comp_ip = defaultdict(float)
    other_ip = defaultdict(float)
    per_param = {}
    total = 0.0
    for name, a, b in named_pairs:
        if a is None or b is None:
            continue
        ip = float((a * b).sum().item())
        per_param[name] = ip
        total += ip
        idx, comp, other = _classify(name)
        if idx is not None:
            layer_ip[idx] += ip
            if comp is not None:
                comp_ip[(idx, comp)] += ip
        else:
            other_ip[other] += ip
    return {
        "total": total,
        "per_layer": {str(k): v for k, v in sorted(layer_ip.items())},
        "per_component": {f"L{i}.{c}": v for (i, c), v in comp_ip.items()},
        "other": {k: v for k, v in other_ip.items()},
        "per_param": per_param,
    }


def _pairwise_layer_dw_cosine(named_deltas):
    """Pairwise cosine matrix of per-layer Δw vectors.

    Each layer's Δw is the concatenation of all that layer's parameter deltas.
    Returns dict {f"L{i}-L{j}": cos_value}, symmetric, including self (cos=1).
    Skips non-layer params (embed_tokens, lm_head, final_norm).
    """
    layer_deltas = defaultdict(dict)   # L -> {within-layer-suffix: tensor}
    for name, delta in named_deltas:
        if delta is None: continue
        m = _LAYER_PAT.search(name)
        if not m: continue
        idx = int(m.group(1))
        suffix = name[m.end():]   # e.g. "self_attn.q_proj.weight"
        layer_deltas[idx][suffix] = delta
    layers = sorted(layer_deltas.keys())
    if not layers:
        return {}
    # per-layer total norm
    norms = {L: math.sqrt(sum(t.pow(2).sum().item() for t in layer_deltas[L].values())) for L in layers}
    cos_mat = {}
    for i, L in enumerate(layers):
        for j, M in enumerate(layers):
            if j < i:
                cos_mat[f"L{L}-L{M}"] = cos_mat[f"L{M}-L{L}"]   # symmetric copy
                continue
            common = set(layer_deltas[L].keys()) & set(layer_deltas[M].keys())
            ip = sum(float((layer_deltas[L][k] * layer_deltas[M][k]).sum().item()) for k in common)
            denom = norms[L] * norms[M] + 1e-30
            cos_mat[f"L{L}-L{M}"] = ip / denom
    return cos_mat


class GradNormLogger(TrainerCallback):
    def __init__(
        self,
        output_dir: str,
        every_n_steps: int = 1,
        log_per_param: bool = True,
        log_weight_norms: bool = True,
        log_weight_delta: bool = True,
        log_grad_cosine_with_prev: bool = True,
        log_retain: bool = True,
        log_layer_dw_cosine: bool = True,
    ):
        self.output_dir = output_dir
        self.every_n_steps = max(1, every_n_steps)
        self.log_per_param = log_per_param
        self.log_weight_norms = log_weight_norms
        self.log_weight_delta = log_weight_delta
        self.log_grad_cosine_with_prev = log_grad_cosine_with_prev
        self.log_retain = log_retain
        self.log_layer_dw_cosine = log_layer_dw_cosine
        self.fp = None
        self._init_weights: Dict[str, torch.Tensor] = {}
        self._prev_forget_grads: Dict[str, torch.Tensor] = {}

    def on_train_begin(self, args, state, control, **kwargs):
        os.makedirs(self.output_dir, exist_ok=True)
        rank = getattr(args, "process_index", 0)
        path = os.path.join(self.output_dir, f"grad_norms_rank{rank}.jsonl")
        self.fp = open(path, "a", buffering=1)

        model = _unwrap(kwargs.get("model"))
        n_total = sum(1 for _ in model.named_parameters())
        n_trainable = sum(1 for _, p in model.named_parameters() if p.requires_grad)
        n_in_layers = sum(1 for n, p in model.named_parameters()
                          if p.requires_grad and _LAYER_PAT.search(n))

        meta = {
            "_meta": True,
            "process_index": rank,
            "world_size": getattr(args, "world_size", 1),
            "every_n_steps": self.every_n_steps,
            "log_per_param": self.log_per_param,
            "log_weight_norms": self.log_weight_norms,
            "log_weight_delta": self.log_weight_delta,
            "log_grad_cosine_with_prev": self.log_grad_cosine_with_prev,
            "log_retain": self.log_retain,
            "n_named_parameters_total": n_total,
            "n_named_parameters_trainable": n_trainable,
            "n_named_parameters_in_layers": n_in_layers,
            "model_class": type(model).__name__,
        }
        self.fp.write(json.dumps(meta) + "\n")
        logger.info(f"[GradNormLogger] meta: total={n_total} trainable={n_trainable} in_layers={n_in_layers} class={type(model).__name__}")

        if self.log_weight_delta:
            for n, p in model.named_parameters():
                if p.requires_grad:
                    self._init_weights[n] = p.detach().to("cpu", torch.float32).clone()
            logger.info(f"[GradNormLogger] snapshotted {len(self._init_weights)} init weights to CPU.")

        reset_retain_buffer()
        reset_forget_buffer()

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        if self.fp is None:
            return
        if state.global_step % self.every_n_steps != 0 and state.global_step != 1:
            return

        model = _unwrap(kwargs.get("model"))
        if model is None:
            return

        # --------------- forget grads ---------------
        # Prefer module-level FORGET_BUFFER (populated by *_GradLog
        # compute_loss via torch.autograd.grad). Fall back to param.grad if the
        # buffer is empty — this lets the older path keep working when
        # LOG_GRAD_NORMS is off in the trainer subclass.
        forget_buf_active = FORGET_BUFFER["count"] > 0
        if forget_buf_active:
            n_acc = FORGET_BUFFER["count"]
            forget_grads_cpu = FORGET_BUFFER["grads"]
            forget_loss_avg = FORGET_BUFFER["loss"] / n_acc
            named_forget = [(n, forget_grads_cpu[n] / n_acc if n in forget_grads_cpu else None)
                             for n, p in model.named_parameters() if p.requires_grad]
        else:
            named_forget = [(n, p.grad) for n, p in model.named_parameters() if p.requires_grad]
            forget_loss_avg = None

        forget_norms = _norms_from_named_grads(named_forget)

        record = {
            "step": int(state.global_step),
            "epoch": float(state.epoch) if state.epoch is not None else None,
            "lr": float(kwargs["optimizer"].param_groups[0]["lr"]) if kwargs.get("optimizer") else None,
            "total_grad_norm_forget": forget_norms["total"],
            "per_layer_grad_norm_forget": forget_norms["per_layer"],
            "per_component_grad_norm_forget": forget_norms["per_component"],
            "other_grad_norm_forget": forget_norms["other"],
        }
        if forget_buf_active:
            record["forget_loss"] = forget_loss_avg
            record["forget_accum_count"] = n_acc
        if self.log_per_param:
            record["per_param_grad_norm_forget"] = forget_norms["per_param"]

        # --------------- retain grads + G-effect ---------------
        if self.log_retain and RETAIN_BUFFER["count"] > 0:
            n_acc_r = RETAIN_BUFFER["count"]
            retain_grads_cpu = RETAIN_BUFFER["grads"]
            retain_loss_avg = RETAIN_BUFFER["loss"] / n_acc_r

            named_retain = [(n, retain_grads_cpu[n] / n_acc_r if n in retain_grads_cpu else None)
                             for n, p in model.named_parameters() if p.requires_grad]
            retain_norms = _norms_from_named_grads(named_retain)

            # G-effect: <g_forget, g_retain>, both on CPU fp32, per-param
            named_pairs = []
            forget_dict = (
                {n: (forget_grads_cpu[n] / n_acc) if n in forget_grads_cpu else None
                 for n, _ in [(n, p) for n, p in model.named_parameters() if p.requires_grad]}
                if forget_buf_active else
                {n: (p.grad.detach().to("cpu", torch.float32) if p.grad is not None else None)
                 for n, p in model.named_parameters() if p.requires_grad}
            )
            for (n, p) in model.named_parameters():
                if not p.requires_grad: continue
                gf = forget_dict.get(n)
                gr = retain_grads_cpu[n] / n_acc_r if n in retain_grads_cpu else None
                if gf is None or gr is None: continue
                named_pairs.append((n, gf, gr))
            ip = _inner_products_named(named_pairs)

            def _cos(ip_d, fnorm_d, rnorm_d):
                out = {}
                for k, v in ip_d.items():
                    a = fnorm_d.get(k, 0.0); b = rnorm_d.get(k, 0.0)
                    out[k] = (v / (a * b)) if (a > 0 and b > 0) else 0.0
                return out

            geffect_cosine_layer = _cos(ip["per_layer"], forget_norms["per_layer"], retain_norms["per_layer"])
            geffect_cosine_comp  = _cos(ip["per_component"], forget_norms["per_component"], retain_norms["per_component"])

            lr_now = record["lr"] or 0.0
            record.update({
                "retain_loss": retain_loss_avg,
                "retain_accum_count": n_acc_r,
                "total_grad_norm_retain": retain_norms["total"],
                "per_layer_grad_norm_retain": retain_norms["per_layer"],
                "per_component_grad_norm_retain": retain_norms["per_component"],
                "other_grad_norm_retain": retain_norms["other"],
                "total_geffect": ip["total"],
                "per_layer_geffect": ip["per_layer"],
                "per_component_geffect": ip["per_component"],
                "per_layer_geffect_cosine": geffect_cosine_layer,
                "per_component_geffect_cosine": geffect_cosine_comp,
                "predicted_retain_loss_change": -lr_now * ip["total"],
                "per_layer_predicted_retain_change": {k: -lr_now * v for k, v in ip["per_layer"].items()},
            })
            if self.log_per_param:
                record["per_param_grad_norm_retain"] = retain_norms["per_param"]
                record["per_param_geffect"] = ip["per_param"]

            reset_retain_buffer()

        # reset forget buffer at end of optimizer step
        if forget_buf_active:
            reset_forget_buffer()

        # --------------- weight norms ---------------
        if self.log_weight_norms:
            named_weights = [(n, p.detach()) for n, p in model.named_parameters() if p.requires_grad]
            wnorms = _norms_from_named_grads(named_weights)
            record["per_layer_weight_norm"] = wnorms["per_layer"]
            record["per_component_weight_norm"] = wnorms["per_component"]
            record["total_weight_norm"] = wnorms["total"]

        # --------------- weight delta from initial (expensive) ---------------
        if self.log_weight_delta and self._init_weights:
            delta_iter = []
            for n, p in model.named_parameters():
                if not p.requires_grad: continue
                if n not in self._init_weights: continue
                w_now = p.detach().to("cpu", torch.float32)
                delta = w_now - self._init_weights[n]
                delta_iter.append((n, delta))
            dnorms = _norms_from_named_grads(delta_iter)
            record["per_layer_weight_delta_from_init"] = dnorms["per_layer"]
            record["per_component_weight_delta_from_init"] = dnorms["per_component"]
            record["total_weight_delta_from_init"] = dnorms["total"]

            # --------------- pairwise per-layer Δw cosine matrix (inter-layer coupling) ---------------
            if self.log_layer_dw_cosine:
                record["per_layer_pair_dw_cosine"] = _pairwise_layer_dw_cosine(delta_iter)

        # --------------- per-layer cosine of forget grad with previous step ---------------
        if self.log_grad_cosine_with_prev and self._prev_forget_grads:
            cur_gf = {}
            if forget_buf_active:
                # Read gradients captured in named_forget before the buffer reset.
                for n, g in named_forget:
                    if g is not None:
                        cur_gf[n] = g.detach().to("cpu", torch.float32) if isinstance(g, torch.Tensor) else g
            else:
                cur_gf = {n: p.grad.detach().to("cpu", torch.float32)
                          for n, p in model.named_parameters() if p.requires_grad and p.grad is not None}
            named_pairs = []
            for n, gnow in cur_gf.items():
                if n in self._prev_forget_grads:
                    named_pairs.append((n, gnow, self._prev_forget_grads[n]))
            ip_prev = _inner_products_named(named_pairs)
            tmp_prev = [(n, self._prev_forget_grads[n]) for n in cur_gf.keys() if n in self._prev_forget_grads]
            prev_aggr = _norms_from_named_grads(tmp_prev)
            prev_norms_d = prev_aggr["per_layer"]
            cos_prev_layer = {}
            cur_norms_d = forget_norms["per_layer"]
            for k, v in ip_prev["per_layer"].items():
                a = cur_norms_d.get(k, 0.0); b = prev_norms_d.get(k, 0.0)
                cos_prev_layer[k] = (v / (a * b)) if (a > 0 and b > 0) else 0.0
            record["per_layer_grad_cosine_with_prev"] = cos_prev_layer
            self._prev_forget_grads = cur_gf
        elif self.log_grad_cosine_with_prev:
            if forget_buf_active:
                self._prev_forget_grads = {n: g for n, g in named_forget if g is not None}
            else:
                self._prev_forget_grads = {n: p.grad.detach().to("cpu", torch.float32)
                                            for n, p in model.named_parameters() if p.requires_grad and p.grad is not None}

        self.fp.write(json.dumps(record) + "\n")

    def on_train_end(self, args, state, control, **kwargs):
        if self.fp is not None:
            self.fp.close()
            self.fp = None
