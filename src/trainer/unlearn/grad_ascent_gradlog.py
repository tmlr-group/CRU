"""GradAscent variant that captures BOTH forget and retain grads via
torch.autograd.grad on the unwrapped inner module.

Activated when LOG_GRAD_NORMS=1 (forget) and/or LOG_GRAD_RETAIN=1 (retain).

Why autograd.grad rather than param.grad:
- param.grad is set by the trainer's accelerator.backward(), which under
  DeepSpeed Stage 1 routes through DS engine — the engine reduces gradients
  into its own flat buckets and the original param.grad ends up None.
- torch.autograd.grad runs on the inner module's forward graph and returns
  fresh tensors regardless of DDP/DS engine state.

Why retain_graph=True for forget capture:
- super().compute_loss returns the forget loss tensor whose graph is the
  trainer's authoritative forward. We capture grads on this graph but keep
  it alive so the trainer's subsequent accelerator.backward() can use it
  for the actual optimizer update.
"""

import os
import torch
from trainer.unlearn.grad_ascent import GradAscent
from trainer.grad_log_callback import RETAIN_BUFFER, FORGET_BUFFER


def _unwrap(m):
    while hasattr(m, "module"):
        m = m.module
    return m


def _accumulate_into(buffer, inner_model, named_params, grads, loss_value):
    n_captured = 0
    for (n, _), g in zip(named_params, grads):
        if g is None:
            continue
        n_captured += 1
        gf = g.detach().to("cpu", torch.float32)
        if n in buffer["grads"]:
            buffer["grads"][n] = buffer["grads"][n] + gf
        else:
            buffer["grads"][n] = gf
    buffer["loss"] += float(loss_value)
    buffer["count"] += 1
    return n_captured


def _accumulate_retain_grads(inner_model, retain_loss):
    params_with_names = [(n, p) for n, p in inner_model.named_parameters() if p.requires_grad]
    params = [p for _, p in params_with_names]
    grads = torch.autograd.grad(
        retain_loss, params, retain_graph=False, create_graph=False, allow_unused=True
    )
    n = _accumulate_into(RETAIN_BUFFER, inner_model, params_with_names, grads, retain_loss.detach())
    if RETAIN_BUFFER["count"] <= 2:
        import logging
        logging.getLogger(__name__).info(
            f"[retain-grad] params={len(params)} captured={n} retain_loss={float(retain_loss.detach()):.4f}"
        )


def _accumulate_forget_grads(inner_model, forget_loss):
    """Capture forget gradient with retain_graph=True so the trainer's
    subsequent accelerator.backward(forget_loss) still works."""
    params_with_names = [(n, p) for n, p in inner_model.named_parameters() if p.requires_grad]
    params = [p for _, p in params_with_names]
    grads = torch.autograd.grad(
        forget_loss, params, retain_graph=True, create_graph=False, allow_unused=True
    )
    n = _accumulate_into(FORGET_BUFFER, inner_model, params_with_names, grads, forget_loss.detach())
    if FORGET_BUFFER["count"] <= 2:
        import logging
        logging.getLogger(__name__).info(
            f"[forget-grad] params={len(params)} captured={n} forget_loss={float(forget_loss.detach()):.4f}"
        )


class GradAscent_GradLog(GradAscent):
    def compute_loss(self, model, inputs, return_outputs=False):
        # ---- retain measurement (independent forward on inner) ----
        if os.environ.get("LOG_GRAD_RETAIN", "0") == "1" and "retain" in inputs:
            inner = _unwrap(model)
            retain_inputs = {
                "input_ids": inputs["retain"]["input_ids"],
                "attention_mask": inputs["retain"]["attention_mask"],
                "labels": inputs["retain"]["labels"],
            }
            retain_outputs = inner(**retain_inputs)
            retain_loss = retain_outputs.loss
            _accumulate_retain_grads(inner, retain_loss)
            del retain_outputs, retain_loss
            torch.cuda.empty_cache()

        # ---- forget loss via super (returns tensor with graph through wrapper) ----
        result = super().compute_loss(model, inputs, return_outputs=True)
        if isinstance(result, tuple):
            forget_loss, forget_outputs = result
        else:
            forget_loss, forget_outputs = result, None

        # ---- forget grad measurement (autograd.grad with retain_graph=True) ----
        if os.environ.get("LOG_GRAD_NORMS", "0") == "1":
            inner = _unwrap(model)
            _accumulate_forget_grads(inner, forget_loss)

        if return_outputs:
            return forget_loss, forget_outputs
        return forget_loss
