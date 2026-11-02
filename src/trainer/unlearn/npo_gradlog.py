"""NPO variant that captures both forget and retain grads via autograd.grad.

Same pattern as GradAscent_GradLog. The forget loss returned by NPO.compute_loss
is the gamma * DPO-style loss; we capture autograd.grad(forget_loss, params)
for measurement. retain_graph=True keeps the graph alive for the trainer's
subsequent accelerator.backward()."""

import os
import torch
from trainer.unlearn.npo import NPO
from trainer.unlearn.grad_ascent_gradlog import (
    _unwrap,
    _accumulate_retain_grads,
    _accumulate_forget_grads,
)


class NPO_GradLog(NPO):
    def compute_loss(self, model, inputs, return_outputs=False):
        if os.environ.get("LOG_GRAD_RETAIN", "0") == "1" and "retain" in inputs:
            inner = _unwrap(model)
            retain_inputs = {
                "input_ids": inputs["retain"]["input_ids"],
                "attention_mask": inputs["retain"]["attention_mask"],
                "labels": inputs["retain"]["labels"],
            }
            retain_loss = self.compute_retain_loss(model=inner, retain_inputs=retain_inputs)
            _accumulate_retain_grads(inner, retain_loss)
            del retain_loss
            torch.cuda.empty_cache()

        result = super().compute_loss(model, inputs, return_outputs=True)
        if isinstance(result, tuple):
            forget_loss, forget_outputs = result
        else:
            forget_loss, forget_outputs = result, None

        if os.environ.get("LOG_GRAD_NORMS", "0") == "1":
            inner = _unwrap(model)
            _accumulate_forget_grads(inner, forget_loss)

        if return_outputs:
            return forget_loss, forget_outputs
        return forget_loss
