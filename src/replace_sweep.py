"""Evaluate independent single-layer replacements with two model loads."""
from pathlib import Path
from copy import deepcopy

import hydra
import torch
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM

from cru_paths import prepare_evaluation, resolve_input_path
from layer_ops import copy_layers, _LAYER


def evaluate_layers(model, donor, num_layers, evaluate, seed=None):
    """Keep only one original layer on CPU and restore it after each candidate."""
    for layer in range(num_layers):
        originals = []
        for name, parameter in model.named_parameters():
            match = _LAYER.search(name)
            if match and int(match.group(1)) == layer:
                originals.append((parameter, parameter.detach().cpu().clone()))
        try:
            copy_layers(model, donor, [layer])
            if seed is not None:
                seed()
            evaluate(model, layer)
        finally:
            with torch.no_grad():
                for parameter, original in originals:
                    parameter.copy_(original)


@hydra.main(version_base=None, config_path="../configs", config_name="eval")
def main(cfg: DictConfig):
    prepare_evaluation(cfg)
    from evals import get_evaluators
    from model import get_model
    from trainer.utils import seed_everything

    if cfg.replace is not None or cfg.choice is not None:
        raise ValueError("Replace sweep evaluates all layers; omit replace and choice")
    if not cfg.unlearn_model:
        raise ValueError("unlearn_model must point to the donor checkpoint")
    seed_everything(cfg.seed)
    model, tokenizer = get_model(deepcopy(cfg.model))
    donor = AutoModelForCausalLM.from_pretrained(
        resolve_input_path(cfg.unlearn_model), torch_dtype="auto"
    )
    evaluators = get_evaluators(cfg.eval)

    def evaluate(candidate, layer):
        print(f"Evaluating independent layer {layer}/{model.config.num_hidden_layers - 1}")
        for evaluator in evaluators.values():
            evaluator.evaluate(
                model=candidate,
                tokenizer=tokenizer,
                template_args=cfg.model.template_args,
                output_dir=str(Path(cfg.paths.output_dir) / f"layer-{layer}"),
                overwrite=True,
            )

    evaluate_layers(
        model, donor, model.config.num_hidden_layers, evaluate,
        seed=lambda: seed_everything(cfg.seed),
    )


if __name__ == "__main__":
    main()
