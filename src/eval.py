"""Evaluate a checkpoint or a single-layer CRU replacement."""
import os

import hydra
from omegaconf import DictConfig

from trainer.utils import seed_everything
from model import get_model
from cru_paths import prepare_evaluation, CRU_ROOT
from evals import get_evaluators


@hydra.main(version_base=None, config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig):
    prepare_evaluation(cfg)
    seed_everything(cfg.seed)
    model, tokenizer = get_model(
        cfg.model,
        replace=cfg.replace,
        unlearn_model_path=cfg.unlearn_model,
        choice=cfg.choice,
    )
    for evaluator in get_evaluators(cfg.eval).values():
        evaluator.evaluate(
            model=model,
            tokenizer=tokenizer,
            template_args=cfg.model.template_args,
        )


if __name__ == "__main__":
    os.chdir(CRU_ROOT)
    main()
