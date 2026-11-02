import torch
from typing import Dict, Any
from omegaconf import DictConfig, OmegaConf
from transformers import Trainer, TrainingArguments

from trainer.base import FinetuneTrainer
from trainer.unlearn.grad_ascent import GradAscent
from trainer.unlearn.grad_diff import GradDiff
from trainer.unlearn.npo import NPO
from trainer.unlearn.dpo import DPO
from trainer.unlearn.simnpo import SimNPO
from trainer.unlearn.rmu import RMU
from trainer.unlearn.weighted_grad_ascent import WeightedGradAscent
from trainer.unlearn.tnpo import TNPO
from trainer.unlearn.wtnpo import WTNPO
from trainer.unlearn.flat import FLAT
from trainer.unlearn.satimp import SatImp
from trainer.unlearn.altpo import AltPO
from trainer.unlearn.grad_ascent_gradlog import GradAscent_GradLog
from trainer.unlearn.npo_gradlog import NPO_GradLog

import logging

logger = logging.getLogger(__name__)

TRAINER_REGISTRY: Dict[str, Any] = {}


def _register_trainer(trainer_class):
    TRAINER_REGISTRY[trainer_class.__name__] = trainer_class


def load_trainer_args(trainer_args: DictConfig, dataset):
    # Recursively convert DictConfig -> plain dict so any nested overrides
    # (e.g. gradient_checkpointing_kwargs added via "+trainer.args...") are
    # JSON-serializable when HF Trainer later logs / saves trainer state.
    if isinstance(trainer_args, DictConfig):
        trainer_args = OmegaConf.to_container(trainer_args, resolve=True)
    else:
        trainer_args = dict(trainer_args)
    warmup_epochs = trainer_args.pop("warmup_epochs", None)
    if warmup_epochs:
        batch_size = trainer_args["per_device_train_batch_size"]
        grad_accum_steps = trainer_args["gradient_accumulation_steps"]
        num_devices = torch.cuda.device_count()
        dataset_len = len(dataset)
        trainer_args["warmup_steps"] = int(
            (warmup_epochs * dataset_len)
            // (batch_size * grad_accum_steps * num_devices)
        )

    trainer_args = TrainingArguments(**trainer_args)
    return trainer_args


def load_trainer(
    trainer_cfg: DictConfig,
    model,
    train_dataset=None,
    eval_dataset=None,
    tokenizer=None,
    data_collator=None,
    evaluator=None,
    template_args=None,
):
    trainer_args = trainer_cfg.args
    method_args = trainer_cfg.get("method_args", {})
    trainer_args = load_trainer_args(trainer_args, train_dataset)
    trainer_handler_name = trainer_cfg.get("handler")
    assert trainer_handler_name is not None, ValueError(
        f"{trainer_handler_name} handler not set"
    )
    trainer_cls = TRAINER_REGISTRY.get(trainer_handler_name, None)
    assert trainer_cls is not None, NotImplementedError(
        f"{trainer_handler_name} not implemented or not registered"
    )
    trainer = trainer_cls(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        args=trainer_args,
        evaluator=evaluator,
        template_args=template_args,
        **method_args,
    )
    logger.info(
        f"{trainer_handler_name} Trainer loaded, output_dir: {trainer_args.output_dir}"
    )
    return trainer, trainer_args


# Register Finetuning Trainer
_register_trainer(Trainer)
_register_trainer(FinetuneTrainer)

# Register Unlearning Trainer
_register_trainer(GradAscent)
_register_trainer(GradDiff)
_register_trainer(NPO)
_register_trainer(DPO)
_register_trainer(SimNPO)
_register_trainer(RMU)

_register_trainer(WeightedGradAscent)
_register_trainer(TNPO)
_register_trainer(WTNPO)
_register_trainer(FLAT)
_register_trainer(SatImp)
_register_trainer(AltPO)

# GradLog variants (added for per-layer / G-effect gradient measurement runs)
_register_trainer(GradAscent_GradLog)
_register_trainer(NPO_GradLog)
