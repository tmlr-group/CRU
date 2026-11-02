from pathlib import Path as _CRUPath
import sys as _cru_sys
_cru_root = _CRUPath(__file__).resolve().parents[1]
if str(_cru_root / "src") not in _cru_sys.path:
    _cru_sys.path.insert(0, str(_cru_root / "src"))
from cru_paths import CRU_ROOT
if __name__ == '__main__':
    import os as _cru_os
    _cru_os.chdir(CRU_ROOT)

import os
import hydra
from omegaconf import DictConfig
from data import get_data, get_collators
from model import get_model
from trainer import load_trainer
from evals import get_evaluator
from trainer.utils import seed_everything


@hydra.main(version_base=None, config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig):
    """Entry point of the code to train models
    Args:
        cfg (DictConfig): Config to train
    """
    seed_everything(cfg.trainer.args.seed)
    mode = cfg.get("mode", "train")
    model_cfg = cfg.model
    template_args = model_cfg.template_args
    assert model_cfg is not None, "Invalid model yaml passed in train config."
    model, tokenizer = get_model(model_cfg)

    # Load Dataset
    data_cfg = cfg.data
    data = get_data(
        data_cfg, mode=mode, tokenizer=tokenizer, template_args=template_args
    )

    # Load collator
    collator_cfg = cfg.collator
    collator = get_collators(collator_cfg, tokenizer=tokenizer)

    # Get Trainer
    trainer_cfg = cfg.trainer
    assert trainer_cfg is not None, ValueError("Please set trainer")

    # Get Evaluator
    evaluator = None
    eval_cfgs = cfg.get("eval", None)
    if eval_cfgs and cfg.trainer.args.get("do_eval", True):
        assert len(eval_cfgs) <= 1, ValueError(
            "Only one evaluation supported while training"
        )
        eval_name, eval_cfg = next(iter(eval_cfgs.items()))
        evaluator = get_evaluator(
            eval_name,
            eval_cfg,
            template_args=template_args,
            model=model,
            tokenizer=tokenizer,
        )

    trainer, trainer_args = load_trainer(
        trainer_cfg=trainer_cfg,
        model=model,
        train_dataset=data.get("train", None),
        eval_dataset=data.get("eval", None),
        tokenizer=tokenizer,
        data_collator=collator,
        evaluator=evaluator,
        template_args=template_args,
    )

    # ------------------------------------------------------------------
    # Optional: per-step per-layer gradient logger (env-var gated; off by
    # default so existing scripts behave identically).
    # Enable with LOG_GRAD_NORMS=1.  See src/trainer/grad_log_callback.py
    # for the full set of toggles.
    # ------------------------------------------------------------------
    if os.environ.get("LOG_GRAD_NORMS", "0") == "1":
        # Diagnostic: report and (optionally) repair requires_grad.
        # Some path in the trainer/accelerate setup is freezing 108/146 params
        # for these multi-GPU runs, leaving only 4 of 16 layers trainable.
        # Force re-enable so the unlearning trains the full model the same way
        # the original tofu_unlearn.sh runs do.
        n_before = sum(1 for p in trainer.model.parameters() if p.requires_grad)
        n_total = sum(1 for _ in trainer.model.parameters())
        for p in trainer.model.parameters():
            p.requires_grad = True
        n_after = sum(1 for p in trainer.model.parameters() if p.requires_grad)
        print(f"[gradlog] requires_grad before={n_before}/{n_total} after force={n_after}/{n_total}")

        from trainer.grad_log_callback import GradNormLogger
        trainer.add_callback(
            GradNormLogger(
                output_dir=os.path.join(trainer_args.output_dir, "grad_log"),
                every_n_steps=int(os.environ.get("GRAD_LOG_EVERY", "1")),
                log_per_param=os.environ.get("GRAD_LOG_PER_PARAM", "1") == "1",
                log_weight_norms=os.environ.get("GRAD_LOG_WEIGHT_NORMS", "1") == "1",
                log_weight_delta=os.environ.get("GRAD_LOG_WEIGHT_DELTA", "1") == "1",
                log_grad_cosine_with_prev=os.environ.get("GRAD_LOG_GRAD_COSINE", "1") == "1",
                log_retain=os.environ.get("LOG_GRAD_RETAIN", "1") == "1",
            )
        )

    if trainer_args.do_train:
        trainer.train()
        trainer.save_state()
        trainer.save_model(trainer_args.output_dir)

    if trainer_args.do_eval:
        trainer.evaluate(metric_key_prefix="eval")


if __name__ == "__main__":
    main()
