"""Evaluate every ranked prefix, loading the two source models only once."""
import hashlib
import json
import math
from pathlib import Path
from copy import deepcopy

import hydra
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM

from cru_paths import prepare_evaluation, resolve_input_path
from layer_ops import copy_layers
from rank import load_ranking
from sweep_resume import prepare_sweep, completed, archive_partial, mark_complete


def evaluate_prefixes(
    model, donor, order, evaluate, save=None, seed=None, skip=(), finish=None,
):
    """Extend each prefix by one layer; non-layer parameters always stay Full FT."""
    for k, layer in enumerate(order, start=1):
        copy_layers(model, donor, [layer])
        if k in skip:
            continue
        if seed is not None:
            seed()
        evaluate(model, k)
        if save is not None:
            save(model, k)
        if finish is not None:
            finish(k)


@hydra.main(version_base=None, config_path="../configs", config_name="merge_sweep")
def main(cfg: DictConfig):
    evaluation_identity = prepare_evaluation(cfg)
    from evals import get_evaluators
    from model import get_model
    from trainer.utils import seed_everything

    if cfg.replace is not None or cfg.choice is not None:
        raise ValueError("Merge uses the complete ranking; do not set replace or choice")
    if not cfg.unlearn_model:
        raise ValueError("unlearn_model must point to the donor checkpoint")
    if not isinstance(cfg.save_models, bool):
        raise ValueError("save_models must be true or false")

    base_path = Path(resolve_input_path(cfg.model.model_args.pretrained_model_name_or_path))
    donor_path = Path(resolve_input_path(cfg.unlearn_model))
    ranking_path = Path(cfg.ranking)
    ranking_bytes = ranking_path.read_bytes()
    ranking = load_ranking(ranking_path, base_path)
    if evaluation_identity:
        source_identity = ranking.get("evaluation") or {}
        if (source_identity.get("role") != "validation"
                or source_identity.get("split_manifest_sha256") != evaluation_identity["split_manifest_sha256"]
                or source_identity.get("reference_sha256") != evaluation_identity.get("reference_sha256")
                or evaluation_identity["role"] not in {"validation", "test"}):
            raise ValueError("Merge requires a validation ranking from the same split manifest and reference")
    output = Path(cfg.paths.output_dir)
    prepare_sweep(output, cfg, [base_path, donor_path], ranking_bytes)
    expected_folders = {f"k-{k}" for k in range(1, ranking['num_layers'] + 1)}
    if any(path.name not in expected_folders for path in output.glob("k-*")):
        raise ValueError("Unexpected candidate directory in sweep output")
    done = set()
    for k in range(1, ranking['num_layers'] + 1):
        folder = output / f"k-{k}"
        if completed(folder):
            done.add(k)
        else:
            archive_partial(folder)
    (output / "ranking.json").write_bytes(ranking_bytes)
    if len(done) == ranking['num_layers']:
        print(f"All {len(done)} prefixes already complete: {output}")
        return

    seed_everything(cfg.seed)
    model, tokenizer = get_model(deepcopy(cfg.model))
    # Keep the donor on CPU; move only the next layer to the evaluation device.
    donor = AutoModelForCausalLM.from_pretrained(donor_path, torch_dtype="auto")
    evaluators = get_evaluators(cfg.eval)

    def evaluate(candidate, k):
        eval_dir = output / f"k-{k}" / "eval"
        print(f"Evaluating k={k}/{ranking['num_layers']}: {ranking['ranking'][:k]}")
        for evaluator in evaluators.values():
            evaluator.evaluate(
                model=candidate,
                tokenizer=tokenizer,
                template_args=cfg.model.template_args,
                output_dir=str(eval_dir),
                overwrite=True,
            )
        # A completion marker must never turn a missing/invalid score into success.
        for evaluator in evaluators.values():
            for suffix in ("EVAL", "SUMMARY"):
                path = evaluator.get_logs_file_path(str(eval_dir), suffix=suffix)
                values = json.loads(Path(path).read_text())
                if not isinstance(values, dict) or not values:
                    raise ValueError(f"Empty evaluation artifact: {path}")
                if suffix == "SUMMARY":
                    for metric in ("model_utility", "forget_quality"):
                        if not math.isfinite(float(values[metric])):
                            raise ValueError(f"Non-finite {metric}: {path}")
        provenance = dict(
            protocol="fullft_base_v2",
            base_model=str(base_path.resolve()),
            layer_donor=str(donor_path.resolve()),
            k=k,
            selected_layers=ranking["ranking"][:k],
            ranking=str((output / "ranking.json").resolve()),
            ranking_sha256=hashlib.sha256(ranking_bytes).hexdigest(),
            base_config_sha256=ranking["base_config_sha256"],
            model_saved=False,
        )
        eval_dir.mkdir(parents=True, exist_ok=True)
        (eval_dir / "CRU_MERGE.json").write_text(json.dumps(provenance, indent=2) + "\n")

    def save(candidate, k):
        model_dir = output / f"k-{k}" / "model"
        candidate.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
        record = output / f"k-{k}" / "eval" / "CRU_MERGE.json"
        provenance = json.loads(record.read_text())
        provenance["model_saved"] = True
        contents = json.dumps(provenance, indent=2) + "\n"
        (model_dir / "CRU_MERGE.json").write_text(contents)
        record.write_text(contents)

    evaluate_prefixes(
        model, donor, ranking["ranking"], evaluate,
        save=save if cfg.save_models else None,
        seed=lambda: seed_everything(cfg.seed),
        skip=done,
        finish=lambda k: mark_complete(output / f"k-{k}"),
    )
    print(f"Sweep complete: {output}. Models saved: {cfg.save_models}")


if __name__ == "__main__":
    main()
