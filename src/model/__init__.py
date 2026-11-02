from pathlib import Path as _CRUPath
import sys as _cru_sys

_cru_root = _CRUPath(__file__).resolve().parents[2]
if str(_cru_root / "src") not in _cru_sys.path:
    _cru_sys.path.insert(0, str(_cru_root / "src"))

from transformers import AutoModelForCausalLM, AutoTokenizer
from omegaconf import DictConfig, open_dict
import os
import torch
import logging
import re
from cru_paths import CRU_ROOT, resolve_input_path
from layer_ops import copy_layers

logger = logging.getLogger(__name__)

def extract_model_info(name):
    method_map = {'GradAscent': 'GA', 'NPO': 'NPO', 'RMU': 'RMU', 'SimNPO': 'SimNPO', 'AltPO': 'AltPO', 'SatImp': 'SatImp'}

    if "muse" in name:
        parts = name.split('_')
        if len(parts) >= 4:
            task_type = parts[2].upper()
            method = method_map.get(parts[-1], None)
            return (task_type, method)

    size_match = re.search(r'(\d+[bB])', name)
    size = size_match.group(1) if size_match else "Phi"

    method_match = re.search(r'_(GradAscent|SimNPO|AltPO|SatImp|RMU|NPO)$', name)
    method = method_map.get(method_match.group(1), None) if method_match else None

    return size, method


def get_dtype(model_args):
    with open_dict(model_args):
        torch_dtype = model_args.pop("torch_dtype", None)
    if model_args.get("attn_implementation", None) == "flash_attention_2":
        # This check handles https://github.com/Dao-AILab/flash-attention/blob/7153673c1a3c7753c38e4c10ef2c98a02be5f778/flash_attn/flash_attn_triton.py#L820
        # If you want to run at other precisions consider running "training or inference using
        # Automatic Mixed-Precision via the `with torch.autocast(device_type='torch_device'):`
        # decorator" or using an attn_implementation compatible with the precision in the model
        # config.
        assert torch_dtype in ["float16", "bfloat16"], ValueError(
            f"Invalid torch_dtype '{torch_dtype}' for the requested attention "
            f"implementation: 'flash_attention_2'. Supported types are 'float16' "
            f"and 'bfloat16'."
        )
    if torch_dtype == "float16":
        return torch.float16
    elif torch_dtype == "bfloat16":
        return torch.bfloat16
    return torch.float32


# ----------------------------------------------------------------------
# Optional ablation: selectively freeze layers via env vars.
# Default (no env vars set): no freezing — all params trainable.
#
#   FREEZE_LAYERS=mod:N           → keep only layers where layer_num % N == 0
#   FREEZE_LAYERS=keep:0,4,8,12   → keep only listed layers trainable
#   FREEZE_LAYERS=only:7          → alias for keep:7
#   FREEZE_LAYERS=drop:2,3,5      → freeze listed layers, keep the rest
#   FREEZE_NON_LAYER=1            → also freeze non-layer params (lm_head etc.)
#                                   embed_tokens / model.norm always pass through
#                                   unchanged (matches prior project convention).
#   MODEL_VERBOSE_PARAMS=1        → also print every param's requires_grad
# ----------------------------------------------------------------------
def _apply_freeze_layers(model):
    spec = os.environ.get("FREEZE_LAYERS", "").strip()
    if not spec:
        return False
    freeze_non_layer = os.environ.get("FREEZE_NON_LAYER", "0") == "1"

    if spec.startswith("mod:"):
        mod = int(spec.split(":", 1)[1])
        keep = lambda i: (i % mod == 0)
    elif spec.startswith("keep:") or spec.startswith("only:"):
        ks = {int(x) for x in spec.split(":", 1)[1].split(",") if x.strip()}
        keep = lambda i, s=ks: i in s
    elif spec.startswith("drop:"):
        ds = {int(x) for x in spec.split(":", 1)[1].split(",") if x.strip()}
        keep = lambda i, s=ds: i not in s
    else:
        raise ValueError(f"Unknown FREEZE_LAYERS spec: {spec!r}")

    for name, param in model.named_parameters():
        if "model.embed_tokens.weight" in name or "model.norm.weight" in name:
            continue
        m = re.search(r"model\.layers\.(\d+)", name)
        if m:
            param.requires_grad = keep(int(m.group(1)))
        else:
            param.requires_grad = (not freeze_non_layer)
    n_train = sum(1 for _, p in model.named_parameters() if p.requires_grad)
    n_total = sum(1 for _ in model.named_parameters())
    logger.info(f"[get_model] FREEZE_LAYERS={spec} freeze_non_layer={freeze_non_layer}: trainable {n_train}/{n_total}")
    return True


def get_model(model_cfg: DictConfig, replace: int = None, unlearn_model_path: str = None, choice: str = None):
    """Load a model, optionally copying one transformer layer from an unlearned checkpoint.

    The CRU pipeline uses this only during the replacement-evaluation stage.
    Final selected-layer composition is handled by ``src/merge.py``.
    """
    assert model_cfg is not None and model_cfg.model_args is not None, (
        "Model config not found or model_args absent in configs/model."
    )
    model_args = model_cfg.model_args
    tokenizer_args = model_cfg.tokenizer_args
    with open_dict(model_args):
        model_args.pretrained_model_name_or_path = resolve_input_path(
            model_args.pretrained_model_name_or_path
        )
    with open_dict(tokenizer_args):
        tokenizer_args.pretrained_model_name_or_path = resolve_input_path(
            tokenizer_args.pretrained_model_name_or_path
        )
    model = AutoModelForCausalLM.from_pretrained(
        torch_dtype=get_dtype(model_args), **model_args
    )
    _apply_freeze_layers(model)

    if replace is not None:
        if choice != "equal":
            raise ValueError(
                "The CRU pipeline supports choice=equal for per-layer replacement; "
                "use src/merge.py for selected-layer composition."
            )
        if not unlearn_model_path:
            raise ValueError("unlearn_model is required when replace is set")
        source_path = resolve_input_path(unlearn_model_path)
        source = AutoModelForCausalLM.from_pretrained(source_path, torch_dtype="auto")
        copied = copy_layers(model, source, [replace])
        logger.info("Copied %d parameter tensors from layer %d", copied, replace)

    return model, get_tokenizer(tokenizer_args)


def _add_or_replace_eos_token(tokenizer, eos_token: str) -> None:
    is_added = tokenizer.eos_token_id is None
    num_added_tokens = tokenizer.add_special_tokens({"eos_token": eos_token})

    if is_added:
        logger.info("Add eos token: {}".format(tokenizer.eos_token))
    else:
        logger.info("Replace eos token: {}".format(tokenizer.eos_token))

    if num_added_tokens > 0:
        logger.info("New tokens have been added, make sure `resize_vocab` is True.")


def get_tokenizer(tokenizer_cfg: DictConfig):
    tokenizer_cfg = dict(tokenizer_cfg)
    tokenizer_cfg["pretrained_model_name_or_path"] = resolve_input_path(
        tokenizer_cfg.get("pretrained_model_name_or_path")
    )
    try:
        tokenizer = AutoTokenizer.from_pretrained(**tokenizer_cfg)
    except Exception as e:
        error_message = (
            f"{'--' * 40}\n"
            f"Error {e} fetching tokenizer using AutoTokenizer.\n"
            f"Tokenizer requested from path: {tokenizer_cfg.get('pretrained_model_name_or_path', None)}\n"
            f"Full tokenizer config: {tokenizer_cfg}\n"
            f"{'--' * 40}"
        )
        raise RuntimeError(error_message)

    if tokenizer.eos_token_id is None:
        logger.info("replacing eos_token with <|endoftext|>")
        _add_or_replace_eos_token(tokenizer, eos_token="<|endoftext|>")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Setting pad_token as eos token: {}".format(tokenizer.pad_token))

    return tokenizer
