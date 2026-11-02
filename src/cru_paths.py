"""Paths for repository files, input data, and optional evaluators."""

import os
from pathlib import Path


CRU_ROOT = Path(__file__).resolve().parents[1]


def _configured_path(name: str, default: Path) -> Path:
    value = Path(os.environ.get(name, str(default))).expanduser()
    return value if value.is_absolute() else (CRU_ROOT / value).resolve()


CRU_INPUT_ROOT = _configured_path("CRU_INPUT_ROOT", CRU_ROOT)
CRU_LM_EVAL_ROOT = _configured_path("CRU_LM_EVAL_ROOT", CRU_ROOT / "lm-evaluation-harness")

# Hydra receives the same dynamically discovered roots as Python entry points.
os.environ["CRU_ROOT"] = str(CRU_ROOT)
os.environ["CRU_INPUT_ROOT"] = str(CRU_INPUT_ROOT)


def resolve_input_path(value):
    """Respect absolute inputs; prefer CRU files for relative paths and preserve Hub IDs.

    This helper is for reads only. Output paths must stay relative to CRU_ROOT.
    """
    if value is None:
        return None
    raw = os.fspath(value)
    path = Path(raw).expanduser()
    if path.is_absolute():
        return str(path)

    local = CRU_ROOT / path
    if local.exists():
        return str(local.resolve())
    if path.parts and path.parts[0] in {"saves", "saves_q", "data", "datasets", "outputs"}:
        external_input = CRU_INPUT_ROOT / path
        return str(external_input if external_input.exists() else local)
    if raw.startswith(("./", "../", "~/")):
        return str(local.resolve())
    return raw


def resolve_output_path(value):
    """Resolve generated artifacts under CRU, never inside read-only inputs."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = CRU_ROOT / path
    path = path.resolve()
    if CRU_INPUT_ROOT.resolve() == CRU_ROOT.resolve():
        return path
    try:
        path.relative_to(CRU_INPUT_ROOT.resolve())
    except ValueError:
        return path
    raise ValueError("Output paths must not point into the read-only input directory")


def prepare_evaluation(cfg, output=None):
    """Record original shell evaluation data and the shared retain95 reference."""
    import hashlib
    import json
    tofu = cfg.eval.get("tofu")
    if tofu is None or tofu.forget_split not in {"forget04", "forget01"}:
        return None
    data_root = os.environ.get("CRU_EVAL_DATA_ROOT")
    if not data_root:
        raise ValueError("Held-out evaluation requires CRU_EVAL_DATA_ROOT")
    root = Path(data_root)
    manifest_raw = (root / "split_manifest.json").read_bytes()
    manifest = json.loads(manifest_raw)
    if manifest.get("protocol") != "legacy_sh_v1":
        raise ValueError("Unexpected evaluation split protocol")
    for name, record in manifest["files"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"Evaluation split changed: {root / name}")
    validation = tofu.forget_split == "forget04"
    expected = dict(forget_split="forget04" if validation else "forget01",
                    forget_perturb_split="forget04_perturbed" if validation else "forget01_perturbed",
                    retain_perturb_split="retain_perturbed_320" if validation else "retain_perturbed_80",
                    holdout_split="holdout04" if validation else "holdout01",
                    retain_split="holdout04" if validation else "retain99")
    for key, value in expected.items():
        if tofu.get(key) != value:
            raise ValueError(f"Held-out evaluation requires {key}={value}")
    identity = dict(protocol="legacy_sh_v1", role="validation" if validation else "test",
                    splits=expected, split_manifest_sha256=hashlib.sha256(manifest_raw).hexdigest())
    reference = tofu.get("retain_logs_path")
    if reference:
        reference = Path(resolve_input_path(reference))
        if not reference.is_file():
            raise ValueError(f"Missing original retain95 reference logs: {reference}")
        identity["reference_sha256"] = hashlib.sha256(reference.read_bytes()).hexdigest()
    target = Path(output or cfg.paths.output_dir)
    target.mkdir(parents=True, exist_ok=True)
    marker = target / "evaluation_protocol.json"
    if marker.exists():
        if json.loads(marker.read_text()) != identity:
            raise ValueError("Evaluation protocol changed; use a new output directory")
    elif any(target.glob("*SUMMARY.json")) or any(target.glob("layer-*")) or any(target.glob("k-*")):
        raise ValueError("Existing results lack split provenance; use a new output directory")
    marker.write_text(json.dumps(identity, indent=2) + "\n")
    return identity
