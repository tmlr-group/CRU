"""Prepare deterministic train/validation subsets for MUSE Books or MUSE News."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
from cru_paths import CRU_ROOT, resolve_output_path  # noqa: E402


DEFAULT_CONFIGS = ("knowmem", "verbmem", "privleak")


def prepare_dataset(dataset_name: str, config: str, fraction: float, seed: int, output_dir: Path) -> tuple[Path, Path]:
    from datasets import DatasetDict, load_dataset

    full_dataset = load_dataset(f"muse-bench/MUSE-{dataset_name}", config)
    train_subset = DatasetDict()
    heldout_subset = DatasetDict()
    for split, dataset in full_dataset.items():
        shuffled = dataset.shuffle(seed=seed)
        cutoff = int(len(shuffled) * fraction)
        train_subset[split] = shuffled.select(range(cutoff))
        heldout_subset[split] = shuffled.select(range(cutoff, len(shuffled)))

    dataset_suffix = "_books" if dataset_name == "Books" else ""
    percent_train = int(round(fraction * 100))
    percent_heldout = 100 - percent_train
    train_path = output_dir / f"{config}{percent_train}{dataset_suffix}"
    heldout_path = output_dir / f"{config}{percent_heldout}{dataset_suffix}"
    if train_path == heldout_path:
        train_path = output_dir / f"{config}{percent_train}_train{dataset_suffix}"
        heldout_path = output_dir / f"{config}{percent_heldout}_heldout{dataset_suffix}"
    train_subset.save_to_disk(str(train_path))
    heldout_subset.save_to_disk(str(heldout_path))
    print(f"{config}: {train_path} ({sum(len(ds) for ds in train_subset.values())} rows)")
    print(f"{config}: {heldout_path} ({sum(len(ds) for ds in heldout_subset.values())} rows)")
    return train_path, heldout_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("books", "news"), required=True)
    parser.add_argument("--configs", nargs="+", choices=DEFAULT_CONFIGS, default=DEFAULT_CONFIGS)
    parser.add_argument("--fraction", type=float, default=0.8, help="fraction saved in the first subset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=CRU_ROOT / "datasets")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not 0.0 < args.fraction < 1.0:
        raise SystemExit("--fraction must be between 0 and 1")
    dataset_name = args.dataset.title()
    output_dir = resolve_output_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for config in args.configs:
        prepare_dataset(dataset_name, config, args.fraction, args.seed, output_dir)


if __name__ == "__main__":
    main()
