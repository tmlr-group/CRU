from pathlib import Path
import os
import sys

CRU_ROOT = Path(__file__).resolve().parents[2]
os.chdir(CRU_ROOT)
sys.path.insert(0, str(CRU_ROOT))

"""
Create 4-block cross-validation splits of forget05 / forget05_perturbed / holdout05.

forget05 has 200 questions. We divide into 5 blocks of 40:
  blk0: test = indices   0-39,  val = indices  40-199
  blk1: test = indices  40-79,  val = indices   0-39 + 80-199
  blk2: test = indices 80-119,  val = indices  0-79 + 120-199
  blk3: test = indices 120-159, val = indices 0-119 + 160-199

This utility generates blocks 0–3 only; block 4 is not generated.

Saved to datasets/:
  forget_blkX_v, forget_blkX_t
  forget_blkX_v_perturbed, forget_blkX_t_perturbed
  holdout_blkX_v, holdout_blkX_t
"""

from datasets import load_dataset, DatasetDict

BLOCK_SIZE = 40
N_TOTAL = 200
N_BLOCKS = 4  # run blocks 0-3 only

print("Loading forget05, forget05_perturbed, holdout05 ...")
forget_ds = load_dataset("locuslab/TOFU", "forget05")["train"]
forget_perturb_ds = load_dataset("locuslab/TOFU", "forget05_perturbed")["train"]
holdout_ds = load_dataset("locuslab/TOFU", "holdout05")["train"]

assert len(forget_ds) == N_TOTAL, f"Expected {N_TOTAL}, got {len(forget_ds)}"
assert len(forget_perturb_ds) == N_TOTAL
assert len(holdout_ds) == N_TOTAL

print(f"forget05: {len(forget_ds)}, forget05_perturbed: {len(forget_perturb_ds)}, holdout05: {len(holdout_ds)}")

for k in range(N_BLOCKS):
    test_start = k * BLOCK_SIZE
    test_end = (k + 1) * BLOCK_SIZE

    test_idx = list(range(test_start, test_end))
    val_idx = list(range(0, test_start)) + list(range(test_end, N_TOTAL))

    assert len(test_idx) == BLOCK_SIZE
    assert len(val_idx) == N_TOTAL - BLOCK_SIZE

    splits = [
        (forget_ds,         f"datasets/forget_blk{k}_v",             val_idx),
        (forget_ds,         f"datasets/forget_blk{k}_t",             test_idx),
        (forget_perturb_ds, f"datasets/forget_blk{k}_v_perturbed",   val_idx),
        (forget_perturb_ds, f"datasets/forget_blk{k}_t_perturbed",   test_idx),
        (holdout_ds,        f"datasets/holdout_blk{k}_v",            val_idx),
        (holdout_ds,        f"datasets/holdout_blk{k}_t",            test_idx),
    ]

    for src_ds, save_path, indices in splits:
        subset = src_ds.select(indices)
        DatasetDict({"train": subset}).save_to_disk(save_path)
        print(f"  Saved {save_path} ({len(subset)} samples)")

    print(f"Block {k} done: test=[{test_start},{test_end}), val size={len(val_idx)}")

print("\nAll block datasets created.")
