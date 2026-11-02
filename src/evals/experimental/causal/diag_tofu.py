from pathlib import Path as _CRUPath
import sys as _cru_sys
import os as _cru_os
_cru_root = _CRUPath(__file__).resolve().parents[4]
if str(_cru_root / "src") not in _cru_sys.path:
    _cru_sys.path.insert(0, str(_cru_root / "src"))
from cru_paths import CRU_ROOT, CRU_INPUT_ROOT
if __name__ == "__main__":
    _cru_os.chdir(CRU_ROOT)

"""Quick diagnostic: print TOFU split fields + first example."""
from datasets import load_dataset
for split in ["retain_perturbed", "retain95", "forget05_perturbed", "forget05"]:
    try:
        ds = load_dataset("locuslab/TOFU", split)["train"]
        ex0 = {k: (str(v)[:80]) for k, v in ds[0].items()}
        print(f"[{split}] N={len(ds)}")
        print(f"  fields: {list(ds.features)}")
        print(f"  ex0   : {ex0}")
    except Exception as e:
        print(f"[{split}] ERR {type(e).__name__}: {e}")
