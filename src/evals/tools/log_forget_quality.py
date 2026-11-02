from pathlib import Path as _CRUPath
import sys as _cru_sys
_cru_root = _CRUPath(__file__).resolve().parents[3]
if str(_cru_root) not in _cru_sys.path:
    _cru_sys.path.insert(0, str(_cru_root))

import math
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--number", type=float, required=True)

args = parser.parse_args()

print(round(math.log(args.number), 4))