from pathlib import Path as _CRUPath
import sys as _cru_sys
_cru_root = _CRUPath(__file__).resolve().parents[2]
if str(_cru_root / "src") not in _cru_sys.path:
    _cru_sys.path.insert(0, str(_cru_root / "src"))
from cru_paths import CRU_INPUT_ROOT, CRU_ROOT
if __name__ == '__main__':
    import os as _cru_os
    _cru_os.chdir(CRU_ROOT)

import json
import os
import random

def get_rank(values):
    sorted_values = sorted([(-v, i) for i, v in enumerate(values)], key=lambda x: (x[0], x[1]))
    ranks = [0]*len(values)
    current_rank = 1
    for i, (val, idx) in enumerate(sorted_values):
        if i > 0 and val != sorted_values[i-1][0]:
            current_rank = i + 1
        ranks[idx] = current_rank
    return ranks

fq = []
mu = []

for x in range(512):
    path = f"{CRU_INPUT_ROOT}/saves/unlearn/tofu_Llama-3.2-1B-Instruct_forget05_NPO/eval-attn-{x}/TOFU_SUMMARY.json"
    with open(path, 'r') as f:
        data = json.load(f)
        fq.append(data['forget_quality'])
        mu.append(data['model_utility'])

fq_ranks = get_rank(fq)
mu_ranks = get_rank(mu)

sum_ranks = [fq_ranks[i] + mu_ranks[i] for i in range(512)]

random.seed(42)
combined = list(enumerate(sum_ranks))
random.shuffle(combined)
sorted_indices = [x[0] for x in sorted(combined, key=lambda x: (x[1], -x[0]))]

print("fq_ranks:", fq_ranks)
print("mu_ranks:", mu_ranks)
print("sum_sorted:", sorted_indices)