from pathlib import Path as _CRUPath
import sys as _cru_sys
_cru_root = _CRUPath(__file__).resolve().parents[3]
if str(_cru_root / "src") not in _cru_sys.path:
    _cru_sys.path.insert(0, str(_cru_root / "src"))
from cru_paths import CRU_INPUT_ROOT, CRU_ROOT
if __name__ == '__main__':
    import os as _cru_os
    _cru_os.chdir(CRU_ROOT)

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

checkpoint_path_1 = str(CRU_INPUT_ROOT / 'saves/finetune/tofu_Llama-3.2-1B-Instruct_full')
checkpoint_path_2 = str(CRU_INPUT_ROOT / 'saves/unlearn/tofu_Llama-3.2-1B-Instruct_forget05_RMU')

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model1 = AutoModelForCausalLM.from_pretrained(checkpoint_path_1).to(device)
model2 = AutoModelForCausalLM.from_pretrained(checkpoint_path_2).to(device)

layer_stats = {}

for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
    diff = torch.mean(torch.abs(param1.data - param2.data)).item()
    
    layer_name = name1
    if layer_name not in layer_stats:
        layer_stats[layer_name] = {
            'total_diff': 0,
            'param_count': 0
        }
    
    layer_stats[layer_name]['total_diff'] += diff
    layer_stats[layer_name]['param_count'] += 1

    flat_param1 = param1.data.view(-1)
    flat_param2 = param2.data.view(-1)
    cosine_sim = F.cosine_similarity(flat_param1, flat_param2, dim=0).item()

    layer_stats[layer_name]['cosine_total'] = layer_stats[layer_name].get('cosine_total', 0) + cosine_sim
    layer_stats[layer_name]['cosine_count'] = layer_stats[layer_name].get('cosine_count', 0) + 1

print("Every: ")
for layer_name, stats in layer_stats.items():
    avg_diff = stats['total_diff'] / stats['param_count']
    print(f"{layer_name}: {avg_diff}")

total_avg = sum(s['total_diff'] for s in layer_stats.values()) / sum(s['param_count'] for s in layer_stats.values())
print(f"\ntotal: {total_avg}")

print("\nCosine Similarity:")
for layer_name, stats in layer_stats.items():
    avg_cosine = stats['cosine_total'] / stats['cosine_count']
    print(f"{layer_name}: {avg_cosine}")

total_cosine = sum(s['cosine_total'] for s in layer_stats.values()) / sum(s['cosine_count'] for s in layer_stats.values())
print(f"\nTotal Cosine Similarity: {total_cosine}")