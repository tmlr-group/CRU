from pathlib import Path as _CRUPath
import sys as _cru_sys
_cru_root = _CRUPath(__file__).resolve().parents[2]
if str(_cru_root / "src") not in _cru_sys.path:
    _cru_sys.path.insert(0, str(_cru_root / "src"))
from cru_paths import CRU_INPUT_ROOT, CRU_ROOT
if __name__ == '__main__':
    import os as _cru_os
    _cru_os.chdir(CRU_ROOT)

import os
import json
import math

MODEL_CONFIGS = {
    "1B-GA": {
        "base_dir": str(CRU_INPUT_ROOT / 'saves/unlearn/tofu_Llama-3.2-1B-Instruct_forget05_GradAscent'),
        "evals_values": list(range(16)),
        "prefix": "G"
    },
    "1B-NPO": {
        "base_dir": str(CRU_INPUT_ROOT / 'saves/unlearn/tofu_Llama-3.2-1B-Instruct_forget05_NPO'),
        "evals_values": list(range(16)),
        "prefix": "N"
    },
    "1B-SimNPO": {
        "base_dir": str(CRU_INPUT_ROOT / 'saves/unlearn/tofu_Llama-3.2-1B-Instruct_forget05_SimNPO'),
        "evals_values": list(range(16)),
        "prefix": "S"
    },
    "1B-RMU": {
        "base_dir": str(CRU_INPUT_ROOT / 'saves/unlearn/tofu_Llama-3.2-1B-Instruct_forget05_RMU'),
        "evals_values": list(range(16)),
        "prefix": "R"
    },
    "3B-GA": {
        "base_dir": str(CRU_INPUT_ROOT / 'saves/unlearn/tofu_Llama-3.2-3B-Instruct_forget05_GradAscent'),
        "evals_values": list(range(28)),
        "prefix": "G"
    },
    "3B-NPO": {
        "base_dir": str(CRU_INPUT_ROOT / 'saves/unlearn/tofu_Llama-3.2-3B-Instruct_forget05_NPO'),
        "evals_values": list(range(28)),
        "prefix": "N"
    },
    "3B-SimNPO": {
        "base_dir": str(CRU_INPUT_ROOT / 'saves/unlearn/tofu_Llama-3.2-3B-Instruct_forget05_SimNPO'),
        "evals_values": list(range(28)),
        "prefix": "S"
    },
    "7B-GA": {
        "base_dir": str(CRU_INPUT_ROOT / 'saves/unlearn/tofu_Llama-2-7b-chat-hf_forget05_GradAscent'),
        "evals_values": list(range(32)),
        "prefix": "G"
    },
    "7B-SimNPO": {
        "base_dir": str(CRU_INPUT_ROOT / 'saves/unlearn/tofu_Llama-2-7b-chat-hf_forget05_SimNPO'),
        "evals_values": list(range(32)),
        "prefix": "S"
    },
    "7B-NPO": {
        "base_dir": str(CRU_INPUT_ROOT / 'saves/unlearn/tofu_Llama-2-7b-chat-hf_forget05_NPO'),
        "evals_values": list(range(32)),
        "prefix": "N"
    },
    "Phi-GA": {
        "base_dir": str(CRU_INPUT_ROOT / 'saves/unlearn/tofu_Phi-3.5-mini-instruct_forget05_GradAscent'),
        "evals_values": list(range(32)),
        "prefix": "G"
    },
    "Phi-NPO": {
        "base_dir": str(CRU_INPUT_ROOT / 'saves/unlearn/tofu_Phi-3.5-mini-instruct_forget05_NPO'),
        "evals_values": list(range(32)),
        "prefix": "N"
    },
    "Phi-SimNPO": {
        "base_dir": str(CRU_INPUT_ROOT / 'saves/unlearn/tofu_Phi-3.5-mini-instruct_forget05_SimNPO'),
        "evals_values": list(range(32)),
        "prefix": "S"
    }
}

def get_all_sorted_indices(model_names, metric_name):
    data = []
    
    for model_name in model_names:
        config = MODEL_CONFIGS[model_name]
        base_dir = config["base_dir"]
        evals_values = config["evals_values"]
        prefix = config["prefix"]
        
        for idx in evals_values:
            file_path = os.path.join(base_dir, f"evals-equal-new-{idx}", "TOFU_SUMMARY.json")
            if not os.path.exists(file_path):
                file_path = os.path.join(base_dir, f"evals-equal-{idx}", "TOFU_SUMMARY.json")
            
            if os.path.exists(file_path):
                with open(file_path, "r") as f:
                    info = json.load(f)
                    value = info.get(metric_name)
                    identifier = f"{prefix}{idx}"
                    data.append((identifier, value))
            else:
                identifier = f"{prefix}{idx}"
                data.append((identifier, None))
    
    valid_data = [(identifier, val) for identifier, val in data if val is not None]
    valid_data.sort(key=lambda x: (-x[1], x[0]))
    return [x[0] for x in valid_data]

comparison_groups = [
    {
        "name": "1B-all-methods",
        "models": ["1B-GA", "1B-NPO"],
        "description": "Compare all methods on 1B model"
    },
    {
        "name": "3B-all-methods", 
        "models": ["3B-GA", "3B-NPO"],
        "description": "Compare all methods on 3B model"
    },
    {
        "name": "7B-all-methods",
        "models": ["7B-GA", "7B-NPO"],
        "description": "Compare all methods on 7B model"
    },
    {
        "name": "Phi-all-methods",
        "models": ["Phi-GA", "Phi-NPO"],
        "description": "Compare all methods on Phi-3.5 model"
    }
]

output_dir = "select"
os.makedirs(output_dir, exist_ok=True)

for group in comparison_groups:
    group_name = group["name"]
    models = group["models"]
    description = group["description"]
    
    print(f"\n{'='*60}")
    print(f"Processing: {group_name}")
    print(f"Description: {description}")
    print(f"Models: {models}")
    print(f"{'='*60}")
    
    model_utility_sorted = get_all_sorted_indices(models, "model_utility")
    forget_quality_sorted = get_all_sorted_indices(models, "forget_quality")
    
    utility_rank = {identifier: i for i, identifier in enumerate(model_utility_sorted)}
    quality_rank = {identifier: i for i, identifier in enumerate(forget_quality_sorted)}
    
    common_identifiers = set(utility_rank.keys()).intersection(quality_rank.keys())
    
    combined = []
    for identifier in common_identifiers:
        combined.append((utility_rank[identifier] + quality_rank[identifier], identifier))
    combined.sort(key=lambda x: (x[0], x[1]))
    
    k = len(common_identifiers)
    selected_identifiers = [identifier for (_, identifier) in combined[:k]] if combined else []
    
    used_layer_indices = set()
    unique_selected_identifiers = []
    for identifier in selected_identifiers:
        layer_idx = int(identifier[1:])
        if layer_idx not in used_layer_indices:
            unique_selected_identifiers.append(identifier)
            used_layer_indices.add(layer_idx)
    
    file_path = os.path.join(output_dir, f"{group_name}-cross-method.txt")
    
    with open(file_path, "w") as f:
        f.write(f"model_utility: {','.join(model_utility_sorted)}\n")
        f.write(f"forget_quality: {','.join(forget_quality_sorted)}\n")
        f.write(f"selected_identifiers: {','.join(unique_selected_identifiers)}\n")
    
    with open(file_path, "r") as f:
        lines = f.readlines()
        model_utility = lines[0].split(": ")[1].strip().split(",")
        forget_quality = lines[1].split(": ")[1].strip().split(",")
        selected = lines[2].split(": ")[1].strip().split(",") if lines[2].split(": ")[1].strip() else []
    
    print(f"Model Utility ({len(model_utility)}): {model_utility[:10]}{'...' if len(model_utility) > 10 else ''}")
    print(f"Forget Quality ({len(forget_quality)}): {forget_quality[:10]}{'...' if len(forget_quality) > 10 else ''}")
    print(f"Selected ({len(selected)}): {selected[:10] if selected and len(selected) > 10 else selected}")

print(f"\n{'='*60}")
print(f"Results saved to directory: {output_dir}")
print(f"{'='*60}")