from pathlib import Path as _CRUPath
import sys as _cru_sys
_cru_root = _CRUPath(__file__).resolve().parents[2]
if str(_cru_root / "src") not in _cru_sys.path:
    _cru_sys.path.insert(0, str(_cru_root / "src"))
from cru_paths import CRU_ROOT
if __name__ == '__main__':
    import os as _cru_os
    _cru_os.chdir(CRU_ROOT)

# import json
# import datasets
# from pathlib import Path

# def load_jsonl(path):
#     data = []
#     with open(path, 'r', encoding='utf-8') as f:
#         for line in f:
#             if line.strip():
#                 data.append(json.loads(line))
#     return data

# def convert_alt_jsonl_to_arrow(jsonl_path, output_dir, split_name="forget05"):
#     # 读取 JSONL
#     alt_data = load_jsonl(jsonl_path)
#     # 增加 index 字段
#     for idx, item in enumerate(alt_data):
#         item["index"] = idx
#     # 转为 Arrow
#     dataset = datasets.Dataset.from_list(alt_data)
#     output_path = Path(output_dir)
#     output_path.mkdir(parents=True, exist_ok=True)
#     dataset.save_to_disk(str(output_path))
#     print(f"Saved to {output_path}, num samples: {len(dataset)}")

# if __name__ == "__main__":
#     convert_alt_jsonl_to_arrow(
#         "../open-unlearning/methods/alternate-preference-optimization/paper_models/tofu_ft_llama2-7b/forget05/alt5_seed_0.json",
#         "./datasets/forget05_alt"
#     )

import datasets

# 假设你的 Arrow 数据集目录为 ./datasets/forget05_alt
dataset = datasets.load_from_disk("./datasets/forget05_alt")

# 打印数据集长度
print(f"Total samples: {len(dataset)}")

# 打印前三个样本
for i in range(3):
    print(f"Sample {i}: {dataset[i]}")