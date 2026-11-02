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
import pandas as pd
import json

df = pd.read_parquet(str(CRU_INPUT_ROOT / 'data/wmdp/wmdp-corpora/train-00000-of-00001.parquet'))

output_path = 'data/wmdp/wmdp-corpora/cyber-forget-corpus.jsonl'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, 'w', encoding='utf-8') as f:
    for _, row in df.iterrows():
        text_content = row['text']
        f.write(json.dumps(text_content, ensure_ascii=False) + '\n')
