from pathlib import Path as _CRUPath
import sys as _cru_sys
_cru_root = _CRUPath(__file__).resolve().parents[3]
if str(_cru_root / "src") not in _cru_sys.path:
    _cru_sys.path.insert(0, str(_cru_root / "src"))
from cru_paths import resolve_input_path


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Print scalar values from a TensorBoard event file.")
    parser.add_argument("event_file", help="Path to the TensorBoard event file")
    args = parser.parse_args()
    path = _CRUPath(resolve_input_path(args.event_file))
    if not path.is_file():
        parser.error(f"Event file does not exist: {path}")
    import tensorflow as tf
    for event in tf.compat.v1.train.summary_iterator(str(path)):
        for value in event.summary.value:
            if value.HasField("simple_value"):
                print(f"tag: {value.tag}, value: {value.simple_value}")


if __name__ == "__main__":
    main()
