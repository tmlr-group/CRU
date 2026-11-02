"""Reproduce select_data_tofu.py slices and the original shell evaluation inputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path

SOURCES = {
    "forget05": (("forget04", 0, 160),),
    "forget05_perturbed": (("forget04_perturbed", 0, 160),),
    "holdout05": (("holdout04", 40, 200),),
    "retain_perturbed": (("retain_perturbed_320", 0, 320), ("retain_perturbed_80", 320, 400)),
    "forget01": (("forget01", 0, None),),
    "forget01_perturbed": (("forget01_perturbed", 0, None),),
    "holdout01": (("holdout01", 0, None),),
}


def prepare(output, load_source):
    # Four model/method jobs may prepare the same immutable splits concurrently.
    import fcntl
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_name(output.name + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _prepare(output, load_source)


def _prepare(output, load_source):
    files, sources = {}, {}
    for name, targets in SOURCES.items():
        rows = list(load_source(name))
        raw = json.dumps(rows, ensure_ascii=False, sort_keys=True, allow_nan=False)
        sources[name] = dict(count=len(rows), sha256=hashlib.sha256(raw.encode()).hexdigest())
        for target, start, stop in targets:
            if stop is not None and len(rows) < stop:
                raise ValueError(f"{name} has fewer rows than the original slice requires")
            subset = rows[start:stop]
            files[target + ".json"] = json.dumps(subset, ensure_ascii=False, allow_nan=False) + "\n"
    manifest = dict(protocol="legacy_sh_v1", sources=sources,
                    files={name: dict(count=len(json.loads(raw)), sha256=hashlib.sha256(raw.encode()).hexdigest())
                           for name, raw in files.items()})
    marker = output / "split_manifest.json"
    if output.exists() and any(output.iterdir()):
        if not marker.is_file() or json.loads(marker.read_text()) != manifest:
            raise ValueError("Existing split data differs; choose a new output directory")
        for name, raw in files.items():
            if not (output / name).is_file() or (output / name).read_text() != raw:
                raise ValueError(f"Split file changed: {output / name}")
        return manifest
    output.mkdir(parents=True, exist_ok=True)
    for name, raw in files.items():
        (output / name).write_text(raw)
    marker.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[2] / "datasets/legacy_sh")
    args = parser.parse_args()
    def load_source(name):
        staged = os.environ.get("CRU_TOFU_DATA_ROOT")
        if staged:
            raw = (Path(staged) / (name + ".json")).read_text()
            try:
                rows = json.loads(raw)
            except json.JSONDecodeError:
                rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
            if not isinstance(rows, list):
                raise ValueError(f"Expected JSON array or JSONL for {name}")
            return rows
        from datasets import load_dataset
        return load_dataset("locuslab/TOFU", name, split="train")
    manifest = prepare(args.output_dir, load_source)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
