"""Validate sweep identity and commit completed candidates without losing partial logs."""
import hashlib
import json
from pathlib import Path
import uuid

from omegaconf import OmegaConf


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_sweep(output, cfg, checkpoints, ranking_bytes):
    # Detect replaced local checkpoints without reading many GB of weight data.
    files = {}
    for checkpoint in checkpoints:
        checkpoint = Path(checkpoint).resolve()
        for path in sorted(checkpoint.rglob('*')):
            if path.is_file():
                stat = path.stat()
                files[str(path)] = [stat.st_size, stat.st_mtime_ns]
    reference = Path(str(cfg.retain_logs_path))
    identity = dict(
        config={
            key: OmegaConf.to_container(cfg[key], resolve=True)
            if OmegaConf.is_config(cfg[key]) else cfg[key]
            for key in cfg if key != 'paths'
        },
        checkpoint_files=files,
        ranking_sha256=hashlib.sha256(ranking_bytes).hexdigest(),
        reference_sha256=file_hash(reference) if reference.is_file() else None,
    )
    protocol_file = output / 'evaluation_protocol.json'
    if protocol_file.is_file():
        identity['evaluation_protocol'] = json.loads(protocol_file.read_text())
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / 'sweep_manifest.json'
    if manifest.exists():
        if json.loads(manifest.read_text()) != identity:
            raise ValueError('Sweep inputs changed; choose a new output directory')
    else:
        if any(output.glob('k-*')):
            raise ValueError('Existing sweep has no resume manifest; use a new directory')
        atomic_json(manifest, identity)


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def completed(folder):
    marker = folder / 'COMPLETE.json'
    if not marker.is_file():
        return False
    record = json.loads(marker.read_text())
    for name, expected in record.items():
        path = folder / name
        if not path.is_file() or file_hash(path) != expected:
            raise ValueError(f'Completed result changed or is missing: {path}')
    if not record:
        raise ValueError(f'Empty completion record: {marker}')
    return True


def archive_partial(folder):
    if folder.exists():
        archive = folder.parent / 'incomplete'
        archive.mkdir(exist_ok=True)
        folder.rename(archive / f'{folder.name}-{uuid.uuid4().hex[:12]}')


def mark_complete(folder):
    records = {}
    for path in sorted(folder.rglob('*')):
        if path.is_file() and path.name != 'COMPLETE.json':
            records[str(path.relative_to(folder))] = file_hash(path)
    atomic_json(folder / 'COMPLETE.json', records)
