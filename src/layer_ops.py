"""Copy complete transformer layers while preserving all other parameters."""
import re

import torch

_LAYER = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


def copy_layers(model, donor, layers):
    """Validate selected tensors before modifying the destination model."""
    layers = set(layers)
    source_params = dict(donor.named_parameters())
    transfers = []
    found = set()
    for name, target in model.named_parameters():
        match = _LAYER.search(name)
        if match is None or int(match.group(1)) not in layers:
            continue
        source = source_params.get(name)
        if source is None or source.shape != target.shape:
            raise ValueError(f"Missing or incompatible donor parameter: {name}")
        transfers.append((target, source))
        found.add(int(match.group(1)))
    if found != layers:
        raise ValueError(f"Missing transformer layers: {sorted(layers - found)}")
    with torch.no_grad():
        for target, source in transfers:
            target.copy_(source.to(device=target.device, dtype=target.dtype))
    return len(transfers)
