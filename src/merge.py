"""Build and save a CRU hybrid checkpoint from selected transformer layers."""
import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer

from rank import load_ranking
from layer_ops import copy_layers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--unlearned-model", type=Path, required=True)
    parser.add_argument("--ranking", type=Path, required=True)
    parser.add_argument("--k", type=int, required=True,
                        help="Number of layers to copy from the fixed ranking prefix")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    ranking = load_ranking(args.ranking, args.base_model)
    if not 1 <= args.k <= ranking["num_layers"]:
        parser.error(f"--k must be between 1 and {ranking['num_layers']}")
    layers = set(ranking["ranking"][:args.k])
    base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype="auto")
    unlearned = AutoModelForCausalLM.from_pretrained(args.unlearned_model, torch_dtype="auto")
    copied = copy_layers(base, unlearned, layers)
    args.output.mkdir(parents=True, exist_ok=True)
    base.save_pretrained(args.output)
    AutoTokenizer.from_pretrained(args.base_model).save_pretrained(args.output)
    provenance = dict(protocol="fullft_base_v2", base_model=str(args.base_model.resolve()),
                      layer_donor=str(args.unlearned_model.resolve()), k=args.k,
                      selected_layers=ranking["ranking"][:args.k],
                      ranking=str(args.ranking.resolve()),
                      ranking_sha256=hashlib.sha256(args.ranking.read_bytes()).hexdigest(),
                      base_config_sha256=ranking["base_config_sha256"])
    (args.output / "CRU_MERGE.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Saved k={args.k} CRU checkpoint to {args.output}; copied {copied} tensors "
          f"from ranked layers {ranking['ranking'][:args.k]}")


if __name__ == "__main__":
    main()
