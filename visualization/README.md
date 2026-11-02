# Replace and Merge plots

Run commands from the repository root after the corresponding evaluations finish.
Both commands read saved results without loading models or running inference.

## Merge curves

Stage 5 automatically runs `curves/plot_merge.py` and writes `merge_curves.png`,
`merge_curves.pdf`, and `merge_curves.csv` into the Merge sweep directory.
It shows MU and signed FQ in separate panels for every ranked prefix, k=1..L.
To regenerate these files:

```bash
TASK=tofu_Llama-3.2-1B-Instruct_forget05_NPO
python visualization/curves/plot_merge.py \
  --results-dir "outputs/legacy_sh/merge_sweep/$TASK"
```

## Replace/Merge comparison

`curves/plot_replace_merge.py` reads stage 3's `layer-*/TOFU_SUMMARY.json`
and stage 5's `ranking.json` and `k-*/eval/` results directly:

```bash
TASK=tofu_Llama-3.2-1B-Instruct_forget05_NPO
python visualization/curves/plot_replace_merge.py \
  --replace-dir "outputs/legacy_sh/replace/$TASK" \
  --merge-dir "outputs/legacy_sh/merge_sweep/$TASK"
```

The output is `replace_merge.png`, `replace_merge.pdf`, and `replace_merge.csv`
under the Merge directory's `figures/` subdirectory. Use `--output-dir` to change it.
The left panel shows all physical layers, 0..L-1; the right shows every cumulative
prefix, k=1..L, in the fixed ranking. Each panel has separate MU and FQ axes.
The CSV preserves the plotted metric values and distinguishes layer indices from k.

Optional references must come from evaluations using the same model, data splits,
and retain reference as the sweep:

- `--baseline-summary PATH`: Full-FT `TOFU_SUMMARY.json`, adding an actual k=0 point.
- `--original-summary PATH`: Unlearn `TOFU_SUMMARY.json`, adding dashed reference lines only for older results where both panels use the same evaluation protocol. This option is rejected for the validation/test comparison.

Without these inputs, no baseline point or reference line is added. See
[reference evaluation commands](../scripts/README.md#original-and-cumulative-merge-evaluation).

Both commands require a complete Merge sweep and verify its fixed-ranking provenance.
The comparison also verifies each Replace summary against the hash saved during ranking.
Missing, changed, or non-finite results cause an error. Neither command selects k,
changes the ranking, or modifies evaluation results.

Under the current protocol, the left panel shows Replace on validation data and the right panel shows Merge on test data. Panel titles distinguish the two; their scores are not a direct same-split comparison. Older results can still be plotted without being labeled as the current protocol. The optional Full FT baseline must use the Merge test splits and is added only to the right panel.
