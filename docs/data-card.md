# Data Card

Generated from `merge_and_dedupe.py`.

## Sources after dedupe
- **exploretom**: 2000
- **simpletom**: 1000
- **synth**: 2911
- **synth_phase1**: 991

## Leakage audit
- Records dropped (>0.6 Jaccard with any ToMBench eval question): 0
- Records dropped (internal near-dup >0.7): 802

## Per-source max-Jaccard vs ToMBench eval
| Source | n | mean | p95 | max |
|---|---|---|---|---|
| hi_tom | 0 | 0.000 | 0.000 | 0.000 |
| exploretom | 2000 | 0.000 | 0.000 | 0.000 |
| simpletom | 1000 | 0.000 | 0.000 | 0.000 |
| socialiqa | 0 | 0.000 | 0.000 | 0.000 |
| synth | 2917 | 0.000 | 0.000 | 0.000 |
| synth_phase1_faux_pas | 799 | 0.000 | 0.000 | 0.000 |
| synth_phase1_hinting | 243 | 0.000 | 0.000 | 0.000 |
| synth_phase1_scalar | 449 | 0.000 | 0.000 | 0.000 |
| synth_phase1_so_belief | 296 | 0.000 | 0.000 | 0.000 |