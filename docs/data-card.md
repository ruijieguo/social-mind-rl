# Data Card

Generated from `merge_and_dedupe.py`.

## Sources after dedupe
- **exploretom**: 2000
- **simpletom**: 1000
- **synth**: 2911

## Leakage audit
- Records dropped (>0.6 Jaccard with any ToMBench eval question): 0
- Records dropped (internal near-dup >0.7): 6

## Per-source max-Jaccard vs ToMBench eval
| Source | n | mean | p95 | max |
|---|---|---|---|---|
| hi_tom | 0 | 0.000 | 0.000 | 0.000 |
| exploretom | 2000 | 0.000 | 0.000 | 0.000 |
| simpletom | 1000 | 0.000 | 0.000 | 0.000 |
| socialiqa | 0 | 0.000 | 0.000 | 0.000 |
| synth | 2917 | 0.000 | 0.000 | 0.000 |