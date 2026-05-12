# Qwen3-8B ToMBench RLVR

Post-train Qwen/Qwen3-8B with GRPO on ROLL to maximize ToMBench accuracy.

**Design spec:** `docs/superpowers/specs/2026-05-11-qwen3-8b-tombench-rlvr-design.md`

## Quick start

```bash
cp configs/deploy.env.example configs/deploy.env
# edit configs/deploy.env with your TRAIN host info

export DEEPSEEK_API_KEY=...
export DASHSCOPE_API_KEY=...

make env-check
make build-data
make baseline
make pipeline-stage1
make pipeline-stage2
```

## Documents
- `docs/runbook.md` — operational runbook for each stage
- `docs/data-card.md` — training data sources, licenses, dedupe audit
- `docs/eval-protocol.md` — three evaluation protocols defined precisely
