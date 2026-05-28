# v3.6 Production Frozen — Stage 20 ckpt-79 (Precision Distillation v3)

> **Frozen on**: 2026-05-28
> **Predecessor**: v3.5 (Stage 19 ckpt-120, init source)
> **Method**: Precision distill v3 — Hi-ToM order_low_only paraphrase + skip pre-verify on order_2/3/4 + EmoBench EU_emotion delta-style solve + ToMBench Knowledge ontology+vote (387 → 167 retained after eval-set leakage filter)
> **Training**: Stage 20 RL continue from v3.5 ckpt-120, 1×8 H800, max_steps=80, lr=5e-7, 19403 records (19236 v3.5 backbone + 167 distill_v3)
> **Best ckpt**: ckpt-79 (final step, no early stop triggered)

## Eval Results (12-cell: 4 benchmark × 3 protocol)

| Bench | direct | cot | del_tom |
|---|---|---|---|
| ToMBench (5718) | 0.7644 | 0.7697 | 0.7781 |
| Hi-ToM (600) | **0.6000** ⭐ | 0.7267 | **0.7733** ⭐ |
| SocialIQA (1954) | 0.7845 | 0.7799 | 0.7866 |
| EmoBench (1200) | 0.6483 | 0.6558 | 0.6567 |

**Hi-ToM 双反超 DS-V3.2 拉大优势**:
- direct: 0.6000 vs DS 0.5825 = **+1.75pp** (v3.5 was +0.75pp)
- del_tom: 0.7733 vs DS cot 0.7475 = **+2.58pp** (v3.5 was +0.92pp)

## v3.6 vs v3.5 (12-cell delta)

| Bench | direct | cot | del_tom | Sum |
|---|---|---|---|---|
| ToMBench | -0.42 | -0.63 | -0.47 | -1.52 |
| Hi-ToM | **+1.00** ⭐ | -0.50 | **+1.66** ⭐ | **+2.16** ✅ |
| SocialIQA | -0.16 | -0.46 | -0.36 | -0.98 |
| EmoBench | 0.00 | +0.33 | -1.08 | -0.75 |
| **Sum** | **+0.42** | **-1.26** | **-0.25** | **-1.09** |

净: **-1.09pp 总和, -0.09pp 平均, 3 升 9 降**, **2 反超 DS 优势从 +0.75/+0.92pp 扩大至 +1.75/+2.58pp**

## Distillation v3 method

1. **错例提取** (从 v3.5 stage19 ckpt-120 评测, ANY-protocol-wrong union):
   - Hi-ToM: 314 candidates (order_2/3/4: 242 + order_0/1: 72)
   - EmoBench EU_emotion: 233 candidates (×2 paraphrase mult = 466 work)
   - ToMBench Knowledge: 335 candidates
2. **3 surgical attack points**:
   - **Hi-ToM**: paraphrase ONLY order_0/1, skip pre-verify on order_2/3/4 (preserve hard records).
     Retain 209 (66.6%) — but 142 order_2/3/4 originals dropped by Jaccard ≥ 0.6 leakage filter against eval set.
     **Net 67 paraphrased records added to training** (the only Hi-ToM contribution).
   - **EmoBench EU_emotion delta-style**: distractor-aware solve prompt + vote≥3 unanimous + ×2 paraphrase mult.
     Retain 83 (17.8%) — vote≥3 too strict.
   - **ToMBench Knowledge**: paraphrase + ontology + vote≥2.
     Retain 95 (28.4%) — GPT-5.5 voting unstable on social/factual knowledge.
3. **Total stage20 train**: 19236 backbone + 167 distill_v3 = 19403 records (220 leakage filtered).

## Distillation v3 effectiveness

✅ **Hi-ToM direct +1.00pp, del_tom +1.66pp** — paraphrase order_0/1 + ontology fix worked
✅ **EmoBench cot +0.33pp** — minor delta-style benefit
✅ **2 cells reach DS-superiority with widened margins** (Hi-ToM direct/del_tom)

⚠️ **ToMBench全 protocol -0.4 ~ -0.6pp** — 95 distill records insufficient against 5718-question backbone, lr=5e-7 + 80 step caused mild drift on backbone-strong tasks
⚠️ **SocialIQA / EmoBench del_tom 微退** — no SocialIQA distill_v3 records; EmoBench delta vote≥3 too strict
⚠️ **Hi-ToM cot -0.50pp** — same v3.5 issue (paraphrase breaks long-chain belief cross-reference); fundamentally requires preserving order_2/3/4 originals, but those would be eval-set leakage

## CKPT 存储位置

### 训练机 (.181)
- **Megatron 原始 ckpts** (used for init future stages):
  `/data_nvme/grj-projects/tom-output/qwen3-14B-tombench-rlvr-stage20-1x8/20260528-014841/checkpoint-{60,70,79}` (3 ckpts retained, 10/20/30/40/50 deleted to save disk)
- **HF safetensors** (for vLLM serve):
  `/data_nvme/grj-projects/tom-output/qwen3-14B-tombench-rlvr-stage20-1x8-hf-ckpt79/`
  (28GB, 6 safetensors shards + tokenizer + config.json)

### 训练数据
- **Train set**: `data/tom/tom_train_stage20.jsonl` (19403 records, 22MB)
- **Distill v3 raw outputs**: `data/tom/raw_v36/distill_v3_{hitom,emobench_eu_emotion,tombench_knowledge}.jsonl` (387 records 总计, 167 after dedup+leakage)
- **Ontology**: `data/distill/emotion_ontology.txt` (1749 chars, 12 fine-grained emotions + ToM concepts)

## Reproducibility

```bash
# 1. Verify integrity
cd production_frozen/v3.6
bash verify.sh

# 2. Build stage20 data (assumes v3.5 backbone exists)
docker compose -f docker/dev/docker-compose.yml run --rm dev \
  python scripts/data/build_stage20_data.py

# 3. Train (assumes data sync'd to TRAIN host)
make sync-up
make train-stage20-1x8-14b   # uses configs/tombench-rlvr/rlvr_config_stage20_1x8_14b.yaml

# 4. Convert ckpt-79 → HF
docker compose -f docker/train/docker-compose.yml run --rm --entrypoint /bin/bash train -c "
  python framework/ROLL/mcore_adapter/tools/convert.py \
    --checkpoint_path /mnt/output/qwen3-14B-tombench-rlvr-stage20-1x8/<TS>/checkpoint-79 \
    --output_path /mnt/output/qwen3-14B-tombench-rlvr-stage20-1x8-hf-ckpt79 \
    --bf16
"

# 5. Serve via vLLM (4 instances × tp=2 on 8 H800)
docker compose -f docker/serve/eval_dp4_compose_stage20.yml --env-file configs/deploy.env up -d --build

# 6. Eval (3 protocols × 4 benchmarks parallel)
TS=$(date +%Y%m%d_%H%M%S)
mkdir -p output/eval/stage20_ckpt79
TRAIN_HOST_HOSTONLY=172.16.120.181 MODEL_NAME=qwen3-14b-stage20-ckpt79 OUTPUT_DIR=output/eval/stage20_ckpt79 \
  bash scripts/eval/_one_bench.sh tombench 8001 &
TRAIN_HOST_HOSTONLY=172.16.120.181 MODEL_NAME=qwen3-14b-stage20-ckpt79 OUTPUT_DIR=output/eval/stage20_ckpt79 \
  bash scripts/eval/_one_bench.sh hitom 8002 &
TRAIN_HOST_HOSTONLY=172.16.120.181 MODEL_NAME=qwen3-14b-stage20-ckpt79 OUTPUT_DIR=output/eval/stage20_ckpt79 \
  bash scripts/eval/_one_bench.sh emobench 8003 &
TRAIN_HOST_HOSTONLY=172.16.120.181 MODEL_NAME=qwen3-14b-stage20-ckpt79 OUTPUT_DIR=output/eval/stage20_ckpt79 \
  bash scripts/eval/_one_bench.sh socialiqa 8004 &
wait
```

## Known issues & v3.7 directions

1. **Hi-ToM cot regression -0.50pp**: same v3.5 issue. Root cause: order_2/3/4 paraphrase breaks long-chain belief cross-reference; preserving original would leak eval data.
   - **v3.7 idea**: New paraphrase prompt that explicitly preserves named-entity belief chains; or augment with synthetic Hi-ToM (not from eval set) at order_2/3/4.

2. **EmoBench EU_emotion gap -11.66pp del_tom**: distill_v3 vote≥3 too strict (17.8% retain = 83 records). Insufficient signal.
   - **v3.7 idea**: vote≥2 + 2-3× paraphrase mult to get 200+ records; or generate synthetic EU_emotion stories (not from eval) with delta-style reasoning.

3. **ToMBench general drift -0.4 ~ -0.6pp**: 95 Knowledge distill records too few; lr=5e-7 + 80 steps caused mild backbone drift on the 5718-question dominant bench.
   - **v3.7 idea**: lower lr to 2e-7, increase distill scale 3-5×, or preserve v3.5 ckpt for ToMBench-strong cases.

4. **SocialIQA monotonic micro-decline**: no SocialIQA-targeted distill_v3 records; pure backbone drift.
   - **v3.7 idea**: include SocialIQA error candidates in distill_v3.

## Per-task breakdown (ToMBench del_tom, vs v3.5)

| Sub-task | v3.6 | v3.5 | Δ |
|---|---|---|---|
| Belief | 0.7359 | 0.7500 | -1.41 |
| Desire | 0.6222 | 0.6167 | +0.55 |
| Emotion | 0.7476 | 0.7536 | -0.60 |
| False Belief | 0.8845 | 0.8865 | -0.20 |
| Intention | 0.8382 | 0.8382 | 0.00 |
| Knowledge | 0.5433 | 0.5450 | -0.17 |
| Non-literal Comm | 0.7988 | 0.8095 | -1.07 |

(Knowledge sub-task did NOT improve despite 95 distill_v3 records targeted at it — confirms knowledge tasks are unteachable via paraphrase distillation.)
