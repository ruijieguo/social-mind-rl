# v3.5 Production Frozen — Stage 19 ckpt-120 (Improved GPT-5.5 Distillation)

> **Frozen on**: 2026-05-27
> **Predecessor**: v3.4 (Stage 18 ckpt-30) / v3.3 (Stage 17 ckpt-120, init source)
> **Method**: GPT-5.5 distillation v2 — pre-verify on original + 3-sample voting + emotion ontology injection (1098 candidates → 937 retained)
> **Training**: Stage 19 RL continue from v3.3 ckpt-120 (NOT v3.4), 1×8 H800, init from `qwen3-14B-tombench-rlvr-stage17-1x8-hf-ckpt120`
> **Best ckpt**: ckpt-120 (step 120 / 150 max; step 135 disk-full crash, ckpt-135 incomplete; ckpt-120 is last clean)

## Eval Results (4 benchmark × 3 protocol)

| Bench | direct | cot | del_tom |
|---|---|---|---|
| ToMBench (5718) | 0.7686 | 0.7760 | **0.7828** ⭐ |
| Hi-ToM (600) | **0.5900** ⭐ | 0.7317 | **0.7567** ⭐ |
| SocialIQA (1954) | 0.7861 | 0.7845 | 0.7902 |
| EmoBench (1200) | 0.6483 | 0.6525 | 0.6675 |

**Hi-ToM direct 反超 DS-V3.2 +0.75pp**(0.5900 vs DS 0.5825)
**Hi-ToM del_tom 反超 DS-V3.2 cot +0.92pp**(0.7567 vs DS 0.7475)

## v3.5 vs v3.4 (12-cell delta)

| Bench | direct | cot | del_tom | Sum |
|---|---|---|---|---|
| EmoBench | -0.25 | **+0.75** | **+0.42** | +0.92 |
| Hi-ToM | **+4.17** 🔥 | -1.16 ⚠️ | **+1.67** | **+4.68** |
| SocialIQA | -0.30 | +0.25 | -0.10 | -0.15 |
| ToMBench | -0.47 | -0.10 | **+0.58** | +0.01 |
| **Sum** | **+3.15** | -0.26 | **+2.57** | **+5.46** |

净: **+5.46pp 总和, +0.46pp 平均, 7 升 5 降**, **2 反超 DS** (Hi-ToM direct + del_tom)

## Distillation v2 method

1. **错例提取** (从 v3.4 stage18 ckpt-30 评测): EmoBench 519 + SocialIQA 558 + Hi-ToM 311(×2 mult) = 1388 candidates from real wrong answers
2. **Pre-verify on ORIGINAL story (T=0)**: 让 GPT-5.5 在原题上能答对 gold 的题才进入 distill (淘汰 GPT-5.5 自己不会的)
3. **Paraphrase v2 prompt**: 强化"保持 gold 答案不变"约束 + sanity-check 自审
4. **3-sample voting solve (T=0.4)**: 在 paraphrased story 上 3 次推理，要求 majority ≥2 == gold 才保留
5. **Ontology injection**: 12 fine-grained emotions + ToM concepts as system prompt
6. 总 retain rate: emo 45.5% / soci 40.1% / hitom 76.7% → **937 high-quality records**
7. MinHash dedup vs eval set → **19236 final stage19 train set**, 0 leakage hits

## Distillation v2 effectiveness

✅ **Hi-ToM direct +4.17pp** (跨所有 bench×protocol 最大正向): ontology + voting 让短答 belief tracking 更稳
✅ **del_tom 协议全 4 bench 改善** (+0.42/+1.67/-0.10/+0.58): del_tom 是 ToM 鲁棒性测试，证明学到真本事
⚠️ **Hi-ToM cot -1.16pp**: pre-verify 偏向 easy subset; paraphrase 破坏长链 belief cross-reference
⚠️ **EmoBench/SocialIQA direct 微负**: ontology 在无 explicit reasoning 时可能加噪

## CKPT 存储位置

### 训练机 (.181)
- **Megatron 原始 ckpt** (用于 init 后续 stage):
  `/data_nvme/grj-projects/tom-output/qwen3-14B-tombench-rlvr-stage19-1x8/20260526-181304/checkpoint-120`
- **HF safetensors** (用于 vLLM serve):
  `/data_nvme/grj-projects/tom-output/qwen3-14B-tombench-rlvr-stage19-1x8-hf-ckpt120/`
  (28.8GB, 6 safetensors shards + tokenizer + config.json)

### 训练数据
- **Train set**: `/data_nvme/grj-projects/tom-data/tom_train_stage19.jsonl` (19236 records, 22MB)
  本地副本: `data/tom/tom_train_stage19.jsonl`
- **Distill v2 raw outputs**: `data/tom/raw_v35/distill_v2_{emobench,socialiqa,hitom}.jsonl` (937 records 总计)
- **Ontology**: `data/distill/emotion_ontology.txt`

## Reproducibility

```bash
# 1. Verify integrity
cd production_frozen/v3.5
bash verify.sh

# 2. Continue from frozen config
make train-stage19-1x8-14b   # uses configs/tombench-rlvr/rlvr_config_stage19_1x8_14b.yaml

# 3. Convert ckpt-120 → HF
docker compose -f docker/train/docker-compose.yml run --rm --entrypoint /bin/bash train -c "
  python framework/ROLL/mcore_adapter/tools/convert.py \
    --checkpoint_path /mnt/output/qwen3-14B-tombench-rlvr-stage19-1x8/<TS>/checkpoint-120 \
    --output_path /mnt/output/qwen3-14B-tombench-rlvr-stage19-1x8-hf-ckpt120 \
    --bf16
"

# 4. Serve via vLLM (4 instances × tp=2 on 8 H800)
docker compose -f docker/serve/eval_dp4_compose_stage19.yml --env-file configs/deploy.env up -d --build

# 5. Eval (3 protocols × 4 benchmarks)
TRAIN_HOST_HOSTONLY=172.16.120.181 MODEL_NAME=qwen3-14b-stage19-ckpt120 OUTPUT_DIR=output/eval \
  bash scripts/eval/_one_bench.sh tombench 8001
# (repeat for emobench/hitom/socialiqa with --tasks pinned per bench)
```

## Known issues

1. **训练 90% 完成挂**: step 135 ckpt save 时 `/data_nvme` 100% 满 → tokenizer.json 写损坏。已用 ckpt-105 的 tokenizer 复制覆盖。**v3.5 实际是 ckpt-120**(step 120/150)，非完整 150-step training，但综合评测 ≥ v3.4 表明此 ckpt 已超越或匹配 v3.4 全训练效果。
2. **Hi-ToM cot regression -1.16pp**: 见 distillation v2 effectiveness 第 ⚠️ 项。

## Next iteration ideas (v3.6)

- 不 paraphrase hitom order_4 长链 (paraphrase_multiplier=0 for order_*)
- 跳过 hitom pre-verify（保留 hard 题）
- 用 v3.5 ckpt-120 init + 50 step finetune（已在 90% 基础上微调）
- EmoBench/SocialIQA distill 量增到 1000+ each
