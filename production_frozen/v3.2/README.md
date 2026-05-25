# Production Frozen v3.2 — Stage 16 ckpt-270 快照

> **状态**: Stage 16 (v3.3 训练方案) 任务针对性数据扩充成功，**ckpt-270** 在 Hi-ToM 大幅提升，整体 800-q 验证集峰值 = **0.6825**
> **创建日期**: 2026-05-25
> **基线对比**: v3.1 (Stage 14b ckpt-199, 2026-05-22) → **v3.2 (Stage 16 ckpt-270, 2026-05-25)**
> **核心提升**: Hi-ToM 全 3 协议大幅突破 (+3.33~+4.83pp)，达成"训练数据结构性补齐"的设计目标

## 头条结果（全 4 benchmark × 3 protocol）

### v3.2 vs v3.1 (12 cells)

| Benchmark | Protocol | v3.1 | **v3.2** | Δ |
|---|---|---|---|---|
| ToMBench (n=5718) | direct | 0.7721 | 0.7692 | -0.29pp |
| ToMBench | cot | 0.7754 | 0.7704 | -0.50pp |
| ToMBench | del_tom | 0.7875 | 0.7831 | -0.44pp |
| **Hi-ToM** (n=600) | direct | 0.5417 | **0.5783** | **+3.66pp** ⭐ |
| **Hi-ToM** | cot | 0.6667 | **0.7150** | **+4.83pp** ⭐⭐ |
| **Hi-ToM** | del_tom | 0.7217 | **0.7550** | **+3.33pp** ⭐ |
| SocialIQA (n=1954) | direct | 0.7876 | 0.7881 | +0.05pp |
| SocialIQA | cot | 0.7861 | 0.7810 | -0.51pp |
| SocialIQA | del_tom | 0.7892 | 0.7835 | -0.57pp |
| EmoBench (n=1200) | direct | 0.6483 | 0.6492 | +0.09pp |
| EmoBench | cot | 0.6575 | 0.6517 | -0.58pp |
| EmoBench | del_tom | 0.6775 | 0.6600 | -1.75pp ⚠️ |

### v3.2 vs DeepSeek-v4-pro (gap 距离)

| Benchmark | Protocol | v3.2 | DeepSeek | Gap |
|---|---|---|---|---|
| ToMBench | direct | 0.7692 | 0.8080 | -3.88pp |
| ToMBench | cot | 0.7704 | 0.7140 ‡ | **+5.64pp ✅** (DS 是 subset500 + truncation bug) |
| ToMBench | del_tom | 0.7831 | 0.8069 | -2.38pp |
| Hi-ToM | direct | 0.5783 | 0.8033 | -22.50pp |
| **Hi-ToM** | **cot** | **0.7150** | **0.7475** | **-3.25pp** (从 -8.08pp 缩 60%) |
| Hi-ToM | del_tom | 0.7550 | — (hang) | **N/A ✅** |
| SocialIQA | direct | 0.7881 | 0.8101 | -2.20pp |
| SocialIQA | cot | 0.7810 | 0.8106 | -2.96pp |
| SocialIQA | del_tom | 0.7835 | 0.8188 | -3.53pp |
| EmoBench | direct | 0.6492 | 0.7625 | -11.33pp |
| EmoBench | cot | 0.6517 | 0.7550 | -10.33pp |
| EmoBench | del_tom | 0.6600 | 0.7733 | -11.33pp |

‡ DeepSeek ToMBench cot 仅评测 subset500，且 v1 reasoning_content 被 max_tokens=2048 截断破坏。

### 训练曲线 (800-q 混合 val)

| step | val_all | Δ累计 | Δ增量 | 备注 |
|---|---|---|---|---|
| 0 (init = v3.1 ckpt-199) | 0.6088 | — | — | baseline |
| 30 | 0.6200 | +1.13 | +1.13 | warmup |
| 60 | 0.6212 | +1.25 | +0.12 | LR 完成 warmup |
| 90 | 0.6612 | +5.25 | **+4.00 ⭐** | Hi-ToM 大幅突破 |
| 120 | 0.6637 | +5.50 | +0.25 | |
| 150 | 0.6712 | +6.25 | +0.75 | first peak |
| 180 | 0.6575 | +4.88 | -1.37 | dip (variance) |
| 210 | 0.6687 | +6.00 | +1.12 | bounce back |
| 240 | 0.6737 | +6.50 | +0.50 | second peak |
| **270** | **0.6825** | **+7.38** | **+0.88 🥇** | **NEW PEAK = 选择 ckpt** |
| 299 | (final) | — | — | 训练结束未跑 val |

**ckpt 选择**: ckpt-270 在 800-q 混合 val 上达到峰值 0.6825（+7.38pp vs v3.1 init），是 production_frozen v3.2 的入选 ckpt。后续 ckpt-299 比 270 多 29 步但 LR 已降到 1.05e-7（极小），更新影响微弱，未必有提升。

## 核心方法论

### 关键洞察

v3.1 vs DeepSeek 残留 gap 来自**训练数据的结构性缺口**，不是数量问题：

1. **Hi-ToM 类型: 0 条** — 训练数据全是 4 选项 MCQ，eval 是 15 选项
2. **EmoBench EU_emotion (6-opt 情绪命名): 0 条** — 训练数据全是 4 选项
3. **ROLL reward worker bug**: `_BOXED = re.compile(r"\\boxed\{([A-D])\}")` 仅匹配 A-D，导致即使加 Hi-ToM/EU_emotion 数据，正确答案 \boxed{K} 或 \boxed{F} 也被 reward 判 r_fmt=0 (silently zero)

### 三大改进

1. **Reward worker 扩展 A-Z + task-aware l_max** (`tom_mcq_reward_worker_FROZEN.py`):
   - `_BOXED` 正则: A-D → A-Z（支持 6/15-opt MCQ）
   - 新增 `l_max_long` (默认 = `l_max`)，Hi-ToM 长 CoT 不被惩罚（v3.2: l_max_long=512）
   - source/task 字段检测: source=hitom* 或 task=order_* 自动走 l_max_long
   - 新增 18 个单元测试，全过

2. **数据扩充 (~17.1k → 14.4k after dedup)**:
   - 1200 Hi-ToM 训练数据 (5 orders × 240) 来自 bigai-ai/ToM-RL hi_tom_3000.csv
   - 1500 DeepSeek-v4-pro 合成: EU_emotion 500 (6-opt) + EU_cause 300 + EA 300 + SocialIQA 400
   - MinHash 4-gram Jaccard ≤0.6 vs 4 个 eval 集 → **0 leakage**

3. **Stage 16 配置**:
   - init from v3.1 ckpt-199, LR 5e-7 (was 1e-6)
   - response_length 256→512, max_model_len 2048→2304
   - max_steps 200→300, save_steps 50→30
   - difficulty_low 0.15→0.05 (Hi-ToM 高阶题低 pass rate 也入梯度)

### v3.2 训练数据组成 (14439 records)

| Source | Count | 说明 |
|---|---|---|
| synth (Phase 1 GPT-4) | 2795 | v3.0 backbone |
| exploretom_v2 | 1805 | v3.0 |
| **hitom_synth** | **1200** | **v3.2 新加** |
| synth_gpt55_phase_d_hot | 1180 | v3.0 |
| synth_gpt55_phase_c | 1171 | v3.0 |
| synth_gpt55 | 1139 | v3.0 |
| simpletom | 1000 | v3.0 |
| synth_zh | 971 | v3.0 |
| synth_phase1 | 878 | v3.0 |
| synth_gpt55_phase_b_zh | 800 | v3.0 |
| **synth_emobench_eu_emotion** | **500** | **v3.2 新加** |
| **synth_socialiqa** | **400** | **v3.2 新加** |
| **synth_emobench_eu_cause** | **300** | **v3.2 新加** |
| **synth_emobench_ea** | **300** | **v3.2 新加** |

## 文件清单

```
production_frozen/v3.2/
├── README.md                    本文档
├── SHA256SUMS.txt               所有文件 + 远端 HF model 的 SHA-256 校验
├── verify.sh                    本地复现验证脚本
├── configs/
│   ├── rlvr_config_14b_stage14b_FROZEN.yaml   v3.1 (Stage 14b)
│   └── rlvr_config_14b_stage16_FROZEN.yaml    v3.2 (Stage 16) ⭐
├── scripts/
│   ├── build_hitom_train.py                   Hi-ToM 数据生成
│   ├── synth_emobench_socialiqa.py            DeepSeek 合成 EmoBench/SocialIQA
│   ├── build_stage16_data.py                  v3.2 训练数据合并
│   ├── run_tombench.py                        ToMBench eval
│   ├── run_generic_mcq.py                     Hi-ToM/SocialIQA/EmoBench eval
│   ├── extractors_generic.py                  通用 N-opt extractor
│   └── tom_mcq_reward_worker_FROZEN.py        ⭐ A-Z + task-aware l_max
├── docs/
│   ├── v3.2_training_plan_2026-05-25.md       完整训练方案
│   └── full_eval_report_2026-05-24.md         7-model × 4-benchmark 全量报告
└── eval/
    ├── stage16_ckpt270_tombench.json          全 5718 × 3 protocols
    ├── stage16_ckpt270_hitom.json             v1 (direct only, 4096 cap)
    ├── stage16_ckpt270_hitom_v2.json          v2 (cot+del_tom @ 8192 max_len)
    ├── stage16_ckpt270_socialiqa.json
    ├── stage16_ckpt270_socialiqa_v2.json
    ├── stage16_ckpt270_emobench.json
    └── stage16_ckpt270_emobench_v2.json
```

## TRAIN host 模型路径

```
/data_nvme/grj-projects/tom-output/qwen3-14B-tombench-rlvr-stage16-1x8-hf-ckpt270
```

包含 6 shards × ~5GB = 28.5GB safetensors + tokenizer + config.json。

复现方式:
```bash
ssh h800@172.16.120.181 'cd /data_nvme/grj-projects/tom-output/qwen3-14B-tombench-rlvr-stage16-1x8-hf-ckpt270 && \
   sha256sum *.safetensors *.json *.txt' \
| diff - <(grep stage16-1x8-hf-ckpt270 production_frozen/v3.2/SHA256SUMS.txt | sed "s|qwen3-14B-tombench-rlvr-stage16-1x8-hf-ckpt270/||")
```

## 复现头条数字 (vLLM 数值抖动 ±0.001)

```bash
# 1. Serve via vLLM 0.8.4 from train image
ssh h800@172.16.120.181 'docker run -d --name vllm-serve --gpus all --ipc=host --shm-size 16g \
  --entrypoint /bin/bash \
  -v /data_nvme/grj-projects/tom-output:/mnt/output \
  -p 8000:8000 \
  qwen3-tom-train:latest \
  -c "python -m vllm.entrypoints.openai.api_server \
    --model /mnt/output/qwen3-14B-tombench-rlvr-stage16-1x8-hf-ckpt270 \
    --host 0.0.0.0 --port 8000 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 8192 \
    --served-model-name qwen3-14b-stage16-ckpt270"'

# 2. Run eval (DEV side)
docker compose -f docker/dev/docker-compose.yml run --rm \
  -e OPENAI_API_KEY=dummy \
  -v $PWD:/workspace -w /workspace dev \
  python production_frozen/v3.2/scripts/run_tombench.py \
    --backend openai --base-url http://172.16.120.181:8000/v1 \
    --model qwen3-14b-stage16-ckpt270 \
    --data data/tom/tombench_eval.jsonl \
    --protocols direct,cot,del_tom \
    --concurrency 32 --del-tom-n 8 \
    --output /tmp/repro_tombench.json
```

预期:
- ToMBench direct ≈ 0.7692 ✓
- ToMBench cot ≈ 0.7704 ✓
- ToMBench del_tom ≈ 0.7831 ✓
- Hi-ToM cot ≈ 0.7150 ✓
- Hi-ToM del_tom ≈ 0.7550 ✓

## 后续 (v3.3 in progress)

v3.3 = Stage 17 = continue from v3.2 ckpt-270 + 4500 条针对性数据 (+ Hi-ToM direct compressed-CoT 范式 + EU_emotion 1500 扩充 + Belief distillation 600)。目标在所有 12 cell 上 ≥ DeepSeek-v4-pro。详见 `docs/v3.3_training_plan_2026-05-25.md`（仍在仓库 docs/）。
