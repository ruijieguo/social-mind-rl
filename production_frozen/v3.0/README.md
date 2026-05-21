# Production Frozen v3.0 — Stage 12 (Track E) 快照

> **状态**: Stage 11 v2 系列完成，Stage 12 + del_tom = **0.7823** 创项目记录
> **创建日期**: 2026-05-21
> **基线对比**: v1.0 (Stage 8, 2026-05-20) → v3.0 (Stage 12, 2026-05-21)
> **核心提升**: del_tom 0.7762 → **0.7823** (+0.61pp)，cot 0.7594 → **0.7690** (+0.96pp)

## 头条结果（full 5718 raw）

| 协议 | Stage 8 (v1.0) | **Stage 12 (v3.0)** | Δ |
|---|---|---|---|
| direct | 0.7594 | **0.7660** | +0.66pp |
| cot | 0.7594 | **0.7690** | +0.96pp |
| **del_tom** | 0.7762 | **0.7823** | **+0.61pp** ⭐ |

**与外部基线**：
- GPT-5.5 (zero-shot): 0.8349（仍领先 5.26pp，从原来的 7.55pp 收窄）
- deepseek-v4-pro: 0.8080（仍领先 2.57pp，从原来的 4.86pp 收窄）

## 目录结构

```
production_frozen/v3.0/
├── README.md                    # 本文件
├── SHA256SUMS.txt               # 校验和
├── configs/
│   ├── rlvr_config_14b_stage12_FROZEN.yaml         # Track E 训练配置
│   └── rlvr_config_14b_stage11d_track_d_FROZEN.yaml # Track D 续训配置（参考）
├── data/
│   ├── tom_train_14b_stage12.jsonl                 # 12519 条整合训练集
│   └── raw/
│       ├── exploretom_v2_track_b.jsonl             # 2000 条 (Track B 原料)
│       └── synth_gpt55_phase_d_hot_track_c.jsonl   # 1260 条 (Track C 原料)
├── scripts/
│   ├── convert_exploretom_v2.py                    # ExploreToM v2 → 标准格式
│   ├── synth_gpt55_phase_d_hot.py                  # GPT-5.5 HOT 合成
│   ├── merge_stage11_train.py                      # 合并训练集
│   ├── launch_stage12_train.sh                     # Track E 启动器
│   ├── launch_stage11d_train.sh                    # Track D 启动器
│   ├── run_tombench.py                             # 评测框架
│   └── extractors.py                               # del_tom 实现
├── eval/
│   └── 14b_stage12_full5718.json                   # 三协议 5718 题完整结果
├── logs/
│   ├── train_stage12_14b.log.gz                    # Stage 12 训练日志
│   ├── train_stage11d_track_d_14b.log.gz           # Track D 训练日志
│   └── eval_stage12_14b.log.gz                     # 评测日志
└── docs/
    └── stage11_v2_final_report_zh.md               # 完整中文技术报告
```

## 校验

```bash
cd production_frozen/v3.0
bash verify.sh
```

期望输出：所有文件 `OK`，每条记录数与 headline 匹配。

## 模型权重位置（不在 git 中）

```
TRAIN host: h800@172.16.120.181

Megatron 原始 ckpt:
  /data_nvme/grj-projects/tom-output/qwen3-14B-tombench-rlvr-stage12-1x8/20260520-174926/checkpoint-349/

HF 推理用模型 (28 GB):
  /data_nvme/grj-projects/tom-output/qwen3-14B-tom-hf-stage12/

Track D ckpt (对照组):
  /data_nvme/grj-projects/tom-output/qwen3-14B-tombench-rlvr-stage11d-1x8/20260520-111250/checkpoint-349/
```

如需保存到本地：

```bash
mkdir -p /path/to/archive
rsync -avz --progress \
  -e "ssh -i ~/.ssh/id_ed25519" \
  h800@172.16.120.181:/data_nvme/grj-projects/tom-output/qwen3-14B-tom-hf-stage12/ \
  /path/to/archive/qwen3-14B-tom-hf-stage12/
```

## 与 v1.0 的关系

v3.0 是增量快照，**依赖 v1.0** 的若干文件：

| 资源 | 来源 |
|---|---|
| 评测数据 `tombench_eval.jsonl` (5718) | `production_frozen/data/tombench_eval.jsonl` |
| 评测子集 `subset500.jsonl` | `production_frozen/data/tombench_eval_subset500.jsonl` |
| 评测净化集 `clean_eval.jsonl` (4551) | `production_frozen/data/tombench_eval_clean.jsonl` |
| Stage 8 训练数据基础 9259 条 | `production_frozen/data/tom_train_14b_stage8.jsonl` |
| Stage 8 HF 模型（v3.0 训练 init） | TRAIN host `qwen3-14B-tom-hf-stage8/` |
| Reward worker | `production_frozen/scripts/tom_mcq_reward_worker.py`（v1.0 版本仍在用） |

v3.0 在 v1.0 的基础上新增：
- 2000 条 Track B (ExploreToM v2)
- 1260 条 Track C (GPT-5.5 HOT-targeted)
- 整合训练集 12519 条
- Stage 12 训练配置 + 启动脚本
- Stage 12 ckpt-349 (Megatron + HF)
- Stage 12 评测结果 (5718 三协议)
- Track A `del_tom` 评测协议（在 v1.0 的 `run_tombench.py` 已支持，v3.0 这份是当前最新版）

## 端到端复现

### 训练（1×8 H800，~6 小时）

```bash
# 1. 数据准备
cp production_frozen/v3.0/data/tom_train_14b_stage12.jsonl /mnt/data/tom_train.jsonl

# 2. 配置准备
cp production_frozen/v3.0/configs/rlvr_config_14b_stage12_FROZEN.yaml \
   configs/tombench-rlvr/rlvr_config_stage12_1x8_14b.yaml

# 3. 启动训练
bash production_frozen/v3.0/scripts/launch_stage12_train.sh

# 4. Megatron → HF
docker run --rm --gpus all --ipc host --shm-size 8gb \
  --cap-add SYS_PTRACE --cap-add SYS_ADMIN \
  -v $(pwd):/workspace \
  -v /data_nvme/grj-projects/tom-output:/mnt/output \
  -e PYTHONPATH=/workspace/framework/ROLL/mcore_adapter/src \
  -w /workspace --entrypoint python qwen3-tom-train:latest \
  framework/ROLL/mcore_adapter/tools/convert.py \
  --checkpoint_path /mnt/output/qwen3-14B-tombench-rlvr-stage12-1x8/<timestamp>/checkpoint-349 \
  --output_path /mnt/output/qwen3-14B-tom-hf-stage12 --bf16
```

### 评测（DEV 端）

```bash
# 1. 起 vLLM 服务（TRAIN 端）
ssh h800@172.16.120.181 'docker run --rm -d --name eval-serve-stage12 \
  --gpus device=0 --ipc host --shm-size 16gb -p 8000:8000 \
  -v /data_nvme/grj-projects/tom-output:/mnt/output \
  --entrypoint python qwen3-tom-train:latest \
  -m vllm.entrypoints.openai.api_server \
  --model /mnt/output/qwen3-14B-tom-hf-stage12 \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 1 --gpu-memory-utilization 0.85 \
  --max-model-len 4096 --served-model-name qwen3-14b-tom-stage12'

# 2. 跑评测（DEV 端）
docker compose -f docker/dev/docker-compose.yml run --rm \
  -e OPENAI_API_KEY=dummy dev \
  python production_frozen/v3.0/scripts/run_tombench.py \
    --backend openai \
    --base-url http://172.16.120.181:8000/v1 \
    --model qwen3-14b-tom-stage12 \
    --data production_frozen/data/tombench_eval.jsonl \
    --protocols direct,cot,del_tom \
    --concurrency 32 \
    --output reproduce_stage12.json
```

期望输出（±0.001 vLLM 数值抖动）：
```
direct: 0.7660
cot:    0.7690
del_tom: 0.7823
```

### 数据合成（可选，需 OpenAI API key）

```bash
# Track B：ExploreToM v2 转换（需先下载 v2 源数据）
python production_frozen/v3.0/scripts/convert_exploretom_v2.py \
  --input data/tom/raw/exploretom_v2_source.jsonl \
  --output exploretom_v2_track_b.jsonl

# Track C：GPT-5.5 HOT 靶向合成
OPENAI_API_KEY=sk-... python production_frozen/v3.0/scripts/synth_gpt55_phase_d_hot.py \
  --output synth_gpt55_phase_d_hot.jsonl \
  --target_n_per_class 250

# 合并
python production_frozen/v3.0/scripts/merge_stage11_train.py \
  --base production_frozen/data/tom_train_14b_stage8.jsonl \
  --add exploretom_v2_track_b.jsonl synth_gpt55_phase_d_hot.jsonl \
  --output tom_train_stage12.jsonl
```

## 关键不变量（绝对不要改，否则结果不复现）

### 训练超参（与 Stage 8 一致）
- `whiten_advantages: true`
- `add_token_level_kl: false`
- `loss_agg_mode: "seq-mean-token-mean"`
- `pg_clip_low: 0.20, pg_clip_high: 0.28` (DAPO Clip-Higher)
- `dual_clip_loss: true`
- `difficulty_low_threshold: 0.1, high_threshold: 0.95`
- `response_length: 256`
- `prompt_length: 1024` (14B; 14B 受 KV cache 限制)
- `pretrain: <stage 8 HF path>` (NO SFT cold start)
- `distrib_optim_fully_reshardable_mem_efficient: true` (防 ckpt save OOM)

### 数据
- 12519 条 = 9259 (= Stage 8 数据原样) + 2000 (Track B) + 1260 (Track C)
- 0 条与评测集 5718 重合（merge 脚本会校验）

### 硬件
- 1×8 H800 80GB SXM (NVLink full mesh)
- TP=2, PP=1
- 80GB VRAM 必须

## Track A `del_tom` 协议说明

del_tom 是评测时使用的 prompt 工程协议，**不需要重训**。流程：

1. 让模型先做 cot 推理拿到初步答案。
2. 删除被认为最不可能正确的若干选项（基于初步推理 + 启发式）。
3. 在缩减选项集上重新让模型回答。
4. N=8 次独立采样，多数投票。

实现见 `scripts/extractors.py:vote_del_tom()` 和 `scripts/run_tombench.py` 的 `protocol == "del_tom"` 分支。

del_tom 在 Stage 8 上能给 +2.16pp（0.7594 → 0.7810），在 Stage 12 上给 +1.33pp（0.7690 cot → 0.7823 del_tom）。增益减小是预期，因为 Stage 12 模型在 cot 协议下已经更鲁棒。

## 失败实验（不在 v3.0，但保留警示）

这些训练阶段虽然产生了模型但都比 Stage 8 弱：

| 阶段 | 结果 | 失败原因 |
|---|---|---|
| Stage 9 (SFT 冷启 + KL + 长 CoT) | 14B raw 0.7429, **-1.51pp vs s8** | SFT init 洗掉了 Stage 8 RLVR 学到的策略 |
| Stage 10 (weighted_sum + entropy) | step 214 终止, val 0.666 < s8 0.706 | entropy bonus 让模型学到错乱的 hedging |
| 11d Track D（仅续训） | subset500 平滑 ~0.73 | 不如 Track E，证明数据是瓶颈 |

详情见 `docs/stage9_retro.md`、`docs/stage10_plan_evidence_based.md`、`docs/final_project_report.md`、`docs/stage11_report.md`。

## Git 标签

```bash
git tag -a v3.0-production -m "Stage 12 + del_tom = 0.7823 (project record)"
git push origin v3.0-production
```

恢复到本快照：

```bash
git checkout v3.0-production -- production_frozen/v3.0/
```

## 详细技术报告

中文完整版：[`docs/stage11_v2_final_report_zh.md`](docs/stage11_v2_final_report_zh.md)

包括：
- 五条赛道（A/B/C/D/E）的设计动机与结果
- Track E 完整 val 轨迹（含 step 200 凹陷分析）
- 与历史所有阶段的对比表
- 失败实验教训
- 后续可探索方向（del_tom on Stage 12 / N=16 投票 / 续训 Stage 12 等）

最后更新：2026-05-21 02:30 UTC
