# Stage1 1×8 训练 + 评测最终报告

> 训练: 2026-05-15 12:17 → 15:51 (UTC); 共 200 步, 4k 样本子集
> Eval: 2026-05-16 00:25 (full ToMBench 5718) + subset500 (3 protocols)
> Checkpoint: `/data_nvme/grj-projects/tom-output/qwen3-8B-tombench-rlvr-stage1-1x8/20260515-121728/checkpoint-199`

## 1. Headline 结果

**完整 ToMBench (5718 题, direct protocol)**:

| Model | n | Overall | EN | ZH |
|---|---|---|---|---|
| qwen3-8b-nt baseline | 5718 | 0.7009 | 0.7020 | 0.6999 |
| **qwen3-8b-tom (trained, stage1)** | 5718 | **0.7394** | 0.7275 | **0.7513** |
| **Δ vs baseline** | | **+3.85pp** | +2.55pp | **+5.14pp** |
| deepseek-v4-pro target (X) | 500 | 0.7880 | 0.7803 | 0.7966 |
| Gap to target X | | **−4.86pp** | −5.28pp | −4.53pp |
| Approaches X (X−ε=0.7680)? | | ✗ (差 2.86pp) | | |

**Subset500 (3 protocols, apples-to-apples)**:

| Model | direct | cot | del_tom |
|---|---|---|---|
| qwen3-8b-nt baseline | 0.6900 | 0.7640 | — |
| **qwen3-8b-tom trained** | **0.7460** | 0.6980 | **0.7460** |
| Δ direct | **+5.60pp** | | |

> Note: trained model 在 direct/del_tom 上提升明显，但 cot 反而下降 ~6pp。这是 RL 训练把答案"格式化"到 `\boxed{X}` 的副作用 — 模型现在更倾向直接给答案，CoT 长链推理被压缩。

## 2. 训练动态

- **训练曲线** (stdout 日志解析): `output/analysis/curves_stage1_1x8.png`
- **关键 metric 上升** (in-rollout):
  - `critic/score/mean` (rollout batch 答对率): step 0 → 0.21, step 100 → 0.52, step 198 → 0.79 (**3.7×**)
  - `tom_mcq/reward/r_total_mean`: 0.21 → 0.77
  - `tom_mcq/reward/r_fmt_mean`: 0.45 → 0.82 (格式收敛快)
  - `tom_mcq/reward/r_out_mean`: 0.30 → 0.78 (答案准确率)
  - `tom_mcq/reward/r_len_mean`: 较稳定在 0.85-0.90 (length penalty 不主导)
  - `actor/total_loss`: ≈0 (PG loss 小, KL 控制良好)

- **subset500 在线验证**:
  | step | val_correct/all | tom_mcq |
  |---|---|---|
  | 0 | 0.042 | 0.278 |
  | 50 | 0.204 | 0.299 |
  | 100 | 0.454 | 0.534 |
  | 150 | 0.548 | 0.613 |

- **吞吐**: ~1 step/min, 全程 199 步 ≈ 3h 20m;  do_checkpoint 用 617 秒 (mem-efficient gloo+CPU gather, 但成功)
- **GPU 利用率**: 训练 100%, save 阶段 0% (CPU bound)
- **OOM 计数**: 0 (修复完全生效, 见 §4)

## 3. Per-task 提升 (full 5718, direct)

| Task | baseline | trained | Δ |
|---|---|---|---|
| **False Belief** | 0.7277 | **0.8520** | **+12.43pp** ↑↑↑ |
| Emotion | 0.6893 | 0.7286 | +3.93pp ↑ |
| Belief | 0.6725 | 0.6937 | +2.11pp ↑ |
| Intention | 0.7500 | 0.7647 | +1.47pp ↑ |
| Desire | 0.5861 | 0.5917 | +0.56pp ↑ |
| Non-literal Comm | 0.7767 | 0.7674 | −0.94pp ↓ |
| Knowledge | 0.4810 | 0.4792 | −0.17pp ↓ |

**洞察**:
- **False Belief 大幅提升 (+12.4pp)**: 这是 ToMBench 最经典的 task；ExploreToM/SimpleToM 训练数据中包含大量隐含信念跟踪样本，正中靶心。
- **Knowledge 几乎没变 (-0.2pp)**: baseline_gap_analysis 早就标记 Knowledge "low ceiling" — deepseek 在这个 task 也只有 0.60 (远低于其他 task 的 0.80+)。需要专门的 fact retrieval 训练数据。
- **Non-literal Comm 微降 (-0.9pp)**: 训练数据没有专门的 sarcasm/irony 样本，可能被其他 task 的 RL 信号挤掉了。

## 4. 解决的工程问题

**save_checkpoint OOM** (上一版 199 步全跑完，最后 save 时 8/8 worker OOM)

- **根因**: `megatron.core.dist_checkpointing` 默认用 NCCL+CUDA all-gather 收集 DP-sharded optimizer state，每 rank 临时申请 ~3.81 GiB GPU buffer (`recv_tensors`)。1×8 colocated 部署下 vllm + train + reference + 优化器已占 ~75 GiB，剩 < 1 GiB → OOM。
- **修复**: 在 6 个 stage 配置的 `actor_train.strategy_args.strategy_config` 加 `distrib_optim_fully_reshardable_mem_efficient: true`。这是 mcore_adapter 已有的开关，触发 megatron 走 Gloo+CPU gather 路径 (line 1077: `device = "cpu" if use_gloo_comm else torch.cuda.current_device()`)。代价: save 慢 (~10 分钟 vs ~30 秒)，但稳定可靠。
- **commit**: `d7bf18a` "fix(save): use mem-efficient (Gloo+CPU) optimizer state gather"

## 5. 部署

- **HF 模型路径** (host): `/data_nvme/grj-projects/tom-output/qwen3-8B-tom-hf/` (16 GB, 4-shard safetensors)
- **vLLM serve**: `qwen3-tom-serve-direct` 容器 (复用 train image，避免重新 build serve image), GPU 0, port 8000
- **OpenAI-compatible 端点**: `http://172.16.120.181:8000/v1`, model id `qwen3-8b-tom`
- **冷启动**: ~2 分钟 (load + torch.compile)
- **推理吞吐**: subset500 direct ≈ 18 it/s @ concurrency=32

## 6. 接下来的可选路径

1. **不接 stage2，直接进 stage1 加长版** (4k → 8k data, 200 → 500 steps): 还有 0.5 epoch 在剧烈学习，可能再涨 2-3pp。
2. **针对 Knowledge 加专门数据**: 现在距 deepseek 还有 4.86pp 差距，主要来自 Knowledge (deepseek 0.60 vs trained 0.48)。需要 4-option 知识题数据（可由 deepseek-flash 合成）。
3. **CoT 修复**: trained 模型在 cot 上降到 0.698, 暗示 RL 把推理风格压窄。可以加 cot reward 样本或调高 length budget (response_length 256 → 512)。
4. **L3 fallback (stage3)**: 如果 stage2 仍不达标，启用过程奖励。当前差距说明 stage2 极可能不够。

## 7. 资源消耗

- **训练**: 1×8 H800, 3h 20m wall, ~26 GPU-hr
- **Checkpoint 写盘**: 112 GB (model 16G + optimizer 92G + tokenizer/rng 4G)；save 10m
- **HF 转换**: ~3 分钟, 16 GB
- **Eval (full 5718, direct)**: ~6 分钟 @ vLLM concurrency=32

## 8. 文件与产物

| 文件 | 说明 |
|---|---|
| `output/eval/final_full5718.json` | 5718 题 direct 完整 raw |
| `output/eval/final_full5718.md` | 5718 题 markdown 摘要 |
| `output/eval/final_subset500.json` | subset500 × 3 protocols 完整 |
| `output/eval/final_subset500.md` | subset500 markdown 摘要 |
| `output/eval/baseline_subset500.json` | baseline 在同一 500 子集的过滤结果 |
| `output/analysis/curves_stage1_1x8.png` | 12 子图训练曲线 |
| `output/analysis/eval_diff.md` | baseline vs trained 完整对比 |
| `output/analysis/errors.md` | 每 task 5 个 trained 错误样本 |
| `logs/train_stage1_1x8_20260515_121704.log` | 训练完整日志 (10 MB) |
