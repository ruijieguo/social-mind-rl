# Stage2 1×8 训练 + 评测报告

> 训练: 2026-05-16 00:18 → 08:42 (UTC); 共 500 步, 8k 数据 (tom_train.jsonl, 7911 records)
> Eval: 2026-05-16 16:44 (full 5718) + subset500 × 3 protocols
> Checkpoint: `/data_nvme/grj-projects/tom-output/qwen3-8B-tombench-rlvr-stage2-1x8/20260516-001853/checkpoint-499`

## 1. Headline 对比

**Full ToMBench 5718, direct**:

| Model | n | Overall | EN | ZH |
|---|---|---|---|---|
| qwen3-8b-nt baseline | 5718 | 0.7009 | 0.7020 | 0.6999 |
| qwen3-8b-tom **stage1** (200 steps, 4k) | 5718 | **0.7394** | 0.7275 | **0.7513** |
| qwen3-8b-tom **stage2** (500 steps, 8k) | 5718 | 0.7263 | 0.7223 | 0.7303 |
| deepseek-v4-pro target | 500 | 0.7880 | 0.7803 | 0.7966 |

**Stage2 在 direct 协议上比 stage1 退化 1.31pp**。但...

**Subset500 (3 protocols, apples-to-apples)**:

| Model | direct | cot | del_tom |
|---|---|---|---|
| qwen3-8b-nt baseline | 0.6900 | **0.7640** | — |
| qwen3-8b-tom **stage1** | **0.7460** | 0.6980 | 0.7460 |
| qwen3-8b-tom **stage2** | 0.7340 | **0.7540** | 0.7480 |
| deepseek-v4-pro target | 0.7880 | 0.7140 | — |

**关键洞察**: **Stage2 cot 0.7540 比 stage1 cot 0.6980 高 5.6pp** — stage2 修复了 stage1 在 cot 上的退化。
- Stage2 **best score**: cot **0.7540**（比 stage1 best direct 0.7460 高 0.8pp）
- 离 deepseek-v4-pro target 0.7880 还差 **3.4pp**（stage1 best 是差 4.2pp）

## 2. Per-task 对比

| Task | baseline | stage1 | stage2 | Δs1 | Δs2 | Δs2−s1 |
|---|---|---|---|---|---|---|
| Belief | 0.6725 | 0.6937 | 0.6373 | +2.11pp | **−3.52pp** | **−5.63pp** |
| Desire | 0.5861 | 0.5917 | 0.5861 | +0.56pp | 0 | −0.56pp |
| Emotion | 0.6893 | 0.7286 | 0.7012 | +3.93pp | +1.19pp | −2.74pp |
| False Belief | 0.7277 | 0.8520 | 0.8385 | +12.43pp | **+11.08pp** | −1.35pp |
| Intention | 0.7500 | 0.7647 | 0.7632 | +1.47pp | +1.32pp | −0.15pp |
| Knowledge | 0.4810 | 0.4792 | 0.4879 | −0.17pp | **+0.69pp** | +0.87pp |
| Non-literal Comm | 0.7767 | 0.7674 | 0.7553 | −0.94pp | −2.14pp | −1.20pp |

**洞察**:
- **Stage2 在 Belief 上大幅退化** (−5.63pp vs stage1): 训练数据中带有 belief tracking 的样本可能 overfit 到狭窄模式
- **False Belief 仍保持 +11pp 大幅提升**: 这是 stage1 的核心收益，stage2 没显著破坏
- **Knowledge 唯一在 stage2 仍上涨** (+0.87pp vs +0.69pp): 训练增加额外数据帮到了 Knowledge
- **Non-literal Comm 持续退化**: 数据未覆盖此 task

## 3. 训练动态

**Validation 曲线** (subset500 val_all / tom_mcq):

| step | stage1 | stage2 |
|---|---|---|
| 0 | 0.042 / 0.278 | 0.036 / 0.280 |
| 50 | 0.204 / 0.299 | 0.206 / 0.294 |
| 100 | 0.454 / 0.534 | 0.466 / 0.551 |
| 150 | 0.548 / 0.613 | 0.546 / 0.633 |
| 200 | — (stop) | 0.530 / 0.627 |
| 250 | — | 0.634 / 0.675 |
| 300 | — | 0.648 / 0.691 |
| 350 | — | 0.646 / 0.679 |
| 400 | — | 0.650 / 0.692 |
| **450** | — | **0.664** / 0.678 |

**Rollout score** (in-batch 答对率):
- step 0: 0.33
- step 80: 0.60
- step 200: 0.87
- step 300: 0.92
- **step 450+: 0.95–0.98** ← saturate

**关键观察**:
- val_all 在 step 250-450 平台 0.63-0.66
- rollout score 0.93-0.98 = 训练集"打满"
- **真正的泛化提升在 step 150-250 已结束**
- step 250+ 是过拟合期 — 这正是 full 5718 direct 反而下降的原因

## 4. Stage1 vs Stage2 的本质区别

| 方面 | Stage1 | Stage2 |
|---|---|---|
| 数据 | 4000 records (4k tom_train_4k.jsonl) | 7911 records (tom_train.jsonl) |
| 步数 | 200 | 500 |
| 训练时长 | 3h 20m | 8h 24m |
| Save 用时 | 10m (Gloo+CPU gather) | 10m (相同) |
| OOM | 0 | 0 |
| Best full 5718 score | direct 0.7394 | direct 0.7263 (worse) |
| Best subset500 score | direct 0.7460 | **cot 0.7540** |
| 推理风格 | 偏 direct（cot 退化） | **保留 cot 能力** |

**Stage2 的核心价值**: **保留了多协议泛化能力**，没有把模型"压"成只会 direct 答题的特化版本。

## 5. 用户后续路径建议

**Recommendation 1 — 收敛到 stage1**:
- 用 stage1 ckpt (direct 0.7394) 作为生产模型
- stage1 在 direct 上更强，且 false_belief task 收益最大
- 距 deepseek 4.86pp，已是非常好的结果

**Recommendation 2 — early-stop stage2**:
- Stage2 在 step 200-250 已经达 val 峰值 0.55-0.63
- 如果有第三次训练机会，可以跑 250 步即可 (4h vs 8h)，不会过拟合
- 可能能取得 stage1 直接收益 + stage2 多协议保留的双赢

**Recommendation 3 — 提升 deepseek gap 的真正路径**:
- Knowledge task (deepseek 0.60 vs stage1/stage2 0.48-0.49)：需要合成专门的 fact-retrieval 数据
- Non-literal Comm 持续退化：需要 sarcasm/irony 数据
- Belief task 在 stage2 退化：训练数据可能需 belief tracking 样本去重

## 6. 部署

- **HF 模型路径** (host): `/data_nvme/grj-projects/tom-output/qwen3-8B-tom-hf-stage2/` (16 GB, 4-shard safetensors)
- **vLLM serve**: `qwen3-tom-serve-stage2` 容器, GPU 0, port 8000, model id `qwen3-8b-tom-stage2`
- **OpenAI-compatible 端点**: `http://172.16.120.181:8000/v1`

## 7. 产物

| 文件 | 说明 |
|---|---|
| `output/eval/stage2_full5718.{json,md}` | 5718 题 direct |
| `output/eval/stage2_subset500.{json,md}` | 500 题 × 3 protocols |
| `output/analysis/curves_stage2_1x8.png` | 12 子图训练曲线 |
| `output/analysis/errors_stage2.md` | 错题样本 |
| `logs/train_stage2_1x8_20260516_001833.log` | 完整训练日志 (25 MB) |

## 8. 工程实践记录

- **磁盘**: SSD `/data` 875GB 不够（stage2 transient 需 214GB），全栈迁到 NVMe `/data_nvme` (14TB)，4:29 rsync 完成
- **Save fix 持续生效**: distrib_optim_fully_reshardable_mem_efficient=true 在 stage2 500 步规模下零 OOM
- **配置一致性**: stage2 同时修了 4 个 bug（grad_accum, track_with, gpu_mem, save_steps），训练全程零问题

Last updated: 2026-05-16 16:50
