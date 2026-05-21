# Stage 13 训练计划与执行报告

> **状态**: 进行中（2026-05-21 14:04 启动）
> **依据**: Stage 12 完成后的错误分析

## 一、Stage 12 错误分析摘要

Stage 12 + del_tom 在 5718 题上仍错 1245 题。按 task 分布：

| Task | Stage 12 acc | 剩余错误 | 占总错误 |
|---|---|---|---|
| **Knowledge** | 0.5260 | 274 | 22.0% |
| **Non-literal Comm** | 0.8102 | 284 | 22.8% |
| Emotion | 0.7607 | 201 | 16.1% |
| False Belief | 0.8946 | 156 | 12.5% |
| Desire | 0.6083 | 141 | 11.3% |
| Intention | 0.8294 | 116 | 9.3% |
| Belief | 0.7430 | 73 | 5.9% |

**最严重子任务**：
- Knowledge / Attention_Links：38 题中错 27 题（71% 错误率）
- Knowledge / Pretend_Play_Links：60 题中错 26 题（43%）
- Non-literal Comm / pas_Recognition_Test：1120 题中错 240 题（21%，但绝对量最大）

## 二、Stage 13 方案选择

| 方案 | 投入 | 预期增益 | 风险 | 是否启动 |
|---|---|---|---|---|
| **A. 续训 Stage 12（无新数据）** | 6h 训练 | +0.5~1pp | 低 | ✅ Stage 13a 已启动 |
| B. 靶向合成 + 训练 | 2-3h 合成 + 6h 训练 | +1.5~2pp | 中 | ⏸ 待 OpenAI key |
| C. del_tom N=16 投票 | 1h 评测 | +0.2~0.4pp | 低 | 💡 候选 |
| D. GPT-5.5 蒸馏 SFT | 17h | +1~3pp | 高 | ❌ 风险大（Stage 9 失败前例）|

**选 A**：
- 不依赖外部 API
- Track D 在 Stage 8 上证明了"续训"在 350 步后仍能涨 ~+2.5pp
- Stage 12 val 轨迹（0.7060→0.7640）单调上升（除 step 200 噪声），暗示未完全收敛
- 失败成本低（回退到 Stage 12 即可）

## 三、Stage 13 配置

文件：`configs/tombench-rlvr/rlvr_config_stage13_1x8_14b.yaml`

变更（相对 Stage 12）：
- `pretrain`: `qwen3-14B-tom-hf-stage8` → **`qwen3-14B-tom-hf-stage12`**（关键）
- `reward_pretrain`: 同上
- `exp_name`: `stage12-1x8` → `stage13-1x8`
- `max_steps`: 350 → **250**（避免过度续训）
- `save_steps`: 350 → **250**

其他 100% 与 Stage 12 一致（GRPO 超参、whiten_advantages、DAPO Clip-Higher、TP=2、`distrib_optim_fully_reshardable_mem_efficient` 等）。

## 四、预期与判停

**好情况**（继续训练 → 预期）：
- step 50 val ≥ 0.7640 → 训练有效，继续
- step 100 val ≥ 0.78 → 显著提升
- step 250 final → 0.78~0.80（subset500），全量 del_tom 0.79~0.80

**坏情况**（早停/回退）：
- step 50 val < 0.74 → 暗示数据已经被学完，召回 Stage 12
- step 100 val 与 step 50 持平 → 提早 kill，回退 Stage 12

**最可能情况**（部分提升）：
- step 50/100/150 慢速上升 5pp 内 → 训完 250 步评测决定是否提升 baseline

## 五、并行准备

无论 Stage 13a 结果，已经准备：
1. **Stage 12 错误集合**：`output/analysis/stage12_errors.jsonl`（1245 题，按 task/subtype/lang 标签）
2. **靶向合成 prompt 模板**：基于现有 `scripts/data/synth_gpt55_phase_d_hot.py`，可换 OpenAI key 立即跑（Stage 13b 数据预备）
3. **del_tom N=16 评测**：当前 N=8，扩展到 N=16 仅需修改 `run_tombench.py` 一行参数

## 六、时间表

| 时间 | 事件 |
|---|---|
| T+0 (14:04) | Stage 13 启动 |
| T+30min | step 50 val（决策点 1：是否继续）|
| T+1h | step 100 val（决策点 2：是否能撑过 step 200）|
| T+2h30 | step 200 val |
| T+4h | step 250 完成 + ckpt 保存 |
| T+5h | Megatron→HF 转换 |
| T+5h30 | vLLM 启动 + 评测开始 |
| T+7h | 评测完成（direct + cot + del_tom，5718 题）|

最后更新：2026-05-21 14:08 UTC
