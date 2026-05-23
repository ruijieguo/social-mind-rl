# Stage 13 失败回顾：续训 Stage 12 在 step 100 退化

> **状态**：失败，已于 step 100 终止
> **决策**：放弃续训路线，转向针对性新数据合成（Stage 14）

## 一、训练轨迹

| step | val (subset500) | Δ from init | 备注 |
|---|---|---|---|
| 0 (init = Stage 12 ckpt-349) | 0.7540 | — | 与之前以为的 0.7640 不一致——0.7640 是 step 300 峰值；step 349 已经回退到 0.7540 |
| 50 | 0.7560 | **+0.20pp** | 几乎无提升 |
| 100 | **0.7360** | **-1.80pp** | **退化，触发停训规则** |

## 二、失败原因分析

**根本原因**：Stage 12 已经把 12519 条数据榨干。

证据链：
1. Stage 12 的 step 300 → 349 期间 val 从 0.7640 退到 0.7540（已经是过拟合信号）
2. Stage 13 续训 step 50 涨幅仅 +0.20pp（远低于 Track D 在 Stage 8 上的 +1.20pp）
3. Stage 13 step 100 直接退化 -1.80pp

对比 Track D（续训 Stage 8）轨迹：
| step | Track D | Stage 13 |
|---|---|---|
| 0 | 0.7080 | 0.7540 |
| 50 | 0.7200 (+1.20pp) | 0.7560 (+0.20pp) |
| 100 | 0.7280 (+2.00pp) | 0.7360 (-1.80pp) ⚠️ |

**关键差异**：Track D 时 Stage 8 没有训练过 ExploreToM v2 / HOT 数据，所以学到的策略仍有适用空间。Stage 13 时 Stage 12 已经在完整 12519 数据上训完，再多走 100 步只是把模型推向"训练集过拟合"方向，subset500 立刻反映出来。

## 三、教训

1. **续训不是免费的**：模型可能已经收敛，再训只会过拟合。Track D 成功不代表 Stage 13 也能成。Track D 的成功条件是"原始训练数据未充分利用"，而 Stage 12 已经充分利用。
2. **Stage 12 的 step 300 → 349 退化是早期信号**：当时被忽略，以为是 noise；现在看来是真实的过拟合开始。Stage 12 实际最优 ckpt 是 **step 300**，不是 step 349。
3. **判停规则有效**：定的 step 50 ≥ 0.7640 / step 100 与 step 50 持平 / 退化任一触发就停 —— Stage 13 在 step 100 退化触发，及时止损（节省了 ~3h）。

## 四、Stage 12 真实最优 ckpt 评估

**问题**：Stage 12 我们生产的 HF 模型（`qwen3-14B-tom-hf-stage12`）是从 ckpt-349 转换的，但 step 349 在 subset500 上 0.7540，比 step 300 的 0.7640 低 1pp。如果当初保存了 step 300 ckpt，**全量评测可能再涨 ~+1pp（del_tom 0.7823 → 0.79+）**。

但 Stage 12 配置 `save_steps: 350`，没有保存 step 300 ckpt。重新训一遍以拿 step 300 的成本（6h）暂时不值得，留作 backlog。

## 五、Stage 14 备选方向

按 ROI 排序：

### 方案 A：Stage 14 = 靶向 Knowledge / Desire 数据 + 训练 ⭐ 推荐
- **数据**：用 GPT-5.5 针对 Knowledge (Attention_Links + Pretend_Play_Links)、Desire 子任务合成 1500-2000 条
- **训练**：从 Stage 12 ckpt-349 init，max_steps=200，新数据混入旧数据后随机洗牌
- **预期**：del_tom 0.7823 → 0.795~0.81
- **依赖**：OpenAI API key（用户提供）

### 方案 B：Stage 12 重训取 step 300 ckpt
- 与 Stage 12 配置完全相同，只把 `save_steps` 改成 50（每 50 步存一次）
- 训练完后取 step 300 那个 ckpt
- **成本**：6h 训练 + 1h 评测
- **预期**：del_tom 0.7823 → 0.79~0.795（白送 ~+1pp）
- **价值低**：增量小，不依赖外部资源但占 GPU

### 方案 C：del_tom N=16 投票（仅评测）
- 当前 N=8，扩展到 N=16
- **成本**：1.5h（推理量翻倍）
- **预期**：del_tom 0.7823 → 0.785~0.79
- **价值**：小但快，可立即验证

### 方案 D：换基座（Qwen3-32B / Llama-3.3-70B）
- 完全重启项目
- **成本**：高（数据 + 训练 + 评测都要重做）
- 不在本轮考虑范围

## 六、决策建议

**立刻执行**：方案 C（del_tom N=16），1.5h 内有结果，**作为 Stage 14 等待新数据期间的填空**。

**等用户给 OpenAI key**：方案 A（Stage 14 靶向合成 + 训练）是预期最大增益的路径。

**不立刻做**：方案 B（重训取 step 300 ckpt）—— 收益有限，先看方案 A 是否能直接超过它。

## 七、Git + 任务状态

- Stage 13 配置 + launch 脚本保留在 `configs/`、`scripts/` 中（commit `467a0c0`）
- Stage 13 训练日志：`logs/train_stage13_1x8_14b_20260521_140427.log`
- Task #101 标记为 completed（结论：失败 retro 已写）
- 不更新 production_frozen v3.0（Stage 13 没产出有用 ckpt）

最后更新：2026-05-21 14:50 UTC
