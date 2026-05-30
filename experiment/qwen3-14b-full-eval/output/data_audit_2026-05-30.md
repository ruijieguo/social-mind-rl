# 训练数据审计（v3.1 / v3.5 RLVR 数据）— 2026-05-30

> 触发：eval 显示训练只帮 ToMBench、其余全退化，怀疑数据。审计对象
> `tom_train_stage14_weighted.jsonl`（v3.1，14408 条）+ `tom_train_stage19.jsonl`（v3.5，19236 条）。
> RLVR 数据 = 纯 prompt（system+user）+ ground_truth，无思考 trace（trace 是 rollout 时生成的）。

## 🔴 严重 BUG：26% 训练数据被静默丢弃（含全部 ExploreToM + 全部高阶 HOT 合成）

### 现象
- v3.1：14408 条里 **3682 条（26%）`tag=None`** → **全部不参与训练**。
- 这 3682 条 = **ExploreToM v2 全部 2160 条** + **GPT-5.5 高阶 ToM 合成 (`synth_gpt55_phase_d_hot`) 全部 1522 条**。
- v3.5 同样丢弃 ExploreToM(1805) + HOT(1180) = 2985 条。**整条训练线（stage14-21）从未训练过这两个旗舰高阶数据集。**

### 根因（代码已定位）
`framework/ROLL/roll/pipeline/rlvr/rlvr_pipeline.py:100`：
```python
row["domain"] = tag_2_domain.get(row["tag"], "math_rule")   # tag=None → "math_rule"
```
`tag_2_domain = {"tom_mcq": "tom_mcq"}`（由 reward worker 的 `tag_included` 推出）。然后只有
`domain_interleave_probs` 里的 domain（=`{tom_mcq}`）会建数据集训练，`math_rule` 域的数据被
`dataset.filter` **直接过滤掉**。`assert len>0` 只看 tom_mcq（有 10726 条），所以**静默通过、无报错**。

→ 真正根因是**数据构建时的打标 bug**：merge 时漏给 `exploretom_v2` 和 `synth_gpt55_phase_d_hot`
打 `tag=tom_mcq`，导致它们 domain 落到 math_rule 被丢。

### 为什么这正好解释了 eval 结果
被丢的 3682 条任务分布：**Belief 1610、False Belief 648、Knowledge 576**、Desire 299 …——
全是**高阶、最难**的 ToM 内容（HOT = Higher-Order ToM；ExploreToM 是真实高阶数据集）。
模型只在**剩下较易的 synth + simpletom** 上训练。于是：
- **Belief 反而退化**（base 0.7711 → v3.1 0.7394）：最难的 Belief 例子被丢了。
- **Hi-ToM 崩塌**：深层推理数据双重缺失（短思考枷锁 + 高阶数据被丢）。
- 这与"短思考枷锁"是**叠加**的两个独立病因。

### 修复（极简，数据 100% 合法）
被丢记录**全部是合法 MCQ**（gold 字母都在选项里，3682/3682 通过校验）。修复 = 给这两个 source
补 `tag: tom_mcq` 即可，**白捡 26% 最高价值数据**。这是跑 Plan A 前**必须先做**的（否则 Plan A 用
同一份数据，会丢一样的 26%）。

## 🟠 质量问题 1：gold 字母严重失衡（A 偏置）

| 分布 | A | B | C | D |
|---|---|---|---|---|
| **训练 stage14** | **45%** | 31% | 14% | **10%** |
| ToMBench eval | 27% | 34% | 22% | 18% |
| SocialIQA eval | 33% | 33% | 34% | — |
| EmoBench eval | 32% | 22% | 22% | 14% |

训练 gold 45% 是 A、只有 10% 是 D，而所有 eval 集都接近均匀。RLVR 在这种分布上会学出一个
**偏向 A 的答案先验**（不确定时猜 A），这个先验**和任何 eval 集都不匹配**（ToMBench 实际 B 最多），
损害校准与泛化。CLAUDE.md 声称 `rebalance_synth.py` 已均衡到 ~25%，但 task-weighted 重采样后
**又被打回 45/31/14/10**——重采样没有保持字母均衡。
- **修复**：重采样后重新打乱选项位置使 gold ~均匀；或对每条随机轮换选项顺序（同步更新 gold + messages）。

## 🟡 质量问题 2：30% 重复 prompt

stage14：14408 条里有 **4295 条是重复 prompt 的额外副本**（≈30%）。这是 task-weighted 重采样
（Knowledge ×1.92、Desire ×1.65 …）**有意**的过采样，但副作用是：有效多样性下降，且和 gold 失衡叠加，
让"弱任务"的特定模式被过度强化。可接受，但若配合 Plan A 重训，建议改为**按样本难度**而非
"任务×整数倍"过采样，减少完全相同 prompt 的硬复制。

## 🟢 无泄漏（好消息）

train ∩ eval：question_id 重叠 = 0；story+question 文本前缀重叠 = 0（4 个 eval 集全部 0/N）。
MinHash 去重有效，ToMBench/Hi-ToM/SocialIQA/EmoBench 评测干净，分数可信。

## 影响优先级与对 Plan A 的修订

| # | 问题 | 严重度 | 修复成本 | 是否阻塞 Plan A |
|---|---|---|---|---|
| 1 | 26% 高阶数据被丢（tag bug） | 🔴 高 | 极低（补 tag） | **是，必须先修** |
| 2 | gold A 偏置 45% | 🟠 中 | 中（轮换选项） | 建议一并修 |
| 3 | 30% 硬重复 | 🟡 低 | 低（改重采样） | 可选 |
| 4 | 泄漏 | 🟢 无 | — | — |

**Plan A 数据修订**：用修复 #1（+可选 #2）后的新数据 `tom_train_stage22_planA.jsonl` 替换
`tom_train_stage14_weighted.jsonl`。预期：恢复 26% 高阶数据 + 解开思考枷锁 + KL 锚定，三管齐下
才是对"训练损害泛化"的完整反击。
