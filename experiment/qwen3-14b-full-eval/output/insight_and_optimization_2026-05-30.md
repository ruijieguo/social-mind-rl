# Qwen3-14B 评测洞察 + 优化方案

> 基于 `full_eval_report_qwen3-14b_2026-05-30.md`（base / v3.5 / v3.1 / deepseek-v4-pro
> × ToMBench / EmoBench / SocialIQA / Hi-ToM × direct / direct_think / cot）与
> Stage 12→19 训练过程的联合分析。
> 日期：2026-05-30。

---

## 1. 一句话结论

**当前的 14B RLVR 训练在优化一个狭窄的代理目标（ToMBench 准确率 + 256-token 思考预算），
而这个代理目标与"通用心智推理能力"负相关。** 在 4 benchmark 综合均值上，**未训练的 base
反而是最强的本地模型**（0.7603 > v3.5 0.7389 > v3.1 0.7305），训练是**净负收益**——它用
泛化能力换了 ToMBench 的局部分数。

---

## 2. 核心证据

### 2.1 最优协议均值：训练让模型整体变差

| 模型 | ToMBench | EmoBench | SocialIQA | Hi-ToM | **均值** |
|---|---|---|---|---|---|
| **base** | 0.7609 | **0.6875** | **0.7979** | **0.7950** | 0.7603 |
| v3.5 | 0.7705 | 0.6608 | 0.7876 | 0.7367 | 0.7389 |
| v3.1 | **0.7816** | 0.6483 | 0.7886 | 0.7033 | 0.7305 |
| deepseek-v4-pro | **0.8286** | **0.7717** | **0.8178** | 0.7733¹ | **0.7979** |

¹ deepseek Hi-ToM 被截断压低（direct_think 16%、cot 24% 截断，8192 仍不够长篇推理）。

**训练只在 ToMBench 上赢了 base（+2pp），其余 3 个 benchmark 全部输给 base。**
v3.1 是最极端的：ToMBench 最高、但 Hi-ToM 最低（0.7033）——典型的过拟合。

### 2.2 机制：RLVR 把"思考链"压缩了 4-5×

cot 协议下的平均输出长度（chars）：

| benchmark | base | v3.5 | v3.1 | deepseek² |
|---|---|---|---|---|
| ToMBench | 3409 | 700 | 869 | 227 |
| **Hi-ToM** | **11331** | 2600 | 4415 | 337 |
| SocialIQA | 3181 | 816 | 931 | 213 |
| EmoBench | 4097 | 845 | 1082 | 225 |

² deepseek 的可见 content 短，是因为推理在独立的 `reasoning_content` 字段（隐藏），不可比。

**根因坐实于配置**：
- `rlvr_config_stage14_1x8_14b.yaml`（v3.1）：**`response_length: 256`**
- `rlvr_config_stage19_1x8_14b.yaml`（v3.5）：`response_length: 512`

RLVR rollout 被截在 256/512 token，模型被奖励"在极短预算内答对"→ 学会了把思考压到极致。
CLAUDE.md 明确写过 256 是"为了惩罚 ToMBench 上的啰嗦思考"——但这正是**能力税**的来源。

### 2.3 烟枪证据：Hi-ToM 退化随推理深度单调放大

Hi-ToM 分阶（cot），需要 "A 以为 B 以为 …" 的多跳递归推理：

| order | base | v3.5 | v3.1 | 说明 |
|---|---|---|---|---|
| order_0（事实） | 1.0000 | 1.0000 | 1.0000 | 无需推理，不受影响 |
| order_1 | **0.8167** | 0.6917 | 0.6667 | 训练 **-15pp** |
| order_2 | **0.8083** | 0.7333 | 0.7333 | 训练 -7.5pp |
| order_3 | **0.7000** | 0.6000 | 0.5750 | 训练 -10~12pp |
| order_4 | **0.6500** | 0.6583 | 0.5417 | v3.1 -11pp |

**退化幅度与所需推理深度成正比**：order_0（不推理）零损失，order_1+ 越深损失越大。
base 的 11k-char 长链能扛住深度递归；压缩后的 2.6-4.4k 链扛不住。这把"短思考有利 ToMBench、
有害深层推理"的机制钉死了。

### 2.4 ToMBench 内部也不是全赢：训练提升的是高频任务

ToMBench 分任务（cot）：

| task | base | v3.5 | v3.1 | deepseek | 训练效果 |
|---|---|---|---|---|---|
| Intention | 0.7897 | 0.8235 | **0.8412** | 0.8985 | ✅ +5pp（训练数据高频）|
| False Belief | 0.8797 | 0.8804 | **0.9020** | 0.9169 | ✅ +2pp |
| Knowledge | 0.5190 | 0.5294 | 0.5519 | **0.6626** | ✅ +3pp，但离 deepseek 差 11pp |
| Non-literal | 0.7747 | 0.7901 | 0.7894 | 0.8242 | ✅ +1.5pp |
| Emotion | 0.7452 | 0.7464 | 0.7560 | 0.8107 | ≈ |
| Desire | 0.5778 | **0.6111** | 0.6028 | 0.6333 | 混合 |
| **Belief** | **0.7711** | 0.7324 | 0.7394 | 0.8627 | ❌ **-3~4pp（训练反而退化）** |

训练把分数堆在 Intention/False-Belief（单跳、训练高频），却**让 Belief 退化**。

### 2.5 EmoBench：情绪理解（EU_emotion）退化最重

| task | base | v3.5 | v3.1 | deepseek |
|---|---|---|---|---|
| EA（情绪应用）| 0.7275 | 0.7050 | 0.6600 | 0.7400 |
| EU_cause（情绪成因）| 0.7725 | 0.7775 | 0.7675 | **0.8475** |
| **EU_emotion（情绪识别）**| **0.5625** | 0.5000 | 0.5000 | **0.6950** |

EU_emotion 是所有人的弱项，但**训练把 base 的 0.5625 压到 0.50**，而 deepseek 在这项领先 14pp。

---

## 3. 战略洞察

1. **"对标 deepseek 还差 8.7pp" 是 ToMBench 单点叙事。** 在 4-benchmark 均值上，base
   已是最强本地模型，离 deepseek 仅 3.8pp（0.7979 vs 0.7603）。真正的差距集中在
   **ToMBench-Knowledge/Belief、EmoBench-EU_emotion**，**不在 Hi-ToM**（base 已与 deepseek 持平）。

2. **当前训练的"最佳 ckpt"评判标准选错了。** v3.1 是按 ToMBench del_tom 选出的"项目记录"，
   但它恰恰是泛化最差的 ckpt。用 4-benchmark 均值评判，base > v3.5 > v3.1，结论完全反转。

3. **256-token 思考预算是核心枷锁。** 它对 ToMBench（浅）无害甚至有利，但对 Hi-ToM order-4
   （base 要 ~3-4k token）是毁灭性的。这是一个经典的 **specification gaming / 能力税**。

4. **加 Hi-ToM/Emo/Social 数据（stage16-19）只是 SFT-distill，没进 RLVR reward。** 它部分
   止血（v3.5 的 Hi-ToM order_1-3 > v3.1），但因为思考预算和 reward 形状没变，救不回 base 的水平。

---

## 4. 优化方案（按性价比排序）

### Plan D — 模型汤 / WiSE-FT（**立刻可做，零训练，最高性价比**）
base 赢 3/4、v3.1 赢 ToMBench → **权重空间插值 `θ = (1-α)·base + α·v3.1`** 很可能 Pareto 占优。
- 做法：α ∈ {0.2,0.3,0.5,0.7} 扫一遍，4-benchmark 均值选最优。无需训练，几小时出结果。
- 预期：拿到 base 的 Hi-ToM/Social/Emo + v3.1 的 ToMBench，均值有望 >0.77 反超所有现有 ckpt。
- 复用本实验框架：转好权重后直接 `04_run_eval.sh` 加一列。

### Plan A — 解开思考预算枷锁（**下一轮训练第一优先级**）
- `response_length` 从 256/512 提到 **2048-4096**（Hi-ToM order-4 需 ~3-4k token）。
- reward 改为 **correctness 为主 + 轻度长度正则**，并对高阶/长上下文样本**豁免长度惩罚**
  （按 task/order 给分层预算），避免再惩罚必要的长链。
- 注意 `TomMcqRewardWorker` 的 `l_max` 需同步上调（CLAUDE.md 已警告二者耦合）。

### Plan C — 加强对 base 的 KL 锚定（防漂移）
- 当前 `add_token_level_kl: false`。退化本质是策略偏离 base 太远丢了通用推理。
- 开 token-level KL 或加大 KL 系数，把 base 的长链能力锚住；代价是 ToMBench 增益变小，
  但换回 Hi-ToM/Emo/Social，符合"优化均值而非 ToMBench"的新目标。

### Plan B — 多 benchmark 进 RLVR reward（不只 SFT）
- 把 Hi-ToM(15-opt)、SocialIQA(3-opt)、EmoBench(4-opt) 的 MCQ 直接纳入 reward worker，
  让 RL **直接优化 4-benchmark 综合**，而不是只 ToMBench 形状的 reward + 旁路 SFT 数据。

### Plan E — 改 ckpt 选择 + 训练内验证
- 训练循环的 validation 从"ToMBench subset"换成 **4-benchmark 均值**（哪怕各取小子集）。
- 早停按均值，不按 ToMBench；否则还会选出下一个 v3.1。

### 针对 deepseek 差距的定点突破（若仍以 deepseek 为目标）
差距集中在 3 个格子，定点造数据：
- **ToMBench-Knowledge**（base 0.52 vs ds 0.66，-14pp）：知识归因 / scalar implicature 类。
- **ToMBench-Belief**（训练已退化到 0.74 vs ds 0.86）：先用 Plan A/C 止住退化，再补 Belief 数据。
- **EmoBench-EU_emotion**（0.50 vs ds 0.70，-20pp）：情绪识别本体（`data/distill/emotion_ontology.txt`）已有，做定向蒸馏。

---

## 5. 建议执行顺序

1. **今天**：跑 Plan D 模型汤（base⊕v3.1 α 扫描）→ 大概率立刻拿到比所有现有 ckpt 都好的均值模型。
2. **下一轮训练**：Plan A（放开思考预算）+ Plan C（KL 锚定）+ Plan E（均值选 ckpt），从 base 重训
   或从 Stage 12 续训，目标 **4-benchmark 均值 > base 0.7603** 而非 ToMBench 单点。
3. **中期**：Plan B（多 benchmark reward）+ 对 deepseek 三个弱格的定点数据。

> 一句话：**别再用 256-token 预算去刷 ToMBench 了——它在偷偷摧毁模型的深层心智推理能力。
> 先用模型汤把泛化捡回来，再用"放开思考 + 锚定 base + 综合均值选 ckpt"重训。**
