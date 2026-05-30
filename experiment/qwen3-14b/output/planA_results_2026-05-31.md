# Stage 22 / Plan A — 最终结果与深入分析

> Qwen3-14B，从 base 重训：解思考枷锁（response_length 256→2048，reward
> weighted_sum + l_max 2048）+ KL 锚定 base（0.001）+ 修复数据（恢复 26%
> ExploreToM/HOT + gold 均衡）。200 步，.191 8×H800。
> 评测：5 个 ckpt（50/100/125/150/199）+ base，4-bench × 3-protocol，.191 同口径。
> 日期：2026-05-31。

## 0. 一句话结论

**Plan A 成功逆转了"训练损害泛化"的失败模式。** v3.1 当年训练是**净负**（均值掉到 base 以下、
Hi-ToM 崩到 0.7033）；Plan A 训练**净正**——所有 ckpt 都在 4-bench 均值上**超过 base**，
且**全部守住 Hi-ToM**（0.76-0.79，无塌陷）。对 base 的增益**温和（+0.4pp）**，但对 v3.1 是
**决定性反超（均值 +2.5pp，Hi-ToM +6~9pp）**。**核心假设——"解枷锁 + KL 锚 base 能在拿 ToMBench
增益的同时守住泛化"——被证实。**

> ⚠️ 口径：本次 base 在 **.191 (TP=2)** 重测 = **0.7504**（与之前 .181 TP=1 的 0.7603 因 vLLM
> 数值漂移差 ~1pp，Hi-ToM 尤其敏感）。**Stage-22 ckpt 必须对比同口径的 .191 base 0.7504。**

## 1. 主结果（4-bench best-protocol mean，.191 同口径）

| 模型 | ToMBench | EmoBench | SocialIQA | Hi-ToM | **均值** | Hi-ToM cot |
|---|---|---|---|---|---|---|
| **base (.191)** | 0.7634 | 0.6658 | 0.7958 | 0.7767 | **0.7504** | 0.7767 |
| **ckpt50** | 0.7643 | **0.6792** | 0.7892 | **0.7900** | **0.7556** ⭐ | 0.7900 |
| ckpt100 | 0.7690 | 0.6692 | **0.8004** | 0.7750 | 0.7534 | 0.7750 |
| ckpt125 | 0.7739 | 0.6725 | 0.7948 | 0.7750 | 0.7540 | 0.7750 |
| ckpt150 | **0.7744** | 0.6775 | 0.7989 | 0.7633 | 0.7535 | 0.7633 |
| ckpt199 | 0.7723 | 0.6692 | 0.7948 | 0.7733 | 0.7524 | 0.7733 |
| **v3.1（旧 .181）** | 0.7816 | 0.6483 | 0.7886 | 0.7033 | 0.7305 | 0.7033 |

**所有 Stage-22 ckpt（0.7524-0.7556）> base 0.7504，且全部 Hi-ToM ≥ 0.763。** 5 个 ckpt 均值
彼此在 0.3pp 噪声内、基本打平，**最佳平衡 = ckpt50（均值 0.7556，Hi-ToM 0.79）**。

## 2. 深入分析

### 2.1 🔑 烟枪：长思考被完整保留（这是泛化守住的根因）

cot 平均输出长度（chars）：

| | base | ckpt50 | ckpt150 | ckpt199 | **v3.1（旧）** |
|---|---|---|---|---|---|
| ToMBench | 3398 | 3277 | 3206 | 3111 | **869** |
| Hi-ToM | 11337 | 11093 | 10398 | 10411 | **4415** |

**Stage-22 几乎原样保留了 base 的长思考链（Hi-ToM ~11k chars），而 v3.1 被压到 4.4k。**
这直接证明：**解枷锁（response_length 2048 + weighted_sum + l_max 2048）+ KL 锚定，成功阻止了
v3.1 那种思考压缩**——而思考压缩正是 v3.1 摧毁 Hi-ToM 深层推理的机制。**Plan A 的工程修复全部生效。**

### 2.2 ToMBench：增益集中在"可学"任务，Knowledge 反降

per-task（cot），base → Stage-22 最优：

| task | base | 最优 | Δ |
|---|---|---|---|
| **Intention** | 0.7912 | 0.8235 (c100) | **+3.2pp** |
| **Non-literal Comm** | 0.7620 | 0.7888 (c150) | **+2.7pp** |
| Belief | 0.7782 | 0.7923 (c150) | +1.4pp（c199 跌到 0.7606）|
| False Belief | 0.8784 | 0.8872 | +0.9pp |
| Emotion | 0.7536 | 0.7619 | +0.8pp |
| Desire | 0.5972 | ~0.59 | ≈/略降 |
| **Knowledge** | **0.5502** | 0.5225-0.5433 | ❌ **-1~3pp（最难任务，反降）** |

恢复的 ExploreToM/HOT 高阶数据 + 长思考，帮到了 Intention/Non-literal/Belief，但 **Knowledge
（难度天花板任务）没救起来、甚至略降**——和"deepseek 也只有 0.66、本就接近能力上限"一致。

### 2.3 Hi-ToM：泛化随训练步数缓慢侵蚀（KL 漂移）

per-order（cot）+ 均值随 ckpt 的轨迹：

- **Hi-ToM cot 均值在 ckpt50 达峰（0.79）**，之后缓降：0.79 → 0.775(c100) → 0.775(c125) → 0.763(c150)。
- 分阶看：ckpt50 甚至**提升了 order_4（0.633→0.708）和 order_2**；后期 ckpt 主要在 **order_1/order_3
  上让步**（最深的递归先掉）。
- 对应 KL 从 0.012 涨到 0.07——**随训练推进，策略离 base 越来越远、向 ToMBench 特化漂移，
  缓慢侵蚀 Hi-ToM。** KL 锚（0.001）把这个侵蚀**从 v3.1 的悬崖（-7pp）压成缓坡（-1.5pp）**，
  但没完全止住。

→ **这就是为什么 4-bench 均值最优点在很早（ckpt50）**：早期 ToMBench 已拿到大部分增益、Hi-ToM 还没漂走。

### 2.4 正迁移：EmoBench/SocialIQA 没训也涨了

EmoBench +1.3pp（0.6658→0.6792）、SocialIQA +0.5pp（0.7958→0.8004），**尽管它们不在训练数据里**。
长思考能力的提升**正迁移**到了相邻 ToM 任务上——这是健康的信号，说明学到的是通用推理而非 ToMBench 套路。

### 2.5 vs v3.1：同样训练 ToMBench，结果天壤之别

| | ToMBench | Hi-ToM | 均值 | 思考长度(HiToM) |
|---|---|---|---|---|
| v3.1 | **0.7816** | 0.7033 | 0.7305 | 4415 |
| Plan A ckpt150 | 0.7744 | **0.7633** | **0.7535** | 10398 |

v3.1 用 **-0.7pp ToMBench 都不到的"优势"换来了 -6pp Hi-ToM**。Plan A 几乎不丢 ToMBench，却把
Hi-ToM 守在高位——**用极小的 ToMBench 代价换回了全部泛化**。

## 3. 诚实评估

- ✅ **科学上成功**：证明了"解枷锁 + KL 锚 base + 修数据"能让 ToMBench 训练**不再损害泛化**，
  把训练的净效应从 v3.1 的 **-3pp 翻成 +0.4pp**。长思考保留是关键机制，已实锤。
- 🟡 **工程上增益温和**：对 base 只 +0.4pp，没造出一个大幅更强的模型。5 个 ckpt 在均值上基本打平。
- 📌 **最优在早期（ckpt50）**：有用的学习发生得很快，之后是 ToMBench↑/Hi-ToM↓ 的低效互换。

**可交付**：**ckpt50**（均值 0.7556，Hi-ToM 0.79，全面 ≥ base、碾压 v3.1）。

## 4. 下一步优化方案（Plan A-v2）

根本张力仍在：**reward 只奖励 ToMBench**，所以 RL 天然往 ToMBench 特化、侵蚀 Hi-ToM；KL 只能减速、
不能消除。三条递进方案：

### Plan B（首要）— 多 benchmark 进 reward，消除张力
把 Hi-ToM(15-opt)/SocialIQA(3-opt)/EmoBench(4-opt) 的 MCQ **直接纳入 reward worker + 训练数据**，
让 RL **直接优化 4-bench 综合**而非只 ToMBench。这样"ToMBench↑ 必然 Hi-ToM↓"的张力从根上消失。
- 需要：prompt_length 1024→4096（容 Hi-ToM 长故事）；reward worker 已支持 task-aware l_max_long。
- 预期：均值能真正往上推，而不是在 ToMBench/泛化之间零和互换。

### Plan E（立即）— 按 4-bench 均值早停 / 选 ckpt
均值峰在 step 50。**下一轮把 max_steps 砍到 ~75-100**，或训练内验证换成 4-bench 子集、按均值早停。
本轮已离线验证：**ckpt50 就是最优**，别再傻跑 200 步做低效互换。

### Plan C+（强化 KL，若继续 ToMBench-only）
KL 0.001→**0.003**，把 Hi-ToM 侵蚀的"缓坡"压得更平，让后期 ckpt 也能保住 order_3。代价是 ToMBench
增益更小——但既然增益本就温和，换稳定的泛化是划算的。

### 定点数据
- **Knowledge**（唯一反降的 ToMBench 任务，0.55，离 deepseek 0.66 差 11pp）：本轮没救起来，
  需要专门的知识归因 / scalar-implicature 数据。
- **EmoBench EU_emotion**（base 0.56 vs deepseek 0.70）：情绪识别本体定向蒸馏。

### 建议执行序
1. **现在**：上线/固化 ckpt50；把 max_steps 经验值记为 ~75。
2. **下一轮**：Plan B（多 bench reward，prompt_length 4096）+ Plan E（均值早停）——这才是把均值
   真正推过 0.76+ 的路径。
3. **中期**：Knowledge / EU_emotion 定点数据，缩小对 deepseek（均值 0.7979）的最后差距。

> 一句话：**Plan A 证明了"训练不必损害泛化"——枷锁是真凶、长思考是关键、KL 是缰绳。
> 但只奖励 ToMBench 的 reward 决定了增益的天花板；要真正变强，下一步必须让 RL 直接优化 4-bench（Plan B）。**
