# Bad-case deep dive — informing next iteration

> 基于 stage2 (5718 题 direct) + deepseek-v4-pro (500 题 direct) 的细粒度对比分析。目标：找出"close-able gap"是什么，"hard ceiling"是什么，以及最有效的下一步策略。

## 1. 核心数字回顾

**Per-task accuracy** (direct):

| Task | baseline | stage1 | stage2 | deepseek | s2-vs-ds gap |
|---|---|---|---|---|---|
| Belief | 0.6725 | 0.6937 | 0.6373 | 0.8000 | **-16.27pp** |
| Desire | 0.5861 | 0.5917 | 0.5861 | 0.6389 | -5.28pp |
| Emotion | 0.6893 | 0.7286 | 0.7012 | 0.7093 | -0.81pp |
| False Belief | 0.7277 | 0.8520 | 0.8385 | 0.8615 | -2.30pp |
| Intention | 0.7500 | 0.7647 | 0.7632 | 0.8136 | -5.04pp |
| Knowledge | 0.4810 | 0.4792 | 0.4879 | 0.6000 | -11.21pp |
| Non-literal Comm | 0.7767 | 0.7674 | 0.7553 | 0.8433 | -8.80pp |

**500-subset overlap analysis** (where deepseek 也在):

- both correct:                  328 (65.6%) — 已掌握
- only deepseek correct (gap):    66 (13.2%) — **理论可关闭的上限**
- only stage2 correct (bonus):    39 (7.8%) — 我们已经赢的
- both wrong (hard ceiling):      67 (13.4%) — 题目难/有歧义/labeling noise

**=> 即使所有 closeable 都拿下，subset500 上限是 +13.2pp** (从 0.734 → 0.866)。Realistic 目标：吃到一半左右 = +6-7pp。

## 2. Stage1 → Stage2 退化（关键洞察）

| Task | stage2 净增 (s1→s2) |
|---|---|
| Belief | **-16** (32 lost, 16 gained) |
| Emotion | **-23** (54 lost, 31 gained) |
| False Belief | **-20** (82 lost, 62 gained) ⚠️ |
| Non-literal Comm | **-18** (100 lost, 82 gained) |
| Knowledge | **+5** ✓ (33 lost, 38 gained) |
| Intention | -1 |
| Desire | -2 |

**stage2 在 Belief、Emotion、False Belief 上大幅损失**，但 stage1 在这些 task 上是大赢家。说明：
- 多训 300 步把 stage1 用 4k 数据稳稳学会的题，反而又"忘"了
- **过拟合不是简单的训练集 vs 测试集**，而是把 robust 信号当 noise 给 unlearn 了

特别看 False Belief 的 82 个退化（Sally-Anne 经典题）：

```
[False_Belief_Task_17_zh, gold=B, s1=B ✓, s2=A ✗]
故事: 小丽和悠悠在地下室，他们看到手提箱、背包和储物柜。他们发现帽子在手提箱里，
      然后悠悠离开，小丽把帽子移动到了储物柜。
问: 在悠悠回来后，悠悠会认为小丽会在哪里找帽子？

stage2 推理: "悠悠知道小丽移动了帽子，所以她会知道帽子现在在储物柜里。" ← 错！
            (悠悠当时已离开，根本不知道小丽移动了)
gold B = 手提箱 (因为悠悠没见到搬动)
```

**stage2 把 false-belief 反推搞反了**: 它认为"悠悠知道小丽移动了"，混淆了 1st-order 和 2nd-order belief。stage1 还能答对——多训 300 步反而把这个能力训坏。这是经典的 RL **mode collapse**：模型学到了"猜东西在 storage_locker" 的浅启发，而不是真正追踪谁知道什么。

## 3. Close-able 错误的分类（66 个 + 类外推 5718）

按错误**原因**分类（不是 task）：

### 3.1 信念追踪错误 (Belief / False Belief / Intention) — ~25-30%

**模式 A — 把"应该不知道"错当"应该知道"** (False Belief Sally-Anne 反例)
```
[False_Belief_Task_320_zh, gold=C 苹果, s2=D 香蕉]
韩梅梅找到标签"香蕉"的公文包，打开发现是苹果，放回。小明进入但不知内容。
问：韩梅梅期望小明打开后会找到什么？

stage2: "韩梅梅知道标签是香蕉，所以她可能以为里面是香蕉" ← 自己绕进去了
gold C = 苹果 (韩梅梅知道里面真实是苹果，她预期小明也会发现苹果)
```

**模式 B — 把意图信号当字面信号** (Hinting / Belief)
```
[Hinting_Task_Test_55_zh, gold=A, s2=B]
小张问怎么准备演讲稿，小洪说"隔壁新开奶茶店听说不错，这会嘴太干了"
问：小洪真正想说什么？

stage2: "用奶茶店作为借口表示不想帮"
gold A = 想让小张请喝奶茶（暗示而非拒绝）
```

**模式 C — 不能识别 hidden coordination / 排挤**
```
[Ambiguous_Story_Task_187_zh, gold=A, s2=C]
晓明和柳翠提前策划，冬儿走进时晓明故意说"那景点有食物中毒"，
柳翠看着晓明偷偷露出微笑，晓明点了点头。
问：冬儿怎么想？

stage2: "晓明在关心他的安全"
gold A = 失望，感受到不欢迎 (model misses the conspiratorial wink-and-nod)
```

### 3.2 字面解读 Scalar Implicature (Knowledge) — ~15-20%

```
[Scalar_Implicature_Test_107_zh, gold=A 15人, s2=B 16人]
20人排队，小李说"大多数是汉堡，小部分是三明治，几乎没人买沙拉"。
小张数发现只有4人三明治。问推测汉堡多少？

stage2: "20-4=16人" ← 纯数学
gold A = 15 (理解 "几乎没人买沙拉" 是 ~1，所以汉堡 ≈ 20-4-1=15)
```

**Stage2 完全没学会 "almost no" / "几乎没有" 的 pragmatic implicature**。这一类 Scalar Implicature 题在 closeable 6 个 Knowledge 错误里占 **5/6**。

### 3.3 Faux-pas 识别失误 (Non-literal Comm) — ~30-35%

```
[Faux-pas_Recognition_Test_128_en, gold=A, s2=B]
李先生在王阿姨家说"你的酱排骨好吃，但我老婆做的更好"。王阿姨保持微笑回应。
问：有人说不该说的话吗？

stage2: "王阿姨没生气，所以没有失礼"  ← 把"对方没生气"当"没失礼"
gold A = 有人失礼 (李先生当着主人面赞美自家配偶手艺，是经典 faux-pas)
```

**模式**: stage2 用"对方没明显反应"来推"没失礼"，但 faux-pas 的定义是说话人**无心**冒犯，所以 victim 通常会礼貌掩饰。模型不懂这个 social convention。

```
[Faux-pas_Recognition_Test_165_zh, gold=D 没有人失礼, s2=C]
公司给小赵生日惊喜，小陈递礼物说"希望喜欢这小心意"，小赵谢谢，
刘哥说"现在享受蛋糕的时刻"。问哪句不合适？

stage2 杜撰: "刘哥可能在蛋糕没拿出来时就说太早了" ← 凭空想象问题
gold D = 没人失礼
```

**反向问题**: stage2 倾向把任何题都解读为"有失礼"。看 confusion matrix:

| gold↓ pred→ | A | B | C | D |
|---|---|---|---|---|
| A | 0 | 307 | 98 | 109 |
| B | **204** | 0 | 87 | 132 |
| C | 126 | 110 | 0 | 105 |
| D | 70 | 107 | 108 | 0 |

A → B 错 (307 例) + B → A 错 (204 例) 是最大错误源，加在一起 ~33% 的错误。对 faux-pas 题这就是"过度归因"vs"漏归因"的对换。

### 3.4 情感归因错误 (Emotion / Desire) — ~15-20%

```
[Moral_Emotions_33_zh, gold=B 矛盾担忧, s2=A 满意]
张伟拿捐款给班级用，王芳知道但没告诉李华，因为认为对班级有益。
问王芳情绪？

stage2: "她认为自己做得对，所以满意"
gold B = 矛盾担忧 (她在"班级利益 vs 道德"间挣扎，moral conflict 才符合人的真实状态)
```

```
[Unexpected_Outcome_Test_41_zh, gold=B 内疚, s2=D 被侵犯]
林婷以为同学嘲笑她偷看日记，但实际日记本在自己柜子里，同学也没看。
问林婷情绪？

stage2: "她以为被人看了，所以感到被侵犯" ← 没读到"实际"
gold B = 内疚 (因为她错怪了同学)
```

模型在涉及**"我之前误会了别人"**的情境下不会更新归因。这是高阶 ToM (我对你的看法的看法)。

### 3.5 Discrepant Desire (Desire) — 100% 的 closeable Desire 错误

3 个 closeable Desire 题，**全部**是 "A 是素食者，邀请爱吃肉的 B 参加活动":

```
[Discrepant_Desires_9_en, gold=A 素食烹饪课, s2=B 烧烤派对]
林峰是素食者，陈楠爱吃肉。林峰邀请陈楠参加活动。问什么活动？

stage2: "林峰会选陈楠喜欢的烧烤" ← 错位推理：选 host 的活动，不是 guest 的
gold A = 素食烹饪课 (host 邀请别人参加 host 想做的事)
```

模型默认 "邀请方会迁就被邀请方"——这是常识 prior，但 ToMBench 这一类题特意考"邀请方坚持自己 desire"的设定。

## 4. 真正的 "hard ceiling" (67 个 both-wrong)

按 task 分布:

| Task | both-wrong / n | rate |
|---|---|---|
| Knowledge | 11/35 | **31.4%** |
| Emotion | 20/86 | 23.3% |
| Desire | 8/36 | 22.2% |
| Belief | 3/20 | 15.0% |
| Intention | 6/59 | 10.2% |
| Non-literal Comm | 10/134 | 7.5% |
| False Belief | 9/130 | 6.9% |

**Knowledge 的 31.4% both-wrong** 让 task ceiling 显著低于其他 task。例如：

```
[Scalar_Implicature_Test_117_zh, gold=D 55本, deepseek=None, s2=A 60本]
"80本小说，大部分是散文，一些是历史书，几乎没有科幻"
读者发现历史20本。推测散文多少？

gold D = 55 (即 80 - 20 历史 - 5 假定科幻 = 55)
但 gold = 55 vs A=60 / C=62 / B=65，差异极小，且推理路径"几乎没有=5"很主观。
deepseek 完全 timeout 输出 None 也说明它觉得 ambiguous。
```

**这类题不是模型不够强，是 ground truth label 本身依赖一个不明显的 implicature 假设。** 我们硬训练只会过拟合 ToMBench-specific 的标注偏见。

## 5. 模型行为特征

- **100% 题都被 thinking** (stage2 cot/direct 都是) — 训练把 think tag 烙进了所有响应
- **错误响应平均 647 字符** vs 正确响应 492 字符 — 错的时候想更多还是更错
- **回答长度上界 8943 字符** — 部分错题响应过长被 truncate，导致最后那个 \\boxed{X} 没输出（少数 None pred 的来源）

## 6. 策略矩阵 — 哪些值得修

| 错误类型 | 数量(%) | 修复难度 | ROI | 建议 |
|---|---|---|---|---|
| Faux-pas A↔B 互换 | 33% | 中 | **高** | 合成专项数据 |
| Scalar implicature | 15% | 低 | **高** | 合成 + careful prompt template |
| False Belief 反推混乱 | 10-15% | 高 | 中 | 数据清洗 + KL penalty 调高 |
| Hinting (intention) | 10% | 中 | 中 | 合成数据 |
| Discrepant Desire | 5% | 中 | 低 | 数据少 + ToMBench-specific 偏见 |
| Hard ceiling (label noise) | 13% | — | — | 接受 |

## 7. 下一步设计建议

### 方案 A: 数据合成 (推荐)

针对四类错误：
1. **Faux-pas 题 800 条**: 涵盖 "speaker 失礼但 listener 礼貌掩饰" 与 "没人失礼" 的双向，强迫模型用"speaker intent + social norm"而不是"listener reaction"判断
2. **Scalar Implicature 题 400 条**: "几乎没有" / "almost no" / "rarely" / "barely any" 的 pragmatic implicature，配上数值题让模型学到"几乎没有 ≈ 0-3 之间"
3. **Hinting 题 300 条**: 间接请求 / 暗示需求的 4 选 1，让模型学会"看上下文，不要只看字面拒绝"
4. **第二阶信念题 300 条**: explicit "A 知道 B 不知道 C" 的题，配清晰的逻辑路径标注

合计 ~1800 条新数据，加上现有 7911 条 = ~9700 条。

### 方案 B: 训练参数调整

- **降低 step 数**: 250 步早停（stage2 的 val plateau 起点）— 既拿 stage2 cot 恢复，又不失 stage1 direct 优势
- **提高 KL penalty**: 现 add_token_level_kl=false，可以打开它保留 base 模型的多样性
- **去掉过短 response cap**: response_length 256 太小，错题里 8% 都因为 reasoning 太长被 truncate

### 方案 C: 协议路由（零训练成本，立即收益）

按 task 用最强协议：
- False Belief / Emotion / Intention → stage1 direct (False Belief 0.852)
- Knowledge / Belief / Desire / Non-literal → stage2 cot (cot 上 0.754)

但天花板大概 +0.5-1pp，不如方案 A 彻底。

## 8. 最佳次轮训练方案 (综合 A+B)

1. **数据**: 现有 7911 + 合成 1800 = ~9700 (Phase 1 优先做 Faux-pas + Scalar)
2. **训练设置**:
   - `max_steps: 300` (stage2 的 val peak 在 step 200-250)
   - `save_steps: 150 + 300` (中间存一个，避免 overfit)
   - `add_token_level_kl: true` (防 mode collapse)
   - `response_length: 384` (从 256 → 384，让长 reasoning 不被 truncate)
   - 其他保持 stage2 同配置
3. **数据混合策略**: 
   - 老 7911 + 新 1800 同时训
   - 用 `tag` 字段标 task，让 ROLL 按 task 均匀采样（避免 Non-literal Comm 数据被淹没）
4. **预期增益**:
   - Faux-pas 错误从 33% 降到 20% → Non-literal Comm task +2-3pp
   - Scalar 错误大部分修复 → Knowledge task +2-4pp
   - 早停 + KL → 保留 False Belief 的 0.852 不退化
   - **预估 stage3 best ≈ 0.78 direct** (距 deepseek 仅 1pp 差距)

