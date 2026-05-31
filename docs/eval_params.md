# 评测参数规范与版本演进

本文档记录 ToM 评测的采样参数规范、已知问题及修复历史。

---

## 当前推荐：v3 参数（2026-05-31）

**适用场景**：需要最高精度的公平对比评测（消除所有已知截断）

### 采样参数

```python
def sampling_params_for_v3(protocol: str) -> dict:
    """v3 评测参数：消除 base 长 thinking 截断 + v10 direct ZH 截断"""
    if protocol == "direct":
        return dict(
            temperature=0.0, 
            top_p=1.0, 
            max_tokens=128,        # ↑ 从 v2 的 64，修复 v10 ZH Scalar_Implicature 推理截断
            n_samples=1, 
            enable_thinking=False
        )
    if protocol == "direct_think":
        return dict(
            temperature=0.0, 
            top_p=1.0, 
            max_tokens=16384,      # ↑ 从 v2 的 8192，容纳 base 最长 thinking (37k chars)
            n_samples=1, 
            enable_thinking=True
        )
    if protocol == "cot":
        return dict(
            temperature=0.6, 
            top_p=0.95, 
            max_tokens=16384,      # ↑ 从 v2 的 8192，消除 base 0.23% 截断
            n_samples=1, 
            enable_thinking=True
        )
    if protocol == "del_tom":
        return dict(
            temperature=0.7, 
            top_p=0.95, 
            max_tokens=16384,      # ↑ 从 v2 的 8192，8 samples 任一截断都影响投票
            n_samples=8, 
            enable_thinking=True
        )
```

### vLLM 部署配置

```bash
# docker-compose.yml 或 vllm serve 参数
--max-model-len 32768              # ↑ 从 v2 的 16384，容纳 16384 output + 16384 input
--gpu-memory-utilization 0.85      # 保持
--tensor-parallel-size 1           # 保持（8B 单卡足够）
```

### 预期改善

| 协议 | v2 截断率 (base) | v3 预期截断率 | 预期 acc 提升 |
|---|---|---|---|
| direct | 0.00% | 0.00% | - |
| direct_think | 0.68% | **0.00%** | +0.3pp |
| cot | 0.23% | **0.00%** | +0.1pp |

**v10 direct 修复**：8/5718 ZH 题不再截断 → +0.14pp

**总收益**：base 公平性 +0.4pp，v10 公平性 +0.14pp

### 成本评估

- **GPU 内存**：32768 ctx 需要 ~24GB (Qwen3-8B FP16)，H800 80GB 足够
- **推理速度**：max_tokens 翻倍但实际输出长度不变（只是上限），速度影响 <5%
- **缓存失效**：v2 → v3 需要重新跑（max_tokens 变化导致 cache key 不同）

---

## 历史版本

### v2 参数（2026-05-28 ~ 2026-05-30）

**修复内容**（相对 v1）：
1. max_tokens 不足：cot 4096→8192, direct_think 2048→8192
2. ToMBench ZH 选项前缀重复：自动 strip `A./B./C./D.` 前缀
3. DashScope reasoning_content 收集：stream 中 hidden thinking 注入 content
4. vLLM max_model_len：8192→16384

**已知问题**：
- base/cot 仍有 0.23% 题被截断（max_tokens=8192 不够，base 最长 33-37k chars）
- base/direct_think 0.68% 题被截断
- v10/direct 8/5718 ZH 题被截断（64 tokens 内开始中文推理）

**实测效果**（v2 vs v1）：
- base/Hi-ToM/direct_think: 0.5150 → 0.6567 (+14.17pp) ⭐⭐
- base/Hi-ToM/cot: 0.6100 → 0.7133 (+10.33pp) ⭐⭐
- base/ToMBench/cot: 0.7387 → 0.7459 (+0.72pp)

**报告**：`experiment/qwen3-8b-full-eval/output/full_eval_report_qwen3-8b_2026-05-29.md`

### v1 参数（2026-05-27 ~ 2026-05-28）

**采样参数**：
- direct: T=0, max_tokens=64, thinking=false
- direct_think: T=0, max_tokens=2048, thinking=true
- cot: T=0.6, max_tokens=4096, thinking=true
- del_tom: T=0.7, max_tokens=4096, thinking=true, n=8

**vLLM**：max_model_len=8192

**已知问题**：
- base/cot 4.42% 题被截断（max_tokens=4096 不够）→ 损失 ~2pp
- base/direct_think 3.52% 题被截断
- ToMBench ZH 选项前缀重复（"A. A. ..."）
- DashScope reasoning_content 未收集

**报告**：`experiment/qwen3-8b-full-eval/output/full_eval_report_qwen3-8b_2026-05-28.md`

---

## 协议定义

### direct (no-think)

**目的**：测试模型"裸答"能力（无 thinking）

**参数**：
- `enable_thinking=false`（关键）
- `temperature=0.0`（greedy）
- `max_tokens=64`（v2）或 `128`（v3，修复 v10 ZH 截断）

**System prompt**：
```
You are a careful reader answering a multiple-choice theory-of-mind question. 
Read the story and the question carefully, then output ONLY your final answer 
in the format \boxed{X} where X is one of A, B, C, D. 
Do not include any explanation, reasoning, or extra text.
```

**Extractor**：`extract_direct` — 第一个 `\boxed{X}`，否则首个有效字母

### direct_think (default-think)

**目的**：与历史 production_frozen/8b/v1.0 (0.7450) 对齐

**参数**：
- `enable_thinking=true`（关键差异）
- `temperature=0.0`（greedy）
- `max_tokens=2048`（v1）→ `8192`（v2）→ `16384`（v3）

**System prompt**：与 direct 相同（"output ONLY"），但模型实际会生成 `<think>...</think>\boxed{X}`

**Extractor**：`extract_cot` — 最后一个 `\boxed{X}`，否则末 200 字符内最后一个有效字母

**为什么要分 direct 和 direct_think**：Qwen3-8B chat template 中 `enable_thinking` 默认 true，历史评测脚本根本没传该参数，所以即使 prompt 写 "ONLY"，模型仍走 thinking 路径。本次 direct 协议显式 `enable_thinking=false`（衡量"裸答"能力），direct_think 显式 `enable_thinking=true`（与历史对齐）。

### cot

**目的**：测试 chain-of-thought 推理能力

**参数**：
- `enable_thinking=true`
- `temperature=0.6`（sampling，引入多样性）
- `top_p=0.95`
- `max_tokens=4096`（v1）→ `8192`（v2）→ `16384`（v3）

**System prompt**：
```
You are a careful reader answering a multiple-choice theory-of-mind question. 
Think step by step about the mental states of the characters, 
then output your final answer in the format \boxed{X} where X is one of A, B, C, D. 
Put your final \boxed{X} on the last line.
```

**Extractor**：`extract_cot` — 最后一个 `\boxed{X}`，否则末 200 字符内最后一个有效字母

### del_tom (8-vote)

**目的**：多数投票提升鲁棒性

**参数**：
- `enable_thinking=true`
- `temperature=0.7`（更高温度增加多样性）
- `top_p=0.95`
- `max_tokens=4096`（v1）→ `8192`（v2）→ `16384`（v3）
- `n_samples=8`（关键）

**System prompt**：与 cot 相同

**Extractor**：每个 sample 各跑 `extract_cot` → 多数投票，平局取字母序最小

---

## 已知陷阱与注意事项

### 1. DashScope API 的 enable_thinking 行为不同

**问题**：DashScope qwen3-8b API 在 `temperature=0.0` + "output ONLY" system prompt 下，会**忽略 `enable_thinking=true` 信号**，直接给 `\boxed{X}` 不 thinking。

**影响**：DashScope/Hi-ToM/direct_think 只有 11.5% 题目真正启用 thinking，导致 acc 0.5483（比 base 0.6567 低 10pp）。

**根本原因**：DashScope 服务端让模型自主决定是否开 `<think>`（softer hint），而本地 vLLM 在 chat template 里硬插入 `<think>\n` 强制开始 thinking。

**修复方案**（如果要让 DashScope/direct_think 与本地等价）：
- 改用 T=0.6（强制 thinking）
- 或改 system prompt 不带 "output ONLY"
- 或往 user message 末尾追加 `\n<think>\n` 手动开 thinking 标签

### 2. base 模型输出长度远超 v10

**数据**（v2 cache，Hi-ToM/cot）：
- base: median 9732 chars, max 35679 chars
- v10: median 3834 chars, max 21204 chars
- DashScope: median 10284 chars, max 34298 chars

**原因**：v10 经过 RLVR 训练后输出更紧凑；base 和 DashScope（同一权重）都保留了原始"啰嗦"风格。

**影响**：v2 的 max_tokens=8192 对 base 不够（0.23-0.68% 截断），对 v10 足够（<0.05% 截断）。

### 3. v10 在 direct 协议下不遵守 "output ONLY"

**问题**：v10 在 ZH "Scalar_Implicature_Test" 题目上，即使 `enable_thinking=false` + system prompt 说 "output ONLY \boxed{X}"，仍开始用中文给推理（`根据故事...`），最终在 64 token 限制下被截断没给到 `\boxed{X}`。

**影响**：v10/tombench/direct 8/5718 题 pred=None（0.14%）

**原因**：v10 RLVR 训练后的副作用，模型学会了即使在 thinking off 时也"思考"。

**修复**：v3 direct max_tokens 64→128

### 4. ToMBench ZH 选项前缀重复

**问题**：ToMBench 中文数据的 `opt_a` 字段值已含 `"A. "`，user prompt 模板再加一次 `"A. "`，导致 ZH 渲染出 `"A. A. ..."`。

**影响**：v1 评测 prompt 不干净，但与 production_frozen 历史评测一致（数据集 `messages` 字段也是双前缀）。

**修复**：v2 在 `prompts.py:_strip_letter_prefix()` 中用正则 `^\s*([A-D])[.．、:：]\s*` 自动 strip 已有字母前缀（EN 不受影响）。

**实测影响**：DashScope/ToMBench/direct_think 从 0.7011 → 0.7055 (+0.44pp)

### 5. 多 boxed 候选的 extractor 行为

**问题**：base 长 thinking 可能多次尝试答案（`\boxed{B} ... \boxed{C}`），extractor 取最后一个。

**数据**（v2）：
- base/cot/tombench: 39/5718 (0.68%) 多 boxed，其中 2 个不同字母
- v10/cot/tombench: 2/5718 (0.03%) 多 boxed，全部同字母

**extractor 逻辑**：取最后一个 `\boxed{X}`（假设模型最后的答案是最终决定）

**验证**：人工检查 inconsistent 案例，extractor 行为合理（模型明确说 "Correct Final Answer" 时取最后）

---

## 评测一致性检查清单

在对比不同模型时，确保以下项完全一致：

- [ ] 采样参数（temperature, top_p, max_tokens, enable_thinking）
- [ ] Prompt 模板（system + user，包括 ZH 前缀处理）
- [ ] Extractor 逻辑（direct vs cot 风格）
- [ ] Backend 版本（vLLM 版本、TP size、max_model_len）
- [ ] 数据集版本（ToMBench/Hi-ToM 的 question_id 集合）
- [ ] 截断率对比（no-boxed 比例，确认公平性）
- [ ] 预测一致率（同 protocol 下不同模型的 pred 一致率，direct 应 >90%）

---

## 参考

- v1 报告：`experiment/qwen3-8b-full-eval/output/full_eval_report_qwen3-8b_2026-05-28.md`
- v2 报告：`experiment/qwen3-8b-full-eval/output/full_eval_report_qwen3-8b_2026-05-29.md`
- v2 审查：本文档 § 历史版本 / v2 参数
- DashScope enable_thinking 行为分析：v2 报告 § 8.9 + 本文档 § 已知陷阱 #1
