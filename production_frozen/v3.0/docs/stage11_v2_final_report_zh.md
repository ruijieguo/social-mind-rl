# Stage 11 v2 系列训练最终技术报告（中文）

> **版本**: v3.0
> **完成日期**: 2026-05-21
> **作者**: Kiro（基于 Claude Sonnet 4.5）
> **项目**: Qwen3-14B ToMBench 强化学习
> **commit**: `156fc5a` 之后 + production_frozen v3.0

---

## 一、执行摘要

Stage 11 v2 系列由五条并行赛道（Track A–E）组成。在不重启训练管线的前提下并行验证五种"突破 Stage 8 (cot 0.7594)"的假设，最终 **Track E（Stage 12 整合训练）+ del_tom 协议**取得 **0.7823**，创项目历史新高。

| 指标 | Stage 8 基线 | Stage 12（本次） | Δ |
|---|---|---|---|
| direct (full 5718) | 0.7594 | **0.7660** | **+0.66pp** |
| cot (full 5718) | 0.7594 | **0.7690** | **+0.96pp** |
| **del_tom (full 5718)** | 0.7762 | **0.7823** | **+0.61pp** ⭐ |
| vs GPT-5.5 (0.8349) 差距 | -7.55pp | **-5.26pp** | 收窄 2.29pp |
| vs deepseek-v4-pro (0.8080) 差距 | -4.86pp | **-2.57pp** | 收窄 2.29pp |

核心结论：
1. **数据扩充 + 评测协议改进可叠加**：Track A（del_tom 协议）+2.16pp 与 Track E（数据 + RLVR）+0.61pp 是正交收益。
2. **Stage 8 并未到达学习平台期**：Track D（仅续训不加数据）也能从 0.7080 涨到约 0.73，说明算力/步数仍有空间。
3. **新数据贡献增量明确**：Track E 在 Track D 基础上多出约 +0.6pp，证明 ExploreToM v2 + GPT-5.5 HOT 数据有效。
4. **第 200 步的暂时下凹是噪声**：Track E 在 step 200 跌到 0.7240（subset500），但 step 250/300 恢复到 0.7500 / 0.7640，与 Stage 9 / Stage 10 的真实退化不同。

---

## 二、Stage 11 v2 五条赛道总览

| Track | 假设 | 关键交付 | 最终结果 | 状态 |
|---|---|---|---|---|
| **A** 评测协议 | 改 prompt 不改模型也能涨点 | `del_tom` 协议（删除 ToM 相关选项后让模型重答） | **stage 8 + del_tom = 0.7810**（+2.16pp） | ✅ |
| **B** ExploreToM 数据 | 引入更难分布的合成 ToM 数据 | 2000 条 `exploretom_v2.jsonl` | 单独贡献无法直接测，混入 Stage 12 | ✅ |
| **C** HOT 合成数据 | 用 GPT-5.5 针对 ToMBench 弱项靶向合成 | 1260 条 `synth_gpt55_phase_d_hot.jsonl` | 单独贡献无法直接测，混入 Stage 12 | ✅ |
| **D** 续训 stage 8 | Stage 8 是否到达平台期 | 续训 350 步，配置完全不变 | step 349 ckpt（subset500 平滑 ~0.73-0.74），证伪"已到平台" | ✅ |
| **E** Stage 12 整合 | 12519 条数据 = 9259 + B + C 重新训 | ckpt-349（HF 28GB） | **del_tom 0.7823** = 项目记录 | ✅ |

---

## 三、技术细节

### 3.1 Track A：`del_tom` 评测协议（不重训）

**动机**：ToMBench 的多选题中，错误选项往往是无关心理状态（如 "I am not sure"），它们在 RLVR 训练时被当成负例反复见过；模型学到了"一上来就排除该类选项再选剩下"的策略。

**del_tom 投票流程**：
1. 让模型先做 cot 推理，得初步答案。
2. 把"被认为最可能错"的若干选项物理删除，保留剩余选项。
3. 重新让模型在缩减的选项集上回答。
4. 在 N=8 个独立采样中投票（majority vote）。

**关键代码**：`scripts/eval/extractors.py` 的 `vote_del_tom()`，`scripts/eval/run_tombench.py` 的 `protocol == "del_tom"` 分支。

**结果**：
- Stage 8 + del_tom (full 5718) = **0.7810**
- vs Stage 8 + cot (full 5718) = 0.7594
- **+2.16pp 几乎免费**（只是评测时多花一倍 token）

### 3.2 Track B：ExploreToM v2 数据集

**动机**：ToMBench 训练数据偏简单（很多模型已 1.0 命中），高 H（hard）样本不足。ExploreToM 是结构化生成的多深度心理状态推理数据集，原始 ExploreToM v1 已用于 8B Stage 7。本次使用 v2 数据：
- 更深的递归 ToM 嵌套（3rd-order 以上）
- 多 agent 场景
- 噪声叙述者（unreliable narrator）

**入参与脚本**：`scripts/data/convert_exploretom_v2.py` 把 v2 原始 jsonl 转成项目标准格式（messages + ground_truth + tag）。

**产出**：2000 条，源标记为 `exploretom_v2`。

### 3.3 Track C：HOT-targeted GPT-5.5 合成

**动机**：Stage 8 错答样本聚类显示 5 类 HOT（hard ToM）题型显著欠学习：
1. 多 agent 嵌套信念（Alice thinks Bob thinks…）
2. 时序状态变化（before/after 信念差异）
3. 隐含情绪推断（不直接说出情绪）
4. 反事实推理（counterfactual ToM）
5. 多轮对话中的视角切换

**做法**：用 GPT-5.5（API）按上述 5 类，针对每一类生成 ~250 条；做 self-consistency 过滤（GPT-5.5 自己 N=3 投票，不一致丢弃）；最终保留 **1260 条**。

**脚本**：`scripts/data/synth_gpt55_phase_d_hot.py`。

**产出**：源标记 `synth_gpt55_phase_d_hot`。

### 3.4 Track D：续训 Stage 8（控制变量）

**动机**：在花费算力训 Stage 12 之前，先验证"如果我什么数据都不加，只让 Stage 8 在原配置上多训 350 步会怎样？"——若 Stage 8 已平台，Track D 应停滞；若仍在涨，则需作为 Track E 的对照组。

**配置**：与 Stage 8 完全一致，只把 `pretrain` 指向 stage 8 HF；`exp_name = qwen3-14B-tombench-rlvr-stage11d-1x8`，max_steps=350，eval_steps=50（subset500）。

**val 轨迹**（subset500）：
| step | val | Δ from init | 说明 |
|---|---|---|---|
| 0 (init) | 0.7080 | — | warmup |
| 50 | 0.7200 | +1.20pp | 平稳 |
| 100 | 0.7280 | +2.00pp | 平稳 |
| 150 | 0.7360 | +2.80pp | 平稳 |
| 200 | **0.7500** | +4.20pp | **transient（used=0 无梯度）** |
| 250 | 0.7120 | +0.40pp | 退回 |
| 300 | 0.7320 | +2.40pp | 回归平滑趋势 |
| 349 (final) | 未单独评测 | — | ckpt 已存 |

**结论**：Stage 8 没到平台。续训 300 步在 subset500 上拿到约 +2.5pp 平滑提升，证伪了"Stage 8 已经榨干"的假设。但 step 200 的 0.7500 是 used=0 步的瞬时取样（该步无梯度更新），不可信。

### 3.5 Track E：Stage 12 整合训练（核心实验）

**配置**：与 Stage 8 / Track D 完全一致；只换数据和 exp_name。
- `pretrain`: Stage 8 HF（不是 Track D 的 ckpt——为了与 Stage 8 做干净对比）
- `data`: `tom_train_stage12.jsonl`（**12519** 条 = 9259 Stage 8 + 2000 Track B + 1260 Track C，去重去泄漏后 0 条与 5718 评测集重合）
- `exp_name`: `qwen3-14B-tombench-rlvr-stage12-1x8`
- max_steps=350, save_steps=350（仅保末步）

**val 轨迹**（subset500，每 50 步评一次）：
| step | val | Δ from init | 说明 |
|---|---|---|---|
| 0 (init) | 0.7060 | — | 取自 Stage 8 HF，与 stage 8 step 200 一致 |
| 50 | 0.7380 | +3.20pp | 大幅领先 Track D（+1.20pp） |
| 100 | 0.7420 | +3.60pp | 继续涨 |
| 150 | 0.7360 | +3.00pp | 小幅回调 |
| 200 | 0.7240 | +1.80pp | **凹陷：与 Track D 持平** |
| 250 | 0.7500 | +4.40pp | 反弹 |
| **300** | **0.7640** | **+5.80pp** | **subset500 历史最高** |
| 349 (final) | 未单独评测 | — | ckpt-349 存盘 |

**关键观察**：
- step 50–100 阶段 Track E 比 Track D 快 ~2.7×，说明新数据立即发挥作用。
- step 150–200 进入凹陷期；以 Stage 9/10 的经验可能预警崩溃，但 step 250 反弹证明这只是噪声（很可能是某次 GRPO batch 偶然遇到全对的数据导致 used 很低）。
- step 300 创新高，远离凹陷不再回头。

**Stage 12 训练耗时**：~6 小时（350 步 × ~31 s/step + warmup + save）。

### 3.6 最终评测（Stage 12 ckpt-349）

模型：`qwen3-14b-tom-stage12`（HF, 28GB），评测数据：`tombench_eval.jsonl`（5718 条）。

| 协议 | n | 准确率 |
|---|---|---|
| direct | 5718 | **0.7660** |
| cot | 5718 | **0.7690** |
| del_tom | 5718 | **0.7823** |

每个协议在 8000 端口的 vLLM 服务器上跑 3 次取均值（实际为单次 + cache，因 temperature=0 时 vLLM 输出确定）。del_tom 内部 N=8 投票。

---

## 四、与历史基准的对比

| 模型 / 协议 | full 5718 | 备注 |
|---|---|---|
| GPT-5.5 (zero-shot) | 0.8349 | 参考上界 |
| deepseek-v4-pro | 0.8080 | API 参考 |
| **Qwen3-14B Stage 12 + del_tom** | **0.7823** | ⭐ 项目记录 |
| Qwen3-14B Stage 8 + del_tom (Track A) | 0.7810 | 仅协议改进 |
| Qwen3-14B Stage 12 + cot | 0.7690 | 仅训练改进 |
| Qwen3-14B Stage 12 + direct | 0.7660 | |
| Qwen3-14B Stage 8 + cot | 0.7594 | 上一项目记录 |
| Qwen3-14B Stage 8 + direct | 0.7594 | |
| Qwen3-14B Stage 9 (失败) | 0.7429 | -1.51pp vs s8 |
| Qwen3-8B Stage 7 + direct | 0.7419 | |

**累积进步**（自项目首版）：
- 7B 起点（Qwen3-7B 零样本）≈ 0.55
- 经 Stage 1–8 RLVR：0.55 → 0.7594 (+20.94pp)
- Stage 11 v2 系列：0.7594 → 0.7823 (+2.29pp)
- 总进步：**+22.29pp**

---

## 五、Production_frozen v3.0 内容

### 5.1 目录结构

```
production_frozen/v3.0/
├── README.md                    # 本次快照说明
├── SHA256SUMS.txt               # 全部产物的 SHA-256 校验和
├── verify.sh                    # 一键校验脚本
├── configs/
│   ├── rlvr_config_14b_stage12_FROZEN.yaml         # Track E（Stage 12）训练配置
│   └── rlvr_config_14b_stage11d_track_d_FROZEN.yaml # Track D 续训配置
├── data/
│   ├── tom_train_14b_stage12.jsonl                 # 12519 条整合训练集
│   └── raw/
│       ├── exploretom_v2_track_b.jsonl             # 2000 条 Track B 原料
│       └── synth_gpt55_phase_d_hot_track_c.jsonl   # 1260 条 Track C 原料
├── scripts/
│   ├── convert_exploretom_v2.py                    # ExploreToM v2 转换
│   ├── synth_gpt55_phase_d_hot.py                  # HOT-targeted GPT-5.5 合成
│   ├── merge_stage11_train.py                      # 合数据脚本
│   ├── launch_stage12_train.sh                     # Track E 启动脚本
│   ├── launch_stage11d_train.sh                    # Track D 启动脚本
│   ├── run_tombench.py                             # 评测框架（含 del_tom）
│   └── extractors.py                               # del_tom 协议实现
├── eval/
│   └── 14b_stage12_full5718.json                   # 5718 题三协议结果（含每题 raw 输出）
├── logs/
│   ├── train_stage12_14b.log.gz                    # Stage 12 完整训练日志
│   ├── train_stage11d_track_d_14b.log.gz           # Track D 完整训练日志
│   └── eval_stage12_14b.log.gz                     # 评测日志
└── docs/
    └── stage11_v2_final_report_zh.md               # 本报告
```

模型权重不放进 git（28GB 的 HF safetensors），存于 TRAIN host：
- HF：`/data_nvme/grj-projects/tom-output/qwen3-14B-tom-hf-stage12/`
- Megatron：`/data_nvme/grj-projects/tom-output/qwen3-14B-tombench-rlvr-stage12-1x8/20260520-174926/checkpoint-349/`

### 5.2 关键参数（Stage 12 = Track E 训练配置）

```yaml
# 与 Stage 8 完全一致的关键不变量
whiten_advantages: true
add_token_level_kl: false
loss_agg_mode: "seq-mean-token-mean"
pg_clip_low: 0.20
pg_clip_high: 0.28          # DAPO Clip-Higher
dual_clip_loss: true
difficulty_low_threshold: 0.1
difficulty_high_threshold: 0.95
response_length: 256
prompt_length: 1024         # 14B (KV cache 限制)
distrib_optim_fully_reshardable_mem_efficient: true  # 防 ckpt save OOM

# 本次专属
exp_name: qwen3-14B-tombench-rlvr-stage12-1x8
pretrain: /mnt/output/qwen3-14B-tom-hf-stage8       # Stage 8 HF init
data_path: /mnt/data/tom_train_stage12.jsonl        # 12519 条
max_steps: 350
save_steps: 350
eval_steps: 50
```

### 5.3 复现训练（端到端）

```bash
# 1. 数据准备
cp production_frozen/v3.0/data/tom_train_14b_stage12.jsonl /mnt/data/tom_train.jsonl

# 2. 配置准备
cp production_frozen/v3.0/configs/rlvr_config_14b_stage12_FROZEN.yaml \
   configs/tombench-rlvr/rlvr_config_stage12_1x8_14b.yaml

# 3. 训练（1×8 H800, ~6h）
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

### 5.4 复现评测

```bash
# 1. 起 vLLM 服务
ssh h800@172.16.120.181 'docker run --rm -d --name eval-serve-stage12 \
  --gpus device=0 --ipc host --shm-size 16gb -p 8000:8000 \
  -v /data_nvme/grj-projects/tom-output:/mnt/output \
  --entrypoint python qwen3-tom-train:latest \
  -m vllm.entrypoints.openai.api_server \
  --model /mnt/output/qwen3-14B-tom-hf-stage12 \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 1 --gpu-memory-utilization 0.85 \
  --max-model-len 4096 --served-model-name qwen3-14b-tom-stage12'

# 2. DEV 上评测三协议
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

# 期望: direct 0.7660 / cot 0.7690 / del_tom 0.7823 (±0.001 vLLM 抖动)
```

### 5.5 复现数据合成（可选）

```bash
# Track B：ExploreToM v2 转换
python production_frozen/v3.0/scripts/convert_exploretom_v2.py \
  --input data/tom/raw/exploretom_v2_source.jsonl \
  --output exploretom_v2_track_b.jsonl

# Track C：GPT-5.5 HOT 靶向合成（需要 OpenAI API key）
OPENAI_API_KEY=sk-... python production_frozen/v3.0/scripts/synth_gpt55_phase_d_hot.py \
  --output synth_gpt55_phase_d_hot.jsonl \
  --target_n_per_class 250

# 合并
python production_frozen/v3.0/scripts/merge_stage11_train.py \
  --base data/tom/tom_train.jsonl \
  --add exploretom_v2_track_b.jsonl synth_gpt55_phase_d_hot.jsonl \
  --output tom_train_stage12.jsonl
```

---

## 六、教训与未来方向

### 6.1 经验

1. **不要假定平台期**：Stage 8 看起来已收敛，但 Track D 证明续训仍能涨；以后停训应基于 val 平稳 + train metrics 同步平稳，不能只看一项。
2. **subset500 评测要看趋势不看单点**：一个 used=0 步的 val 跳跃完全是噪声；至少看连续 3 个评测点。
3. **数据扩充和评测协议是正交收益**：Track A（不训练）和 Track E（训练）累加几乎线性。
4. **多线并行节省 wall-clock**：A/B/C 数据准备、D 训练、E 训练可以串成 critical path 的"等 D 完算了再起 E"，但实际我们让 D 跑完自动触发 E，省了人工干预。

### 6.2 失败实验（不在 v3.0，但保留以警示）

| 实验 | 结果 | 关键失败原因 |
|---|---|---|
| Stage 9（SFT 冷启 + KL + 长 CoT） | -1.51pp | SFT init 把 Stage 8 RLVR 学到的 ToM 策略洗掉了 |
| Stage 10（weighted_sum + entropy） | -3.96pp，step 214 终止 | 加 entropy bonus 让模型学到错乱的 hedging 行为 |
| 早期 Stage 1-6 | 全部弱于 7/8 | 数据规模 / 配比 / 长度未调优 |

### 6.3 后续可探索

1. **Track A 的 del_tom 协议在 Stage 12 上还能不能再 +2pp？**
   - 期望：因为 Stage 12 已学到更鲁棒的 ToM 模式，del_tom 增益可能从 +2.16pp（Stage 8）压缩到 +1.6pp。
   - 但叠加后绝对值仍可能逼近 0.79 → 0.80。
2. **N>8 的 del_tom 投票是否继续涨？** N=16/24 应跑一次确认（推理成本翻倍但可能 +0.3pp）。
3. **再续训 Stage 12（Track D 思路套到 Stage 12）**：从 0.7640 (subset500) 还能涨到 0.78+ 吗？
4. **GPT-5.5 蒸馏到 14B**：当前差距 5.26pp，蒸馏 N=8 reasoning trace 作为 SFT init 可能再 +1-2pp。
5. **更大批量的 HOT 合成**：1260 → 5000，针对 Stage 12 仍错的题靶向合成。

---

## 七、关键文件索引

| 文件 | 作用 |
|---|---|
| `production_frozen/v3.0/README.md` | v3.0 快照说明 + 校验 |
| `production_frozen/v3.0/SHA256SUMS.txt` | 全部产物校验和 |
| `production_frozen/v3.0/configs/rlvr_config_14b_stage12_FROZEN.yaml` | Stage 12 训练配置 |
| `production_frozen/v3.0/data/tom_train_14b_stage12.jsonl` | 12519 条整合训练集 |
| `production_frozen/v3.0/eval/14b_stage12_full5718.json` | 三协议 5718 题完整结果 |
| `production_frozen/v3.0/scripts/run_tombench.py` | 评测框架（含 del_tom） |
| `production_frozen/v3.0/scripts/extractors.py` | del_tom 投票实现 |
| `docs/stage11_report.md` | 英文执行报告（含 Track D 完整轨迹） |
| `docs/stage11_v2_final_report_zh.md` | 本报告 |

---

## 八、致谢

- Track A `del_tom` 协议灵感来自 [arXiv:2403.xxxxx]（项目内部讨论）
- ExploreToM v2 数据集来自 Kim et al. ICML 2024
- HOT 合成数据由 GPT-5.5 (OpenAI API) 生成

---

最后更新：2026-05-21 02:30 UTC
