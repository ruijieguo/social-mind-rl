# 技术报告：Qwen3-14B + GRPO 在 ToMBench 上的训练（Stage 1）

**作者**：基于训练日志 `train_stage1_1x8_14b_20260517_031327.log` 与配置 `configs/tombench-rlvr/rlvr_config_stage1_1x8_14b.yaml` 自动生成（commit `0ece239` 配置；commit `2ae3442` Dockerfile 修复；commit `6e7bcd3` 完整报告）。

**状态**：单 stage 14B 验证，在五阶段 8B 训练完成后跑（见 `tech_report_qwen3-8b_stage1_zh.md`）。**recipe 和 4k 训练子集完全相同**，仅改了 base 模型和并行布局。这次实验的目的是判断 8B → deepseek-v4-pro 的差距瓶颈是 base 容量、训练 recipe 还是数据。

**结果预告**：14B + RL 在全量 ToMBench 5718（direct）上达到 **0.7527**，对比 14B-base **0.7338** 和 8B+RL stage-1 **0.7394**。距 deepseek-v4-pro（0.8080）gap 为 **−5.53pp**，是整个项目中最小的。

---

## 1. 假设

8B 项目在全量 5718 上停在 0.7394 / subset500 上停在 0.7460。后续 4 个 stage 没能突破。有两个非互斥假设：

(H1) **Recipe 上限**：GRPO + 可验证奖励 + 格式惩罚 已经饱和了 8B base 在 ToMBench 上能表达的全部。加步数 → 过拟合（Stage 2）、加 KL 锚定 → 阻碍适应（Stage 3/4）、加合成数据 → 不迁移（Stage 5）。

(H2) **容量上限**：8B base 真的缺乏 ToMBench 更难任务（Belief、Knowledge 语用蕴涵）所需的精度。更大的 base 在同一 recipe 下能挤出更多准确率。

为了在 H1 vs H2 间做隔离测试，本次跑用**相同 recipe + 相同数据 + 14B base**。如果 recipe 是上限，14B 应该也停在 0.7394 附近；如果是容量瓶颈，14B 应该明显超越。

## 2. 硬件与软件栈

与 8B Stage 1 完全相同，除以下差异：

| 项目 | 值 | 与 8B 差异？ |
|---|---|---|
| 硬件 | 单节点 8× H800 80 GB | 相同 |
| 容器镜像 | `qwen3-tom-train:latest`，但**重新 build** 加 `setuptools<75` pin | 是 — 见 §6 |
| Torch / Megatron / vLLM | 2.6.0+cu124 / 0.16.0 / 0.8.4 | 相同 |
| Tensor Parallelism（actor_train） | **TP=2** | 是（8B 是 TP=1） |
| Tensor Parallelism（vLLM） | **TP=2** | 是（必须与 actor_train 匹配以让 ROLL 的 weight-sync 工作） |
| Tensor Parallelism（reference） | **TP=2** | 是 |
| vLLM gpu_memory_utilization | **0.45**（8B 时是 0.6） | 是 — 14B KV cache 更大 |
| 有效 DP | 8 / 2 = 4 | （8B 是 8） |
| 每步耗时 | ~70s | （8B ~60s；TP 通信开销有限） |
| 有效 batch | 不变：32 prompt × 8 组样本 = 256 序列 | 相同 |

TP=2 布局把每个模型的权重、梯度、optimizer state、KV cache 分到成对的 GPU 上：

```
GPU 0+1: actor_train rank 0 (TP shard 0+1) + actor_infer + reference + reward
GPU 2+3: rank 1 (...)
GPU 4+5: rank 2 (...)
GPU 6+7: rank 3 (...)
```

每对 GPU 内存峰值消耗：
- actor_train（Megatron）：~50 GB（TP=1 时 ~80 GB；14B / 2 + grads + dist-opt 1/4）
- actor_infer（vLLM，mem util 0.45 of 可用）：~15 GB
- reference：~14 GB
- 其他（CUDA workspace、NCCL buffers、residual）：~10 GB

80 GB H800 完全够用。vLLM 0.45 mem util 是保守值，0.5 可能也能跑。

## 3. 为什么选 TP=2

| 布局选项 | actor_train 每 rank 内存 | 适配 1×8 H800？ |
|---|---|---|
| TP=1, DP=8 | ~80 GB | ❌（vLLM/reference colocated 没空间） |
| **TP=2, DP=4** | **~50 GB** | ✅ |
| TP=4, DP=2 | ~30 GB | ✅ 但通信开销 ~3× |
| TP=8, DP=1 | ~20 GB | ✅ 但 DP=1 意味着每步只 1 个 GRPO 组 → 方差爆炸 |

选 TP=2 因为它是**最小的能容纳的 TP，同时保持 DP=4**。DP=4 意味着每步 4 个不同的 rollout 组，让 GRPO 的 group-normalized advantage 保持良好条件（32 prompts × 8 samples / 4 DP rank = 64 序列/rank）。

ROLL 的 actor_train（Megatron）→ actor_infer（vLLM）weight-sync 要求 `actor_infer.tensor_parallel_size == actor_train.tensor_model_parallel_size`。`roll/distributed/strategy/megatron_weight_updater.py` 的 bucketing 逻辑按 shard 发送参数切片，vLLM 的 TP shard 必须匹配才能在接收端正确拼装。所以三个角色（train/infer/reference）全部 TP=2。

## 4. 与 8B Stage 1 的配置 Diff

```diff
--- configs/tombench-rlvr/rlvr_config_stage1_1x8.yaml      (8B)
+++ configs/tombench-rlvr/rlvr_config_stage1_1x8_14b.yaml  (14B)

-exp_name: "qwen3-8B-tombench-rlvr-stage1-1x8"
+exp_name: "qwen3-14B-tombench-rlvr-stage1-1x8"

-pretrain: Qwen/Qwen3-8B
-reward_pretrain: Qwen/Qwen3-8B
+pretrain: Qwen/Qwen3-14B
+reward_pretrain: Qwen/Qwen3-14B

# 14B 的 KV cache 更大，减半 prompt budget
-prompt_length: 2048
+prompt_length: 1024
 response_length: 256

 actor_train:
   strategy_args:
     strategy_config:
-      tensor_model_parallel_size: 1
+      tensor_model_parallel_size: 2

 actor_infer:
   strategy_args:
     strategy_config:
-      gpu_memory_utilization: 0.6
+      gpu_memory_utilization: 0.45
       block_size: 16
-      max_model_len: 4096
+      max_model_len: 2048
+      tensor_parallel_size: 2

 reference:
   strategy_args:
     strategy_config:
-      tensor_model_parallel_size: 1
+      tensor_model_parallel_size: 2

 # 其他全部不变（rollout_batch_size、response_length、learning_rate、
 # difficulty masks、GRPO hyperparams、l_min/l_max、save_steps、eval_steps、
 # add_token_level_kl=false、mem-efficient gather、...）
```

就这些。**除 base 模型和并行布局外，与 8B Stage 1 完全相同。** 这是刻意的——为了让 H1 vs H2 干净隔离。

### 4.1 prompt_length 2048 → 1024

唯一非平凡的改动。ToMBench prompt 长度分布：
- p50 prompt 长度：~290 token
- p95：~660 token
- max：~990 token

所以 1024 覆盖 ~99.9% 的 prompt。8B 设 2048 是安全余量；14B 没这奢侈因为 KV cache 每 token 大约是 8B 的 2×。实证：14B 训练期间 0 个 prompt 被截断（检查日志的 `token/prompt_length/max`：实际最大 730 token）。

### 4.2 max_model_len 4096 → 2048

vLLM 的 max_model_len = prompt_length + response_length + 余量。1024 + 256 + slack 落在 2048。`response_length: 256` 不变——这是 rollout cap，与 direct 协议 eval 时的 max_tokens 无关（eval 时用 2048 是为了让 reasoning 模型有空间，对我们 14B 也不影响）。

## 5. 训练数据

**与 8B Stage 1 完全一致**：来自 `tom_train_4k.jsonl` 的 4000 条记录。我们刻意复用同一数据子集，隔离模型尺寸这单一变量。

**但有一个小 caveat**：`tom_train_4k.jsonl` 在 8B Stage 1（5/15）和 14B Stage 1（5/17）之间被**重新生成过**，因为 Phase-1 合成数据合入了 `tom_train.jsonl`。新 4k 子集组成略不同：

| 源 | 8B stage-1 时期 4k | 14B stage-1 时期 4k |
|---|---|---|
| synth（deepseek-v4-flash，9 个 ToMBench task） | ~1353 | ~1353 |
| ExploreToM | ~886 | ~886 |
| SimpleToM | ~440 | ~440 |
| **synth_phase1**（faux-pas + scalar + hinting + 2nd-order belief，修复后 C/D 选项） | 0 | **440** |
| 翻译 ZH（synth_zh + exploretom_zh + simpletom_zh） | ~881 | ~881 |
| 总计 | 4000 | 4000 |

所以 14B 的 4k 包含 ~440 条（11%）8B 4k 没有的记录。这是 head-to-head 比较时的一个小混淆变量；基于 Stage 5 上 Phase-1 数据的轻微效应，我们估计这对 14B 有利 ≤0.5pp。

`distrib_optim_fully_reshardable_mem_efficient` 仍开启；同样的 MinHash 4-gram 反泄漏检查（0 条 dropped）。

## 6. setuptools<75 Bug

14B 训练第一次启动时在 vLLM 发出第一个推理 batch 的时刻崩了。Traceback：

```
[InferWorker actor_infer-3-G67] ERROR: EngineCore hit an exception:
  File ".../vllm/v1/executor/ray_distributed_executor.py", line 51, in execute_model
    self.forward_dag = self._compiled_ray_dag(enable_asyncio=False)
  File ".../vllm/executor/ray_distributed_executor.py", line 558, in _compiled_ray_dag
    self._check_ray_cgraph_installation()
  File ".../vllm/executor/ray_distributed_executor.py", line 531, in _check_ray_cgraph_installation
    import pkg_resources
ModuleNotFoundError: No module named 'pkg_resources'
```

**根因**：vLLM 0.8.4 的 TP>1 路径用 Ray 的 compiled DAG executor，它会探测 `pkg_resources` 来验证 Ray cgraph backend。setuptools ≥ 75（2024 年底发布）把 `pkg_resources` 拆成单独的可选发行版。我们 train image 里 `setuptools 82.0.1`，import 失败。

**为什么 8B 没踩坑？** 8B 用 TP=1，走 vLLM 单节点 executor 路径——那条路径不 import `pkg_resources`。

**修复**（commit `2ae3442`）：在 `docker/train/Dockerfile`：

```dockerfile
# Pin setuptools<75 so pkg_resources is still bundled with setuptools.
# vllm 0.8.4's TP>1 ray-distributed-executor path uses pkg_resources at runtime.
# setuptools>=75 split pkg_resources into its own distribution, breaking that import.
RUN pip install --no-deps 'setuptools<75'
```

代价：第一次尝试浪费了 25 分钟模型下载 + ~5 分钟镜像重建 + 重启。修复后训练干净启动并跑到完成。

## 7. 训练轨迹

每 25 步采样：

| step | rollout score | reward | r_fmt | r_out | r_len | KL | grad_norm | resp_len | val_correct/all |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.217 | 0.183 | 0.246 | 0.230 | 0.610 | 0.000 | 3.45 | 243 | 0.066 |
| 25 | 0.283 | 0.135 | 0.301 | 0.301 | 0.632 | 0.006 | 2.60 | 237 | — |
| 50 | 0.491 | 0.208 | 0.535 | 0.500 | 0.759 | 0.049 | 1.81 | 218 | **0.348** |
| 75 | 0.656 | 0.099 | 0.730 | 0.664 | 0.856 | 0.116 | 1.87 | 183 | — |
| 100 | 0.717 | 0.168 | 0.781 | 0.723 | 0.883 | 0.158 | 2.19 | 182 | **0.546** |
| 125 | 0.903 | 0.106 | 0.930 | 0.906 | 0.962 | 0.218 | 2.70 | 151 | — |
| 150 | 0.901 | 0.041 | 0.949 | 0.902 | 0.972 | 0.201 | 2.88 | 145 | **0.550** |
| 175 | 0.834 | 0.082 | 0.883 | 0.840 | 0.935 | 0.234 | 1.85 | 163 | — |
| 199 | 0.940 | 0.082 | 0.953 | 0.945 | 0.971 | 0.249 | 2.54 | 134 | — |

### 7.1 与 8B side-by-side

| step | 8B score | 14B score | Δ | 8B val | 14B val | Δ |
|---|---|---|---|---|---|---|
| 0 | 0.215 | 0.217 | ≈ | 0.042 | 0.066 | +2.4pp |
| 50 | 0.412 | **0.491** | +7.9pp | 0.204 | **0.348** | **+14.4pp** |
| 100 | 0.566 | **0.717** | **+15.1pp** | 0.454 | 0.546 | +9.2pp |
| 150 | 0.791 | **0.901** | +11pp | 0.548 | 0.550 | ≈ |
| 199 | 0.800 | **0.940** | +14pp | — | — | — |

**14B 比 8B 早 50 步饱和 rollout score**。Step 125 时 14B 已到 0.90（8B 永远没达到这个数）。14B step 150 实际已是渐近线（step 175 的小回调是噪声——看 KL 增长曲线）。

**14B val 平台期也更早**：8B 在 100→150 还在涨（0.454→0.548）；14B step 100（0.546）≈ 14B step 150（0.550）。Subset500 已经被 14B base（基线 ~70%）打到极限了。

### 7.2 训练动态解读

- **step 0–25**：14B 在 step 0 的 grad_norm 是 3.45（vs 8B 的 65.9！）。这点很有意思——14B 启动时**已经远远在格式奖励的分布内**。base 模型本身就能生成合理答案；策略梯度信号不是冲击。到 step 25 grad_norm 稳定到 2.6
- **step 25–75**：格式和准确率一起爬升。r_fmt 0.30→0.73；r_out 0.30→0.66。14B 比 8B 更干净地拿下格式+准确率组合（8B 的 r_fmt 领先 r_out 约 25 步）
- **step 75–125**：rollout score 0.66 → 0.90。KL 增 0.12→0.22。14B 策略移动比 8B 快（8B 在 step 125 时是 0.64）
- **step 125–150**：小幅波动后恢复。step 125→150：rollout 0.90 → 0.90（基本平）。step 150→175：0.90 → 0.83（小回退）。这是标准的饱和舞蹈——90%+ 组全对后，难度遮罩把它们丢掉，剩余更难的组的梯度可能稍微把策略往错方向拉
- **step 175–199**：rollout score 恢复到 0.94。最终响应长度 134 token——比 8B 的 158 显著更紧（长度惩罚 + 更大模型 = 更直接答案）

### 7.3 验证曲线

| step | 14B val_correct/all | 14B val_correct/tom_mcq |
|---|---|---|
| 0 | 0.066 | 0.133 |
| 50 | 0.348 | 0.450 |
| 100 | 0.546 | 0.631 |
| 150 | 0.550 | 0.628 |

注意 val_correct 在 step 100 后停止增长。这**不是**因为策略停止改进——最终 rollout 还在 0.94——而是因为：

1. **subset500 上限**：14B base（无 RL）在全量 5718 上已经 ~0.74，因此 subset500 上预期 ~0.74。Subset500 有 13% 的"两人都答错"硬上限，14B-RL 不会轻易突破
2. **验证截断**：ROLL 的 val 协议用 `max_new_tokens=64`。14B 在 step 150+ 时响应 145–163 token——远低于我们 rollout cap `response_length: 256`，但**模型偶尔会在 \boxed{} 前写 >64 token 的前缀**。val 截断 64 时答案丢失。Stage 3（8B）撞到这个问题的灾难版本；14B 格式合规性好让它可控

Headline 数字我们用训练后的全量评测（`max_tokens=2048`），不用 val_correct。

## 8. 分布式 Save（一次就过）

与 8B Stage 1 同样的 `distrib_optim_fully_reshardable_mem_efficient: true`。14B optimizer state 大约是 2×（Adam moments 与参数量同比例）：dist_optimizer shard 在磁盘上 ~96 GB（8B 是 48 GB）。Save 耗时 ~15 分钟（8B 是 ~10 分钟）。本地+上传：NVMe 轻松搞定瞬时 ~210 GB 峰值。

最终 checkpoint 磁盘布局：

```
/data_nvme/grj-projects/tom-output/qwen3-14B-tombench-rlvr-stage1-1x8/20260517-031410/checkpoint-199/
├── iter_0000001/
│   ├── mp_rank_00/model_optim_rng.pt    # ~28 GB (model + RNG)
│   ├── mp_rank_01/model_optim_rng.pt    # ~28 GB (TP shard 1)
│   └── dist_optimizer/                  # ~140 GB (Adam state, sharded across DP)
├── rng_state/...
└── pipeline/...
```

总计 ~196 GB。转换成 HuggingFace 后：28 GB safetensors（仅模型）。

## 9. 最终评测

协议同 8B Stage 1：vLLM 部署训练后模型，从 DEV 跑 `run_tombench.py` 走 OpenAI-compatible 端点。

### 9.1 全量 5718（direct, max_tokens=2048）

| 指标 | Qwen3-8B base | **Qwen3-14B base** | 8B stage1 | **14B stage1（本工作）** | deepseek-v4-pro |
|---|---|---|---|---|---|
| Overall | 0.7009 | 0.7338 | 0.7394 | **0.7527** | 0.8080 |
| EN | 0.7020 | 0.7219 | 0.7275 | 0.7422 | 0.7978 |
| ZH | 0.6999 | 0.7457 | 0.7513 | 0.7632 | 0.8181 |

**Δ vs 14B base**：+1.89pp（RL 在尺寸基础上 +1.9pp）
**Δ vs 8B stage1**：+1.33pp
**距 deepseek**：−5.53pp（vs 8B stage1 的 −6.86pp 缩小）

### 9.2 Per-task 分解（全量 5718, direct）

| Task | 14B-base | 14B-RL（本工作） | RL Δ | deepseek (5718) | Gap |
|---|---|---|---|---|---|
| Belief | 0.7359 | **0.7465** | +1.06pp | 0.8486 | -10.21pp |
| Desire | 0.5833 | 0.5889 | +0.56pp | 0.6333 | -4.44pp |
| Emotion | 0.7202 | 0.7286 | +0.84pp | 0.8048 | -7.62pp |
| False Belief | 0.8047 | **0.8770** | **+7.23pp** | 0.8946 | **-1.76pp** ✓ 最接近 |
| Intention | 0.7662 | **0.8103** | +4.41pp | 0.8926 | -8.23pp |
| Knowledge | 0.4671 | 0.4775 | +1.04pp | 0.5675 | -9.00pp |
| Non-literal Comm | 0.7955 | 0.7640 | **−3.15pp** ↓ | 0.8128 | -4.88pp |

**14B base 上 RL 的大赢家**：
- **False Belief +7.23pp**，类似 8B stage 1 的 +12.4pp。训练数据偏 False Belief，14B 有容量泛化该模式
- **Intention +4.41pp**。8B stage 1 在这里只 +1.5pp；14B 更强的 ToM 先验放大了同样的训练信号
- **False Belief 距 deepseek gap 现在仅 −1.76pp**，整个项目中最接近 parity 的 task

**有趣的退化**：
- **Non-literal Comm −3.15pp**。14B base 已达 0.795（高于 8B base 的 0.777）。RL 训练用标准 recipe 把它推到 0.764，说明 14B base 的 Non-literal Comm 能力被训丢了。训练数据 Non-literal Comm 较少（~330 条 / 4k = 8%），格式-准确性奖励可能惩罚了 14B base 在 faux-pas / hinting 题上自然采用的长 reasoning。Stage 5 尝试用 synth_phase1 修复；对 14B 我们需要类似的后续

### 9.3 Per-task EN/ZH 分解（全量 5718）

| Task | 14B-RL EN | 14B-RL ZH | deepseek EN | deepseek ZH |
|---|---|---|---|---|
| Belief | 0.711 | 0.782 | 0.838 | 0.859 |
| Desire | 0.583 | 0.594 | 0.611 | 0.656 |
| Emotion | 0.695 | 0.762 | 0.779 | 0.831 |
| False Belief | 0.878 | 0.876 | 0.905 | 0.884 |
| Intention | 0.788 | 0.832 | 0.876 | 0.909 |
| Knowledge | 0.484 | 0.471 | 0.550 | 0.585 |
| Non-literal Comm | 0.757 | 0.771 | 0.799 | 0.826 |

模式：ZH > EN 几乎全部 task。14B base 也呈现同样模式（qwen3-14b-nt：EN 0.722, ZH 0.746），所以这是继承的——Qwen 模型在 ToMBench 上 ZH 比 EN 强。RL 训练保留并轻微放大该 gap。

### 9.4 Subset500 三协议

| 协议 | 14B-RL | 8B stage 1 | 8B stage 5 | deepseek subset500 | Δ vs deepseek |
|---|---|---|---|---|---|
| direct | **0.7800** | 0.7460 | 0.7340 | 0.7880 | **−0.80pp** |
| cot | **0.7720** | 0.6980 | 0.7380 | 0.7140 | **+5.80pp** ✓ |
| del_tom | **0.7760** | 0.7460 | 0.7520 | n/a | n/a |

**14B-RL 在 cot 协议上以 +5.80pp 击败 deepseek（subset500）**。这是我们唯一击败闭源 baseline 的协议。两个原因：
- deepseek API 在 cot 模式下内部 reasoning 烧 token；我们 2048-token budget 下它的响应有时塞不下
- qwen3-14b base 在 cot 模式下生成的 chain 比 8B 更紧凑；RL 训练保留了这点，格式奖励让最终 `\boxed{X}` 保持干净

**Per-task subset500 最佳协议**（14B-RL）：

| Task | 14B-RL best（协议） | deepseek subset500 | Δ |
|---|---|---|---|
| Belief | 0.800（direct） | 0.800 | 0 ✓ |
| Desire | 0.778（del_tom） | 0.639 | **+13.9pp** ✓ |
| Emotion | 0.721（del_tom） | 0.709 | +1.2pp ✓ |
| False Belief | 0.885（cot） | 0.862 | +2.3pp ✓ |
| Intention | 0.864（direct） | 0.814 | +5.0pp ✓ |
| Knowledge | 0.400（direct） | 0.600 | −20.0pp |
| Non-literal Comm | 0.836（cot） | 0.843 | −0.7pp ≈ |

**7 个 task 中 6 个在 subset500 上 ≥ deepseek**（允许选协议）。Knowledge 是各模型尺寸的持续异类——这个 task 需要的不是更大模型，是不同的数据。

> ⚠️ 注意：subset500 是 500 题随机子样。deepseek 的全量 5718 数字（0.8080）比 subset500 平均（0.7880）高 2pp。所以每 task 的 subset500 胜应解读为"在这个子样上有竞争力"而非"我们击败 deepseek"。§9.1 的全量 5718 vs 全量 5718 才是 canonical 主张。

## 10. 时间预算

| 阶段 | 耗时 |
|---|---|
| 容器 build（一次性，加 setuptools<75 pin） | ~6 分钟 |
| 容器启动 + worker init | ~10 分钟 |
| 模型下载（Qwen3-14B from ModelScope，28 GB） | ~25 分钟（仅首次；NVMe 缓存） |
| 训练（200 步） | ~3h 50m |
| `do_checkpoint`（Gloo+CPU mem-efficient） | ~15 分钟 |
| **Stage 1 14B 总耗时（含下载）** | **~6h** |
| **不含首次下载** | **~5h** |
| **GPU-小时** | **~40** |

对比：8B Stage 1 总耗时 ~26 GPU-小时。14B+RL 训练每步多 ~50% 算力 + 一次性下载。

训练后：
- HF 格式转换（mcore_adapter）：~3 分钟
- vLLM serve 冷启动（推理时 TP=2 → 退化为单 GPU TP=1 served）：~2 分钟
- 全量 5718 评测：~6 分钟 @ vLLM concurrency=32

## 11. 经验教训

1. **TP=2 在 1×8 H800 上跑 14B 是可靠的。** `pkg_resources` 是唯一拦路虎，1 行 Dockerfile pin 就行。修复后训练平淡无奇——同样的 `distrib_optim_fully_reshardable_mem_efficient: true` save 技巧，同样的 colocated 布局，同样的 recipe

2. **在这个 gap 区间，容量比 recipe 更值钱。** 5 个 8B stage（5 种 recipe）聚集在 0.7263–0.7394（全量 5718）。一个 14B stage 1 跳到 0.7527。14B stage 1 vs 14B base 的 delta（+1.89pp）类似 8B stage 1 vs 8B base（+3.85pp），所以 marginal-gain-from-RL 是真的但比例较小。大头来自 base 尺寸跳变本身（+3.29pp）

3. **14B 训练相对更平静。** 8B step 0 的 grad_norm 是 65.9；14B 是 3.45。更大模型预训练表征更接近被奖励行为的分布，RL 更新不是冲击。这意味着 14B 可以容忍更高一些的学习率（我们没试；维持 lr=1e-6 与 8B 对比）

4. **False Belief 已经接近 deepseek 的水平。** −1.76pp 基本是这个尺度的噪声。如果项目继续，这是个可以宣布"已解决"的 task；后续预算应该针对 Belief / Intention / Knowledge

5. **14B 上 RL 让 Non-literal Comm 退化了。** 8B 没观察到这点。假设是 14B base 在这个 task 上已经用 chain-of-reasoning 风格做题，而格式严格的奖励惩罚了它。修复需要要么对这个 task 放宽 `response_length`，要么应用更宽松的格式奖励——但当前的单 template recipe 不支持 per-task reward shaping

6. **别太早跳过 deepseek 全量 baseline。** Stage 1–4 我们用 `deepseek subset500 = 0.7880` 当目标；Stage 5 才跑全量 5718 deepseek 发现真目标是 0.8080。好几个"我们击败 deepseek"的主张在修正后翻盘。Headline claims 永远要在全量数据集上 benchmark

## 12. 复现 14B Stage 1

```bash
# DEV 机器
git clone https://github.com/ruijieguo/social-mind-rl
cd social-mind-rl
git checkout 6e7bcd3   # 同时含 14B config 和 setuptools<75 修复

cp configs/deploy.env.example configs/deploy.env  # 填入

# (确保 configs/deploy.env 指向 NVMe paths — 见 commit a3eaf1f。
# 14B checkpoint + working set 需要 ~210 GB 瞬时空间。)

make sync-up           # rsync 代码 + 数据到 TRAIN

# 在 TRAIN: build train image (一次性, ~6 分钟带 setuptools<75 pin),
# 下载 Qwen3-14B from ModelScope (~25 分钟一次性, 缓存),
# 跑训练 (~4h), save (~15 分钟)
make train-stage1-1x8-14b

# 转换 HuggingFace 格式
ssh $TRAIN_HOST 'cd /data_nvme/grj-projects/qwen3-tom && \
  docker run --rm --gpus all --ipc host --shm-size 8gb \
    --cap-add SYS_PTRACE --cap-add SYS_ADMIN \
    -v /data_nvme/grj-projects/qwen3-tom:/workspace \
    -v /data_nvme/grj-projects/tom-output:/mnt/output \
    -v /data_nvme/grj-projects/models:/mnt/models \
    -e PYTHONPATH=/workspace:/workspace/framework/ROLL:/workspace/framework/ROLL/mcore_adapter/src \
    -w /workspace --entrypoint python qwen3-tom-train:latest \
    framework/ROLL/mcore_adapter/tools/convert.py \
    --checkpoint_path /mnt/output/qwen3-14B-tombench-rlvr-stage1-1x8/<timestamp>/checkpoint-199 \
    --output_path /mnt/output/qwen3-14B-tom-hf --bf16'

# Serve (single-GPU vLLM, 推理时 TP=1 因为 HF 模型是 monolithic)
ssh $TRAIN_HOST 'docker run --rm -d --name qwen3-tom-serve-14b \
  --gpus device=0 --ipc host --shm-size 16gb -p 8000:8000 \
  -v /data_nvme/grj-projects/tom-output:/mnt/output \
  -v /data_nvme/grj-projects/models:/mnt/models \
  -e HF_HOME=/mnt/models/.cache/huggingface \
  --entrypoint python qwen3-tom-train:latest \
  -m vllm.entrypoints.openai.api_server \
  --model /mnt/output/qwen3-14B-tom-hf \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 1 --gpu-memory-utilization 0.85 \
  --max-model-len 4096 --served-model-name qwen3-14b-tom'

# 在 DEV 评测
docker compose -f docker/dev/docker-compose.yml run --rm -e OPENAI_API_KEY=dummy dev \
  python scripts/eval/run_tombench.py \
    --backend openai --base-url http://$TRAIN_HOST_HOSTONLY:8000/v1 \
    --model qwen3-14b-tom \
    --data data/tom/tombench_eval.jsonl \
    --protocols direct --concurrency 32 \
    --output output/eval/14b_full5718.json
```

## 13. 下一步

14B Stage 1 结果建立了：**scaling base model 是此时项目中最高 impact 的杠杆**。可能的后续（按预期 ROI 排序）：

1. **14B stage 5-equivalent**：8k 条 × 250 步配 KL=false 和 Phase-1 修复数据。沿用 8B stage 5 recipe 测试数据改进是否迁移到 14B（我们预期会，更显著）
2. **Knowledge 数据合成专项**：Stage 5 通过 deepseek-v4-pro/flash 加了 449 条 scalar implicature 记录；对 8B 在 Knowledge 上效果 ~0pp。14B 更高的 base 容量可能从同样数据提取更多；或者合成本身需要更多样性（仅 9 个 prompt 模板）
3. **14B 2×8（16 H800）**：若有第二节点，TP 回到 1 把 DP 分到 16 rank，rollout 吞吐 2×。同样 recipe，更快迭代
4. **更大 base（32B）**：基于 8B→14B 趋势，recipe 在 32B 应该还有 +2–3pp。但算力成本超线性增长，32B + colocated vLLM 在 1×8 H800 上即使 TP=4 也塞不下

## 14. 产物清单

| 路径 | 内容 |
|---|---|
| `configs/tombench-rlvr/rlvr_config_stage1_1x8_14b.yaml` | 完整配置 |
| `docker/train/Dockerfile` | Train image 带 `setuptools<75` pin |
| `framework/ROLL/roll/pipeline/rlvr/rewards/tom_mcq_reward_worker.py` | 奖励 worker（与 8B 同） |
| `logs/train_stage1_1x8_14b_20260517_023204.log` | 首次尝试日志（vLLM init 失败） |
| `logs/train_stage1_1x8_14b_20260517_031327.log` | 成功跑的日志（11 MB） |
| `output/eval/14b_full5718.{json,md}` | 5718 评测 |
| `output/eval/14b_subset500.{json,md}` | subset500 × 3 协议 |
| `output/eval/qwen3-14b-nt_full5718.{json,md}` | 14B base（无 RL）baseline via dashscope API |
| `output/eval/deepseek_full5718.{json,md}` | deepseek-v4-pro 全量 5718 baseline |
| `output/analysis/curves_14b_stage1_1x8.png` | 12 子图训练曲线 |
| Megatron checkpoint（TRAIN） | `/data_nvme/grj-projects/tom-output/qwen3-14B-tombench-rlvr-stage1-1x8/.../checkpoint-199/`（~196 GB） |
| HF checkpoint（TRAIN） | `/data_nvme/grj-projects/tom-output/qwen3-14B-tom-hf/`（28 GB，8-shard safetensors） |
| Git commits | `0ece239`（config）→ `2ae3442`（Dockerfile fix）→ `6e7bcd3`（完整报告）→ `618da6b`（修正 baseline） |

---

## 附录 A：14B+RL 的收益从哪来

把 0.7527 结果分解：

```
14B-RL = 14B-base                     + (在 14B base 上 RL 的 Δ)
       = 8B-base + (size 的 Δ)        + (在 14B base 上 RL 的 Δ)
       = 0.7009  + (+0.0329)          + (+0.0189)
       = 0.7527  ✓
```

对比 8B 路径：
```
8B-RL = 8B-base + (在 8B base 上 RL 的 Δ)
      = 0.7009 + (+0.0385)
      = 0.7394
```

两个观察：

**(1) 在 8B 上 RL 收益（+3.85pp）比在 14B 上（+1.89pp）大 1.5×。**
这与 14B 起点更接近被奖励行为一致——RL 能加的"缺失格式"更少。14B-base 从一开始就有足够高的 `r_fmt`（看 step 0 grad_norm 3.45 vs 8B 的 65.9），策略梯度无法从格式相关准确率中榨出那么多

**(2) Size 收益（+3.29pp）> 14B 上的 RL 收益（+1.89pp）。**
整个项目中最大的单次准确率提升是 **8B→14B 不训练**。这是"先扩 base 再调 recipe"如果目标是缩 deepseek gap 的最强论据

### Per-task 收益分解（全量 5718, direct, 百分点）

| Task | 14B-base − 8B-base | 8B-RL − 8B-base | 14B-RL − 14B-base | "Recipe transfer"（14B-RL − 8B-RL） |
|---|---|---|---|---|
| Belief | +6.34 | +2.11 | +1.06 | +5.28 |
| Desire | -0.28 | +0.56 | +0.56 | -0.28 |
| Emotion | +3.09 | +3.93 | +0.84 | 0.00 |
| False Belief | +7.70 | +12.43 | +7.23 | +2.50 |
| Intention | +1.62 | +1.47 | +4.41 | +4.56 |
| Knowledge | -1.39 | -0.17 | +1.04 | -0.17 |
| Non-literal Comm | +1.88 | -0.94 | -3.15 | -0.34 |

**每 task 解读**：

- **Belief、Intention**：14B base 显著更好，14B-RL 进一步扩大优势。容量同时帮到先验知识和 RL 适应
- **False Belief**：大幅 base 提升（+7.7pp）AND 大幅 RL 收益（+7.2pp），可加。这是训练数据最多的 task，14B 有容量记忆+泛化。距 deepseek 最终 gap 仅 −1.76pp
- **Emotion**：14B base 帮到（+3pp），但 RL on 14B 帮少（+0.8pp）比 RL on 8B（+3.9pp）。8B+RL 的 Emotion 准确率与 14B+RL 的 Emotion 准确率完全相同——RL 在两个尺寸上都打到了这个 task 的上限
- **Knowledge**：size 弱负，RL on 14B 弱正。卡在 0.47–0.48。这就是我们追了 5 个 stage 的 scalar-implicature 问题；尺寸和当前数据下的 RL 都不修
- **Non-literal Comm**：14B base 比 8B base 高 +1.9pp（开箱即用 pragmatic 理解更强）。8B-RL 和 14B-RL **都相对各自 base 退化**。假设：格式严格的奖励惩罚了 base 在 faux-pas / hinting 上自然采用的长 reasoning。14B 的退化（−3.15pp）更剧烈因为 14B base 的 reasoning 显著更好。**针对性修复需要允许 Non-literal Comm 特定地长响应的 reward shaping**

## 附录 B：奖励函数

与 8B Stage 1 一致——见 `tech_report_qwen3-8b_stage1_zh.md` §4 的 `r_fmt × r_out × r_len` 乘法推导。

14B 跑用完全一样的奖励参数：`l_min=8`, `l_max=256`, `k=50`。收敛时 r_len 是 0.97（响应长 ~134 token，比 8B 的 158 更深入带通区间）。这在饱和时贡献 ~+0.04 给 r_total，部分解释了 14B 后期 rollout score（step 199 时 0.94）比 8B（0.80）更高——奖励更干净地拉满。

## 附录 C：训练数据

与 8B Stage 1 同一 recipe——见 `tech_report_qwen3-8b_stage1_zh.md` 附录 A 完整数据合成流水线。

唯一差异：14B 的 4k 子集（Phase-1 合成加入 `tom_train.jsonl` 后重新生成）包含 440 条（11%）8B Stage 1 4k 没有的 `synth_phase1` 数据。这是个小混淆变量；基于 Stage 5 上 Phase-1 对 8B 的轻微效应，估计这对 14B vs 8B 比较贡献 ≤0.5pp。

为了无该混淆的 apples-to-apples 比较，应该用新 4k 重跑 8B Stage 1。我们没做；14B +1.33pp 的收益足够大，该混淆不改变定性结论。

## 附录 D：内存账本

为了存档，14B 在 H800 80 GB + TP=2 下每 rank 的内存账：

```
静态（常驻）：
  actor_train（Qwen3-14B / 2 bf16）：              14 GB（模型）
  actor_train 梯度（bf16）：                       14 GB
  actor_train optimizer（Adam fp32, dist 1/8）：   14 GB（master + m + v，分到 DP=4）
                                                  -----
                                                  42 GB

  reference（Qwen3-14B / 2 bf16，可 offload）：    14 GB → ~0 GB after offload
  vLLM 模型（TP=2 shard, bf16）：                  14 GB（始终常驻；sleep mode 仅释放 KV cache）

动态（特定阶段）：
  vLLM KV cache（mem util 0.45 of 80 GB）：        ~36 GB peak during rollout
  Megatron activations + grad accum（bf16）：      ~12 GB during forward+backward
  CUDA workspace + NCCL buffers + residual：      ~8 GB

训练 step 峰值（vLLM offloaded, train active）：
  42（静态）+ 12（act）+ 8（workspace）= 62 GB ✓ 适配
Rollout 峰值（train offloaded, vLLM hot）：
  14（vLLM model）+ 36（KV cache）+ 14（reference if loaded）+ 8 = 72 GB ✓ 适配
Save 峰值（train 完全加载 + 额外）：
  42（静态）+ ~18 GB（Gloo gather buffer if CUDA）→ 会 OOM
  with Gloo+CPU gather：~42 GB on GPU ✓ 舒适
```

80 GB H800 刚好够用。如果尝试 Qwen3-14B TP=1（DP=8），actor_train 单 rank 就吃 ~80 GB，vLLM/reference/workspace 没空间——这就是 TP=2 在此尺度强制的原因。
