# Plan A 设计：解开思考预算枷锁（Stage 22 / v4.0）

> 配置文件：`configs/tombench-rlvr/rlvr_config_stage22_planA_1x8_14b.yaml`
> 依据：`insight_and_optimization_2026-05-30.md` §3、§6（模型汤证明缝合无效，必须改训练目标）

## 1. 要解决的根因（三重耦合的长度枷锁）

| 层 | v3.1 (stage14) | 作用 |
|---|---|---|
| `response_length` | 256 | rollout 生成上限，思考被截在 256 token |
| reward `aggregation` | multiplicative `r_fmt×r_out×r_len` | 任一项为 0 → 总 reward 0 |
| reward `l_max` | 256 | `r_len=sigmoid_window` 在 >256 token 处掉到 ~0 → **答对但思考长 = reward 归零** |

三者叠加 = 模型被强烈训练成"压缩思考"。eval 实测：base 在 Hi-ToM 思考 ~11k chars 得 0.795，训练版压到 2.6-4.4k chars 崩到 0.70-0.74，且退化随推理深度单调放大。

## 2. Plan A 的改动（vs v3.1，数据不变做干净 A/B）

| 项 | v3.1 | Plan A | 理由 |
|---|---|---|---|
| `response_length` | 256 | **2048** | 让 rollout 能想长链 |
| reward `aggregation` | multiplicative | **weighted_sum** | 长度变 5% 软偏好，不再是"答对也归零"的硬门 |
| reward 权重 | 0.05/0.85/0.10 | **0.05/0.90/0.05** | correctness 占 90%，长度仅 5% |
| reward `l_max` | 256 | **2048**（`l_max_long` 4096 / `l_max_short` 512） | 长度窗口覆盖整个预算 |
| `use_kl_loss` | off | **true, `kl_loss_coef` 0.001** | **锚定 base**，保住 base 的长链泛化（0.001 是 stage9 验证过的值，stage3 的 0.01 太重） |
| `pretrain` / `reference` | stage12 | **base Qwen3-14B** | 从泛化最优点出发，KL 把它按在这个 basin 里 |
| `actor_infer.max_model_len` | 2048 | **4096** | 容纳 prompt 1024 + response 2048 |
| `validation.max_new_tokens` | 64 | **2048** | 量思考质量，而非 64-token 直答（长思考模型 64 token 几乎得 0） |
| `save_steps` | 50 | **25** | 频繁存档，按 4-bench 均值选 ckpt（Plan E），防 stage17 那种 step100 后崩 |
| `difficulty_*` 窗口 | (0.15, 0.80) | (0.10, 0.90) | 从 base 出发训练集通过率更低更散，放宽 learnable-middle |

**为什么这条路和模型汤不同**：soup 是两个静态权重的线性平均，改不了"思考长度"这个维度（v3.1 本身就是短思考者）。Plan A 用长预算 + 不惩罚长度 + KL 锚定，让模型在 RL 中**主动保持长链**同时学对 ToMBench——这是 soup 到不了的新轴。

## 3. 假设与成功判据

**假设**：从 base + 长思考 + KL 锚定，RL 能在**不付泛化税**的前提下加 ToMBench。

**成功判据（按优先级）**：
1. **主判据**：4-benchmark 最优协议均值 **> base 0.7603**（v3.1 只有 0.7305）。
2. ToMBench cot ≥ 0.77（接近 v3.1 的 0.7816）。
3. **Hi-ToM cot 不塌**：保持 ≥ 0.76（base 0.7950；v3.1 崩到 0.7033）——这是 Plan A 是否真的解开枷锁的核心信号。
4. 训练时 rollout 平均思考长度应明显 > 256（worker 日志 `response_lengths`），证明枷锁确实解开。

**失败信号**：若 Hi-ToM 仍随步数下滑、或均值仍 < base，则说明"ToMBench 增益本质就要付泛化税"，下一步只能靠 Plan B（多 benchmark 直接进 reward）。

## 4. 要盯的风险

- **算力**：response_length 256→2048 ≈ 8× rollout 生成量，每 step 明显变慢。先按 max_steps=200 跑，必要时减。
- **晚期崩塌**：参考 [[stage17_drgrpo_late_collapse]]，step100 后可能 greedy val 崩。save_steps=25 + 按均值选 ckpt 兜底。
- **从 base 起步 ToMBench 偏低**：base ToMBench cot 0.7609 < stage12 起点。若 200 步学不到 0.77，考虑备选方案。

## 5. 备选方案

- **A2（若 A1 ToMBench 上不去）**：`pretrain` 改 stage12（ToMBench 已强），`reference` 仍 = base。保留 ToMBench 起点，KL 往 base 拉回泛化。风险：stage12 已在压缩 basin，KL 可能拉不动长链。
- **A + Plan B**：把 Hi-ToM/SocialIQA/EmoBench 直接纳入 reward worker + 训练数据（prompt_length 需提到 4096 容纳 Hi-ToM 长故事）。最彻底，但偏离干净 A/B。

## 6. 启动方式

```bash
# 在 181 上（base/data/mount 均已就绪）
#   /mnt/models/Qwen3-14B  ← /data_nvme/grj-projects/models/Qwen3-14B (base, 8 shards ✓)
#   /mnt/data/tom_train_stage14_weighted.jsonl ✓   /mnt/data/tombench_eval_subset500.jsonl ✓
# 用与 stage14/19 相同的 RLVR 启动路径（docker/train compose + 这份 config），
# exp_name=qwen3-14B-tombench-rlvr-stage22-planA-1x8，输出到 /mnt/output/$exp_name。

# 跑完每个 ckpt 用本实验框架评 4 benchmark（best-protocol 均值）选最优：
#   转 HF → run_soup_eval.sh 风格起服务 → parallel_eval.py 四项 → soup_summary.py 比较
```

> 一句话：**把"答对但想得长 = reward 归零"这条规则删掉，从 base 起步、KL 锚住长链，看能不能第一次做到"加 ToMBench 不掉泛化"。**
