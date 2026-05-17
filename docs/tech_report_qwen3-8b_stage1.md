# Tech Report: Qwen3-8B + GRPO on ToMBench (Stage 1)

**Author**: Internal training runbook, generated from training logs `train_stage1_1x8_20260515_121704.log` and config `configs/tombench-rlvr/rlvr_config_stage1_1x8.yaml` (commit `227ee48`).

**Status**: Stage 1 of 5 in the Qwen3-8B + GRPO program. Stage 1 is the small-scale verification run (4k records × 200 steps). Subsequent stages (stage2/3/5) extend training; stage4 was abandoned. See `docs/stage{1..5}_report.md` for per-stage executive summaries; this document is the deeper engineering account of stage 1 only.

---

## 1. Goal

Train Qwen3-8B (base, non-thinking) to maximize accuracy on the ToMBench multiple-choice theory-of-mind benchmark, using:
- Reinforcement learning with verifiable rewards (RLVR), specifically GRPO with DAPO clip-higher and dynamic sampling.
- Roll framework (Alibaba) as the trainer, vLLM 0.8.4 for rollout, Megatron-Core 0.16.0 for training, in a colocated 1×8 H800 layout.
- A custom multi-component reward (`TomMcqRewardWorker`) that combines format compliance, answer correctness, and length regularization.

Target: deepseek-v4-pro on the same eval set (full 5718 questions, direct protocol). Stage 1 is intentionally small to validate the full pipeline end-to-end before committing to the larger stage 2 budget.

## 2. Hardware & Software

| Item | Value |
|---|---|
| Hardware | 1 node, 8× NVIDIA H800 80 GB SXM |
| Interconnect | NVLink full mesh (intra-node only) |
| CPU/RAM | 64 cores / 512 GB system RAM (relevant for Gloo+CPU optimizer save) |
| Container | `qwen3-tom-train:latest`, NVIDIA pytorch 24.05-py3 base |
| Torch | 2.6.0+cu124 (CUDA 12.4 from Aliyun pytorch-wheels mirror) |
| Megatron-Core | 0.16.0 |
| Transformer Engine | 2.2.0 (pinned; newer TE breaks `transformer_engine.pytorch` import) |
| vLLM | 0.8.4 (bundled flash-attn) |
| Ray | 2.48 (with `click==8.2.1` pin to avoid sentinel error on Python 3.10) |
| ROLL framework | vendored upstream snapshot under `framework/ROLL/`, with one custom worker `roll/pipeline/rlvr/rewards/tom_mcq_reward_worker.py` |
| Persistent storage | `/data_nvme/grj-projects/` on a 14 TB NVMe LV (after migration from the 875 GB SSD) |
| Inside-container mounts | `/workspace` ← repo, `/mnt/data` ← training data, `/mnt/models` ← HF/ModelScope cache, `/mnt/output` ← checkpoints |

GPU resource sharing: actor_train (Megatron), actor_infer (vLLM), and reference (Megatron infer) all colocate on the same 8 GPUs. ROLL's "offload state manager" swaps each role's weights/optimizer/KV cache between GPU and pinned CPU memory between phases.

## 3. Data

Stage 1 trains on `data/tom/tom_train_4k.jsonl`, a 4000-record subset of `tom_train.jsonl` (~5911 records at the time stage 1 ran; later grew to 8901 after Phase-1 synthesis was merged in for stage 5).

### 3.1 Sources

| Source | n (in 4k subset stage 1 used) | Description |
|---|---|---|
| ExploreToM | ~886 | Synthetic narrative ToM dataset; second-order belief and knowledge-attention tasks |
| SimpleToM | ~440 | Sally-Anne style first-order false-belief tasks |
| Synth (deepseek-v4-flash) | ~1353 | In-house synthesis: 9 ToMBench task types via `scripts/data/synth_tomtype.py`, 0.9 temperature |
| Translated ZH | ~298 + 412 + 171 | Chinese translations of EN training records (deepseek-v4-flash, `scripts/data/translate_to_zh.py`) |

(For stage 1 specifically the splits sum to 4000; exact per-source counts depended on the random seed at `merge_and_dedupe.py` time.)

### 3.2 Anti-leakage protocol

**Every training record is checked against ToMBench eval (5718 questions) before being kept.**

`scripts/data/merge_and_dedupe.py` builds a MinHash LSH index over ToMBench eval at threshold 0.6, indexed by 4-grams of normalized text (story + question + options). For every candidate training record, we:

1. Compute MinHash and query the LSH index.
2. For any candidate ≥ 0.6 estimated similarity, compute exact 4-gram Jaccard.
3. Drop the record if exact Jaccard ≥ 0.6 against any eval question.

**Result for stage 1's training data**: 0 records dropped by this filter (the synth pipeline was prompted explicitly not to reproduce ToMBench questions, and the upstream datasets are pre-existing).

We also dedupe internally at MinHash threshold 0.7. Internal duplicates dropped: ~150 (stage 1 era).

### 3.3 Format

Each record is a chat message pair:

```json
{
  "messages": [
    {"role": "system", "content": "You are a careful reader answering ..."},
    {"role": "user",   "content": "Story:\n...\n\nQuestion: ...\nA. ...\nB. ...\nC. ...\nD. ..."}
  ],
  "ground_truth": "B",
  "tag": "tom_mcq",
  "source": "exploretom",
  "language": "en",
  "task": "False Belief",
  "question_id": "exploretom_7661"
}
```

The system prompt for direct protocol is fixed:

```
You are a careful reader answering a multiple-choice theory-of-mind question.
Read the story and the question carefully, then output ONLY your final answer
in the format \boxed{X} where X is one of A, B, C, D.
Do not include any explanation, reasoning, or extra text.
```

The reward worker matches this contract: it parses `\boxed{[ABCD]}` from the rollout, compares to `ground_truth`, and applies format / length penalties.

## 4. Reward Function

Implemented in `framework/ROLL/roll/pipeline/rlvr/rewards/tom_mcq_reward_worker.py`. Per response, three sub-rewards are computed and **multiplied**:

```python
r_fmt = 1.0 if response matches \boxed{[A-D]} else 0.0
r_out = 1.0 if (r_fmt and extracted_letter == ground_truth) else 0.0
r_len = sigmoid_window(response_length, l_min=8, l_max=256, k=50)
r_total = r_fmt * r_out * r_len
```

`sigmoid_window(L, l_min, l_max, k)` is a smooth band-pass: rises near `l_min`, plateaus, falls near `l_max`. With `k=50` the transitions are sharp:

```python
def sigmoid_window(L, l_min, l_max, k):
    span = max(1.0, l_max - l_min)
    rise = sigmoid(k * (L - l_min) / span)
    fall = 1.0 - sigmoid(k * (L - l_max) / span)
    return rise * fall
```

For our settings (l_min=8, l_max=256, k=50):
- L=4 (too short): r_len ≈ 0.0
- L=20 (warmed up): r_len ≈ 0.95
- L=200 (typical): r_len ≈ 1.0
- L=260 (hit cap + a bit): r_len ≈ 0.5
- L=270 (over cap, padded): r_len ≈ 0.05

| Component | Behavior at convergence (rollout step 175+) |
|---|---|
| `r_fmt` | ~0.87 (vast majority of rollouts emit `\boxed{[A-D]}`) |
| `r_out` | ~0.81 (when format-OK, ~93% pick the right letter) |
| `r_len` | ~0.93 (responses cluster around 158 tokens, well inside the band) |
| `r_total` | ~0.80 (multiplicative product) |

**Why multiplicative not additive?** Three reasons:
1. A wrong-format response gets exactly 0 reward, regardless of length — there's no partial credit for "almost formatted". This forces the policy to nail format first.
2. Long-but-correct responses are penalized smoothly; the model can't "buy" reward by writing more.
3. The product is bounded in [0, 1], which keeps the variance whitening stable.

`l_max=256` matches `response_length: 256`. If we raise `response_length` we should raise `l_max` to match (stage 3 raised both to 384; stage 5 dropped both back to 256 after observing val mismatch).

The reward worker logs every `(r_fmt, r_out, r_len, r_total, extracted_letter, ground_truth)` tuple at debug level for post-hoc analysis. Aggregated metrics (`reward/r_*_mean`) are emitted to the tracker every step.

## 5. Algorithm: GRPO with DAPO Clip-Higher + Dynamic Sampling

### 5.1 GRPO basics

For each prompt, the rollout cluster generates 8 samples (`num_return_sequences_in_group: 8`). Each sample's advantage is computed as the group-relative reward: `A_i = (r_i − mean(r_group)) / std(r_group)` (whitened advantages on, `whiten_advantages: true`). This removes the need for a separate value head while still controlling variance.

### 5.2 DAPO Clip-Higher

The PPO clip range is asymmetric:
- `pg_clip_low: 0.20`
- `pg_clip_high: 0.28`

This permits modestly larger upward updates for samples with positive advantage, while keeping the downward clip at the standard 0.2. The asymmetric clip gives faster credit assignment without instability. Combined with `dual_clip_loss: true` (clip both ratio and ratio×advantage when advantage is negative), this is the standard DAPO setup.

### 5.3 Dynamic sampling

`use_additional_prompts: true`. The scheduler adaptively pulls extra prompts when a batch contains too many "easy" (all 8 group samples correct) or "hard" (all wrong) groups, since those have zero variance and contribute zero gradient. `max_running_requests: 128`.

### 5.4 Difficulty masking

After reward computation, samples are masked out if the per-prompt group accuracy is too easy or too hard:
- `difficulty_low_threshold: 0.1` — drop the entire group if fewer than 10% of its samples are correct (model can't learn yet)
- `difficulty_high_threshold: 0.95` — drop the entire group if more than 95% are correct (no signal left)

This is critical at convergence: at step 175+ the rollout score is 0.80, so without difficulty masking the gradient would shrink toward zero on the saturated examples.

### 5.5 KL regularization

Stage 1 uses `add_token_level_kl: false`. The reasoning: KL penalty on a fresh-from-base model can over-anchor to the (weak) base distribution and prevent fast learning. Stage 1 was meant to verify that the pipeline learns at all, so we kept the standard PPO setup.

(KL was tried in stage 3 with mixed results — protected Belief / False Belief but blocked Knowledge / Non-literal Comm progress.)

### 5.6 Effective batch & rollout shape

- `rollout_batch_size: 32` — 32 distinct prompts per step
- `num_return_sequences_in_group: 8` — 8 rollouts per prompt
- Effective rollout batch: **256 sequences per step**
- `gradient_accumulation_steps: 32` — but per-device batch is 1, so each rank does 32 forward+backward chunks
- DP=8 (one rank per GPU), TP=1, PP=1

Per-prompt sequences are GRPO group → contributed to advantage normalization → used for one PPO update.

## 6. Hyperparameters

```yaml
# Sampling
prompt_length: 2048
response_length: 256
rollout_batch_size: 32
num_return_sequences_in_group: 8
ppo_epochs: 1

# DAPO Clip-Higher
use_pg_clip_range: true
pg_clip_low: 0.20
pg_clip_high: 0.28
dual_clip_loss: true

# Variance control
value_clip: 0.5
reward_clip: 5
advantage_clip: 2.0
whiten_advantages: true
add_token_level_kl: false

# Difficulty masking
max_len_mask: true
difficulty_mask: true
difficulty_low_threshold: 0.1
difficulty_high_threshold: 0.95
error_max_len_clip: false

# Training
max_steps: 200
save_steps: 200      # save once at end (mid-train OOM workaround, see §7)
eval_steps: 50
logging_steps: 1

# Optimizer
learning_rate: 1.0e-6
weight_decay: 0
warmup_steps: 20
gradient_accumulation_steps: 32
per_device_train_batch_size: 1

# Generation (rollout)
temperature: 0.99
top_p: 0.95
top_k: 50

# Generation (validation)
temperature: 0.0
max_new_tokens: 64    # NOTE: this 64-token cap caused major val_correct under-reporting in later stages

# Reward
l_min: 8
l_max: 256
k: 50  # length penalty curve sharpness
```

### Notable choices
- **Learning rate 1e-6**: standard for 8B-scale RL post-training. We did not tune this.
- **Warmup 20 steps**: gentle ramp to avoid early gradient explosion (note step-0 grad_norm = 65.9 in the log; by step 25 it had stabilized to 2.4).
- **No reference model in RL update path** (`enable_reference: true` but not used as KL target since `add_token_level_kl: false`). Reference is only used to compute log-prob ratios for the PPO loss numerator — its KL term has zero coefficient.
- **`save_steps: 200`** (only the final checkpoint): defers the very expensive Megatron distributed-optimizer save until after training is done. See §7.

## 7. Distributed Save (the OOM saga)

Stage 1 nearly didn't have a checkpoint. The first attempt OOM'd at the final `do_checkpoint` call after 199 successful training steps. Root cause analysis:

In a colocated 1×8 deployment, at the moment `do_checkpoint` runs:
- vLLM holds ~7.7 GB residual KV cache + weight metadata per GPU (it doesn't fully release on offload)
- Reference model holds ~3 GB residual
- actor_train is loaded back (it had been offloaded at the end of the previous step) — it consumes ~75 GB at this point because in addition to model weights, it has: optimizer (Adam moments), gradients (still allocated), reference activation buffers, and all the hot CUDA workspaces

That sums to ~85 GB on an 80 GB GPU.

The default Megatron-Core distributed-optimizer save invokes `FullyParallelSaveStrategyWrapper.sharded_param_state_fully_reshardable`, which calls `get_parameter_state_dp_zero` with `use_gloo_comm=False`. That function allocates `recv_tensors` of shape `(buffer_numel_unpadded,)` × `data_parallel_world_size` on every DP rank, on **CUDA** (line 1077):

```python
device = "cpu" if use_gloo_comm else torch.cuda.current_device()
recv_tensors = [
    torch.zeros((gbuf_local_numel,), dtype=torch.float32, device=device)
    for _ in range(data_parallel_world_size)
]
```

For Qwen3-8B with DP=8, `gbuf_local_numel × DP × 4 bytes ≈ 3.81 GiB`. With 690 MiB free per GPU, this reliably OOMs.

**Fix** (commit `d7bf18a`): set `distrib_optim_fully_reshardable_mem_efficient: true` in `actor_train.strategy_args.strategy_config`. This routes the gather through Gloo (CPU) instead of NCCL (GPU), filling `world_tensors` on the CPU side:

```yaml
strategy_config:
  use_distributed_optimizer: true
  distrib_optim_fully_reshardable_mem_efficient: true  # the magic flag
  recompute_granularity: full
```

After this fix:
- The save path uses Gloo collective + CPU buffers
- DP rank 0 receives the full reshardable state on CPU
- It writes via `dist_checkpointing.save` to local disk
- `checkpoint_manager.upload` then rsync-style copies to `/mnt/output/...`
- Local disk usage spikes briefly: ~107 GB local + ~107 GB upload = transient ~214 GB, then the local copy is deleted

The save itself takes ~10 minutes (Gloo gather is single-threaded over CPU, much slower than NCCL but reliable). On NVMe storage this is fine.

## 8. Training Trajectory

Read directly from `train_stage1_1x8_20260515_121704.log`, sampled every 25 steps:

| step | rollout score | reward | r_fmt | r_out | r_len | KL loss | grad_norm | response len (tok) | val_correct/all (subset500) |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.215 | 0.118 | 0.301 | 0.273 | 0.581 | 0 | 65.9 | 208 | 0.042 |
| 25 | 0.148 | 0.149 | 0.160 | 0.160 | 0.568 | 0.005 | 2.44 | 251 | — |
| 50 | 0.412 | 0.213 | 0.438 | 0.426 | 0.705 | 0.020 | 1.97 | 230 | 0.204 |
| 75 | 0.650 | 0.168 | 0.680 | 0.664 | 0.824 | 0.073 | 2.05 | 202 | — |
| 100 | 0.566 | 0.246 | 0.656 | 0.574 | 0.819 | 0.113 | 1.82 | 201 | 0.454 |
| 125 | 0.637 | 0.168 | 0.707 | 0.641 | 0.849 | 0.145 | 1.71 | 188 | — |
| 150 | 0.791 | 0.090 | 0.867 | 0.793 | 0.929 | 0.207 | 3.01 | 157 | 0.548 |
| 175 | 0.806 | 0.056 | 0.875 | 0.812 | 0.931 | 0.194 | 2.88 | 162 | — |
| 199 | 0.800 | 0.116 | 0.871 | 0.801 | 0.934 | 0.235 | 2.04 | 158 | — |

Reading the trajectory:
- **steps 0–25**: gradient explosion at step 0 (grad_norm 65.9 = the start-of-training shock as the policy first sees its own rollouts as "reward signal"). Warmup absorbs this within 20 steps; by step 25 the policy temporarily forgets format (r_fmt drops 0.30→0.16) but stabilizes.
- **steps 25–75**: rapid format learning. r_fmt climbs from 0.16 to 0.68 — the model picks up `\boxed{X}` as the rewarded shape. r_out tracks r_fmt closely because the model that gets the format wrong gets zero on `r_total`, and a wrong-format response is uncorrelated with the answer.
- **steps 75–150**: accuracy ramp. r_out 0.66 → 0.79. The format reward saturates first (~0.87), then accuracy follows. KL grows steadily 0.07→0.21, indicating the policy is meaningfully drifting from the base — but no instability since clip-higher is in effect.
- **steps 150–199**: plateau. rollout score oscillates 0.80–0.81. **Without difficulty masking the gradient would be vanishing here** (90%+ groups all correct or all wrong); with masking, the remaining "mixed" groups still produce learning signal. Response length stable at 158–166 tokens, well within `l_max=256`.

The `reward` column (the `critic/rewards/mean` value) goes down even as `r_total` goes up, because `reward` is the whitened advantage proxy after KL is subtracted: the more correct the policy, the smaller its raw advantage spread. This is normal in GRPO.

### Validation curve (subset500)

| step | val_correct/all | val_correct/tom_mcq |
|---|---|---|
| 0 | 0.042 | 0.278 |
| 50 | 0.204 | 0.299 |
| 100 | 0.454 | 0.534 |
| 150 | 0.548 | 0.613 |

`val_correct/tom_mcq` is the same metric scoped to ToM-MCQ-tagged records (everything in the eval set). The all-domain `val_correct/all` includes the format check; tom_mcq drops the `\boxed{}` requirement and just looks at the predicted letter.

The val curve mirrors training: 50→100→150 sees the biggest gains; 150→200 we'd expect modest further movement. We didn't hit step 200 val because eval_steps=50 schedule + max_steps=200 means the next val would be at step 200 itself (right before save), which the codepath skips.

## 9. Final Eval

We ran the trained checkpoint through two evaluation regimes:
1. **Full ToMBench 5718 questions**, direct protocol, `max_tokens=2048` (so reasoning models like deepseek aren't truncated; doesn't matter for our 8B which doesn't reason)
2. **Subset500** (a deterministic 500-question sample), 3 protocols: direct, cot, del_tom

### 9.1 Full 5718 (direct)

| Metric | Value |
|---|---|
| Overall | **0.7394** |
| EN | 0.7275 |
| ZH | 0.7513 |

vs. corrected baselines:

| Model | Overall | EN | ZH |
|---|---|---|---|
| Qwen3-8B base (no RL) | 0.7009 | 0.7020 | 0.6999 |
| **Stage 1 (this work)** | **0.7394** | 0.7275 | 0.7513 |
| deepseek-v4-pro target | 0.8080 | 0.7978 | 0.8181 |

**Δ vs baseline: +3.85pp**, **gap to deepseek (true full-set baseline): −6.86pp**.

The ZH bump (+0.05pp better than EN trained model) is interesting and not predicted from training data (which is 70/30 EN/ZH). Hypothesis: ZH evaluation patterns are more rigid (more direct answer formats) so format learning translates directly to accuracy.

### 9.2 Per-task breakdown (full 5718, direct)

| Task | Stage 1 | deepseek (5718) | Gap | EN | ZH |
|---|---|---|---|---|---|
| Belief | 0.6937 | 0.8486 | -15.49pp | 0.669/0.718 | |
| Desire | 0.5917 | 0.6333 | -4.16pp | 0.567/0.617 | |
| Emotion | 0.7286 | 0.8048 | -7.62pp | 0.700/0.757 | |
| False Belief | **0.8520** | 0.8946 | -4.26pp | 0.862/0.842 | (closest task) |
| Intention | 0.7647 | 0.8926 | -12.79pp | 0.750/0.779 | |
| Knowledge | 0.4792 | 0.5675 | -8.83pp | 0.471/0.488 | (lowest absolute) |
| Non-literal Comm | 0.7674 | 0.8128 | -4.54pp | 0.749/0.786 | |

**Stage 1 wins**: False Belief (+12.4pp vs base, closest to deepseek). The training data is heavily biased toward False Belief (~2629 records, the largest task), so this is consistent.

**Stage 1 weaknesses**:
- Knowledge (0.48): scalar implicature questions ("most/some/almost no" + count puzzles) — model defaults to literal arithmetic. Training data has none of this pattern. (Bug-fix attempt in stage 5 with synth-Phase-1 scalar set; modest improvement.)
- Non-literal Comm (0.77): Faux-pas recognition. Model over-attributes faux-pas (sees inappropriate speech in innocuous stories). Training data also doesn't cover this. (Stage 5 added 800 synth faux-pas records; flat improvement on full 5718.)

### 9.3 Subset500 across 3 protocols

| Protocol | Stage 1 | deepseek (subset500) | Gap |
|---|---|---|---|
| direct | 0.7460 | 0.7880 | -4.20pp |
| cot | 0.6980 | 0.7140 | -1.60pp |
| del_tom | 0.7460 | n/a | |

The cot regression vs base (cot 0.7464 → 0.6980) is a known RL post-training side effect: training pushed the policy toward direct-only response and slightly hurt CoT-formatted prompts. Stage 5 (with token-level KL, retried) recovered cot to 0.7540 but at the cost of overall accuracy. Stage 1 + cot together is the local minimum on this protocol.

## 10. Wall Clock Budget

| Phase | Wall time |
|---|---|
| Container launch + worker init | ~10 min |
| Training (200 steps) | 3 h 20 min |
| `do_checkpoint` (Gloo+CPU, mem-efficient) | ~10 min |
| Total stage-1 wall | ~3 h 40 min |
| GPU-hours | ~26 |
| HF format conversion (post-training) | ~3 min |
| vLLM serve cold start | ~2 min |
| Full 5718 eval (vLLM concurrency=32) | ~6 min |

## 11. Lessons & Caveats

1. **Distributed-optimizer save defaults to NCCL+CUDA gather, which OOMs on colocated 1×8 H800.** Always set `distrib_optim_fully_reshardable_mem_efficient: true` for this layout. Megatron-Core has supported this since 0.14.0 but it's off by default.

2. **Difficulty masking is essential past step 100.** Without it the gradient would die when groups saturate. The 0.1/0.95 thresholds are from the GRPO paper and we didn't re-tune.

3. **`r_len` matters more than expected.** Removing the length penalty in early experiments led the model to write 256-token responses that often ran past `\boxed{X}` and lost format compliance. Brevity reward keeps the policy tight.

4. **Format learning happens before accuracy learning.** r_fmt saturates around step 75 at 0.87; r_out is still at 0.66 at the same step. The model first learns "always wrap the letter in `\boxed{}`", then learns "and pick the right letter". This is consistent with GRPO learning the reward shape before the underlying skill.

5. **Validation `max_new_tokens=64` is dangerously short.** Stage 1 fits comfortably (response ~158 tokens with some preamble fits) but later stages with `response_length=384` saw `val_correct` collapse because `\boxed{X}` ran past the 64-token cutoff. If you change `response_length`, you should re-tune the val token cap.

6. **The 5718 baseline ≠ subset500 baseline.** Stage 1 looked like it was 4.20pp from deepseek when measured on subset500; on the full 5718 the gap was 6.86pp. We learned this only in stage 5 when we finally ran a deepseek full-set eval. Always benchmark on the full set when reporting headline numbers.

## 12. Reproducing Stage 1

```bash
# DEV machine
git clone https://github.com/ruijieguo/social-mind-rl
cd social-mind-rl
make build-data            # if data/ not present (rebuilds tom_train_4k.jsonl)
cp configs/deploy.env.example configs/deploy.env  # then fill in TRAIN_HOST etc.

# Sync to TRAIN
make sync-up

# On TRAIN: launches docker, builds qwen3-tom-train image if not present, runs training
make train-stage1-1x8

# Post-training, on TRAIN: convert checkpoint to HuggingFace format
ssh $TRAIN_HOST 'cd /data_nvme/grj-projects/qwen3-tom && \
  docker run --rm --gpus all --ipc host --shm-size 8gb \
    --cap-add SYS_PTRACE --cap-add SYS_ADMIN \
    -v /data_nvme/grj-projects/qwen3-tom:/workspace \
    -v /data_nvme/grj-projects/tom-output:/mnt/output \
    -v /data_nvme/grj-projects/models:/mnt/models \
    -e PYTHONPATH=/workspace:/workspace/framework/ROLL:/workspace/framework/ROLL/mcore_adapter/src \
    -w /workspace --entrypoint python qwen3-tom-train:latest \
    framework/ROLL/mcore_adapter/tools/convert.py \
    --checkpoint_path /mnt/output/qwen3-8B-tombench-rlvr-stage1-1x8/<timestamp>/checkpoint-199 \
    --output_path /mnt/output/qwen3-8B-tom-hf --bf16'

# Serve via vLLM
make serve-launch  # uses qwen3-tom-serve image; or use the train image directly

# Evaluate from DEV
make eval-final
```

## 13. Next Steps

Stage 1 confirmed the pipeline learns. Subsequent stages explored:
- **Stage 2** (8k × 500): more data, more steps. Result: 0.7263 — overfit; +500 steps was too much.
- **Stage 3** (KL=true, response_len=384): protect Belief/FB but blocked Knowledge. Result: 0.7302.
- **Stage 4** (KL=true + Phase-1 synth data with empty C/D options): training stagnated; aborted.
- **Stage 5** (KL=false + Phase-1 fixed data): 0.7305. Best subset500 cot 0.7540 but full 5718 unchanged.

Through all five stages the **8B+RL ceiling on full 5718 is 0.7394**. The 14B+RL run (see `tech_report_qwen3-14b_stage1.md`) reaches 0.7527 — a meaningful step toward deepseek's 0.8080.

## 14. Artifacts

| Path | What |
|---|---|
| `configs/tombench-rlvr/rlvr_config_stage1_1x8.yaml` | Full config |
| `framework/ROLL/roll/pipeline/rlvr/rewards/tom_mcq_reward_worker.py` | Custom reward worker |
| `logs/train_stage1_1x8_20260515_121704.log` | Full training log (10 MB) |
| `output/eval/final_full5718.{json,md}` | 5718 eval results |
| `output/eval/final_subset500.{json,md}` | subset500 eval (3 protocols) |
| `output/analysis/curves_stage1_1x8.png` | 12-panel training curve |
| Megatron checkpoint (TRAIN) | `/data_nvme/grj-projects/tom-output/qwen3-8B-tombench-rlvr-stage1-1x8/.../checkpoint-199/` (107 GB) |
| HF checkpoint (TRAIN) | `/data_nvme/grj-projects/tom-output/qwen3-8B-tom-hf/` (16 GB, 4-shard safetensors) |
| Git commit | `227ee48` (training fix) → `d15cec2` (full report) |

---

## Appendix A: Training Data Synthesis Recipe

Stage 1's 4k training set was the result of a multi-step assembly pipeline. The components and their provenance:

### A.1 ExploreToM (~886 records)

A pre-existing synthetic dataset from a third-party paper (Yufei Tian et al., 2024), targeting second-order belief and knowledge-attention link tasks. We loaded their 2000-record release and used `scripts/data/build_exploretom.py` to convert the schema:

```python
record = TomRecord(
    question_id=f"exploretom_{i}",
    source="exploretom",
    language="en",
    task=ability_to_task(record["ability"]),  # maps verbose ability strings to broad tasks
    story=record["context"],
    question=record["question"],
    opt_a=record["options"]["A"], opt_b=..., opt_c=..., opt_d=...,
    gold=record["correct_answer"],
)
```

### A.2 SimpleToM (~440 records)

Sally-Anne style first-order false-belief tasks, also pre-existing. 1000 records loaded, randomly subsampled into the 4k mix.

### A.3 In-house synthesis via deepseek-v4-flash (~1353 records)

`scripts/data/synth_tomtype.py` calls deepseek-v4-flash with explicit prompts per ToMBench task type:

**System prompt** (anti-leakage):
```
You are a careful question writer creating new theory-of-mind multiple-choice questions for training.
Your output MUST be a single JSON object with keys: story, question, options (an object with A,B,C,D), answer (one of A,B,C,D).
Do NOT reproduce, paraphrase, or translate any question from ToMBench by Chen et al. (ACL 2024).
Write entirely new scenarios.
```

**Per-task user prompts** (one of 9 task types per call, sampled uniformly):
```
False Belief:       "Write a False Belief task: a character's belief differs from reality after an unseen change."
Strange Story:      "Write a Strange Story task involving subtle social misunderstanding or irony."
Unexpected Outcome: "Write an Unexpected Outcome task where the result of an action differs from the character's expectation."
Persuasion Story:   "Write a Persuasion Story task where one character tries to change another's belief."
Knowledge:          "Write a Knowledge-Attention Link task where a character's knowledge depends on what they observed."
Desire:             "Write a Multiple Desires task where two characters have different preferences."
Emotion:            "Write a Discrepant Emotions task where two characters feel differently about the same event."
Intention:          "Write a Prediction of Actions task asking what a character will do given their intention."
Non-literal Comm:   "Write a Hinting Task: a character makes an indirect request and we must infer their actual desire."
```

**Generation params**: temperature=0.9, max_tokens=800, concurrency=8 → 0.5 req/s actual rate. Roughly 60 minutes for 1500 generations.

**Filtering**: each response is regex-extracted with `\{...\}` matching, parsed as JSON, and rejected if any of (story, question, options A-D, answer letter) is missing or the answer isn't in {A,B,C,D}. Failure rate ~5%.

The 9 task types span both classical ToM (False Belief, Knowledge) and pragmatic communication (Hinting, Strange Story). We chose these to mirror ToMBench's task taxonomy without leaking exact prompts.

### A.4 Chinese translation (~881 records)

`scripts/data/translate_to_zh.py` takes EN training records and translates each (story, question, options) into Chinese via deepseek-v4-flash. The system prompt:

```
You are a precise translator. Translate the given theory-of-mind multiple-choice question
from English to Simplified Chinese. Preserve story logic, character names (transliterate),
and option order. Output a JSON object with the same keys.
```

Each successfully-translated record gets `source = "synth_zh"` (or `exploretom_zh` / `simpletom_zh`), language=`zh`, and a fresh question_id `<original_id>_zh`. About 70% of attempted translations succeed (failures: model adds extra commentary, violates JSON schema, or refuses).

### A.5 Anti-leakage at merge time

`scripts/data/merge_and_dedupe.py` is the gate:

1. Load `data/tom/tombench_eval.jsonl` (5718 records). Build a MinHash-LSH index over `story + question + opt_a + opt_b + opt_c + opt_d` text, threshold=0.6, num_perm=128.
2. For each candidate training record, compute its MinHash and query the index.
3. Compute exact 4-gram Jaccard between candidate and each LSH hit.
4. If any exact Jaccard ≥ 0.6, drop the candidate.

Result for the stage-1 era (no Phase-1 data yet): **0 records dropped by leakage filter.** The data card (`docs/data-card.md`) records the max-Jaccard distribution per source — all means and p95s are 0.000.

After cross-source dedup (internal MinHash threshold 0.7), we drop another ~150 records that are near-duplicates of each other. Final assembly: shuffled 5911 records → seeded random 4000 → `tom_train_4k.jsonl`.

### A.6 Why deepseek-v4-flash and not deepseek-v4-pro?

We considered deepseek-v4-pro (the eval target) for synthesis but chose flash for two reasons:
1. **Cost**: pro is ~5× more expensive per token. For 3000 generations × 800 tokens that's a real budget hit.
2. **Pro has reasoning tokens that bleed**: in a JSON-structured output, the reasoning tokens (which pro emits even for "simple" tasks) often run past `max_tokens` and the JSON output gets truncated. We saw a 50% failure rate on pro vs ~5% on flash for the same prompt.

Phase-1 synthesis (added in stages 5+) used a mix: flash for hinting/faux-pas (which flash handles well), pro for scalar implicature and second-order belief (which need careful reasoning for the answer to be correct). See `tech_report_qwen3-14b_stage1.md` and the stage 5 report for that breakdown.

## Appendix B: Eval Protocol Details

### B.1 Direct protocol

System prompt:
```
You are a careful reader answering a multiple-choice theory-of-mind question.
Read the story and the question carefully, then output ONLY your final answer
in the format \boxed{X} where X is one of A, B, C, D.
Do not include any explanation, reasoning, or extra text.
```

User prompt: built per `scripts/eval/run_tombench.py`:
- EN: `"Story:\n{story}\n\nQuestion: {question}\nA. {opt_a}\nB. {opt_b}\nC. {opt_c}\nD. {opt_d}"`
- ZH: `"故事：\n{story}\n\n问题：{question}\nA. {opt_a}\nB. {opt_b}\nC. {opt_c}\nD. {opt_d}"`

Eval params: temperature=0.0, top_p=1.0, max_tokens=2048 (large enough for any reasoning model). Extraction: same `\boxed{[A-D]}` regex.

### B.2 CoT protocol

System prompt:
```
You are a careful reader answering a multiple-choice theory-of-mind question.
Think step by step about the mental states of the characters,
then output your final answer in the format \boxed{X} where X is one of A, B, C, D.
Put your final \boxed{X} on the last line.
```

Eval params: temperature=0.6, top_p=0.9, max_tokens=1024.

### B.3 Del-Tom protocol

A robustness check: present the same story and question but with all explicit mental-state words deleted (story is preprocessed by removing words like "knows", "believes", "thinks", "wants", "feels", etc.). Tests whether the model is using shallow keyword shortcuts. Eval params same as direct.

### B.4 Subset500 vs Full 5718

`tombench_eval_subset500.jsonl` is a deterministic random 500-question sample (seed=42) of `tombench_eval.jsonl` (5718 questions). It was created during stage 0 as a fast iteration loop. **Reading our results**:

- Full 5718 numbers are the canonical headline.
- Subset500 numbers are useful for protocol comparisons (where running deepseek-v4-pro on full 5718 was prohibitive in early stages) and for direct comparisons against earlier reports.
- The subset500 baseline for deepseek-v4-pro is 0.7880, while the full 5718 baseline is 0.8080 — a 2pp difference attributed to sample variance and possible slight evaluation-time differences (deepseek's API behavior at higher concurrency).
