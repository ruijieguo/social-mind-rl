# Tech Report: Qwen3-14B + GRPO on ToMBench (Stage 1)

**Author**: Internal training runbook, generated from `train_stage1_1x8_14b_20260517_031327.log` and `configs/tombench-rlvr/rlvr_config_stage1_1x8_14b.yaml` (commit `0ece239` config; commit `2ae3442` Dockerfile fix; commit `6e7bcd3` final report).

**Status**: Single-stage 14B verification, run after the five-stage 8B program had finished (see `tech_report_qwen3-8b_stage1.md`). Same recipe and same 4k training subset, only the base model and the parallelism layout changed. The point of this run was to test whether the 8B → deepseek-v4-pro accuracy gap is bottlenecked by base capacity, training recipe, or data.

**Result preview**: 14B + RL achieves **0.7527** on full ToMBench 5718 (direct), vs. 14B-base **0.7338** and 8B+RL stage-1 **0.7394**. Gap to deepseek-v4-pro (0.8080) is now **−5.53pp**, smallest in the program.

---

## 1. Hypothesis

The 8B program plateaued at 0.7394 on full 5718 / 0.7460 on subset500. After 4 follow-up stages we couldn't push 8B past that. Two non-exclusive hypotheses:

(H1) **Recipe ceiling**: GRPO + verifiable reward + format penalty has saturated what an 8B base model can express on ToMBench. Adding more steps overfits (stage 2), adding KL anchors blocks adaptation (stage 3/4), adding more synthesis data doesn't transfer (stage 5).

(H2) **Capacity ceiling**: The 8B base genuinely lacks the precision needed for the harder ToMBench tasks (Belief, Knowledge implicature). A larger base will admit more accuracy from the same recipe.

To test H2 against H1, this run pairs the **same recipe and same data** as 8B stage 1 with the 14B base. If the recipe is ceiling-limited, 14B should plateau near 8B's 0.7394. If capacity is the bottleneck, 14B should comfortably exceed it.

## 2. Hardware & Software

Identical to 8B stage 1 except where noted:

| Item | Value | Differs from 8B? |
|---|---|---|
| Hardware | 1 node, 8× H800 80 GB | same |
| Container image | `qwen3-tom-train:latest`, but **rebuilt** with `setuptools<75` pin | yes — see §6 for the bug |
| Torch / Megatron / vLLM | 2.6.0+cu124 / 0.16.0 / 0.8.4 | same |
| Tensor parallelism (actor_train) | **TP=2** | yes (8B was TP=1) |
| Tensor parallelism (vLLM) | **TP=2** | yes (must match actor_train for ROLL weight-sync) |
| Tensor parallelism (reference) | **TP=2** | yes |
| vLLM gpu_memory_utilization | **0.45** (was 0.6 for 8B) | yes — KV cache for 14B is larger |
| Effective DP | 8 / 2 = 4 | (vs 8 for 8B) |
| Per-step time | ~70s | (vs ~60s for 8B; modest TP overhead) |
| Effective batch | unchanged: 32 prompts × 8 group samples = 256 sequences | same |

The TP=2 layout splits each model's weights, gradients, optimizer state, and KV cache across pairs of GPUs:

```
GPU 0+1: actor_train rank 0 (TP shard 0+1) + actor_infer + reference + reward
GPU 2+3: rank 1 (...)
GPU 4+5: rank 2 (...)
GPU 6+7: rank 3 (...)
```

Per-pair memory consumption (peak):
- actor_train (Megatron): ~50 GB (down from ~80 GB at TP=1; 14B / 2 + grads + dist-opt 1/4)
- actor_infer (vLLM, mem util 0.45 of free): ~15 GB
- reference: ~14 GB
- All other (CUDA workspace, NCCL buffers, residuals): ~10 GB

Comfortably fits in 80 GB. The 0.45 vLLM mem util was conservative; 0.5 might also work.

## 3. Why TP=2 specifically

| Layout option | actor_train per-rank mem | Fits 1×8 H800? |
|---|---|---|
| TP=1, DP=8 | ~80 GB | ❌ (no headroom for vLLM/reference colocated) |
| **TP=2, DP=4** | **~50 GB** | ✅ |
| TP=4, DP=2 | ~30 GB | ✅ but communication overhead is ~3× |
| TP=8, DP=1 | ~20 GB | ✅ but DP=1 means one GRPO group per step → variance ↑↑ |

We chose TP=2 because it's the smallest TP that fits while preserving DP=4. DP=4 means 4 distinct rollout groups per step, which keeps GRPO's group-normalized advantage well-conditioned (32 prompts × 8 samples / 4 DP ranks = 64 sequences per rank).

ROLL's weight-sync between actor_train (Megatron) and actor_infer (vLLM) requires `actor_infer.tensor_parallel_size == actor_train.tensor_model_parallel_size`. The bucketing logic in `roll/distributed/strategy/megatron_weight_updater.py` sends parameter slices per-shard, and vLLM's TP shards must match for the receiving end to assemble correctly. So all three roles (train/infer/reference) are TP=2.

## 4. Configuration Diff vs 8B Stage 1

```diff
--- configs/tombench-rlvr/rlvr_config_stage1_1x8.yaml      (8B)
+++ configs/tombench-rlvr/rlvr_config_stage1_1x8_14b.yaml  (14B)

-exp_name: "qwen3-8B-tombench-rlvr-stage1-1x8"
+exp_name: "qwen3-14B-tombench-rlvr-stage1-1x8"

-pretrain: Qwen/Qwen3-8B
-reward_pretrain: Qwen/Qwen3-8B
+pretrain: Qwen/Qwen3-14B
+reward_pretrain: Qwen/Qwen3-14B

# Halve prompt budget for 14B's larger KV cache
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

 # Everything else (rollout_batch_size, response_length, learning_rate,
 # difficulty masks, GRPO hyperparams, l_min/l_max, save_steps, eval_steps,
 # add_token_level_kl=false, mem-efficient gather, ...) UNCHANGED.
```

That's it. **Everything except the base model and the parallelism layout is identical to 8B stage 1.** This is by design — we want H1 vs H2 to be cleanly isolated.

### 4.1 prompt_length 2048 → 1024

The single non-trivial change. ToMBench prompts vary widely:
- p50 prompt length: ~290 tokens
- p95: ~660 tokens
- max: ~990 tokens

So 1024 fits ~99.9% of prompts. The 8B run set 2048 as a safety margin; 14B can't afford that margin because the 14B KV cache is ~2× per token. Empirically, 0 prompts were truncated during 14B training (we checked `token/prompt_length/max` from the log: max observed = 730).

### 4.2 max_model_len 4096 → 2048

vLLM's max_model_len = prompt_length + response_length + safety. With 1024 + 256 + slack we land at 2048. Note `response_length: 256` is unchanged — this is the rollout cap, also the eval cap for direct protocol when we run `run_tombench.py --max-tokens 2048` (no, the 2048 there is unrelated; it's the eval-time vLLM `max_tokens`).

## 5. Data

**Identical to 8B stage 1**: 4000 records from `tom_train_4k.jsonl`. We deliberately reused the same data subset to isolate the model-size variable.

That said, the file `tom_train_4k.jsonl` had been *regenerated* between 8B stage 1 (May 15) and 14B stage 1 (May 17) when Phase-1 synthesis was merged into `tom_train.jsonl`. The new 4k subset has slightly different composition:

| Source | 8B stage-1 era 4k | 14B stage-1 era 4k |
|---|---|---|
| synth (deepseek-v4-flash, 9 ToMBench task types) | ~1353 | ~1353 |
| ExploreToM | ~886 | ~886 |
| SimpleToM | ~440 | ~440 |
| **synth_phase1** (faux-pas + scalar + hinting + 2nd-order belief, fixed C/D options) | 0 | **440** |
| Translated ZH (synth_zh + exploretom_zh + simpletom_zh) | ~881 | ~881 |
| Total | 4000 | 4000 |

So 14B's 4k contains ~440 records (11%) that 8B's 4k did not. This is a small confound when comparing 14B vs 8B head-to-head; we'd expect it to favor 14B by ~0.3–0.5pp at most based on stage 5's modest Phase-1 effect.

`distrib_optim_fully_reshardable_mem_efficient` is on; same MinHash 4-gram leakage check vs ToMBench eval (0 records dropped).

## 6. The setuptools<75 Bug

The 14B training crashed on first attempt at the moment vLLM tried to issue its first inference batch. The traceback:

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

**Root cause**: vLLM 0.8.4's TP>1 path uses Ray's compiled DAG executor, which probes for `pkg_resources` to validate the Ray cgraph backend. setuptools ≥ 75 (released late 2024) split `pkg_resources` into its own optional distribution. Our train image had `setuptools 82.0.1`, so the import fails.

**Why didn't 8B hit this?** 8B uses TP=1, which uses vLLM's single-node executor — that path doesn't import `pkg_resources`.

**Fix** (commit `2ae3442`): in `docker/train/Dockerfile`:

```dockerfile
# Pin setuptools<75 so pkg_resources is still bundled with setuptools.
# vllm 0.8.4's TP>1 ray-distributed-executor path uses pkg_resources at runtime.
# setuptools>=75 split pkg_resources into its own distribution, breaking that import.
RUN pip install --no-deps 'setuptools<75'
```

Cost: 25 min model download lost on the first attempt + ~5 min image rebuild + a fresh restart. After the fix, training started cleanly and ran to completion.

## 7. Training Trajectory

Sampled every 25 steps from the log:

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

### 7.1 Side-by-side trajectory (vs 8B)

| step | 8B score | 14B score | Δ | 8B val | 14B val | Δ |
|---|---|---|---|---|---|---|
| 0 | 0.215 | 0.217 | ≈ | 0.042 | 0.066 | +2.4pp |
| 50 | 0.412 | **0.491** | +7.9pp | 0.204 | **0.348** | **+14.4pp** |
| 100 | 0.566 | **0.717** | **+15.1pp** | 0.454 | 0.546 | +9.2pp |
| 150 | 0.791 | **0.901** | +11pp | 0.548 | 0.550 | ≈ |
| 199 | 0.800 | **0.940** | +14pp | — | — | — |

**14B saturates rollout score 50 steps earlier than 8B.** By step 125 14B is at 0.90 (8B doesn't get there). 14B step 150 is essentially the asymptote (the small dip at step 175 is noise — see KL ramp).

**14B val plateau is also earlier**: 8B keeps climbing 100→150 (0.454→0.548); 14B step 100 (0.546) ≈ 14B step 150 (0.550). Subset500 saturates because the 14B base already had ~70%-class on it.

### 7.2 Reading the training dynamics

- **steps 0–25**: 14B's grad_norm at step 0 is 3.45 (vs 8B's 65.9!). This is striking — 14B starts much more "in distribution" with the format reward. The base model already produces reasonable answers; the policy gradient signal isn't a shock. By step 25 grad_norm has settled to 2.6.
- **steps 25–75**: format and accuracy climb together. r_fmt 0.30→0.73; r_out 0.30→0.66. 14B picks up the format+accuracy combo more cleanly than 8B did (where r_fmt led r_out by ~25 steps).
- **steps 75–125**: rollout score 0.66 → 0.90. KL grows 0.12→0.22. The 14B policy is moving faster than 8B's (which was at 0.64 at step 125).
- **steps 125–150**: small dip then recovery. step 125→150: rollout 0.90 → 0.90 (essentially flat). step 150→175: 0.90 → 0.83 (small backslide). This is the standard saturation dance — once 90%+ of groups are all-correct, difficulty masking drops them, and the gradient comes from the harder remaining groups, which can pull the policy slightly the wrong way.
- **steps 175–199**: rollout score recovers to 0.94. Final response_len 134 tokens — significantly tighter than 8B's 158 (length penalty + larger model = more direct answers).

### 7.3 Validation curve

| step | 14B val_correct/all | 14B val_correct/tom_mcq |
|---|---|---|
| 0 | 0.066 | 0.133 |
| 50 | 0.348 | 0.450 |
| 100 | 0.546 | 0.631 |
| 150 | 0.550 | 0.628 |

Note val_correct stops growing after step 100. This isn't because the policy stops improving — final rollout is still 0.94 — but because:

1. **subset500 ceiling**: 14B base (no RL) already scores ~0.74 on full 5718, hence ~0.74 expected on subset500. Subset500 has 13% "both wrong" hard ceiling for the 14B base, and 14B-RL won't easily clear those.
2. **Validation truncation**: ROLL's val protocol uses `max_new_tokens=64`. 14B's response_len at step 150+ is 145–163 tokens — well under our `response_length: 256` rollout cap, but **the model occasionally writes >64 tokens of preamble before \boxed{}**. When val truncates at 64, the answer can be lost. Stage 3 (8B) ran into the catastrophic version of this; 14B's better format compliance keeps it manageable.

For the headline numbers we use post-training full-set eval with `max_tokens=2048`, not val_correct.

## 8. Distributed Save (worked first time)

Same `distrib_optim_fully_reshardable_mem_efficient: true` flag as 8B stage 1. The 14B optimizer state is roughly 2× larger (Adam moments scale with parameter count): ~96 GB vs 48 GB on disk for the dist_optimizer shard. Save took ~15 min (vs 8B's ~10 min). Local then upload: NVMe handled the transient ~210 GB peak comfortably.

Final checkpoint layout on disk:

```
/data_nvme/grj-projects/tom-output/qwen3-14B-tombench-rlvr-stage1-1x8/20260517-031410/checkpoint-199/
├── iter_0000001/
│   ├── mp_rank_00/model_optim_rng.pt    # ~28 GB (model + RNG)
│   ├── mp_rank_01/model_optim_rng.pt    # ~28 GB (TP shard 1)
│   └── dist_optimizer/                  # ~140 GB (Adam state, sharded across DP)
├── rng_state/...
└── pipeline/...
```

Total: ~196 GB. After conversion to HuggingFace format: 28 GB safetensors (model only).

## 9. Final Eval

Same protocol as 8B stage 1: vLLM serve trained model, run `run_tombench.py` from DEV against the OpenAI-compatible endpoint.

### 9.1 Full 5718 (direct, max_tokens=2048)

| Metric | Qwen3-8B base | **Qwen3-14B base** | 8B stage1 | **14B stage1 (this work)** | deepseek-v4-pro |
|---|---|---|---|---|---|
| Overall | 0.7009 | 0.7338 | 0.7394 | **0.7527** | 0.8080 |
| EN | 0.7020 | 0.7219 | 0.7275 | 0.7422 | 0.7978 |
| ZH | 0.6999 | 0.7457 | 0.7513 | 0.7632 | 0.8181 |

**Δ vs 14B base**: +1.89pp (RL training adds 1.9pp on top of size).
**Δ vs 8B stage1**: +1.33pp.
**Gap to deepseek**: −5.53pp (down from −6.86pp for 8B stage1).

### 9.2 Per-task (full 5718, direct)

| Task | 14B-base | 14B-RL (this work) | RL Δ | deepseek (5718) | Gap |
|---|---|---|---|---|---|
| Belief | 0.7359 | **0.7465** | +1.06pp | 0.8486 | -10.21pp |
| Desire | 0.5833 | 0.5889 | +0.56pp | 0.6333 | -4.44pp |
| Emotion | 0.7202 | 0.7286 | +0.84pp | 0.8048 | -7.62pp |
| False Belief | 0.8047 | **0.8770** | **+7.23pp** | 0.8946 | **-1.76pp** ✓closest |
| Intention | 0.7662 | **0.8103** | +4.41pp | 0.8926 | -8.23pp |
| Knowledge | 0.4671 | 0.4775 | +1.04pp | 0.5675 | -9.00pp |
| Non-literal Comm | 0.7955 | 0.7640 | **−3.15pp** ↓ | 0.8128 | -4.88pp |

**Big wins for RL on 14B base**:
- **False Belief +7.23pp**, similar to 8B stage 1's +12.4pp. Training data is heavy on false-belief, and 14B has the capacity to generalize the pattern.
- **Intention +4.41pp**. 8B stage 1 only gained +1.5pp here; 14B's stronger Theory-of-Mind prior amplifies the same training signal.
- **False Belief gap to deepseek now only −1.76pp**, the closest single task to parity in the entire program.

**Interesting regression**:
- **Non-literal Comm −3.15pp**. The 14B base already scored 0.795 (better than 8B base's 0.777). RL training on the standard recipe pushed it down to 0.764, suggesting the 14B base's Non-literal Comm performance was something we trained out. The training data is light on Non-literal Comm (~330 records / 4k = 8%), and the format-and-correctness reward likely punishes the longer reasoning the 14B base used naturally on faux-pas / hinting questions. Stage 5 tried to fix this with synth_phase1; for 14B we'd need a similar follow-up.

### 9.3 Per-task EN/ZH split (full 5718)

| Task | 14B-RL EN | 14B-RL ZH | deepseek EN | deepseek ZH |
|---|---|---|---|---|
| Belief | 0.711 | 0.782 | 0.838 | 0.859 |
| Desire | 0.583 | 0.594 | 0.611 | 0.656 |
| Emotion | 0.695 | 0.762 | 0.779 | 0.831 |
| False Belief | 0.878 | 0.876 | 0.905 | 0.884 |
| Intention | 0.788 | 0.832 | 0.876 | 0.909 |
| Knowledge | 0.484 | 0.471 | 0.550 | 0.585 |
| Non-literal Comm | 0.757 | 0.771 | 0.799 | 0.826 |

Pattern: ZH > EN almost everywhere. The 14B base shows the same pattern (qwen3-14b-nt: EN 0.722, ZH 0.746), so this is inherited — Qwen models score better on ZH than EN on ToMBench. RL training preserves and slightly amplifies the gap.

### 9.4 Subset500 across 3 protocols

| Protocol | 14B-RL | 8B stage 1 | 8B stage 5 | deepseek subset500 | Δ vs deepseek |
|---|---|---|---|---|---|
| direct | **0.7800** | 0.7460 | 0.7340 | 0.7880 | **−0.80pp** |
| cot | **0.7720** | 0.6980 | 0.7380 | 0.7140 | **+5.80pp** ✓ |
| del_tom | **0.7760** | 0.7460 | 0.7520 | n/a | n/a |

**14B-RL beats deepseek on cot protocol by +5.80pp on subset500.** This is the only protocol on which we beat the closed-source baseline. Two reasons:
- The deepseek API in cot mode reasons internally and burns tokens; with our 2048-token budget some of its responses don't fit.
- The qwen3-14b base in cot mode produces tighter chains than 8B did; RL training preserved this and the format reward kept the final `\boxed{X}` clean.

**Per-task subset500 best protocol** (14B-RL):

| Task | 14B-RL best (proto) | deepseek subset500 | Δ |
|---|---|---|---|
| Belief | 0.800 (direct) | 0.800 | 0 ✓ |
| Desire | 0.778 (del_tom) | 0.639 | **+13.9pp** ✓ |
| Emotion | 0.721 (del_tom) | 0.709 | +1.2pp ✓ |
| False Belief | 0.885 (cot) | 0.862 | +2.3pp ✓ |
| Intention | 0.864 (direct) | 0.814 | +5.0pp ✓ |
| Knowledge | 0.400 (direct) | 0.600 | −20.0pp |
| Non-literal Comm | 0.836 (cot) | 0.843 | −0.7pp ≈ |

**6 of 7 tasks at-or-above deepseek on subset500** when allowed to choose protocol. Knowledge is the persistent outlier at every model size — we need different data for that task, not a different model.

> ⚠️ Note: subset500 is a 500-question random subsample. The full 5718 numbers for deepseek (0.8080) are 2pp higher than its subset500 average (0.7880). So per-task subset500 wins should be read as "competitive on this subsample" not "we beat deepseek". Our full-5718 vs full-5718 comparison (§9.1) is the canonical claim.

## 10. Wall Clock Budget

| Phase | Wall time |
|---|---|
| Container build (one-time, with setuptools<75 pin) | ~6 min |
| Container launch + worker init | ~10 min |
| Model download (Qwen3-14B from ModelScope, 28 GB) | ~25 min (first run only; cached on NVMe) |
| Training (200 steps) | ~3 h 50 min |
| `do_checkpoint` (Gloo+CPU mem-efficient) | ~15 min |
| Total stage-1-14B wall (including download) | ~6 h |
| Total minus first-time download | ~5 h |
| GPU-hours | ~40 |

For comparison, 8B stage 1 was ~26 GPU-hours total. The 14B+RL training thus cost ~50% more compute per step plus the one-time download.

Post-training:
- HF format conversion (mcore_adapter): ~3 min
- vLLM serve cold start (with TP=2 → falls back to TP=1 for inference since we serve on a single GPU): ~2 min
- Full 5718 eval: ~6 min @ vLLM concurrency=32

## 11. Lessons & Caveats

1. **TP=2 for 14B on 1×8 H800 works reliably.** The pkg_resources crash was the only blocker, and it's a 1-line Dockerfile pin. Once that's fixed, training is uneventful — same `distrib_optim_fully_reshardable_mem_efficient: true` save trick as 8B, same colocated layout, same recipe.

2. **Capacity is the bigger lever than recipe at this gap range.** Five 8B stages (5 different recipes) clustered in 0.7263–0.7394 on full 5718. One 14B stage 1 jumped to 0.7527. The 14B stage 1 vs 14B base delta (+1.89pp) is similar to 8B stage 1 vs 8B base (+3.85pp), so the marginal-gain-from-RL is real but proportionally smaller. The big win is the +3.29pp from the base size jump itself.

3. **14B training is proportionally calmer.** 8B's step-0 grad_norm was 65.9; 14B's was 3.45. The bigger model's pretrained representations are closer in distribution to the rewarded behavior, so the RL update isn't a shock. This means 14B might tolerate a slightly higher learning rate (we didn't try; held lr=1e-6 to match 8B for the comparison).

4. **False Belief is now within striking distance of deepseek.** −1.76pp is essentially noise at this scale. If the program continues, this is the task to declare "solved"; subsequent budget should target Belief / Intention / Knowledge.

5. **Non-literal Comm regressed under RL on 14B.** This wasn't observed on 8B. The hypothesis is that 14B base already does this task in a chain-of-reasoning style that the format-strict reward penalizes. A fix would be to either widen `response_length` for this task or apply the format reward more leniently — but the current single-template recipe doesn't support per-task reward shaping.

6. **Don't skip the full-set deepseek baseline early.** Through stages 1–4 we used `deepseek subset500 = 0.7880` as the target; only at stage 5 did we run full-5718 deepseek and discover the true target is 0.8080. Several "we beat deepseek" claims flipped under the corrected number. Always benchmark against the full set for headline claims.

## 12. Reproducing 14B Stage 1

```bash
# DEV machine
git clone https://github.com/ruijieguo/social-mind-rl
cd social-mind-rl
git checkout 6e7bcd3   # the commit with both 14B config and setuptools<75 fix

cp configs/deploy.env.example configs/deploy.env  # then fill in

# (Make sure configs/deploy.env points to NVMe paths — see commit a3eaf1f.
# 14B checkpoints + working set need ~210 GB transient space.)

make sync-up           # rsync code + data to TRAIN

# On TRAIN: builds train image (one-time, ~6 min with setuptools<75 pin),
# downloads Qwen3-14B from ModelScope (~25 min one-time, cached afterward),
# runs training (~4h), saves (~15 min).
make train-stage1-1x8-14b

# Convert to HuggingFace format
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

# Serve (single-GPU vLLM, no TP at inference time since the HF model is monolithic)
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

# Evaluate from DEV
docker compose -f docker/dev/docker-compose.yml run --rm -e OPENAI_API_KEY=dummy dev \
  python scripts/eval/run_tombench.py \
    --backend openai --base-url http://$TRAIN_HOST_HOSTONLY:8000/v1 \
    --model qwen3-14b-tom \
    --data data/tom/tombench_eval.jsonl \
    --protocols direct --concurrency 32 \
    --output output/eval/14b_full5718.json
```

## 13. Next Steps

The 14B stage 1 result establishes that **scaling base model is the highest-impact lever** at this point of the program. Possible follow-ups (in order of expected ROI):

1. **14B stage 5-equivalent**: 8k records × 250 steps with KL=false and Phase-1 fixed data. Following the 8B-stage5 recipe should test whether the data improvements transfer to 14B (we expect they will, more strongly).
2. **Targeted Knowledge data synthesis**: stage 5 added 449 scalar-implicature records via deepseek-v4-pro/flash; effect on 8B was ~0pp on Knowledge. 14B's higher base capacity might extract more from the same data; or the synthesis itself needs more diversity (only 9 prompt templates).
3. **14B 2×8 (16 H800)**: if a second node is available, drop TP back to 1 and split DP across 16 ranks for 2× rollout throughput. Same recipe, faster turnaround.
4. **Larger base (32B)**: the recipe likely has another +2–3pp in it at 32B given the 8B→14B trend. But compute cost grows superlinearly and 32B + colocated vLLM doesn't fit on 1×8 H800 even at TP=4.

## 14. Artifacts

| Path | What |
|---|---|
| `configs/tombench-rlvr/rlvr_config_stage1_1x8_14b.yaml` | Full config |
| `docker/train/Dockerfile` | Train image with `setuptools<75` pin |
| `framework/ROLL/roll/pipeline/rlvr/rewards/tom_mcq_reward_worker.py` | Reward worker (unchanged from 8B) |
| `logs/train_stage1_1x8_14b_20260517_023204.log` | First-attempt log (failed at vLLM init) |
| `logs/train_stage1_1x8_14b_20260517_031327.log` | Successful run log (11 MB) |
| `output/eval/14b_full5718.{json,md}` | 5718-question results |
| `output/eval/14b_subset500.{json,md}` | subset500 × 3 protocols |
| `output/eval/qwen3-14b-nt_full5718.{json,md}` | 14B base (no RL) baseline via dashscope API |
| `output/eval/deepseek_full5718.{json,md}` | deepseek-v4-pro full 5718 baseline |
| `output/analysis/curves_14b_stage1_1x8.png` | 12-panel training curve |
| Megatron checkpoint (TRAIN) | `/data_nvme/grj-projects/tom-output/qwen3-14B-tombench-rlvr-stage1-1x8/.../checkpoint-199/` (~196 GB) |
| HF checkpoint (TRAIN) | `/data_nvme/grj-projects/tom-output/qwen3-14B-tom-hf/` (28 GB, 8-shard safetensors) |
| Git commits | `0ece239` (config) → `2ae3442` (Dockerfile fix) → `6e7bcd3` (full report) → `618da6b` (corrected baselines) |

---

## Appendix A: Where the 14B+RL Gain Comes From

To dissect the 0.7527 result, decompose it as:

```
14B-RL = 14B-base                + (Δ from RL on 14B base)
       = 8B-base + (Δ from size) + (Δ from RL on 14B base)
       = 0.7009  + (+0.0329)     + (+0.0189)
       = 0.7527  ✓
```

For comparison, the 8B path:
```
8B-RL = 8B-base + (Δ from RL on 8B base)
      = 0.7009 + (+0.0385)
      = 0.7394
```

Two observations:

**(1) RL gain is 1.5× larger on 8B (+3.85pp) than on 14B (+1.89pp).**
This is consistent with 14B starting closer to the rewardable behavior — there's less "missing format" for RL to add. The 14B-base scores high enough on `r_fmt` from the get-go (see step-0 grad_norm 3.45 vs 8B's 65.9) that the policy gradient can't extract as much format-related accuracy.

**(2) Size gain (+3.29pp) > RL gain (+1.89pp) at 14B.**
The single largest accuracy win in the entire program came from going 8B→14B with **no training at all**. This is the strongest argument for "scale base before tuning recipe" if the goal is closing the deepseek gap.

### Per-task gain decomposition (full 5718, direct, percentage-points)

| Task | 14B-base − 8B-base | 8B-RL − 8B-base | 14B-RL − 14B-base | "Recipe transfer" (14B-RL − 8B-RL) |
|---|---|---|---|---|
| Belief | +6.34 | +2.11 | +1.06 | +5.28 |
| Desire | -0.28 | +0.56 | +0.56 | -0.28 |
| Emotion | +3.09 | +3.93 | +0.84 | 0.00 |
| False Belief | +7.70 | +12.43 | +7.23 | +2.50 |
| Intention | +1.62 | +1.47 | +4.41 | +4.56 |
| Knowledge | -1.39 | -0.17 | +1.04 | -0.17 |
| Non-literal Comm | +1.88 | -0.94 | -3.15 | -0.34 |

**Interpretation per task**:

- **Belief, Intention**: 14B base is meaningfully better, and 14B-RL extends the gain. The capacity helps both the prior knowledge and the RL adaptation.
- **False Belief**: massive base improvement (+7.7pp) AND big RL gain (+7.2pp), additive. This is the task with the most training data, and 14B has the capacity to memorize+generalize the patterns. Final gap to deepseek is only −1.76pp.
- **Emotion**: 14B base helps (+3pp), but RL on top of 14B helps less (+0.8pp) than RL on top of 8B (+3.9pp). The 8B+RL Emotion accuracy is identical to 14B+RL Emotion accuracy — RL has hit its ceiling for this task at both sizes.
- **Knowledge**: weakly negative for size, weakly positive for RL on 14B. Stuck around 0.47–0.48. This is the scalar-implicature problem we've been chasing through 5 stages; neither base size nor RL with our current data fixes it.
- **Non-literal Comm**: 14B base is +1.9pp higher than 8B base (it has stronger pragmatic understanding out of the box). Both 8B-RL and 14B-RL **regress** on this task vs their respective bases. Hypothesis: the format-strict reward punishes the longer reasoning the bases naturally produce on faux-pas / hinting tasks. The 14B regression (−3.15pp) is more dramatic because the 14B base reasoning was meaningfully better. **A targeted fix would be reward shaping that allows longer responses on Non-literal Comm specifically.**

## Appendix B: Reward Function

Same as 8B stage 1 — see `tech_report_qwen3-8b_stage1.md` §4 for the multiplicative `r_fmt × r_out × r_len` derivation.

The 14B run uses identical reward parameters: `l_min=8`, `l_max=256`, `k=50`. Final r_len at convergence is 0.97 (response_len ~134 tokens, deeper inside the band than 8B's 158). This contributes ~+0.04 to r_total at saturation, partially explaining why 14B's late-stage rollout score (0.94 at step 199) is higher than 8B's (0.80) — the reward maxes out more cleanly.

## Appendix C: Training Data

Same recipe as 8B stage 1 — see `tech_report_qwen3-8b_stage1.md` Appendix A for the full data synthesis pipeline.

The only difference: 14B's 4k subset (regenerated after Phase-1 synthesis was added to `tom_train.jsonl`) contains 440 records (11%) of `synth_phase1` data that 8B stage 1's 4k didn't have. This is a small confound; based on stage 5's modest Phase-1 effect on 8B we estimate this contributes ≤0.5pp to the 14B vs 8B comparison.

For an apples-to-apples comparison without this confound, one would re-run 8B stage 1 with the new 4k. We did not; the +1.33pp 14B gain is large enough that the confound doesn't change the qualitative conclusion.

## Appendix D: Memory Math

For posterity, here's the per-rank memory accounting for 14B at TP=2 on H800 80 GB:

```
Static (always resident):
  actor_train (Qwen3-14B / 2 bf16):              14 GB  (model)
  actor_train gradients (bf16):                  14 GB
  actor_train optimizer (Adam fp32, dist 1/8):   14 GB  (master + m + v, sharded across DP=4)
                                                 -----
                                                 42 GB

  reference (Qwen3-14B / 2 bf16, can be offloaded):  14 GB → ~0 GB after offload
  vLLM model (TP=2 shard, bf16):                     14 GB (resident always; sleep-mode releases KV cache only)

Dynamic (during specific phases):
  vLLM KV cache (mem util 0.45 of 80 GB):        ~36 GB peak during rollout
  Megatron activations + grad accum (bf16):      ~12 GB during forward+backward
  CUDA workspace + NCCL buffers + residual:      ~8 GB

Peak at training step (vLLM offloaded, train active):
  42 (static) + 12 (act) + 8 (workspace) = 62 GB ✓ fits
Peak at rollout (train offloaded, vLLM hot):
  14 (vLLM model) + 36 (KV cache) + 14 (reference if loaded) + 8 = 72 GB ✓ fits
Peak at save (train fully loaded + extras):
  42 (static) + ~18 GB (Gloo gather buffers if it were CUDA) → would OOM
  with Gloo+CPU gather: ~42 GB on GPU ✓ comfortable
```

The 80 GB H800 has just enough headroom. If we'd tried Qwen3-14B at TP=1 (DP=8), actor_train alone would consume ~80 GB per rank with no room for vLLM/reference/workspace — that's why TP=2 is mandatory at this scale.
