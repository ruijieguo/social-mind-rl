# Path B — Qwen3.5/Qwen3.6-27B training image progress checkpoint

> **Purpose**: durable on-disk anchor for the path-B image work so future
> sessions can rehydrate from a file, not from a chat summary. Update this
> file at every meaningful state transition (commit, build success, smoke
> result).

Last updated: 2026-05-26

## Problem this solves

Smoke run `make train-stage1-27b-1x8-smoke` crashes at vLLM model loading:

```
KeyError: 'qwen3_5'
ValueError: model type 'qwen3_5' not recognized
```

Architecture-level, not config-level. Qwen3.6-27B reports
`model_type: qwen3_5` (hybrid: 48 Gated-Delta-Net linear-attn layers +
16 full-attn layers, `Qwen3_5ForConditionalGeneration`). The existing
training image (`qwen3-tom-train:latest`) pins:
- `transformers>=4.51,<4.55` — first qwen3_5 support is in transformers 4.57
- `vllm==0.8.4` — first qwen3_5 model definition is in vllm 0.10

Neither version registers `qwen3_5`, so the image cannot load this base.

## Decision: path B (new image tag, leave `:latest` alone)

Chosen 2026-05-26. Alternatives considered:
- **Path A** — bump `:latest` in place. Rejected: .181 Stage 18 (14B v3.4)
  is still running on `:latest`; any image change risks contaminating it.
- **Path C** — preprocess the base ckpt to remap `model_type` to a
  recognized arch. Rejected: qwen3_5 GDN layers have no equivalent in
  qwen2/qwen3 — would silently mis-load.

Path B: separate Dockerfile.qwen35 → builds `qwen3-tom-train:qwen35`.
`:latest` untouched. Switch Make targets only after the new image is green.

## Pin set (locked)

| Package | Base image (`:latest`) | qwen35 image | Why changed |
|---|---|---|---|
| transformers | `>=4.51,<4.55` | `>=4.57,<5.0` | qwen3_5 registry + GDN config |
| vllm | `==0.8.4` | `>=0.10,<0.12` | qwen3_5 model definition |
| causal-conv1d | (absent) | `>=1.4` | GDN Mamba-style 1D conv kernel |
| megatron-core | `==0.16.0` | `==0.16.0` | UNCHANGED — exposes gated_delta_net layer spec at this version |
| torch / TE / deepspeed | unchanged | unchanged | cu124 stack stays the same |

All other constraints (torch 2.6.0+cu124, transformer-engine 2.2.0,
deepspeed 0.16.4, opencv-python-headless 4.11.0.86, click==8.2.1,
setuptools<75) are preserved byte-equivalent.

## State machine

```
[ ] designs landed:    Dockerfile.qwen35 + docker-compose.qwen35.yml on disk
                       — DONE 2026-05-26 (this commit cycle)
[ ] git committed:     `feat/qwen3.6-27b-tom` branch, message "wip: qwen35 image scaffolding"
                       — pending user "commit it" sign-off
[ ] sync-up to .191:   rsync the two new files to /home/h800/grj-projects/qwen3-tom/docker/train/
                       — pending
[ ] image built:       `docker compose -f docker/train/docker-compose.qwen35.yml build train`
                       on .191. Expected ~25-40 min (torch wheel download is the long pole).
                       — pending
[ ] qwen3_5 probe:     final RUN line in Dockerfile.qwen35 asserts
                       `'qwen3_5' in transformers.CONFIG_MAPPING`. Build must succeed.
                       — pending
[ ] Make targets:      switch experiment/qwen3.6-27b/Make targets (or top-level
                       Makefile lines 139-153) to use docker-compose.qwen35.yml
                       — pending
[ ] smoke green:       `make train-stage1-27b-1x8-smoke` runs 3 steps, no OOM,
                       r_out_mean != 0, ckpt-3 lands on .191
                       — pending
[ ] full run:          `make train-stage1-27b-1x8` (300 steps, ~22h)
                       — pending
```

## Files

| Path | Status | Notes |
|---|---|---|
| `docker/train/Dockerfile.qwen35` | drafted, not committed | Delta-from-`Dockerfile`: only constraints block + vllm install line + causal-conv1d install + final probe line changed |
| `docker/train/docker-compose.qwen35.yml` | drafted, not committed | Delta-from-`docker-compose.yml`: only `image` + `dockerfile` fields changed |
| `Makefile` (lines 139-153) | NOT YET TOUCHED | After image green, swap `docker-compose.yml` → `docker-compose.qwen35.yml` in `train-stage1-27b-1x8{,-smoke}` targets |
| `framework/ROLL/**` | DO NOT TOUCH | Constraint from project CLAUDE.md |
| `configs/deploy.env` | DO NOT TOUCH | .181-scoped while Stage 18 runs |

## Runtime invariants .181/.191 must continue to satisfy

- .181 Stage 18 14B run keeps `docker run … qwen3-tom-train:latest …` — the
  new `:qwen35` tag must not displace it.
- .191 has 8 idle H800s before image build kicks off (bootstrap script
  asserts `< 1 GiB` used per card).
- .191 path layout: `/home/h800/grj-projects/{qwen3-tom,tom-data,models,tom-output}`.
- Branch: `feat/qwen3.6-27b-tom` (.181 may be ahead on `main` from Stage 18 work).

## Open risks for the build itself

1. **causal-conv1d source build can be slow** on cu124 + torch 2.6 — first
   time we install it. If it deadlocks like flash-attn did, the GDN path
   in vllm/transformers will fail at load time. Fallback: try the prebuilt
   wheel `causal-conv1d-cuda12x` from an alternative mirror.
2. **vLLM ≥0.10 changed gpu_memory_utilization accounting** — preallocates
   more KV cache pages upfront than 0.8.4. The smoke config sets 0.40
   already. If the new image OOMs at 0.40, drop to 0.35 (one-line change
   in `rlvr_config_stage1_27b_1x8_smoke.yaml`).
3. **mcore_adapter + transformers 4.57** — mcore_adapter targets the
   transformers HF model surface. If its internal API uses a function
   that was deprecated between 4.54 → 4.57, the install will succeed but
   the first forward will fail. No way to know without trying.

## Recovery strategies if compaction wipes the conversation again

If this file exists and the build is in flight or staged, the next session
should:

1. `Read` this file as the FIRST action (anchors task state).
2. `git status` + `git log -5 feat/qwen3.6-27b-tom` to learn what's committed.
3. ssh .191 `docker images | grep qwen3-tom-train` to learn whether
   `:qwen35` exists already.
4. If draft files exist but uncommitted → propose committing them and proceed.
5. If image exists → propose running smoke.
6. Update this file with new state before doing anything else.
