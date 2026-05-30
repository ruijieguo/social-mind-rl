"""ToM MCQ reward worker for ROLL RLVR pipeline.

Implements the L2 reward:
    R = R_fmt × R_out × R_len

The pure functions (extract_boxed_letter, sigmoid_window, tom_mcq_reward_fn)
are exposed at module level and importable without the full ROLL stack
(which may not be installable in the DEV container).
"""
from __future__ import annotations
import json
import json
import math
import os
import re
from typing import Optional, Tuple

# Guard ROLL imports so pure functions are testable in environments
# without the full ROLL/CUDA stack (e.g. macOS DEV container).
try:
    import torch
    from roll.configs.worker_config import WorkerConfig
    from roll.distributed.executor.worker import Worker
    from roll.distributed.scheduler.decorator import Dispatch, register
    from roll.distributed.scheduler.protocol import DataProto
    from roll.models.model_providers import default_tokenizer_provider
    from roll.utils.logging import get_logger
    _ROLL_AVAILABLE = True
    logger = get_logger()
except ImportError:  # ROLL not installable in DEV container
    _ROLL_AVAILABLE = False
    torch = None  # type: ignore
    WorkerConfig = object  # type: ignore
    Worker = object  # type: ignore
    DataProto = object  # type: ignore
    default_tokenizer_provider = None  # type: ignore

    class _DispatchStub:
        ONE_TO_ALL = "one_to_all"
        DP_MP_COMPUTE = "dp_mp_compute"

    Dispatch = _DispatchStub()

    def register(*args, **kwargs):  # type: ignore
        def deco(fn):
            return fn
        return deco

    import logging
    logger = logging.getLogger(__name__)


_BOXED = re.compile(r"\\boxed\{([A-Z])\}")
_VALID_LETTERS = set(chr(ord("A") + i) for i in range(26))


def extract_boxed_letter(response: str) -> Tuple[str, bool]:
    """Return (letter, format_ok). Format_ok requires a valid \\boxed{[A-Z]}."""
    if not response:
        return "", False
    m = _BOXED.search(response)
    if m:
        return m.group(1), True
    return "", False


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def sigmoid_window(L: float, l_min: float, l_max: float, k: float) -> float:
    """Smooth window: rises near l_min, falls near l_max, plateau in between."""
    span = max(1.0, l_max - l_min)
    rise = _sigmoid(k * (L - l_min) / span)
    fall = 1.0 - _sigmoid(k * (L - l_max) / span)
    return rise * fall


def tom_mcq_reward_fn(
    response: str,
    response_token_count: int,
    ground_truth: str,
    l_min: float = 8.0,
    l_max: float = 256.0,
    k: float = 50.0,
    aggregation: str = "multiplicative",
    r_fmt_weight: float = 0.05,
    r_out_weight: float = 0.85,
    r_len_weight: float = 0.10,
) -> Tuple[float, float, float, float]:
    """Compute (r_fmt, r_out, r_len, r_total) for a single response.

    aggregation:
      - "multiplicative" (legacy):   r_total = r_fmt * r_out * r_len
      - "weighted_sum"   (stage9+):  r_total = w_fmt*r_fmt + w_out*r_out + w_len*r_len
        Weighted sum preserves credit for correct-but-format-imperfect responses.
        Default weights (0.05/0.85/0.10) make r_out dominant.
    """
    letter, fmt_ok = extract_boxed_letter(response)
    r_fmt = 1.0 if fmt_ok else 0.0
    r_out = 1.0 if (fmt_ok and letter == ground_truth) else 0.0
    r_len = sigmoid_window(float(response_token_count), l_min, l_max, k)
    if aggregation == "weighted_sum":
        r_total = r_fmt_weight * r_fmt + r_out_weight * r_out + r_len_weight * r_len
    else:
        r_total = r_fmt * r_out * r_len
    return r_fmt, r_out, r_len, r_total


def apply_reward_override(params: dict, overrides: Optional[dict]) -> dict:
    """Merge a JSON override dict into params (pure; no I/O).

    ROLL's ``RewardConfig`` dataclass silently drops unknown YAML keys
    (l_min/l_max/k/l_max_long/l_max_short/aggregation/r_*_weight), so the reward
    params can only be set reliably via this override file, NOT the YAML rewards
    block. Numeric keys are cast to float, aggregation to str; absent keys are
    left unchanged.
    """
    if not overrides:
        return params
    result = dict(params)
    for fkey in (
        "l_min", "l_max", "k", "l_max_long", "l_max_short",
        "r_fmt_weight", "r_out_weight", "r_len_weight",
    ):
        if fkey in overrides:
            result[fkey] = float(overrides[fkey])
    if "aggregation" in overrides:
        result["aggregation"] = str(overrides["aggregation"])
    return result


class TomMcqRewardWorker(Worker):
    """RLVR reward worker for ToM MCQ with R_fmt × R_out × R_len reward.

    Only operational when the ROLL framework is fully importable (TRAIN container).
    In the DEV container the class is a no-op stub.
    """

    def __init__(self, worker_config):
        if not _ROLL_AVAILABLE:
            raise RuntimeError(
                "TomMcqRewardWorker requires the ROLL framework. "
                "Install ROLL or use the DEV stub."
            )
        super().__init__(worker_config=worker_config)
        self.rank_info.dp_rank = self.rank_info.rank
        self.rank_info.dp_size = self.rank_info.world_size
        self.tokenizer = default_tokenizer_provider(
            model_args=self.worker_config.model_args
        )
        # ROLL's RewardConfig drops unknown YAML reward keys, so these getattr calls
        # return the BUILT-IN DEFAULTS (l_max=256, multiplicative) regardless of the
        # YAML. The authoritative values come from a JSON override file written next to
        # the stage config (env TOM_REWARD_OVERRIDE), merged below. The worker LOGS the
        # resolved params at INFO — grep '[tom_mcq_reward] resolved' to VERIFY (don't
        # trust the YAML). See memory: roll_rewardconfig_drops_custom_keys.
        _base = {
            "l_min": float(getattr(self.worker_config, "l_min", 8)),
            "l_max": float(getattr(self.worker_config, "l_max", 256)),
            "k": float(getattr(self.worker_config, "k", 50)),
            "l_max_long": float(getattr(self.worker_config, "l_max_long", 256)),
            "l_max_short": float(getattr(self.worker_config, "l_max_short", 256)),
            "aggregation": str(getattr(self.worker_config, "aggregation", "multiplicative")),
            "r_fmt_weight": float(getattr(self.worker_config, "r_fmt_weight", 0.05)),
            "r_out_weight": float(getattr(self.worker_config, "r_out_weight", 0.85)),
            "r_len_weight": float(getattr(self.worker_config, "r_len_weight", 0.10)),
        }
        override_path = os.environ.get(
            "TOM_REWARD_OVERRIDE", "/workspace/configs/process-reward/tom_reward_override.json"
        )
        if os.path.exists(override_path):
            try:
                with open(override_path) as _f:
                    _base = apply_reward_override(_base, json.load(_f))
                logger.info(f"[tom_mcq_reward] resolved params from override {override_path}: {_base}")
            except Exception as _e:  # noqa: BLE001
                logger.error(f"[tom_mcq_reward] failed to load override {override_path}: {_e}; using defaults {_base}")
        else:
            logger.info(f"[tom_mcq_reward] no override at {override_path}; using YAML/defaults: {_base}")
        self.l_min = _base["l_min"]
        self.l_max = _base["l_max"]
        self.k = _base["k"]
        self.l_max_long = _base["l_max_long"]
        self.l_max_short = _base["l_max_short"]
        self.aggregation = _base["aggregation"]
        self.r_fmt_weight = _base["r_fmt_weight"]
        self.r_out_weight = _base["r_out_weight"]
        self.r_len_weight = _base["r_len_weight"]

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def initialize(self, pipeline_config):
        pass

    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE, clear_cache=False)
    def compute_rewards(self, data):
        response_text_list = self.tokenizer.batch_decode(
            data.batch["responses"], skip_special_tokens=True
        )
        ground_truths = data.non_tensor_batch["ground_truth"]
        # Optional per-sample task tag (e.g. "order_3", "Knowledge"); used to
        # widen l_max for tasks that legitimately need long CoT (Hi-ToM order_2+).
        sources = data.non_tensor_batch.get("source", [""] * len(ground_truths))
        tasks = data.non_tensor_batch.get("task", [""] * len(ground_truths))

        scores: list[float] = []
        r_fmt_list: list[float] = []
        r_out_list: list[float] = []
        r_len_list: list[float] = []
        response_lengths: list[int] = []

        for i, (resp_tokens, gold) in enumerate(zip(data.batch["responses"], ground_truths)):
            response_text = response_text_list[i]
            non_pad = (resp_tokens != self.tokenizer.pad_token_id).sum().item() \
                if self.tokenizer.pad_token_id is not None else len(resp_tokens)
            # Hi-ToM order_2+ needs longer CoT; widen the length window.
            # Hi-ToM direct-style samples (source=*direct*) need shorter — the
            # whole point is forcing compressed in-forward-pass answers.
            src_i = str(sources[i]) if i < len(sources) else ""
            task_i = str(tasks[i]) if i < len(tasks) else ""
            is_direct_style = "direct" in src_i.lower()
            needs_long = (not is_direct_style) and (
                "hitom" in src_i.lower()
                or task_i.startswith("order_")
            )
            if is_direct_style:
                l_max_eff = self.l_max_short
            elif needs_long:
                l_max_eff = self.l_max_long
            else:
                l_max_eff = self.l_max
            r_fmt, r_out, r_len, r_total = tom_mcq_reward_fn(
                response=response_text,
                response_token_count=non_pad,
                ground_truth=str(gold),
                l_min=self.l_min, l_max=l_max_eff, k=self.k,
                aggregation=self.aggregation,
                r_fmt_weight=self.r_fmt_weight,
                r_out_weight=self.r_out_weight,
                r_len_weight=self.r_len_weight,
            )
            scores.append(r_total)
            r_fmt_list.append(r_fmt)
            r_out_list.append(r_out)
            r_len_list.append(r_len)
            response_lengths.append(non_pad)

            try:
                letter, _ = extract_boxed_letter(response_text)
                logger.debug(json.dumps({
                    "r_fmt": r_fmt, "r_out": r_out, "r_len": r_len, "r_total": r_total,
                    "response_length": non_pad,
                    "extracted_letter": letter,
                    "ground_truth": str(gold),
                }, ensure_ascii=False))
            except Exception as e:
                logger.error(f"logging error: {e}")

        scores_tensor = torch.tensor(scores, dtype=torch.float16)
        token_level_rewards = torch.zeros_like(data.batch["responses"], dtype=torch.float16)

        n = max(1, len(scores))
        metrics = {
            "reward/r_fmt_mean": sum(r_fmt_list) / n,
            "reward/r_out_mean": sum(r_out_list) / n,
            "reward/r_len_mean": sum(r_len_list) / n,
            "reward/r_total_mean": sum(scores) / n,
            "reward/response_length_mean": sum(response_lengths) / n,
        }

        output = DataProto.from_dict(tensors={
            "token_level_rewards": token_level_rewards,
            "response_level_rewards": scores_tensor,
            "scores": scores_tensor,
        })
        output.meta_info = {"metrics": metrics}
        return output
