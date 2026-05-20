"""ToM MCQ reward worker for ROLL RLVR pipeline.

Implements the L2 reward:
    R = R_fmt × R_out × R_len

The pure functions (extract_boxed_letter, sigmoid_window, tom_mcq_reward_fn)
are exposed at module level and importable without the full ROLL stack
(which may not be installable in the DEV container).
"""
from __future__ import annotations
import json
import math
import re
from typing import Tuple

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


_BOXED = re.compile(r"\\boxed\{([A-D])\}")
_VALID_LETTERS = {"A", "B", "C", "D"}


def extract_boxed_letter(response: str) -> Tuple[str, bool]:
    """Return (letter, format_ok). Format_ok requires a valid \\boxed{[A-D]}."""
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
        self.l_min = float(getattr(self.worker_config, "l_min", 8))
        self.l_max = float(getattr(self.worker_config, "l_max", 256))
        self.k = float(getattr(self.worker_config, "k", 50))
        # Stage 9+: weighted-sum reward aggregation (vs legacy multiplicative)
        self.aggregation = str(getattr(self.worker_config, "aggregation", "multiplicative"))
        self.r_fmt_weight = float(getattr(self.worker_config, "r_fmt_weight", 0.05))
        self.r_out_weight = float(getattr(self.worker_config, "r_out_weight", 0.85))
        self.r_len_weight = float(getattr(self.worker_config, "r_len_weight", 0.10))

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def initialize(self, pipeline_config):
        pass

    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE, clear_cache=False)
    def compute_rewards(self, data):
        response_text_list = self.tokenizer.batch_decode(
            data.batch["responses"], skip_special_tokens=True
        )
        ground_truths = data.non_tensor_batch["ground_truth"]

        scores: list[float] = []
        r_fmt_list: list[float] = []
        r_out_list: list[float] = []
        r_len_list: list[float] = []
        response_lengths: list[int] = []

        for i, (resp_tokens, gold) in enumerate(zip(data.batch["responses"], ground_truths)):
            response_text = response_text_list[i]
            non_pad = (resp_tokens != self.tokenizer.pad_token_id).sum().item() \
                if self.tokenizer.pad_token_id is not None else len(resp_tokens)
            r_fmt, r_out, r_len, r_total = tom_mcq_reward_fn(
                response=response_text,
                response_token_count=non_pad,
                ground_truth=str(gold),
                l_min=self.l_min, l_max=self.l_max, k=self.k,
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
