"""Parallel evaluation engine for Qwen3-8B full-eval.

Dispatches MCQ questions across N OpenAI-compatible endpoints (e.g., 4 local
vLLM instances of the same model on ports 8001-8004) or to a single remote API
(DashScope qwen3-8b). Supports ToMBench and Hi-ToM with three protocols:
direct / cot / del_tom (8-sample majority vote).

Usage (local vLLM with 4 endpoints):
  python parallel_eval.py \
    --backend local \
    --model qwen3-8b-base \
    --endpoints 127.0.0.1:8001 127.0.0.1:8002 127.0.0.1:8003 127.0.0.1:8004 \
    --benchmark tombench \
    --data /home/h800/grj-projects/qwen3-tom/data/tom/tombench_eval.jsonl \
    --protocols direct,cot,del_tom \
    --output output/tombench/base.json \
    --cache-dir output/cache \
    --concurrency 32

Usage (DashScope API):
  python parallel_eval.py \
    --backend dashscope \
    --model qwen3-8b \
    --benchmark hitom \
    --data /home/h800/grj-projects/qwen3-tom/data/eval/hitom_eval.jsonl \
    --protocols direct,cot,del_tom \
    --output output/hitom/dashscope.json \
    --cache-dir output/cache \
    --concurrency 4
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from openai import OpenAI
from tqdm import tqdm


def _read_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

# Module path: this file is in scripts/, prompts.py sits next to it.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompts import (
    EXTRACT_HITOM_18,
    EXTRACT_TOMBENCH,
    build_messages_generic,
    build_messages_tombench,
    sampling_params_for,
)


# ---------------------------------------------------------------------------
# Client pool
# ---------------------------------------------------------------------------

class ClientPool:
    """Round-robin pool of OpenAI clients (one per endpoint or 1 for remote)."""

    def __init__(self, clients: list[OpenAI], model: str):
        if not clients:
            raise ValueError("ClientPool needs at least one client")
        self._clients = clients
        self.model = model
        self._counter = itertools.count()
        self._lock = threading.Lock()

    def get(self) -> OpenAI:
        with self._lock:
            idx = next(self._counter) % len(self._clients)
        return self._clients[idx]

    @classmethod
    def from_local(cls, endpoints: list[str], model: str, api_key: str = "EMPTY") -> "ClientPool":
        clients = []
        for ep in endpoints:
            if not ep.startswith("http"):
                ep = f"http://{ep}"
            base_url = ep.rstrip("/") + "/v1"
            clients.append(OpenAI(api_key=api_key, base_url=base_url, timeout=120.0))
        return cls(clients, model=model)

    @classmethod
    def from_dashscope(cls, model: str) -> "ClientPool":
        base_url = os.environ.get(
            "DASHSCOPE_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise SystemExit("DASHSCOPE_API_KEY env var not set")
        return cls(
            [OpenAI(api_key=api_key, base_url=base_url, timeout=300.0)],
            model=model,
        )


# ---------------------------------------------------------------------------
# Single chat call with retry + thinking_kwargs
# ---------------------------------------------------------------------------

def chat_once(
    *,
    client: OpenAI,
    model: str,
    messages: list[dict],
    temperature: float,
    top_p: float,
    max_tokens: int,
    enable_thinking: bool,
    requires_stream: bool,
    is_dashscope: bool,
    max_retries: int = 3,
) -> str:
    """Call /chat/completions and return content. Handles DashScope thinking-stream req.

    For DashScope thinking mode (stream): reasoning_content is the hidden CoT,
    content is the visible answer. We wrap reasoning into <think>...</think>
    and prepend to content so the cache files for DashScope thinking responses
    match the local-vLLM format (which puts the entire <think>...</think>\\nanswer
    in `content`). This makes extractor & audit behave identically.
    """
    # DashScope OpenAI-compat endpoint uses top-level `enable_thinking`,
    # NOT chat_template_kwargs (which is the local-vLLM convention).
    if is_dashscope:
        extra_body: dict = {"enable_thinking": enable_thinking}
    else:
        extra_body = {"chat_template_kwargs": {"enable_thinking": enable_thinking}}
    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            if requires_stream:
                stream = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    extra_body=extra_body,
                    stream=True,
                )
                content_chunks: list[str] = []
                reasoning_chunks: list[str] = []
                for ev in stream:
                    if not ev.choices:
                        continue
                    delta = ev.choices[0].delta
                    c = getattr(delta, "content", None) or ""
                    r = getattr(delta, "reasoning_content", None) or ""
                    if c:
                        content_chunks.append(c)
                    if r:
                        reasoning_chunks.append(r)
                content = "".join(content_chunks)
                reasoning = "".join(reasoning_chunks)
                # If we got reasoning back, wrap it in <think>...</think>
                # and prepend so format matches local-vLLM.
                if reasoning:
                    return f"<think>\n{reasoning}\n</think>\n\n{content}"
                return content
            else:
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    extra_body=extra_body,
                )
                return resp.choices[0].message.content or ""
        except Exception as e:
            last_err = e
            msg = str(e).lower()
            is_rate_limited = ("429" in msg or "rate limit" in msg or "limit_requests" in msg)
            if attempt < max_retries - 1:
                sleep = 30.0 * (attempt + 1) if is_rate_limited else (1.5 ** attempt)
                time.sleep(sleep)
                continue
            break
    assert last_err is not None
    raise last_err


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def cache_path(cache_dir: Path, model_id: str, benchmark: str, protocol: str, qid: str, sample_idx: int) -> Path:
    safe = model_id.replace("/", "_").replace(":", "_")
    return cache_dir / benchmark / safe / protocol / f"{qid}__s{sample_idx}.json"


def load_cache(p: Path) -> Optional[str]:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())["content"]
    except Exception:
        return None


def save_cache(p: Path, content: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"content": content}, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Per-question evaluation
# ---------------------------------------------------------------------------

def evaluate_one(
    *,
    pool: ClientPool,
    record: dict,
    benchmark: str,
    protocol: str,
    backend: str,
    cache_dir: Path,
    model_id: str,
    del_tom_n: int,
) -> dict:
    qid = record["question_id"]
    gold = record["gold"]
    if benchmark == "tombench":
        msgs = build_messages_tombench(record, protocol=protocol)
        extract_direct, extract_cot, vote = EXTRACT_TOMBENCH
        n_opts = 4
    else:  # hitom
        msgs = build_messages_generic(record, protocol=protocol)
        n_opts = len(record["options"])
        if n_opts <= 4:
            extract_direct, extract_cot, vote = EXTRACT_TOMBENCH
        else:
            extract_direct, extract_cot, vote = EXTRACT_HITOM_18

    sp = sampling_params_for(protocol, n_samples_default=del_tom_n)
    n_samples = sp["n_samples"]

    # DashScope requires stream mode whenever enable_thinking=true on chat endpoint.
    requires_stream = (backend == "dashscope") and sp["enable_thinking"]

    # direct uses first-boxed extractor (no thinking allowed). direct_think /
    # cot / del_tom all walk thinking, so use last-boxed extractor.
    use_direct_extractor = (protocol == "direct")

    raw: list[str] = []
    answers: list[Optional[str]] = []
    for s_idx in range(n_samples):
        cp = cache_path(cache_dir, model_id, benchmark, protocol, qid, s_idx)
        content = load_cache(cp)
        if content is None:
            client = pool.get()
            content = chat_once(
                client=client,
                model=pool.model,
                messages=msgs,
                temperature=sp["temperature"],
                top_p=sp["top_p"],
                max_tokens=sp["max_tokens"],
                enable_thinking=sp["enable_thinking"],
                requires_stream=requires_stream,
                is_dashscope=(backend == "dashscope"),
            )
            save_cache(cp, content)
        raw.append(content)
        if use_direct_extractor:
            answers.append(extract_direct(content))
        else:
            answers.append(extract_cot(content))

    pred = vote(answers) if protocol == "del_tom" else answers[0]

    return {
        "question_id": qid,
        "language": record.get("language", "en"),
        "task": record.get("task", "?"),
        "n_options": n_opts,
        "gold": gold,
        "pred": pred,
        "correct": pred == gold,
        "protocol": protocol,
        "benchmark": benchmark,
        "model": model_id,
        "answers": answers,
        # Don't store raw responses in result.json — they live in cache. Keep file lean.
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_protocol(
    *,
    pool: ClientPool,
    records: list[dict],
    benchmark: str,
    protocol: str,
    backend: str,
    cache_dir: Path,
    model_id: str,
    del_tom_n: int,
    concurrency: int,
) -> list[dict]:
    out: list[dict] = []
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = {
            ex.submit(
                evaluate_one,
                pool=pool, record=r, benchmark=benchmark, protocol=protocol,
                backend=backend, cache_dir=cache_dir, model_id=model_id,
                del_tom_n=del_tom_n,
            ): r for r in records
        }
        for f in tqdm(as_completed(futs), total=len(futs), desc=f"{model_id}/{benchmark}/{protocol}"):
            try:
                out.append(f.result())
            except Exception as e:
                rec = futs[f]
                print(f"  ERR qid={rec.get('question_id')}: {e}", flush=True)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["local", "dashscope"], required=True)
    p.add_argument("--model", required=True,
                   help="served-model-name (local) or dashscope model id (e.g. qwen3-8b)")
    p.add_argument("--endpoints", nargs="*", default=[],
                   help="local-only: host:port list, e.g. 127.0.0.1:8001 127.0.0.1:8002")
    p.add_argument("--model-id", default=None,
                   help="display id used for caching + output (defaults to --model)")
    p.add_argument("--benchmark", choices=["tombench", "hitom"], required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--protocols", default="direct,cot,del_tom")
    p.add_argument("--output", required=True)
    p.add_argument("--cache-dir", default="output/cache")
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--del-tom-n", type=int, default=8)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    model_id = args.model_id or args.model

    if args.backend == "local":
        if not args.endpoints:
            raise SystemExit("--backend local requires --endpoints")
        pool = ClientPool.from_local(args.endpoints, model=args.model)
    else:
        pool = ClientPool.from_dashscope(args.model)

    records = _read_jsonl(Path(args.data))
    if args.limit:
        records = records[: args.limit]

    cache_dir = Path(args.cache_dir)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    protocols = [s.strip() for s in args.protocols.split(",") if s.strip()]

    all_results: list[dict] = []
    for protocol in protocols:
        t0 = time.time()
        res = run_protocol(
            pool=pool, records=records, benchmark=args.benchmark, protocol=protocol,
            backend=args.backend, cache_dir=cache_dir, model_id=model_id,
            del_tom_n=args.del_tom_n, concurrency=args.concurrency,
        )
        all_results.extend(res)
        correct = sum(1 for x in res if x["correct"])
        n = len(res)
        elapsed = time.time() - t0
        if n == 0:
            print(f"[{model_id}/{args.benchmark}/{protocol}] no successful results  elapsed={elapsed:.1f}s", flush=True)
        else:
            print(f"[{model_id}/{args.benchmark}/{protocol}] acc={correct}/{n}={correct/n:.4f}  "
                  f"elapsed={elapsed:.1f}s", flush=True)

    out_path.write_text(json.dumps(all_results, ensure_ascii=False))
    print(f"wrote {len(all_results)} results -> {out_path}")


if __name__ == "__main__":
    main()
