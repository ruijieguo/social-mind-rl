"""Parallel evaluation engine for Qwen3-14B full-eval.

Dispatches MCQ questions across N OpenAI-compatible vLLM endpoints (4 local
instances of one model on ports 8001-8004) for four benchmarks
(tombench / hitom / socialiqa / emobench) and three protocols
(direct / direct_think / cot).

This version additionally records, per response, the vLLM `finish_reason` and
output length so the aggregator can report **truncation** (finish_reason ==
"length") — the failure mode that silently cost ~2pp in the 8B v1 eval.

Usage:
  python parallel_eval.py \
    --model qwen3-14b-v35 \
    --endpoints 127.0.0.1:8001 127.0.0.1:8002 127.0.0.1:8003 127.0.0.1:8004 \
    --benchmark tombench \
    --data /data/tom/tombench_eval.jsonl \
    --protocols direct,direct_think,cot \
    --output output/tombench/v35.json \
    --cache-dir output/cache \
    --concurrency 64
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from openai import OpenAI
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompts import (  # noqa: E402
    build_messages_generic,
    build_messages_tombench,
    extractors_for,
    sampling_params_for,
)


def _read_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


# ---------------------------------------------------------------------------
# Client pool (round-robin over local vLLM endpoints)
# ---------------------------------------------------------------------------

class ClientPool:
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
            clients.append(OpenAI(api_key=api_key, base_url=base_url, timeout=600.0))
        return cls(clients, model=model)


# ---------------------------------------------------------------------------
# Single chat call with retry. Returns (content, finish_reason).
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
    max_retries: int = 4,
) -> tuple[str, Optional[str]]:
    extra_body = {"chat_template_kwargs": {"enable_thinking": enable_thinking}}
    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                extra_body=extra_body,
            )
            choice = resp.choices[0]
            content = choice.message.content or ""
            finish = getattr(choice, "finish_reason", None)
            return content, finish
        except Exception as e:  # noqa: BLE001
            last_err = e
            if attempt < max_retries - 1:
                time.sleep(1.5 ** attempt)
                continue
            break
    assert last_err is not None
    raise last_err


# ---------------------------------------------------------------------------
# Cache (one file per (model, bench, protocol, qid, sample))
# ---------------------------------------------------------------------------

def cache_path(cache_dir: Path, model_id: str, benchmark: str, protocol: str, qid: str, sample_idx: int) -> Path:
    safe = model_id.replace("/", "_").replace(":", "_")
    return cache_dir / benchmark / safe / protocol / f"{qid}__s{sample_idx}.json"


def load_cache(p: Path) -> Optional[dict]:
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text())
        if "content" not in d:
            return None
        return d
    except Exception:
        return None


def save_cache(p: Path, content: str, finish_reason: Optional[str]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"content": content, "finish_reason": finish_reason}, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Per-question evaluation
# ---------------------------------------------------------------------------

def evaluate_one(
    *,
    pool: ClientPool,
    record: dict,
    benchmark: str,
    protocol: str,
    cache_dir: Path,
    model_id: str,
) -> dict:
    qid = record["question_id"]
    gold = record["gold"]

    if benchmark == "tombench":
        msgs = build_messages_tombench(record, protocol=protocol)
        n_opts = 4
    else:  # hitom / socialiqa / emobench → generic builder
        msgs = build_messages_generic(record, protocol=protocol)
        n_opts = len(record["options"])

    extract_direct, extract_cot, has_boxed, _vote = extractors_for(n_opts)

    sp = sampling_params_for(protocol)
    use_direct_extractor = (protocol == "direct")

    cp = cache_path(cache_dir, model_id, benchmark, protocol, qid, 0)
    cached = load_cache(cp)
    if cached is None:
        client = pool.get()
        content, finish = chat_once(
            client=client,
            model=pool.model,
            messages=msgs,
            temperature=sp["temperature"],
            top_p=sp["top_p"],
            max_tokens=sp["max_tokens"],
            enable_thinking=sp["enable_thinking"],
        )
        save_cache(cp, content, finish)
    else:
        content = cached["content"]
        finish = cached.get("finish_reason")

    pred = extract_direct(content) if use_direct_extractor else extract_cot(content)
    truncated = (finish == "length")

    return {
        "question_id": qid,
        "language": record.get("language", "en"),
        "task": record.get("task", "?"),
        "category": record.get("category"),       # emobench
        "n_options": n_opts,
        "gold": gold,
        "pred": pred,
        "correct": pred == gold,
        "protocol": protocol,
        "benchmark": benchmark,
        "model": model_id,
        "finish_reason": finish,
        "truncated": truncated,
        "has_boxed": has_boxed(content),
        "out_len": len(content),
    }


def run_protocol(
    *,
    pool: ClientPool,
    records: list[dict],
    benchmark: str,
    protocol: str,
    cache_dir: Path,
    model_id: str,
    concurrency: int,
) -> list[dict]:
    out: list[dict] = []
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = {
            ex.submit(
                evaluate_one,
                pool=pool, record=r, benchmark=benchmark, protocol=protocol,
                cache_dir=cache_dir, model_id=model_id,
            ): r for r in records
        }
        for f in tqdm(as_completed(futs), total=len(futs), desc=f"{model_id}/{benchmark}/{protocol}"):
            try:
                out.append(f.result())
            except Exception as e:  # noqa: BLE001
                rec = futs[f]
                print(f"  ERR qid={rec.get('question_id')}: {e}", flush=True)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="served-model-name exposed by vLLM")
    p.add_argument("--endpoints", nargs="+", required=True, help="host:port list")
    p.add_argument("--model-id", default=None, help="display id for caching + output (defaults to --model)")
    p.add_argument("--benchmark", choices=["tombench", "hitom", "socialiqa", "emobench"], required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--protocols", default="direct,direct_think,cot")
    p.add_argument("--output", required=True)
    p.add_argument("--cache-dir", default="output/cache")
    p.add_argument("--concurrency", type=int, default=64)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    model_id = args.model_id or args.model
    pool = ClientPool.from_local(args.endpoints, model=args.model)

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
            cache_dir=cache_dir, model_id=model_id, concurrency=args.concurrency,
        )
        all_results.extend(res)
        correct = sum(1 for x in res if x["correct"])
        trunc = sum(1 for x in res if x["truncated"])
        n = len(res)
        elapsed = time.time() - t0
        if n == 0:
            print(f"[{model_id}/{args.benchmark}/{protocol}] no results  elapsed={elapsed:.1f}s", flush=True)
        else:
            print(f"[{model_id}/{args.benchmark}/{protocol}] acc={correct}/{n}={correct/n:.4f}  "
                  f"truncated={trunc}/{n}={trunc/n:.2%}  elapsed={elapsed:.1f}s", flush=True)

    out_path.write_text(json.dumps(all_results, ensure_ascii=False))
    print(f"wrote {len(all_results)} results -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
