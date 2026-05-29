"""DEV-side eval of deepseek-v4-pro (DeepSeek official API) — the 4th model.

Runs the SAME 4 benchmarks × 3 protocols as the local Qwen3-14B run, reusing the
IDENTICAL prompts (prompts.py), extractors, and sampling params, so deepseek is a
fair fourth column in the report.

Fairness note: `enable_thinking` is a Qwen/vLLM-only knob; deepseek-v4-pro is a
native reasoning model with no such toggle, so we do NOT send it. Every other
parameter is identical to the local models:
  - direct        : T=0, top_p=1, max_tokens=64,   DIRECT system prompt, extract_direct(content)
  - direct_think  : T=0, top_p=1, max_tokens=8192, DIRECT system prompt, extract_cot(content)
  - cot           : T=0.6, top_p=0.95, max_tokens=8192, COT system prompt, extract_cot(content)
The deepseek API returns reasoning in a separate `reasoning_content` field; the
visible answer (with \\boxed{}) is in `content`, which is what we extract from —
exactly as the local models put \\boxed{} after their <think> block.

Backend: https://api.deepseek.com, key from DEEPSEEK_API_KEY, model deepseek-v4-pro.

Run (inside the qwen3-tom-dev image which ships openai+tqdm):
  docker run --rm -i --network host -e DEEPSEEK_API_KEY=$DEEPSEEK_API_KEY \
    -v "$PWD":/work -w /work -e PYTHONPATH=/work/scripts --entrypoint python3 \
    qwen3-tom-dev:latest scripts/deepseek_eval.py \
      --benchmark tombench --data /work/data/tom/tombench_eval.jsonl \
      --protocols direct,direct_think,cot --output output/tombench/deepseek.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
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

MODEL_ID = "deepseek"
DS_MODEL = "deepseek-v4-pro"
DS_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")


def _read_jsonl(path: Path) -> list[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def chat_once(client, messages, *, temperature, top_p, max_tokens, max_retries=5):
    """Return (content, reasoning, finish_reason). Backs off on 429."""
    last_err = None
    for attempt in range(max_retries):
        try:
            r = client.chat.completions.create(
                model=DS_MODEL, messages=messages,
                temperature=temperature, top_p=top_p, max_tokens=max_tokens,
            )
            ch = r.choices[0]
            content = ch.message.content or ""
            reasoning = getattr(ch.message, "reasoning_content", None) or ""
            return content, reasoning, ch.finish_reason
        except Exception as e:  # noqa: BLE001
            last_err = e
            msg = str(e).lower()
            rate = ("429" in msg or "rate" in msg or "limit" in msg or "tpm" in msg or "rpm" in msg)
            if attempt < max_retries - 1:
                time.sleep((30.0 * (attempt + 1)) if rate else (2.0 ** attempt))
                continue
            break
    raise last_err


def cache_path(cache_dir, bench, protocol, qid):
    return cache_dir / bench / MODEL_ID / protocol / f"{qid}__s0.json"


def load_cache(p):
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text())
        return d if "content" in d else None
    except Exception:
        return None


def save_cache(p, content, reasoning, finish):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"content": content, "reasoning": reasoning,
                             "finish_reason": finish}, ensure_ascii=False))


def evaluate_one(client, record, benchmark, protocol, cache_dir):
    qid = record["question_id"]
    gold = record["gold"]
    if benchmark == "tombench":
        msgs = build_messages_tombench(record, protocol=protocol)
        n_opts = 4
    else:
        msgs = build_messages_generic(record, protocol=protocol)
        n_opts = len(record["options"])
    extract_direct, extract_cot, has_boxed, _vote = extractors_for(n_opts)
    sp = sampling_params_for(protocol)

    cp = cache_path(cache_dir, benchmark, protocol, qid)
    cached = load_cache(cp)
    if cached is None:
        content, reasoning, finish = chat_once(
            client, msgs, temperature=sp["temperature"], top_p=sp["top_p"],
            max_tokens=sp["max_tokens"],
        )
        save_cache(cp, content, reasoning, finish)
    else:
        content = cached["content"]; reasoning = cached.get("reasoning", ""); finish = cached.get("finish_reason")

    # Extract from the VISIBLE content only (reasoning_content is separate),
    # mirroring how local models put \boxed{} after their <think> block.
    pred = extract_direct(content) if protocol == "direct" else extract_cot(content)

    return {
        "question_id": qid,
        "language": record.get("language", "en"),
        "task": record.get("task", "?"),
        "category": record.get("category"),
        "n_options": n_opts,
        "gold": gold,
        "pred": pred,
        "correct": pred == gold,
        "protocol": protocol,
        "benchmark": benchmark,
        "model": MODEL_ID,
        "finish_reason": finish,
        "truncated": finish == "length",
        "has_boxed": has_boxed(content),
        "out_len": len(content),
        "reasoning_len": len(reasoning),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", choices=["tombench", "hitom", "socialiqa", "emobench"], required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--protocols", default="direct,direct_think,cot")
    p.add_argument("--output", required=True)
    p.add_argument("--cache-dir", default="output/cache")
    p.add_argument("--concurrency", type=int, default=12)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise SystemExit("DEEPSEEK_API_KEY not set")
    client = OpenAI(api_key=key, base_url=DS_BASE_URL, timeout=600.0)

    records = _read_jsonl(Path(args.data))
    if args.limit:
        records = records[: args.limit]
    cache_dir = Path(args.cache_dir)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    protocols = [s.strip() for s in args.protocols.split(",") if s.strip()]

    all_results = []
    for protocol in protocols:
        t0 = time.time()
        res = []
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = {ex.submit(evaluate_one, client, r, args.benchmark, protocol, cache_dir): r for r in records}
            for f in tqdm(as_completed(futs), total=len(futs), desc=f"deepseek/{args.benchmark}/{protocol}"):
                try:
                    res.append(f.result())
                except Exception as e:  # noqa: BLE001
                    print(f"  ERR qid={futs[f].get('question_id')}: {e}", flush=True)
        all_results.extend(res)
        c = sum(1 for x in res if x["correct"]); t = sum(1 for x in res if x["truncated"]); n = len(res)
        el = time.time() - t0
        print(f"[deepseek/{args.benchmark}/{protocol}] acc={c}/{n}={c/n:.4f}  "
              f"truncated={t}/{n}={t/n:.2%}  elapsed={el:.1f}s", flush=True) if n else None

    out_path.write_text(json.dumps(all_results, ensure_ascii=False))
    print(f"wrote {len(all_results)} results -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
