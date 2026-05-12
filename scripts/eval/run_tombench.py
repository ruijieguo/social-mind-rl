"""Run ToMBench evaluation against any OpenAI-compatible chat backend."""
from __future__ import annotations
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import jsonlines
from tqdm import tqdm

from scripts.eval.clients import BackendSpec, ChatClient, ChatResult
from scripts.eval.extractors import extract_direct, extract_cot, vote_del_tom
from scripts.eval.report import aggregate_results, format_markdown_table


SYSTEM_PROMPT_DIRECT = (
    "You are a careful reader answering a multiple-choice theory-of-mind question. "
    "Read the story and the question carefully, then output ONLY your final answer "
    "in the format \\boxed{X} where X is one of A, B, C, D. "
    "Do not include any explanation, reasoning, or extra text."
)

SYSTEM_PROMPT_COT = (
    "You are a careful reader answering a multiple-choice theory-of-mind question. "
    "Think step by step about the mental states of the characters, "
    "then output your final answer in the format \\boxed{X} where X is one of A, B, C, D. "
    "Put your final \\boxed{X} on the last line."
)


def build_user_prompt_en(*, story, question, opt_a, opt_b, opt_c, opt_d) -> str:
    return (
        f"Story:\n{story}\n\n"
        f"Question: {question}\n"
        f"A. {opt_a}\nB. {opt_b}\nC. {opt_c}\nD. {opt_d}"
    )


def build_user_prompt_zh(*, story, question, opt_a, opt_b, opt_c, opt_d) -> str:
    return (
        f"故事：\n{story}\n\n"
        f"问题：{question}\n"
        f"A. {opt_a}\nB. {opt_b}\nC. {opt_c}\nD. {opt_d}"
    )


def build_direct_messages(*, story, question, opt_a, opt_b, opt_c, opt_d, language: str) -> list[dict]:
    builder = build_user_prompt_zh if language == "zh" else build_user_prompt_en
    user = builder(story=story, question=question,
                   opt_a=opt_a, opt_b=opt_b, opt_c=opt_c, opt_d=opt_d)
    return [
        {"role": "system", "content": SYSTEM_PROMPT_DIRECT},
        {"role": "user", "content": user},
    ]


def build_cot_messages(*, story, question, opt_a, opt_b, opt_c, opt_d, language: str) -> list[dict]:
    builder = build_user_prompt_zh if language == "zh" else build_user_prompt_en
    user = builder(story=story, question=question,
                   opt_a=opt_a, opt_b=opt_b, opt_c=opt_c, opt_d=opt_d)
    return [
        {"role": "system", "content": SYSTEM_PROMPT_COT},
        {"role": "user", "content": user},
    ]


# ----------------------------------------------------------------------
# Caching
# ----------------------------------------------------------------------

def _cache_path(cache_dir: Path, model: str, protocol: str, qid: str, sample_idx: int = 0) -> Path:
    safe_model = model.replace("/", "_").replace(":", "_")
    return cache_dir / f"{safe_model}__{protocol}__{qid}__s{sample_idx}.json"


def _load_cached(p: Path) -> Optional[dict]:
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return None
    return None


def _save_cached(p: Path, data: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False))


# ----------------------------------------------------------------------
# Per-question evaluation
# ----------------------------------------------------------------------

def evaluate_one(
    *,
    client: ChatClient,
    record: dict,
    protocol: str,
    cache_dir: Path,
    model_id_for_cache: str,
) -> dict:
    qid = record["question_id"]
    language = record["language"]
    gold = record["gold"]
    task = record["task"]

    if protocol == "direct":
        messages = build_direct_messages(
            story=record["story"], question=record["question"],
            opt_a=record["opt_a"], opt_b=record["opt_b"],
            opt_c=record["opt_c"], opt_d=record["opt_d"],
            language=language,
        )
        sample_params = dict(temperature=0.0, top_p=1.0, max_tokens=2048)
        n_samples = 1
    elif protocol == "cot":
        messages = build_cot_messages(
            story=record["story"], question=record["question"],
            opt_a=record["opt_a"], opt_b=record["opt_b"],
            opt_c=record["opt_c"], opt_d=record["opt_d"],
            language=language,
        )
        sample_params = dict(temperature=0.6, top_p=0.9, max_tokens=1024)
        n_samples = 1
    elif protocol == "del_tom":
        messages = build_cot_messages(
            story=record["story"], question=record["question"],
            opt_a=record["opt_a"], opt_b=record["opt_b"],
            opt_c=record["opt_c"], opt_d=record["opt_d"],
            language=language,
        )
        sample_params = dict(temperature=0.7, top_p=0.95, max_tokens=1024)
        n_samples = 8
    else:
        raise ValueError(f"unknown protocol: {protocol}")

    answers: list[Optional[str]] = []
    raw_responses: list[str] = []
    for sample_idx in range(n_samples):
        cache_p = _cache_path(cache_dir, model_id_for_cache, protocol, qid, sample_idx)
        cached = _load_cached(cache_p)
        if cached is not None:
            content = cached["content"]
        else:
            res: ChatResult = client.chat(messages, **sample_params)
            content = res.content
            _save_cached(cache_p, {
                "qid": qid, "protocol": protocol, "sample_idx": sample_idx,
                "content": content,
                "prompt_tokens": res.prompt_tokens,
                "completion_tokens": res.completion_tokens,
            })
        raw_responses.append(content)
        if protocol == "direct":
            answers.append(extract_direct(content))
        else:
            answers.append(extract_cot(content))

    if protocol == "del_tom":
        pred = vote_del_tom(answers)
    else:
        pred = answers[0]

    return {
        "question_id": qid,
        "language": language,
        "task": task,
        "gold": gold,
        "pred": pred,
        "model": model_id_for_cache,
        "protocol": protocol,
        "correct": pred == gold,
        "raw_responses": raw_responses,
    }


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def _build_backend(args) -> tuple[BackendSpec, str]:
    """Returns (spec, cache_id)."""
    if args.preset == "baseline-all":
        raise SystemExit("--preset baseline-all expands to multiple runs; use it via the runner script, not as a single backend.")
    if args.backend == "dashscope":
        extra: dict = {}
        if args.thinking is not None:
            extra["enable_thinking"] = args.thinking
        spec = BackendSpec(name="dashscope", model=args.model, extra_body=extra)
        cache_id = f"{args.model}-{'t' if args.thinking else 'nt'}"
    elif args.backend == "deepseek":
        spec = BackendSpec(name="deepseek", model=args.model)
        cache_id = args.model
    elif args.backend == "openai":
        if not args.base_url:
            raise SystemExit("--backend openai requires --base-url")
        spec = BackendSpec(name="openai", model=args.model, base_url=args.base_url)
        cache_id = args.model
    else:
        raise SystemExit(f"unknown backend: {args.backend}")
    return spec, cache_id


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preset", choices=["baseline-all"], default=None,
                   help="run a preset of multiple backends (baseline-all = qwen3-8b nt+t + deepseek-v4-pro)")
    p.add_argument("--backend", choices=["dashscope", "deepseek", "openai"], default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--thinking", type=lambda x: x.lower() == "true", default=None,
                   help="for dashscope qwen3-8b: enable_thinking true|false")
    p.add_argument("--base-url", default=None)
    p.add_argument("--protocols", default="direct",
                   help="comma-separated subset of direct,cot,del_tom")
    p.add_argument("--data", default="data/tom/tombench_eval.jsonl")
    p.add_argument("--output", default="output/eval/result.json")
    p.add_argument("--cache-dir", default="output/eval_cache")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--limit", type=int, default=None,
                   help="evaluate only first N questions (debug)")
    args = p.parse_args()

    if args.preset == "baseline-all":
        return _run_baseline_all(args)

    # Single-backend run
    spec, cache_id = _build_backend(args)
    client = ChatClient(spec=spec)
    return _run_single(args, client, cache_id)


def _run_single(args, client: ChatClient, cache_id: str):
    data_path = Path(args.data)
    cache_dir = Path(args.cache_dir)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    with jsonlines.open(data_path) as reader:
        for r in reader:
            records.append(r)
    if args.limit:
        records = records[: args.limit]

    protocols = [s.strip() for s in args.protocols.split(",") if s.strip()]
    all_results: list[dict] = []
    for protocol in protocols:
        print(f"=== {cache_id} :: {protocol} :: {len(records)} questions ===")
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futures = [
                ex.submit(evaluate_one,
                          client=client, record=r, protocol=protocol,
                          cache_dir=cache_dir, model_id_for_cache=cache_id)
                for r in records
            ]
            for f in tqdm(as_completed(futures), total=len(futures), desc=protocol):
                all_results.append(f.result())

    out_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2))
    print(f"wrote {len(all_results)} eval records to {out_path}")

    agg = aggregate_results(all_results)
    md_path = out_path.with_suffix(".md")
    md_path.write_text(format_markdown_table(agg))
    print(f"wrote markdown summary to {md_path}")


def _run_baseline_all(args):
    """Run Qwen3-8B (nt + t) and deepseek-v4-pro on both direct + cot."""
    # CRITICAL FIX: Clear preset to avoid SystemExit in _build_backend
    args.preset = None

    plans = [
        dict(backend="dashscope", model="qwen3-8b", thinking=False, cache_id="qwen3-8b-nt"),
        dict(backend="dashscope", model="qwen3-8b", thinking=True,  cache_id="qwen3-8b-t"),
        dict(backend="deepseek",  model="deepseek-v4-pro", thinking=None, cache_id="deepseek-v4-pro"),
    ]

    args.protocols = "direct,cot"
    args.output = "output/eval/baseline_combined.json"

    combined: list[dict] = []
    for plan in plans:
        args.backend = plan["backend"]
        args.model = plan["model"]
        args.thinking = plan["thinking"]
        spec, _ = _build_backend(args)
        client = ChatClient(spec=spec)
        cache_id = plan["cache_id"]

        records = []
        with jsonlines.open(args.data) as reader:
            for r in reader:
                records.append(r)
        if args.limit:
            records = records[: args.limit]

        for protocol in args.protocols.split(","):
            protocol = protocol.strip()
            print(f"=== {cache_id} :: {protocol} :: {len(records)} questions ===")
            with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
                futures = [
                    ex.submit(evaluate_one,
                              client=client, record=r, protocol=protocol,
                              cache_dir=Path(args.cache_dir),
                              model_id_for_cache=cache_id)
                    for r in records
                ]
                for f in tqdm(as_completed(futures), total=len(futures), desc=f"{cache_id}/{protocol}"):
                    combined.append(f.result())

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(combined, ensure_ascii=False, indent=2))
    print(f"wrote {len(combined)} records to {out_path}")

    agg = aggregate_results(combined)
    md_path = Path("output/eval/baseline_report.md")
    md_path.write_text(format_markdown_table(agg))
    print(f"wrote markdown report to {md_path}")


if __name__ == "__main__":
    main()
