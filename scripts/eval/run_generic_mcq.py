"""
Generic MCQ benchmark eval: support any number of options (3, 4, 18) and arbitrary
benchmarks beyond ToMBench. Reuses the same backend infrastructure as run_tombench
but with parametric prompt builders and extractors.

Input record schema (jsonl):
  {
    "question_id": str,
    "language": "en" | "zh",
    "task": str (any string, used for per-task aggregation),
    "story": str,
    "question": str,
    "options": [str, str, ...],   # variable length, will be labeled A, B, C, ...
    "gold": str (single letter A-Z)
  }

Output: same schema as run_tombench.py output.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

# Add repo root to path so we can import scripts.eval.*
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.eval.clients import BackendSpec, ChatClient
from scripts.eval.extractors_generic import make_extractors


# ---- Prompt templates ----

SYSTEM_DIRECT_TPL = (
    "You are a careful reader answering a multiple-choice question. "
    "Read the story (if any) and the question carefully, then output ONLY your final answer "
    "in the format \\boxed{{X}} where X is one of {letters}. "
    "Do not include any explanation, reasoning, or extra text."
)

SYSTEM_COT_TPL = (
    "You are a careful reader answering a multiple-choice question. "
    "Think step by step about the question, then output your final answer "
    "in the format \\boxed{{X}} where X is one of {letters}. "
    "Put your final \\boxed{{X}} on the last line."
)


def build_options_block(options: list[str]) -> str:
    return "\n".join(f"{chr(ord('A') + i)}. {o}" for i, o in enumerate(options))


def build_user_prompt(*, story: str, question: str, options: list[str], language: str) -> str:
    opts_block = build_options_block(options)
    if language == "zh":
        return (
            (f"故事：\n{story}\n\n" if story else "")
            + f"问题：{question}\n"
            + opts_block
        )
    return (
        (f"Story:\n{story}\n\n" if story else "")
        + f"Question: {question}\n"
        + opts_block
    )


def build_messages(record: dict, protocol: str) -> list[dict]:
    options = record["options"]
    n = len(options)
    letters = ", ".join(chr(ord("A") + i) for i in range(n))
    sys_tpl = SYSTEM_COT_TPL if protocol in ("cot", "del_tom") else SYSTEM_DIRECT_TPL
    system = sys_tpl.format(letters=letters)
    user = build_user_prompt(
        story=record.get("story", ""),
        question=record["question"],
        options=options,
        language=record.get("language", "en"),
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


# ---- Eval loop ----

def evaluate_one(client: ChatClient, record: dict, protocol: str,
                 model: str, del_tom_n: int = 8) -> dict:
    n_options = len(record["options"])
    extract_direct, extract_cot, vote_del_tom = make_extractors(n_options)

    # Unified v2 protocol (2026-05-28): direct = strictly no thinking on
    # both sides, so the protocol is comparable across self-served Qwen and
    # DeepSeek API. cot/del_tom keep thinking ON.
    #   Qwen3: enable_thinking=False (chat_template_kwargs) + max_tokens=64
    #   DeepSeek: thinking.type=disabled + max_tokens=64
    # Both produce a clean \boxed{X} that fits in <16 tokens.
    extra_no_think_qwen = {"chat_template_kwargs": {"enable_thinking": False}}
    extra_no_think_ds = {"thinking": {"type": "disabled"}}
    is_qwen = model.startswith(("eval-target-", "eval-")) or "qwen" in model.lower()

    # cot/del_tom keep thinking ON; allocate enough budget for either
    # Qwen <think>...</think> in content, or DeepSeek reasoning_content.
    # Capped at 4096 to fit within vLLM's typical max_model_len=8192
    # (input prompts up to ~3500 tokens for Hi-ToM long stories).
    big_tokens = 4096

    if protocol == "del_tom":
        msgs = build_messages(record, "cot")
        sample_params = dict(temperature=0.7, top_p=0.95, max_tokens=big_tokens)
        n_samples = del_tom_n
        extra = None
    elif protocol == "cot":
        msgs = build_messages(record, "cot")
        sample_params = dict(temperature=0.0, top_p=1.0, max_tokens=big_tokens)
        n_samples = 1
        extra = None
    else:  # direct (strictly no thinking, both sides)
        msgs = build_messages(record, "direct")
        sample_params = dict(temperature=0.0, top_p=1.0, max_tokens=64)
        extra = extra_no_think_qwen if is_qwen else extra_no_think_ds
        n_samples = 1

    answers: list[Optional[str]] = []
    for _ in range(n_samples):
        result = client.chat(messages=msgs, extra_body_override=extra, **sample_params)
        content = result.content if hasattr(result, "content") else (result or "")
        if protocol == "direct":
            answers.append(extract_direct(content))
        else:
            answers.append(extract_cot(content))

    if protocol == "del_tom":
        pred = vote_del_tom(answers)
    else:
        pred = answers[0]

    gold = record["gold"]
    return {
        "question_id": record["question_id"],
        "source": record.get("source", "?"),
        "language": record.get("language", "en"),
        "task": record.get("task", "?"),
        "n_options": n_options,
        "gold": gold,
        "pred": pred,
        "correct": pred == gold,
        "protocol": protocol,
        "model": model,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["openai", "deepseek"], required=True)
    p.add_argument("--base-url", default=None)
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True, help="jsonl with options field")
    p.add_argument("--protocols", default="direct,cot,del_tom")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--del-tom-n", type=int, default=8)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    spec = BackendSpec(name=args.backend, model=args.model, base_url=args.base_url)
    client = ChatClient(spec=spec)
    protocols = [x.strip() for x in args.protocols.split(",")]

    records = []
    with open(args.data) as f:
        for line in f:
            records.append(json.loads(line))
    if args.limit:
        records = records[:args.limit]

    results = []
    for protocol in protocols:
        print(f"=== {args.model} :: {protocol} :: {len(records)} questions ===", flush=True)
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futures = {
                ex.submit(evaluate_one, client, r, protocol, args.model, args.del_tom_n): r
                for r in records
            }
            for i, fut in enumerate(as_completed(futures)):
                try:
                    res = fut.result()
                    results.append(res)
                except Exception as e:
                    rec = futures[fut]
                    print(f"  err qid={rec['question_id']}: {e}", flush=True)
                if (i + 1) % 100 == 0:
                    correct = sum(1 for r in results if r.get("protocol") == protocol and r.get("correct"))
                    total = sum(1 for r in results if r.get("protocol") == protocol)
                    print(f"  progress: {i+1}/{len(records)}  acc so far: {correct}/{total}", flush=True)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, ensure_ascii=False)

    # Summary
    print("\n=== Summary ===")
    for protocol in protocols:
        sub = [r for r in results if r.get("protocol") == protocol]
        if sub:
            acc = sum(r["correct"] for r in sub) / len(sub)
            print(f"  {protocol:>10}: {acc:.4f}  (n={len(sub)})")
    print(f"\nWrote → {args.output}")


if __name__ == "__main__":
    main()
