"""SFT cold-start training data builder for ToMBench.

Per docs/improvement_plan_v3.md S1, this is the missing step from our stage 1-8
pipeline. Loads 4000 GPT-5.5 reasoning traces and converts them to the Alpaca-
style format expected by ROLL's SFT pipeline (prompt_key/query_key/response_key).

Input format (from gen_reasoning_traces.py):
  data/tom/raw/reasoning_traces.jsonl
  Each record: {question_id, story, question, options{A/B/C/D}, gold,
                reasoning, final, language, task, source}

Output format (Alpaca-style for ROLL SFT):
  {
    system: <system prompt>,           # system_key
    prompt: <user message>,             # prompt_key
    response: <assistant message>,      # response_key
    question_id: ...,
    language: en/zh,
    task: ...,
  }
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path


SYSTEM_EN = """You are a careful reader answering a multiple-choice theory-of-mind question.
Read the story and the question carefully, then think step by step in
<reasoning>...</reasoning> tags using labelled cognitive operations
([identify], [track], [infer], [conclude], etc.), and output your final answer
in the format \\boxed{X} where X is one of A, B, C, D."""


SYSTEM_ZH = """你是一名细心的读者, 正在回答一道心智理论(Theory of Mind)选择题。
仔细阅读故事和问题, 在 <reasoning>...</reasoning> 标签内一步步推理, 用标注的认知操作
（[识别], [追踪], [推断], [总结] 等）, 最后用 \\boxed{X} 格式输出最终答案,
X 为 A、B、C、D 之一。"""


def make_user_prompt_en(rec):
    return (
        f"Story:\n{rec['story']}\n\n"
        f"Question: {rec['question']}\n"
        f"A. {rec['options']['A']}\n"
        f"B. {rec['options']['B']}\n"
        f"C. {rec['options']['C']}\n"
        f"D. {rec['options']['D']}"
    )


def make_user_prompt_zh(rec):
    return (
        f"故事：\n{rec['story']}\n\n"
        f"问题：{rec['question']}\n"
        f"A. {rec['options']['A']}\n"
        f"B. {rec['options']['B']}\n"
        f"C. {rec['options']['C']}\n"
        f"D. {rec['options']['D']}"
    )


def make_assistant_response(rec):
    """Build the assistant's reasoning + final answer."""
    return f"<reasoning>\n{rec['reasoning']}\n</reasoning>\n\n{rec['final']}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/tom/raw/reasoning_traces.jsonl")
    p.add_argument("--out", default="data/tom/tom_train_sft.jsonl")
    p.add_argument("--shuffle-seed", type=int, default=42)
    args = p.parse_args()

    inp = Path(args.input)
    out = Path(args.out)
    if not inp.exists():
        raise SystemExit(f"input {inp} missing — run gen_reasoning_traces.py first")
    out.parent.mkdir(parents=True, exist_ok=True)

    n_in = n_out = 0
    records = []
    with inp.open() as f:
        for line in f:
            n_in += 1
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("language") == "zh":
                sys_prompt = SYSTEM_ZH
                user = make_user_prompt_zh(rec)
            else:
                sys_prompt = SYSTEM_EN
                user = make_user_prompt_en(rec)
            assistant = make_assistant_response(rec)
            records.append({
                "system": sys_prompt,
                "prompt": user,
                "response": assistant,
                "question_id": rec["question_id"],
                "language": rec["language"],
                "task": rec["task"],
                "source": rec.get("source", ""),
            })

    # Shuffle for SFT
    import random
    random.Random(args.shuffle_seed).shuffle(records)

    with out.open("w", encoding="utf-8") as fp:
        for r in records:
            fp.write(json.dumps(r, ensure_ascii=False) + "\n")
            n_out += 1
    print(f"[build_sft] {n_in} input → {n_out} SFT records → {out}", file=sys.stderr)


if __name__ == "__main__":
    main()

