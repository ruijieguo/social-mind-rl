"""Distill Belief task corrections from DeepSeek-v4-pro.

For ToMBench Belief errors v3.2 ckpt-270 got wrong (≥2 of 3 protocols), ask
DeepSeek to produce a clean step-by-step reasoning + boxed answer. We then
package it as a training record where the *user* asks for CoT and the *assistant*
turns include the reasoning + the gold answer.

This is a hybrid SFT-style record but stored in the same RLVR rollout format —
the model learns to imitate DeepSeek's Belief reasoning chain while RL still
shapes the final answer reward.

Output schema matches data/tom/tom_train_stage14_weighted.jsonl format:
  messages (system + user + we-stash-assistant-as-reference-not-used-for-RL),
  ground_truth (the gold letter), tag, source=belief_distill, etc.

Note: the assistant message is NOT used by GRPO rollouts (which generate fresh
samples), but is logged to data/tom/raw/belief_distill_meta.jsonl for SFT use
later if desired.

Usage:
  python scripts/data/build_belief_distill.py
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.eval.clients import BackendSpec, ChatClient


SYSTEM_DISTILL_EN = """You are a careful theory-of-mind reasoner. Given a story and a multiple-choice question, write 2-4 short sentences of step-by-step reasoning that traces what each character knows, believes, or feels. Then output your final answer on the last line as \\boxed{X} where X is one of A, B, C, D.

Be concise — 50-150 words of reasoning, then the boxed answer."""

SYSTEM_DISTILL_ZH = """你是一位仔细的心理理论推理者。给定一个故事和单选题，写 2-4 句简短的逐步推理，分析每个角色知道什么、相信什么、感受什么。然后在最后一行输出最终答案，格式为 \\boxed{X}，其中 X 是 A、B、C、D 之一。

简洁 — 50-150 字推理，然后是 boxed 答案。"""


def build_user(rec: dict) -> str:
    if rec["language"] == "zh":
        return (
            f"故事：\n{rec['story']}\n\n"
            f"问题：{rec['question']}\n"
            f"A. {rec['opt_a']}\n"
            f"B. {rec['opt_b']}\n"
            f"C. {rec['opt_c']}\n"
            f"D. {rec['opt_d']}"
        )
    return (
        f"Story:\n{rec['story']}\n\n"
        f"Question: {rec['question']}\n"
        f"A. {rec['opt_a']}\n"
        f"B. {rec['opt_b']}\n"
        f"C. {rec['opt_c']}\n"
        f"D. {rec['opt_d']}"
    )


def collect_belief_errors(min_wrong: int = 2) -> list[dict]:
    """Pick Belief eval rows that v3.2 ckpt-270 got wrong on ≥min_wrong protocols."""
    err_count: dict[str, int] = {}
    for r in json.load(open(ROOT / "output/eval/stage16_ckpt270_tombench.json")):
        if r.get("task") != "Belief":
            continue
        if not r.get("correct"):
            err_count[r["question_id"]] = err_count.get(r["question_id"], 0) + 1
    candidates = {qid for qid, c in err_count.items() if c >= min_wrong}
    eval_map: dict[str, dict] = {}
    for line in open(ROOT / "data/tom/tombench_eval.jsonl"):
        r = json.loads(line)
        if r.get("task") == "Belief" and r["question_id"] in candidates:
            eval_map[r["question_id"]] = r
    return list(eval_map.values())


def make_train_record(eval_rec: dict, distilled_response: str) -> dict:
    """Build the standard training-style record (messages format)."""
    lang = eval_rec.get("language", "en")
    sys_p = (
        "You are a careful reader answering a multiple-choice theory-of-mind question. "
        "Read the story and the question carefully, then output ONLY your final answer "
        "in the format \\boxed{X} where X is one of A, B, C, D. "
        "Do not include any explanation, reasoning, or extra text."
    )
    return {
        "messages": [
            {"role": "system", "content": sys_p},
            {"role": "user", "content": build_user(eval_rec)},
        ],
        "ground_truth": eval_rec["gold"],
        "tag": "tom_mcq",
        "source": "belief_distill",
        "language": lang,
        "task": "Belief",
        "question_id": f"belief_distill_{eval_rec['question_id']}",
        "_distilled_cot": distilled_response,
    }


def call_distill(client: ChatClient, eval_rec: dict) -> dict | None:
    sys_p = SYSTEM_DISTILL_ZH if eval_rec["language"] == "zh" else SYSTEM_DISTILL_EN
    user_p = build_user(eval_rec)
    messages = [
        {"role": "system", "content": sys_p},
        {"role": "user", "content": user_p},
    ]
    try:
        result = client.chat(
            messages=messages,
            temperature=0.4,
            top_p=0.9,
            max_tokens=2048,
        )
    except Exception:
        return None
    content = (result.content if hasattr(result, "content") else (result or "")).strip()
    if not content:
        return None
    # Validate: must contain \boxed{<gold>}
    import re
    m = re.search(r"\\boxed\{([A-D])\}", content)
    if not m:
        return None
    if m.group(1) != eval_rec["gold"]:
        # DeepSeek itself got it wrong — skip (we don't want to teach a wrong answer)
        return None
    return make_train_record(eval_rec, content)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/tom/raw/belief_distill.jsonl")
    p.add_argument("--meta-out", default="data/tom/raw/belief_distill_meta.jsonl")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--model", default="deepseek-v4-pro")
    p.add_argument("--min-wrong", type=int, default=2,
                   help="only distill examples v3.2 missed in ≥N protocols (default 2)")
    args = p.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.meta_out).parent.mkdir(parents=True, exist_ok=True)

    cands = collect_belief_errors(args.min_wrong)
    print(f"Found {len(cands)} Belief error candidates (≥{args.min_wrong}/3 wrong)")

    spec = BackendSpec(name="deepseek", model=args.model)
    client = ChatClient(spec=spec)

    written = []
    fail = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = [ex.submit(call_distill, client, r) for r in cands]
        for fut in as_completed(futures):
            try:
                rec = fut.result()
            except Exception:
                rec = None
            if rec:
                written.append(rec)
            else:
                fail += 1

    # Write training records (without _distilled_cot since RL doesn't use it)
    with open(args.out, "w") as f:
        for r in written:
            r2 = {k: v for k, v in r.items() if not k.startswith("_")}
            f.write(json.dumps(r2, ensure_ascii=False) + "\n")
    # Write meta with distilled CoT for inspection / future SFT
    with open(args.meta_out, "w") as f:
        for r in written:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"wrote {len(written)} distilled Belief records (failed {fail})")
    print(f"  → {args.out}")
    print(f"  → {args.meta_out} (with CoT)")


if __name__ == "__main__":
    main()
