"""Phase D: GPT-5.5 reasoning trace generation for SFT cold start.

For each training question, ask GPT-5.5 to:
1. Read story + question + options + gold answer
2. Generate STRUCTURED reasoning (numbered steps) for HOW to arrive at the gold answer
3. Conclude with \boxed{X} matching the gold answer

This gives the SFT stage a teacher signal not just for the answer but for the
reasoning process. Per DeepSeek-R1 / Light-R1 / Magistral, SFT cold-start with
reasoning traces is the standard pipeline missing from our stage1-8.

Two-phase workflow:
  1. Small-scale validation: --limit 50 → manual + GPT-5.5 audit
  2. Full generation: 4000 records covering 7 ToMBench tasks

Output: data/tom/raw/reasoning_traces.jsonl
Each record:
  {
    "question_id": <orig>,
    "story": <orig>,
    "question": <orig>,
    "options": {"A": ..., "B": ..., "C": ..., "D": ...},
    "gold": "B",
    "reasoning": "Step 1: ...\\nStep 2: ...\\nStep 3: ...",
    "final": "\\boxed{B}",
    "language": "en"|"zh",
    "task": <orig>,
    "source": <orig>,
  }
"""
from __future__ import annotations
import argparse
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI


SYSTEM_EN = """You are an expert theory-of-mind tutor producing high-quality reasoning traces for training student models.

You are given a story, a question, four options, and the correct answer (gold). Your job is to write a clear, structured reasoning trace that arrives at the gold answer.

CRITICAL OUTPUT FORMAT (must follow exactly):
```
<reasoning>
Step 1 [name of the mental-state operation]: <one sentence>.
Step 2 [name of the operation]: <one sentence>.
Step 3 [name of the operation]: <one sentence>.
...
Step N [conclude]: <one sentence justifying why the answer is X>.
</reasoning>
<answer>\\boxed{X}</answer>
```

REASONING QUALITY REQUIREMENTS:
- 3-6 numbered steps, each ONE sentence, each labelled with a cognitive operation:
  [identify], [track], [infer], [check], [conclude], [contrast], [recall], [apply]
- The reasoning must ARRIVE at the gold answer X. Do not output a different answer.
- Each step must be a concrete inference, not a restatement of the story.
- Be EXPLICIT about whose mental state you are tracking ("character A believes...", "X does not know...").
- Avoid filler like "Now we need to think about...", "It is clear that...". Get straight to the inference.

IMPORTANT: Output ONLY the <reasoning>...</reasoning><answer>...</answer> blocks. No preamble, no commentary.
"""


SYSTEM_ZH = """你是心智理论(Theory of Mind)专家，正在为学生模型生成高质量的推理过程示范。

给定故事、问题、四个选项、以及正确答案 (gold)。你的任务是写一个清晰、结构化的推理链，最终得出 gold 答案。

输出格式 (必须严格遵守)：
```
<reasoning>
步骤 1 [心智操作名称]: <一句话>。
步骤 2 [操作名称]: <一句话>。
步骤 3 [操作名称]: <一句话>。
...
步骤 N [总结]: <一句话说明为什么答案是 X>。
</reasoning>
<answer>\\boxed{X}</answer>
```

推理质量要求：
- 3-6 步, 每步一句, 每步标注认知操作:
  [识别], [追踪], [推断], [验证], [总结], [对比], [回想], [应用]
- 推理必须得出 gold 答案 X。不能输出不同答案。
- 每步是具体推断, 不是故事复述。
- 明确指出追踪的是谁的心智状态 ("A 相信...", "X 不知道...")。
- 避免"现在我们需要考虑..."这种废话。直接给推断。

重要: 只输出 <reasoning>...</reasoning><answer>...</answer> 部分, 不要任何前置或后置注释。
"""


USER_TMPL_EN = """Story:
{story}

Question: {question}

Options:
A. {opt_a}
B. {opt_b}
C. {opt_c}
D. {opt_d}

Correct answer: {gold}

Write the reasoning trace now (3-6 steps, each labelled, ending with \\boxed{{{gold}}})."""


USER_TMPL_ZH = """故事：
{story}

问题：{question}

选项：
A. {opt_a}
B. {opt_b}
C. {opt_c}
D. {opt_d}

正确答案：{gold}

现在请写出推理过程 (3-6 步, 每步标注认知操作, 以 \\boxed{{{gold}}} 结尾)。"""


_RE_REASONING = re.compile(r"<reasoning>(.*?)</reasoning>", re.DOTALL)
_RE_ANSWER = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_RE_BOXED = re.compile(r"\\boxed\{([A-D])\}")


def parse_response(text: str, gold: str) -> dict | None:
    """Return {"reasoning": str, "final": str} or None on parse fail."""
    if not text:
        return None
    m_r = _RE_REASONING.search(text)
    m_a = _RE_ANSWER.search(text)
    if not m_r or not m_a:
        return None
    reasoning = m_r.group(1).strip()
    answer = m_a.group(1).strip()
    # Verify boxed answer matches gold
    m_b = _RE_BOXED.search(answer)
    if not m_b:
        return None
    if m_b.group(1) != gold:
        return None  # GPT-5.5 disagrees with gold; reject this trace
    # Quality check: at least 3 step lines
    steps = [ln for ln in reasoning.split("\n") if ln.strip().startswith(("Step", "步骤"))]
    if len(steps) < 3:
        return None
    return {"reasoning": reasoning, "final": answer}


def load_eval_prompt_format(record: dict) -> tuple[str, str, dict, str]:
    """Extract story / question / options / gold from a tom_train.jsonl record."""
    msgs = record["messages"]
    user_msg = next(m["content"] for m in msgs if m["role"] == "user")
    # Format is: "Story:\n<story>\n\nQuestion: <q>\nA. <a>\nB. <b>\nC. <c>\nD. <d>"
    # Or ZH: "故事：\n<story>\n\n问题：<q>\nA. <a>..."
    if user_msg.startswith("Story:"):
        sm, qm = "Story:", "Question:"
        lang = "en"
    elif user_msg.startswith("故事："):
        sm, qm = "故事：", "问题："
        lang = "zh"
    else:
        return None
    _, _, after = user_msg.partition(sm)
    story, _, qopts = after.lstrip("\n").partition(qm)
    lines = qopts.strip().split("\n")
    question = lines[0].strip()
    opts = {"A": "", "B": "", "C": "", "D": ""}
    for line in lines[1:]:
        line = line.strip()
        if line.startswith("A."): opts["A"] = line[2:].strip()
        elif line.startswith("B."): opts["B"] = line[2:].strip()
        elif line.startswith("C."): opts["C"] = line[2:].strip()
        elif line.startswith("D."): opts["D"] = line[2:].strip()
    gold = record["ground_truth"]
    return story.strip(), question, opts, gold, lang


def generate_trace(client, record: dict, model: str = "gpt-5.5", max_retries: int = 3) -> dict | None:
    parsed = load_eval_prompt_format(record)
    if parsed is None:
        return None
    story, question, opts, gold, lang = parsed
    if lang == "en":
        system = SYSTEM_EN
        user = USER_TMPL_EN.format(
            story=story, question=question,
            opt_a=opts["A"], opt_b=opts["B"], opt_c=opts["C"], opt_d=opts["D"],
            gold=gold,
        )
    else:
        system = SYSTEM_ZH
        user = USER_TMPL_ZH.format(
            story=story, question=question,
            opt_a=opts["A"], opt_b=opts["B"], opt_c=opts["C"], opt_d=opts["D"],
            gold=gold,
        )
    last_err = ""
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0.3,  # low — we want deterministic structured reasoning
                max_tokens=800,
                timeout=120,
            )
            content = resp.choices[0].message.content or ""
            parsed_trace = parse_response(content, gold)
            if parsed_trace is None:
                last_err = f"parse/gold-mismatch: {content[:120]!r}"
                continue
            return {
                "question_id": record["question_id"],
                "story": story,
                "question": question,
                "options": opts,
                "gold": gold,
                "reasoning": parsed_trace["reasoning"],
                "final": parsed_trace["final"],
                "language": lang,
                "task": record["task"],
                "source": record.get("source", ""),
            }
        except Exception as e:
            last_err = f"{type(e).__name__}: {str(e)[:120]}"
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    print(f"[trace] {record['question_id']} failed: {last_err}", file=sys.stderr)
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/tom/tom_train.jsonl")
    p.add_argument("--out", default="data/tom/raw/reasoning_traces.jsonl")
    p.add_argument("--limit", type=int, default=0, help="0 = no limit. Use 50 for validation.")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--model", default="gpt-5.5")
    p.add_argument("--task-balance", action="store_true",
                   help="Sample evenly across 7 ToMBench tasks (recommended for full run)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true", help="Skip qids already in --out")
    args = p.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY not set")
    client = OpenAI(api_key=api_key, base_url=base_url)

    # Load all training records
    all_records = []
    with open(args.input) as f:
        for line in f:
            all_records.append(json.loads(line))
    print(f"[trace] loaded {len(all_records)} training records", file=sys.stderr)

    # Optional balanced sampling
    rng = random.Random(args.seed)
    if args.task_balance:
        from collections import defaultdict
        by_task = defaultdict(list)
        for r in all_records:
            by_task[r["task"]].append(r)
        per_task = (args.limit // len(by_task)) if args.limit > 0 else 0
        sampled = []
        for task, recs in by_task.items():
            rng.shuffle(recs)
            sampled.extend(recs[:per_task] if per_task else recs)
        rng.shuffle(sampled)
        all_records = sampled
        print(f"[trace] task-balanced sampling: {len(all_records)} records", file=sys.stderr)
    else:
        rng.shuffle(all_records)

    if args.limit > 0:
        all_records = all_records[:args.limit]

    # Resume support
    done_qids = set()
    if args.resume and Path(args.out).exists():
        with open(args.out) as f:
            for line in f:
                try:
                    done_qids.add(json.loads(line)["question_id"])
                except Exception:
                    continue
        print(f"[trace] resume: skipping {len(done_qids)} done", file=sys.stderr)
        all_records = [r for r in all_records if r["question_id"] not in done_qids]

    print(f"[trace] processing {len(all_records)} records (concurrency={args.concurrency})", file=sys.stderr)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume else "w"
    write_lock = threading.Lock()
    counter = {"ok": 0, "fail": 0}
    started = time.time()
    with out_path.open(mode, encoding="utf-8") as fp, ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = [ex.submit(generate_trace, client, rec, args.model) for rec in all_records]
        total = len(futures)
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            with write_lock:
                if r:
                    fp.write(json.dumps(r, ensure_ascii=False) + "\n")
                    fp.flush()
                    counter["ok"] += 1
                else:
                    counter["fail"] += 1
                if i % 25 == 0 or i == total:
                    elapsed = time.time() - started
                    rate = i / elapsed if elapsed > 0 else 0
                    eta = (total - i) / rate if rate > 0 else 0
                    print(f"[trace] {i}/{total} ok={counter['ok']} fail={counter['fail']} "
                          f"rate={rate:.2f}/s elapsed={elapsed:.0f}s eta={eta:.0f}s",
                          file=sys.stderr, flush=True)
    print(f"[trace] DONE: wrote {counter['ok']} traces to {out_path} ({counter['fail']} failed)")


if __name__ == "__main__":
    main()
