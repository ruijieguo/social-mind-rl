"""Build Hi-ToM direct-style training records by deriving from hitom_train_1200.jsonl.

Each Hi-ToM record originally was COT-style (thinks, then \\boxed{X}). For v3.3
we add a paired direct-style version that asks for an answer with no reasoning.
This trains the model to compress belief tracking into a single forward pass —
analogous to how DeepSeek's reasoning_content lets it "cheat" on direct prompts.

Output schema matches data/tom/tom_train_stage14_weighted.jsonl.

Strategy:
- Skip order_0 (already too easy, 100% from existing data)
- For order_1..order_4: derive ONE direct-style sample per existing cot record
- Use a strict no-reasoning system prompt to clearly mark these as direct-style
- Use source="hitom_synth_direct" so reward worker can apply l_max=64 to these.
"""
from __future__ import annotations
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data/tom/raw/hitom_train_1200.jsonl"
OUT = ROOT / "data/tom/raw/hitom_train_direct_1000.jsonl"


DIRECT_SYS_TEMPLATE = (
    "You are a careful reader answering a multiple-choice theory-of-mind question. "
    "Read the story and the question carefully, then output ONLY your final answer "
    "in the format \\boxed{{X}} where X is one of {letters}. "
    "Do not include any explanation, reasoning, or extra text. "
    "You must answer in a single forward pass without thinking out loud."
)


def n_options_from_user_msg(user_content: str) -> int:
    """Count A. B. C. ... lines in the user prompt."""
    matches = re.findall(r"\n([A-Z])\.\s", user_content)
    return len(matches)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    n_in = n_out = 0
    with open(SRC) as f, open(OUT, "w") as fout:
        for line in f:
            r = json.loads(line)
            n_in += 1
            task = r.get("task", "")
            # Skip order_0 (already saturated at 1.00) — focus capacity on hard orders.
            if task == "order_0":
                continue
            if task not in {"order_1", "order_2", "order_3", "order_4"}:
                continue
            # Existing user message has the story + question + options block.
            user_msg = r["messages"][1]["content"]
            n_opts = n_options_from_user_msg(user_msg)
            letters = ", ".join(chr(ord("A") + i) for i in range(n_opts))
            new_record = {
                "messages": [
                    {"role": "system", "content": DIRECT_SYS_TEMPLATE.format(letters=letters)},
                    {"role": "user", "content": user_msg},
                ],
                "ground_truth": r["ground_truth"],
                "tag": r["tag"],
                "source": "hitom_synth_direct",
                "language": r.get("language", "en"),
                "task": task,
                "question_id": f"{r['question_id']}_direct",
            }
            fout.write(json.dumps(new_record, ensure_ascii=False) + "\n")
            n_out += 1
    print(f"Read {n_in} records (cot-style), wrote {n_out} direct-style → {OUT}")


if __name__ == "__main__":
    main()
