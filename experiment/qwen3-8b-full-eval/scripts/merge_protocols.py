"""Merge per-protocol result files into a unified per-(model, benchmark) JSON.

For each (model, benchmark), combine:
  output/{benchmark}/{model}.json         — base direct, cot
  output/{benchmark}/{model}_dt.json      — direct_think
  (and any preserved del_tom records, if v10.json had them)

Then re-run any missing protocols by replaying from cache.
"""
from __future__ import annotations
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path("/work/output")
MODELS = [("base", "qwen3-8b-base"), ("v10", "qwen3-8b-v10"), ("dashscope", "qwen3-8b-api")]
BENCHES = ["tombench", "hitom"]


def main():
    for bench in BENCHES:
        for fname, mid in MODELS:
            main_path = ROOT / bench / f"{fname}.json"
            dt_path = ROOT / bench / f"{fname}_dt.json"

            recs_main = json.load(open(main_path)) if main_path.exists() else []
            recs_dt = json.load(open(dt_path)) if dt_path.exists() else []

            # De-dup by (qid, protocol) — keep latest
            merged: dict = {}
            for r in recs_main + recs_dt:
                merged[(r["question_id"], r["protocol"])] = r
            out = list(merged.values())

            from collections import Counter
            counts = Counter(r["protocol"] for r in out)
            print(f"{bench}/{fname}: {len(out)} records, protocols={dict(counts)}")

            # Backup main + write merged
            if main_path.exists():
                main_path.replace(main_path.with_suffix(".json.bak_premerge"))
            json.dump(out, open(main_path, "w"), ensure_ascii=False)


if __name__ == "__main__":
    main()
