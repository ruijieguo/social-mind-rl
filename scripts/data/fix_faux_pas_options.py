"""Fix Phase-1 faux_pas records to have valid C/D options.

The original synth_phase1_faux_pas.jsonl has empty strings for opt_c and
opt_d, mimicking the ToMBench faux-pas convention but breaking consistency
with the other 7000 training records which all have 4 valid options.

ROLL's training pipeline (and the TomMcqRewardWorker) doesn't crash on
empty options, but it likely degrades:
  - The reward / format reward computes against all 4 letters, so model
    sees "C" or "D" as valid output but can't get meaningful reward.
  - Mode collapse: stage4 rollout score got stuck at 0.17-0.29 vs
    stage3's 0.50 peak, despite same KL configuration.

This script either drops the empty-option records (option=keep_2) or
fills C/D with reasonable distractors (option=fill).

For the targeted training purpose (teach faux-pas attribution), the
2-option style is fine in isolation but diluting effect when 13% of
training data has different structure than the rest.

Decision: drop them and accept that we lose ~800 faux-pas records.
The other 1000 phase1 records (scalar/hinting/so_belief) all have
4 valid options.
"""
import json
import sys
from pathlib import Path


def main():
    src = Path("data/tom/raw/synth_phase1_faux_pas.jsonl")
    dst = Path("data/tom/raw/synth_phase1_faux_pas_fixed.jsonl")
    kept = []
    dropped_empty = 0
    with src.open() as f:
        for line in f:
            r = json.loads(line)
            opts = [r.get("opt_a"), r.get("opt_b"), r.get("opt_c"), r.get("opt_d")]
            if not all(opts):
                # Has empty option(s) — convert to a 4-option style by
                # filling C/D with concrete distractors that sound plausible
                # for "did someone say something inappropriate?":
                #   C = "Maybe" / "Cannot determine"
                #   D = the most-recent quoted line in the story
                # If gold is A (someone WAS inappropriate), gold stays A.
                # If gold is B (no one was), gold stays B.
                if r["language"] == "zh":
                    r["opt_c"] = "无法判断"
                    r["opt_d"] = "故事中所有人的话都需要重新审视"
                else:
                    r["opt_c"] = "Cannot be determined"
                    r["opt_d"] = "All sentences in the story should be reconsidered"
                dropped_empty += 1
            kept.append(r)
    with dst.open("w") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"src: {src} ({sum(1 for _ in src.open())} records)")
    print(f"dst: {dst} ({len(kept)} records, filled C/D for {dropped_empty})")


if __name__ == "__main__":
    main()
