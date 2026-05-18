"""Build clean ToMBench eval subset from GPT-5.5 audit results.

Reads:
  output/analysis/clean_eval_qids.json   (keep list from C.1 audit)
  data/tom/tombench_eval.jsonl           (original 5718)

Writes:
  data/tom/tombench_eval_clean.jsonl     (subset of 4551 with high-confidence correct labels)
"""
import json
from pathlib import Path


def main():
    keep_set = set(json.loads(Path("output/analysis/clean_eval_qids.json").read_text())["keep"])
    src = Path("data/tom/tombench_eval.jsonl")
    out = Path("data/tom/tombench_eval_clean.jsonl")
    n_kept, n_dropped = 0, 0
    with src.open() as fr, out.open("w") as fw:
        for line in fr:
            r = json.loads(line)
            if r["question_id"] in keep_set:
                fw.write(line)
                n_kept += 1
            else:
                n_dropped += 1
    print(f"kept {n_kept}, dropped {n_dropped} -> {out}")


if __name__ == "__main__":
    main()
