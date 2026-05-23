"""Build SocialIQA dev split (1954 records) into unified eval schema.

Output schema (jsonl):
  question_id, source=socialiqa, language=en, task=social_iqa,
  story, question, options[3], gold (A/B/C)

Note: HF datasets >=2.20 removed loading-script support, so we fetch the
official files directly. The validation split has 1954 records.
"""
from __future__ import annotations
import json
import urllib.request
import zipfile
import io
from pathlib import Path


URL = "https://storage.googleapis.com/ai2-mosaic/public/socialiqa/socialiqa-train-dev.zip"


def main():
    out = Path("data/eval/socialiqa_eval.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    cache_dir = Path("data/eval/raw_socialiqa")
    cache_dir.mkdir(parents=True, exist_ok=True)
    dev_jsonl = cache_dir / "dev.jsonl"
    dev_labels = cache_dir / "dev-labels.lst"

    if not dev_jsonl.exists() or not dev_labels.exists():
        zip_path = cache_dir / "socialiqa-train-dev.zip"
        if not zip_path.exists():
            print(f"Downloading {URL} ...")
            urllib.request.urlretrieve(URL, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(cache_dir)
        # Find dev files
        for p in cache_dir.rglob("dev.jsonl"):
            dev_jsonl = p
            break
        for p in cache_dir.rglob("dev-labels.lst"):
            dev_labels = p
            break

    print(f"Dev jsonl: {dev_jsonl}")
    print(f"Dev labels: {dev_labels}")

    questions = [json.loads(l) for l in open(dev_jsonl)]
    labels = [l.strip() for l in open(dev_labels) if l.strip()]
    assert len(questions) == len(labels), f"{len(questions)} != {len(labels)}"
    print(f"Loaded {len(questions)} dev questions")

    n_written = 0
    with open(out, "w") as f:
        for i, (row, label) in enumerate(zip(questions, labels)):
            if label not in {"1", "2", "3"}:
                continue
            gold = chr(ord("A") + int(label) - 1)
            rec = {
                "question_id": f"socialiqa_dev_{i:04d}",
                "source": "socialiqa",
                "language": "en",
                "task": "social_iqa",
                "story": row["context"],
                "question": row["question"],
                "options": [row["answerA"], row["answerB"], row["answerC"]],
                "gold": gold,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"Wrote {n_written} records → {out}")


if __name__ == "__main__":
    main()
