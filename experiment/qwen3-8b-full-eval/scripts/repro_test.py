"""Reproducibility test: re-run v10 direct with historical settings to validate
the difference between this run (max_tokens=64, enable_thinking=false) and the
production_frozen run (max_tokens=2048, enable_thinking=default-true).

Run inside the vllm-openai docker container with /work mounted.
"""
import sys, json, re
sys.path.insert(0, "/work/scripts")
from openai import OpenAI
from prompts import build_messages_tombench

client = OpenAI(api_key="EMPTY", base_url="http://127.0.0.1:8005/v1")
records = [json.loads(l) for l in open("/data/tom/tombench_eval.jsonl")]
recs = {r["question_id"]: r for r in records}

# Questions where NEW (max_tokens=64, thinking=false) got wrong but OLD got right.
test_qids = [
    "Ambiguous_Story_Task_101_en",  # gold=B, OLD=B, NEW=C
    "Ambiguous_Story_Task_126_en",  # gold=C, OLD=C, NEW=A
    "Ambiguous_Story_Task_106_zh",  # gold=C, OLD=C, NEW=B
    "Ambiguous_Story_Task_15_en",   # gold=C, OLD=?, NEW=A
]

print("=== HISTORICAL SETTINGS: max_tokens=2048, NO chat_template_kwargs ===")
for qid in test_qids:
    r = recs[qid]
    msgs = build_messages_tombench(r, protocol="direct")
    resp = client.chat.completions.create(
        model="qwen3-8b-v10", messages=msgs,
        temperature=0.0, top_p=1.0, max_tokens=2048,
    )
    content = resp.choices[0].message.content or ""
    matches = re.findall(r"\\boxed\{([A-D])\}", content)
    pred = matches[-1] if matches else None
    print(f"{qid}: gold={r['gold']} pred={pred}  len={len(content)}")
    print(f"  first 80: {content[:80]!r}")
    print(f"  last 80:  {content[-80:]!r}")
    print()

print("=== NEW SETTINGS: max_tokens=64, enable_thinking=false ===")
for qid in test_qids:
    r = recs[qid]
    msgs = build_messages_tombench(r, protocol="direct")
    resp = client.chat.completions.create(
        model="qwen3-8b-v10", messages=msgs,
        temperature=0.0, top_p=1.0, max_tokens=64,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    content = resp.choices[0].message.content or ""
    matches = re.findall(r"\\boxed\{([A-D])\}", content)
    pred = matches[-1] if matches else None
    print(f"{qid}: gold={r['gold']} pred={pred}  len={len(content)}")
    print(f"  content: {content!r}")
    print()
