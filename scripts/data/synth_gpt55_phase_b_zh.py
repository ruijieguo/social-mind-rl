"""GPT-5.5 ToM synthesis - Phase A.2: 800 records of Chinese ToM data.

ZH gap to GPT-5.5 is 3-5pp larger than EN on Belief / Knowledge / Desire.
Our training data is ~35% ZH. Phase A.2 ships 800 Chinese-only records
focused on the 3 high-gap tasks:
  Belief        300  (1st-order belief, ambiguous story comprehension)
  Knowledge     250  (who-knows-what, scalar implicature ZH idioms)
  Desire        250  (preference resolution, contradictory wants)

All outputs must be in natural Chinese with culturally appropriate scenarios.
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

from scripts.data.schema import TomRecord


SYSTEM = """你是中文 Theory-of-Mind 多选题写作专家。

只输出一个有效的 JSON 对象，键为：story, question, options（对象，键 A/B/C/D）, answer（A/B/C/D 之一）。

硬性要求：
- 绝对不要复现、改写或翻译 ToMBench（陈等，ACL 2024）或任何已发表心智理论基准中的题目。
- 故事必须充分约束唯一正确答案，4 个选项明显区分，仅有一个根据故事无歧义正确。
- 题目要真正考察心智理论（信念、欲望、意图、知识、社会规范），不是事实记忆。
- 不要重复或近似改写选项。
- 角色名要原创（不能用王/李/张三/小明/小红，自创一组）。
- 场景要原创（不要"在教室里"、"在厨房里"，可以用画室、社区中心、运动馆、园艺工坊等）。
- 故事和选项必须为简体中文。
"""


PROMPTS = {
    "belief_zh": """写一道关于角色信念（Belief）的中文心智理论选择题。

设定模板（任选其一）：
1. 一阶错误信念：A 看见物品 X 在容器甲中；A 离开；B 把 X 移到容器乙；A 回来。问：A 会去哪里找？
2. 模糊场景信念推断：角色 A 看到某种暗示性场景（一封信、一张未发出的消息、一个奇怪的眼神），需要推断 A 此刻最可能相信什么。
3. 对话信念：角色 A 对话中暗示自己已经知道某事，问 A 现在心里相信的事实是什么。

关键技能：模型必须区分"现实"与"角色的认知状态"。

要求：
- 5-9 句话。中文要自然流畅，不要直接翻译腔。
- 4 个选项：1 正确（A 的真实信念）；1 错误（A 的真实状况但不是 A 的信念）；2 个似是而非的干扰项。

只输出 JSON。""",

    "knowledge_zh": """写一道关于角色知识（Knowledge）的中文心智理论选择题。

设定模板（任选其一）：
1. "谁知道什么"：故事描述若干角色，每人接触到不同的信息片段。问：在某时刻，谁知道事件 E？谁不知道？
2. 标量含蓄（scalar implicature）：说话人说"几乎没有 X""大部分是 Y""只有少数 Z"。需要根据这些近似数量词推断剩余数量。例如：60 个学生，"大部分喜欢篮球，少数喜欢乒乓球，几乎没人喜欢羽毛球"，已知喜欢乒乓球的有 12 人。问喜欢篮球的最可能有多少人？答案要把"几乎没人"理解为 1-3 人，而不是 0。
3. 知识—注意力联结：角色 A 只在某时刻在场，错过了关键信息。问 A 现在知道什么，不知道什么？

关键技能：把"听到的话/看到的场面"和"由此可推的知识状态"区分开。

要求：
- 5-9 句中文。
- 4 个选项：1 正确；1 字面错误（无视量词的近似含义）；2 干扰项。

只输出 JSON。""",

    "desire_zh": """写一道关于角色欲望（Desire）的中文心智理论选择题。

设定模板（任选其一）：
1. 矛盾欲望：角色 A 同时想要 X 和 Y，但 X 与 Y 不能兼得，故事提供线索表明哪个优先。问：A 最可能选择什么？
2. 隐含偏好：角色 A 嘴上说想要 X，实际行为透露出 A 更想要 Y。问：A 真正的欲望是什么？
3. 说服策略：要劝服 A 做某事，根据 A 的核心动机（如"陪家人"、"被认可"、"省时间"），哪种策略最有效？

关键技能：从行为/说话方式推断深层动机，而非字面诉求。

要求：
- 5-9 句中文。
- 4 个选项：1 正确（真实欲望/最有效策略）；1 字面陷阱（A 嘴上说的）；2 似是而非的干扰项。

只输出 JSON。""",
}


TASK_MAP = {
    "belief_zh": "Belief",
    "knowledge_zh": "Knowledge",
    "desire_zh": "Desire",
}


_FENCE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)
_OBJ = re.compile(r"\{[\s\S]*\}")


def parse_response(text):
    if not text: return None
    m = _FENCE.search(text)
    if m: text = m.group(1)
    m = _OBJ.search(text)
    if not m: return None
    try: return json.loads(m.group(0))
    except Exception: return None


def call_one(client, prompt_kind, model="gpt-5.5", max_retries=3):
    user = PROMPTS[prompt_kind]
    last = ""
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role":"system","content":SYSTEM},{"role":"user","content":user}],
                temperature=0.9, max_tokens=1500, timeout=120,
            )
            content = resp.choices[0].message.content or ""
            obj = parse_response(content)
            if not obj:
                last = f"parse fail: {content[:100]!r}"; continue
            try:
                story, question, opts, answer = obj["story"], obj["question"], obj["options"], obj["answer"]
            except (KeyError, TypeError):
                last = f"missing keys: {list(obj.keys())}"; continue
            if not isinstance(opts, dict) or not all(k in opts for k in "ABCD"):
                last = "bad options"; continue
            answer = str(answer).strip().upper()
            if answer not in {"A","B","C","D"}:
                last = f"bad answer {answer!r}"; continue
            return TomRecord(
                question_id="synth_gpt55_pb_pending",
                source="synth_gpt55_phase_b_zh",
                language="zh",
                task=TASK_MAP[prompt_kind],
                story=str(story), question=str(question),
                opt_a=str(opts["A"]), opt_b=str(opts["B"]),
                opt_c=str(opts["C"]), opt_d=str(opts["D"]),
                gold=answer,
            ), prompt_kind
        except Exception as e:
            last = f"{type(e).__name__}: {str(e)[:120]}"
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt); continue
    print(f"[phase-b-zh] {prompt_kind} failed: {last}", file=sys.stderr)
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-belief", type=int, default=300)
    p.add_argument("--n-knowledge", type=int, default=250)
    p.add_argument("--n-desire", type=int, default=250)
    p.add_argument("--out", default="data/tom/raw/synth_gpt55_phase_b_zh.jsonl")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--model", default="gpt-5.5")
    args = p.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY not set")
    client = OpenAI(api_key=api_key, base_url=base_url)

    plan = []
    for kind, n in [("belief_zh", args.n_belief), ("knowledge_zh", args.n_knowledge), ("desire_zh", args.n_desire)]:
        plan.extend([kind] * n)
    random.shuffle(plan)
    print(f"[phase-b-zh] plan: {len(plan)} calls (concurrency={args.concurrency})", file=sys.stderr)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_lock = threading.Lock()
    counter = {"ok": 0, "fail": 0}
    by_kind = {k: 0 for k in PROMPTS}
    started = time.time()
    with out_path.open("w", encoding="utf-8") as fp, ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = [ex.submit(call_one, client, kind, args.model) for kind in plan]
        total = len(futures)
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            with write_lock:
                if r:
                    rec, kind = r
                    rec.question_id = f"synth_gpt55_pb_{counter['ok']}"
                    fp.write(json.dumps(rec.to_jsonl_dict(), ensure_ascii=False) + "\n")
                    fp.flush()
                    counter["ok"] += 1
                    by_kind[kind] += 1
                else:
                    counter["fail"] += 1
                if i % 25 == 0 or i == total:
                    elapsed = time.time() - started
                    rate = i / elapsed if elapsed > 0 else 0
                    print(f"[phase-b-zh] {i}/{total} ok={counter['ok']} fail={counter['fail']} rate={rate:.2f}/s elapsed={elapsed:.0f}s by_kind={by_kind}",
                          file=sys.stderr, flush=True)
    print(f"[phase-b-zh] DONE: {counter['ok']} records in {out_path}  by_kind={by_kind}")


if __name__ == "__main__":
    main()
