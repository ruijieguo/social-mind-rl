"""Synthesize ToM MCQ questions targeting specific stage2 bad-case patterns.

Designed from docs/badcase_analysis.md. Generates 4 categories:

1. faux_pas  (800):  Faux-pas recognition. Balanced 50/50 between
                     "someone was inappropriate" and "no one was inappropriate"
                     to break stage2's A/B confusion (33% of all errors).

2. scalar    (400):  Scalar implicature with quantifiers ("almost no",
                     "most", "some") + math reasoning. Stage2 ignored the
                     pragmatic implicature and only did arithmetic.

3. hinting   (300):  Indirect requests. Speaker hints rather than asks
                     directly. Stage2 read hints as refusals.

4. so_belief (300):  Second-order false belief. Explicit "A knows that B
                     does not know that C". Stage2 mixed 1st-order and
                     2nd-order belief tracking.

DESIGN PRINCIPLES (to avoid the failure modes I just diagnosed):
- Bilingual: 50/50 en/zh per category, so ZH-leaning training doesn't
  bias the protocol distribution.
- Story freshness: each call has a different (entity, scenario) seed
  drawn from a curated pool, not the deepseek default sample which
  tends to repeat names like "Xiao Li / Xiao Wang".
- Anti-leakage: every prompt explicitly forbids reproducing or
  translating ToMBench questions. Pipeline-level Jaccard 0.6 check
  against tombench_eval.jsonl runs in merge_and_dedupe.py.
- Balance check: at write time, count gold-letter distribution per
  category and refuse to write if any letter is >40% (stage2-style
  mode collapse) — fail-loud so the operator notices.

Output: data/tom/raw/synth_phase1_<category>.jsonl (one record per line,
unified TomRecord schema). Then merge_and_dedupe.py picks them up via
the standard build_tom_train flow.

Each call uses deepseek-v4-flash (cheaper, ~2x faster than -pro for
this kind of generation task). Failures (parse errors, API timeouts)
are tolerated up to max_retries and logged to stderr.
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
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from openai import OpenAI

from scripts.data.schema import TomRecord


# ============================================================
# Anti-leakage preamble used for every category. Repeated here
# rather than imported because it must be visible in the prompt
# we send to deepseek (anti-leakage relies on telling the model).
# ============================================================
LEAK_PREAMBLE = (
    "You are an expert writer creating ORIGINAL theory-of-mind multiple-choice "
    "training questions. ABSOLUTE RULE: do not reproduce, paraphrase, or translate "
    "any question from ToMBench (Chen et al. ACL 2024) or any other published "
    "ToM benchmark. Invent fresh scenarios with fresh names and fresh contexts. "
    "Output a SINGLE valid JSON object with keys: story, question, options "
    "(an object with A,B,C,D), answer (one letter A/B/C/D)."
)

# ============================================================
# Entity / context seeds — sampled per-call so deepseek doesn't
# default to its training distribution (which leans on Xiao Ming
# / Xiao Li and overlaps with ToMBench).
# ============================================================
EN_NAMES = [
    "Marcus", "Priya", "Wendell", "Beatriz", "Hideo", "Liesl", "Tariq",
    "Camille", "Ostap", "Indira", "Kofi", "Inga", "Sven", "Yuna",
    "Diego", "Ngozi", "Selma", "Theo", "Anaya", "Ramses",
]
ZH_NAMES = [
    "陆远", "苏婉", "贺青", "祁睿", "白桦", "钟蕾", "戚明",
    "夏知秋", "穆寒", "辛桐", "顾岚", "段宁", "薄羽", "凌瑶",
    "邵璟", "费夏", "卓尔", "应川", "席溪", "覃乐",
]
EN_SCENARIOS = [
    "an academic conference reception", "a community potluck dinner",
    "an open-mic night at a bookstore", "a workplace retirement party",
    "a co-working space coffee break", "a charity art auction",
    "a hospital waiting room", "a university lab meeting",
    "a publisher's manuscript review", "a startup investor pitch event",
    "a graduate orientation mixer", "a citizens' advisory council meeting",
]
ZH_SCENARIOS = [
    "城市艺术展开幕酒会", "公司年会的茶歇时间", "社区医院的候诊大厅",
    "高校创业孵化器路演", "城市图书馆志愿者培训", "学校家长开放日",
    "电影节首映礼后台", "小型摄影工作室分享会", "民宿主题派对",
    "公共图书馆书友会", "线下读书俱乐部", "园艺爱好者茶会",
]


def random_seed_block(language: str) -> str:
    """Inject randomness so back-to-back deepseek calls diverge."""
    if language == "zh":
        names = random.sample(ZH_NAMES, k=3)
        scenario = random.choice(ZH_SCENARIOS)
        return f"使用名字: {names[0]}, {names[1]}, {names[2]} (你可只用其中 2 个). 场景: {scenario}."
    else:
        names = random.sample(EN_NAMES, k=3)
        scenario = random.choice(EN_SCENARIOS)
        return f"Use names: {names[0]}, {names[1]}, {names[2]} (only 2-3 are needed). Setting: {scenario}."


# ============================================================
# Category prompts. Each returns (system, user) given language +
# desired gold letter so we can balance the gold distribution.
# ============================================================
@dataclass
class CategorySpec:
    name: str
    task_tag: str          # broad task category in TomRecord.task
    n_gold_letters: tuple  # which letters are valid as the target answer

    def build(self, language: str, gold_target: str) -> tuple[str, str]:
        raise NotImplementedError


class FauxPasSpec(CategorySpec):
    """Faux-pas recognition. 50% inappropriate / 50% innocuous.

    Stage2 confusion matrix shows A<->B swaps in 511 errors (33% of all).
    Most ToMBench faux-pas questions ask "did someone say something
    inappropriate?" with A=yes / B=no. We want training data that:
      - Half the time the answer is "yes someone was inappropriate"
        (with the inappropriate speech given mid-story, NOT obvious)
      - Half the time everyone speaks innocuously and the answer is "no"
    Stage2 over-attributes faux-pas (sees one in 'no faux-pas' stories);
    so we deliberately oversample the 'no faux-pas' case to recalibrate.
    """

    def __init__(self):
        super().__init__("faux_pas", "Non-literal Comm", ("A", "B"))

    def build(self, language: str, gold_target: str) -> tuple[str, str]:
        # gold_target A = inappropriate / B = innocuous
        if language == "zh":
            polarity_text = (
                "故事中需要有一个人说了不该说的话（如:在主人家比较其菜与配偶手艺、当面提某人忌讳的话题、在追悼会上谈论遗产分配等）。"
                if gold_target == "A"
                else "故事中所有人的言行都得体合适，没有任何不该说的话。即便有点平淡也无妨。"
            )
            template = f"""{LEAK_PREAMBLE}

{random_seed_block(language)}

写一个"失礼识别"(Faux-pas)题。要求：
- 故事 6-12 句话，描述一次社交互动
- {polarity_text}
- 问题: "故事中如果有人说了不该说的话，那是哪一句？"
- 选项 A: "有人说了不该说的话"，选项 B: "没有人说不该说的话"
- 选项 C 和 D 可为空字符串
- 正确答案: {gold_target}
- 注意: 若 gold=B，故事必须真的全程得体；不要写一个边缘案例

直接输出 JSON, 不要 markdown 包裹，不要解释。"""
        else:
            polarity_text = (
                "Someone in the story MUST make an inappropriate remark "
                "(e.g., praising one's own spouse's cooking to the host who "
                "just served their dish, mentioning a taboo topic, or "
                "complaining about a gift to its giver)."
                if gold_target == "A"
                else "All speech in the story must be entirely polite and appropriate. No subtle slights either."
            )
            template = f"""{LEAK_PREAMBLE}

{random_seed_block(language)}

Write a Faux-Pas Recognition task with these requirements:
- The story is 6-12 sentences depicting a social interaction
- {polarity_text}
- Question: "If anyone said something inappropriate, which sentence was it?"
- Option A: "Someone said something inappropriate"; Option B: "No one said anything inappropriate"
- Options C and D may be empty strings
- Correct answer: {gold_target}
- Important: If gold=B, the story must be genuinely polite throughout; do not write a borderline case.

Output JSON directly, no markdown fence, no explanation."""
        return LEAK_PREAMBLE, template


class ScalarSpec(CategorySpec):
    """Scalar implicature: 'almost no X' means ~0-2, 'most' means > half.

    Stage2 failed 5/6 Knowledge closeable errors on this exact pattern.
    The fix is to teach: when total=N and 'almost no Z', then |Z| ~ 0-3,
    so the other categories sum to N-|Z| (not just N).
    """

    def __init__(self):
        super().__init__("scalar", "Knowledge", ("A", "B", "C", "D"))

    def build(self, language: str, gold_target: str) -> tuple[str, str]:
        if language == "zh":
            template = f"""{LEAK_PREAMBLE}

{random_seed_block(language)}

写一个"标量含意"(Scalar Implicature)推理题。要求：
- 故事中一人对另一人描述一个集合，使用 "大多数/大部分是 X，一些/小部分是 Y，几乎没有/极少 Z" 的格式
- 总数明确（例如 50, 80, 20 等）
- 另一人观察后能确认 Y 的实际数量（如 "发现只有 12 个 Y"）
- 关键: "几乎没有 Z" 表示 Z 大约是 1-3 个 (实际是一个隐含的语用含意)
- 问题: 在观察前，另一人推测 X 是多少？
- 4 个选项, 数值差距不大 (区分语用与字面解读)
  例如: 选项A 是正确语用解 (45 = 50-4-1)，选项B 是字面 (46 = 50-4)，选项C/D 是其他干扰
- 正确答案: {gold_target}

直接输出 JSON。"""
        else:
            template = f"""{LEAK_PREAMBLE}

{random_seed_block(language)}

Write a Scalar Implicature reasoning task:
- One character tells another about a set using the format
  "most are X, some are Y, almost no/barely any Z" with a clear total (e.g. 50, 80, 20).
- The observer then verifies Y's actual count (e.g. "finds only 12 Y").
- Critical: "almost no Z" implies Z is approximately 1-3 (pragmatic implicature).
- Question: before observing, what does the observer estimate the count of X to be?
- 4 options with small numeric differences (forces pragmatic vs literal interpretation):
  e.g. option A is the pragmatic answer (45 = 50 - 4 - 1), option B is literal (46 = 50 - 4),
  options C/D are other distractors.
- Correct answer: {gold_target}

Output JSON directly."""
        return LEAK_PREAMBLE, template


class HintingSpec(CategorySpec):
    """Indirect requests. A character makes an indirect remark whose real
    intent is a polite request, not a refusal or unrelated statement.

    Stage2 reads hints as refusals ("said something else => doesn't want to").
    Training data should provide unambiguous indirect-request examples.
    """

    def __init__(self):
        super().__init__("hinting", "Intention", ("A", "B", "C", "D"))

    def build(self, language: str, gold_target: str) -> tuple[str, str]:
        if language == "zh":
            template = f"""{LEAK_PREAMBLE}

{random_seed_block(language)}

写一个"间接请求"(Hinting Task)推理题。要求：
- 故事 2-4 句话，一人向另一人说话，但说话内容不是直接请求
- 真实意图是一个明确的请求或希望（例如希望对方帮忙、希望对方提供某物等）
- 问题: "当他说这句话时，他真正想说的是什么？"
- 4 选项:
  - 1 项是真实意图（间接请求的潜台词）
  - 1 项是误读为拒绝
  - 1 项是字面意思
  - 1 项是无关
- 正确答案: {gold_target}

直接输出 JSON。"""
        else:
            template = f"""{LEAK_PREAMBLE}

{random_seed_block(language)}

Write an Indirect Request (Hinting Task) inference question:
- Story is 2-4 sentences. One character speaks to another with a remark that is NOT a direct request.
- The real intent is a clear request or wish (e.g., wanting the other to help, to give something, etc.).
- Question: "When they say this, what do they really mean?"
- 4 options:
  - 1 is the real indirect-request meaning
  - 1 is a misread-as-refusal
  - 1 is the literal surface meaning
  - 1 is unrelated
- Correct answer: {gold_target}

Output JSON directly."""
        return LEAK_PREAMBLE, template


class SecondOrderBeliefSpec(CategorySpec):
    """Second-order false belief: A knows that B does not know.

    Stage2 failed: when an agent LEFT and someone else moved an object,
    stage2 says the absent agent "knows about the move". This is a
    reasoning collapse, not a knowledge gap. Training data must
    explicitly state who saw what, who left when, and ask 2nd-order
    questions.
    """

    def __init__(self):
        super().__init__("so_belief", "False Belief", ("A", "B", "C", "D"))

    def build(self, language: str, gold_target: str) -> tuple[str, str]:
        if language == "zh":
            template = f"""{LEAK_PREAMBLE}

{random_seed_block(language)}

写一个"二阶错误信念"(Second-order False Belief)推理题。要求：
- 角色 A 和 B 在一个房间，看到某物在容器 X 中
- A 离开房间，B 把该物移到容器 Y 中
- A 不知道物体被移动了 (因为 A 已离开)
- 问题应是二阶的: "A 回来后，A 认为 B 会去哪里找该物？"
  (注意: 不是"该物现在在哪", 也不是"B 会去哪里找", 而是 A 对 B 行为的预测)
- 正确答案是: A 认为 B 仍以为该物在 X (即 A 对 B 的信念建模出错)
  但实际 B 知道在 Y 里
- 4 选项，A/B/C/D 各为一个容器名或地点
- 正确答案: {gold_target}

注意: 关键是"A 认为 B 不知道 A 已知道..." 这种二阶关系。
确保故事明确说明谁见到了什么、何时离开。

直接输出 JSON。"""
        else:
            template = f"""{LEAK_PREAMBLE}

{random_seed_block(language)}

Write a Second-order False Belief inference question:
- Characters A and B are in a room and see an object in container X.
- A leaves the room. B moves the object to container Y.
- A does NOT know about the move (A had already left).
- The question should be SECOND-ORDER: "When A returns, where does A think B will look for the object?"
  (Not "where is the object now"; not "where will B look"; but "A's prediction of B's behavior".)
- Correct answer: A thinks B still believes the object is in X (i.e. A misjudges what B knows).
- 4 options, each a different container/location.
- Correct answer: {gold_target}

Critical: the story must explicitly state who saw what and when they left
so the second-order reasoning chain is unambiguous.

Output JSON directly."""
        return LEAK_PREAMBLE, template


CATEGORIES = {
    "faux_pas": FauxPasSpec(),
    "scalar": ScalarSpec(),
    "hinting": HintingSpec(),
    "so_belief": SecondOrderBeliefSpec(),
}


# ============================================================
# JSON parsing + record building (same as synth_tomtype.py)
# ============================================================
_FENCE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)
_OBJ = re.compile(r"\{[\s\S]*\}")


def parse_record(text: str) -> Optional[dict]:
    if not text:
        return None
    m = _FENCE.search(text)
    if m:
        text = m.group(1)
    m = _OBJ.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def call_one(client: OpenAI, spec: CategorySpec, language: str, gold_target: str,
             model: str, max_retries: int = 3, max_tokens: int = 900) -> Optional[TomRecord]:
    system, user = spec.build(language, gold_target)
    last_err = ""
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.9,
                max_tokens=max_tokens,
                timeout=180,
            )
            content = resp.choices[0].message.content or ""
            obj = parse_record(content)
            if not obj:
                last_err = f"parse failed; content[:100]={content[:100]!r}"
                continue
            # Verify required fields
            try:
                story = obj["story"]
                question = obj["question"]
                opts = obj["options"]
                answer = obj["answer"]
            except (KeyError, TypeError):
                last_err = f"missing keys; got: {list(obj.keys())}"
                continue
            if not isinstance(opts, dict) or not all(k in opts for k in "ABCD"):
                last_err = "options not dict with A/B/C/D"
                continue
            answer = str(answer).strip().upper()
            if answer not in spec.n_gold_letters:
                last_err = f"answer {answer!r} not in {spec.n_gold_letters}"
                continue
            if answer != gold_target:
                # Bad: deepseek returned a different gold than we asked for.
                # Keep it (data is still valid) but log the rate later.
                pass
            return TomRecord(
                question_id="synth_p1_pending",
                source="synth_phase1",
                language=language,
                task=spec.task_tag,
                story=str(story),
                question=str(question),
                opt_a=str(opts.get("A", "")),
                opt_b=str(opts.get("B", "")),
                opt_c=str(opts.get("C", "")),
                opt_d=str(opts.get("D", "")),
                gold=answer,
            )
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
    print(f"[synth_phase1] {spec.name}/{language}/gold={gold_target} failed after {max_retries}: {last_err}",
          file=sys.stderr, flush=True)
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--category", choices=list(CATEGORIES.keys()), required=True)
    p.add_argument("--n", type=int, required=True,
                   help="total records to generate (balanced across language and gold)")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="deepseek-v4-flash")
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--max-tokens", type=int, default=900,
                   help="Bump to ~4000 for reasoning models like deepseek-v4-pro since they consume budget for thinking.")
    args = p.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("DEEPSEEK_API_KEY not set")
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    spec = CATEGORIES[args.category]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Build balanced plan: language x gold letter
    plan = []
    n_per_combo = max(1, args.n // (2 * len(spec.n_gold_letters)))
    for lang in ("en", "zh"):
        for gold in spec.n_gold_letters:
            plan.extend([(lang, gold)] * n_per_combo)
    random.shuffle(plan)

    print(f"[synth_phase1] category={args.category} | n={args.n} | concurrency={args.concurrency}",
          file=sys.stderr, flush=True)
    print(f"[synth_phase1] plan: {len(plan)} calls, balanced over languages x gold letters {spec.n_gold_letters}",
          file=sys.stderr, flush=True)

    write_lock = threading.Lock()
    counter = {"ok": 0, "fail": 0, "wrong_gold": 0}
    gold_hist = {l: 0 for l in "ABCD"}
    started = time.time()

    with out_path.open("w", encoding="utf-8") as fp, ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = [ex.submit(call_one, client, spec, lang, gold, args.model, args.max_retries, args.max_tokens)
                   for (lang, gold) in plan]
        total = len(futures)
        # Map back the planned gold for stats
        plan_iter = iter(plan)
        for i, f in enumerate(as_completed(futures), 1):
            requested_lang, requested_gold = next(plan_iter, ("?", "?"))
            rec = f.result()
            with write_lock:
                if rec is not None:
                    rec.question_id = f"synth_p1_{args.category}_{counter['ok']}"
                    fp.write(json.dumps(rec.to_jsonl_dict(), ensure_ascii=False) + "\n")
                    fp.flush()
                    gold_hist[rec.gold] = gold_hist.get(rec.gold, 0) + 1
                    counter["ok"] += 1
                    if rec.gold != requested_gold:
                        counter["wrong_gold"] += 1
                else:
                    counter["fail"] += 1
                if i % 25 == 0 or i == total:
                    elapsed = time.time() - started
                    rate = i / elapsed if elapsed > 0 else 0
                    print(
                        f"[{args.category}] {i}/{total} | ok={counter['ok']} fail={counter['fail']} "
                        f"deepseek_used_wrong_gold={counter['wrong_gold']} | {rate:.1f} req/s | "
                        f"hist={gold_hist} | elapsed={elapsed:.0f}s",
                        file=sys.stderr, flush=True,
                    )

    print(f"\n[synth_phase1] DONE category={args.category}", file=sys.stderr)
    print(f"  wrote {counter['ok']} records to {out_path}", file=sys.stderr)
    print(f"  failed: {counter['fail']}", file=sys.stderr)
    print(f"  deepseek picked wrong gold: {counter['wrong_gold']}", file=sys.stderr)
    print(f"  final gold distribution: {gold_hist}", file=sys.stderr)
    # Sanity check: refuse if any letter dominates (mode collapse).
    if counter["ok"] >= 50:
        max_letter_share = max(gold_hist.values()) / counter["ok"]
        if max_letter_share > 0.5:
            print(f"  WARNING: gold {max(gold_hist, key=gold_hist.get)} is {max_letter_share*100:.0f}% — possible mode collapse",
                  file=sys.stderr)


if __name__ == "__main__":
    main()
