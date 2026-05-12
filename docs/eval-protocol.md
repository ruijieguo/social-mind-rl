# Evaluation Protocols

Three protocols. Main score for "approaches/surpasses deepseek-v4-pro" is **protocol 1 (Direct)**.

## Protocol 1 — Direct answer (MAIN SCORE)

**System prompt:**
```
You are a careful reader answering a multiple-choice theory-of-mind question.
Read the story and the question carefully, then output ONLY your final answer
in the format \boxed{X} where X is one of A, B, C, D.
Do not include any explanation, reasoning, or extra text.
```

**User prompt (English):**
```
Story:
{STORY}

Question: {QUESTION}
A. {OPTION-A}
B. {OPTION-B}
C. {OPTION-C}
D. {OPTION-D}
```

**User prompt (Chinese):** same shape, Chinese labels `故事:`, `问题:`.

**Sampling:** `temperature=0.0, top_p=1.0, max_tokens=32`

**Answer extraction:** `\boxed{[A-D]}` first match; fallback to first capital letter A-D in response.

**Why this is the main protocol:** training is done with this exact prompt + extraction; train/test consistency is critical.

## Protocol 2 — CoT (reference score)

**System prompt:** allows step-by-step thinking, requires final `\boxed{X}` on the last line.

**Sampling:** `temperature=0.6, top_p=0.9, max_tokens=1024`

**Answer extraction:** last `\boxed{[A-D]}`; fallback to last capital letter in tail 200 chars.

## Protocol 3 — DEL-ToM (optional)

For belief-class subtasks only (False Belief, Unexpected Outcome, Knowledge). Generates **N=8 CoT samples** at `temperature=0.7`; majority vote, alphabetic tiebreak.

Not used during training. Only for final evaluation enhancement.

## Reporting

For each model:
- Overall accuracy (all 2860 questions × 2 languages = ~5720 records)
- Per-language: EN, ZH
- Per-task (8 ToMBench broad tasks): False Belief, Strange Story, Unexpected Outcome, Persuasion Story, Knowledge, Desire, Emotion, Intention, Non-literal Comm
