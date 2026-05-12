"""Phase 0 environment check.

Verifies:
- DEEPSEEK_API_KEY and DASHSCOPE_API_KEY are set
- Both API endpoints reachable with a minimal echo call
- Local mount points exist (data/, output/, configs/)
"""
import os
import sys
from pathlib import Path


def check_env_vars() -> list[str]:
    issues = []
    for var in ("DEEPSEEK_API_KEY", "DASHSCOPE_API_KEY"):
        if not os.environ.get(var):
            issues.append(f"missing env var: {var}")
    return issues


def check_mount_points() -> list[str]:
    issues = []
    for path in ("data", "output", "configs"):
        p = Path("/workspace") / path
        if not p.exists():
            try:
                p.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                issues.append(f"cannot create {p}: {e}")
    return issues


def check_deepseek() -> list[str]:
    from openai import OpenAI
    issues = []
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        return ["DEEPSEEK_API_KEY empty, skipping deepseek check"]
    try:
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        resp = client.chat.completions.create(
            model="deepseek-v4-pro",
            messages=[{"role": "user", "content": "Say OK"}],
            max_tokens=4,
        )
        content = resp.choices[0].message.content or ""
        print(f"  deepseek-v4-pro reachable; sample reply: {content!r}")
    except Exception as e:
        issues.append(f"deepseek api error: {e}")
    return issues


def check_dashscope() -> list[str]:
    from openai import OpenAI
    issues = []
    api_key = os.environ.get("DASHSCOPE_API_KEY", "")
    if not api_key:
        return ["DASHSCOPE_API_KEY empty, skipping dashscope check"]
    try:
        client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        resp = client.chat.completions.create(
            model="qwen3-8b",
            messages=[{"role": "user", "content": "Say OK"}],
            max_tokens=4,
            extra_body={"enable_thinking": False},
        )
        content = resp.choices[0].message.content or ""
        print(f"  qwen3-8b (non-thinking) reachable; sample reply: {content!r}")
    except Exception as e:
        issues.append(f"dashscope api error: {e}")
    return issues


def main() -> int:
    print("=== Phase 0 environment check ===")
    all_issues: list[str] = []
    for name, fn in [
        ("env vars", check_env_vars),
        ("mount points", check_mount_points),
        ("deepseek api", check_deepseek),
        ("dashscope api", check_dashscope),
    ]:
        print(f"checking {name}...")
        issues = fn()
        for i in issues:
            print(f"  ! {i}")
        all_issues.extend(issues)

    if all_issues:
        print(f"\nFAILED with {len(all_issues)} issue(s)")
        return 1
    print("\nALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
