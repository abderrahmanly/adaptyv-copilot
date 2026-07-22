"""Adaptyv Copilot — terminal chat.

Run with:  python cli.py
A lightweight alternative to the Streamlit UI; same agent, same tools.
"""

from __future__ import annotations

import json
import os

from dotenv import load_dotenv

from adaptyv_copilot.brain import build_client, run_turn
from adaptyv_copilot.tools import AgentContext

load_dotenv()


def main() -> None:
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise SystemExit("Set ANTHROPIC_API_KEY (e.g. in a .env file) first.")

    client = build_client()
    ctx = AgentContext(mode=os.getenv("ADAPTYV_MODE", "simulated"))
    messages: list[dict] = []

    print("🧬 Adaptyv Copilot  (backend: %s)" % ctx.foundry.mode)
    print("Type a goal like 'Find me a strong EGFR binder.'  Ctrl-C to quit.\n")

    while True:
        try:
            user = input("you › ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye 👋")
            return
        if not user:
            continue
        messages.append({"role": "user", "content": user})

        for ev in run_turn(client, ctx, messages):
            if ev["type"] == "text":
                print(f"\ncopilot › {ev['text']}\n")
            elif ev["type"] == "thinking":
                print(f"  💭 {ev['text'][:200].strip()}…")
            elif ev["type"] == "tool_use":
                print(f"  🛠  {ev['name']}({json.dumps(ev['input'])})")
            elif ev["type"] == "tool_result":
                preview = json.dumps(ev["result"])
                print(f"  ↳  {preview[:300]}{'…' if len(preview) > 300 else ''}")
            elif ev["type"] == "error":
                print(f"  ⚠️  {ev['text']}")


if __name__ == "__main__":
    main()
