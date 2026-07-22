"""The agent brain: a manual Claude tool-use loop.

Yields structured events as it works so a UI (Streamlit) or CLI can render the
reasoning, each tool call, and each tool result live. Uses Claude Opus 4.8 with
adaptive thinking.
"""

from __future__ import annotations

import json
from typing import Iterator

import anthropic

from .tools import TOOLS, AgentContext, dispatch

MODEL = "claude-opus-4-8"

SYSTEM_PROMPT = """\
You are Adaptyv Copilot — an AI protein-engineering agent that drives Adaptyv \
Bio's automated wet-lab "Foundry". Adaptyv turns AI-designed protein sequences \
into real experimental data (binding affinity, expression, thermostability) in \
about three weeks. Your job is to run the Design -> Build -> Test -> Learn loop \
on the user's behalf, through the Foundry's API.

You can operate in two ways, and should infer which the user wants:

1. Direct ordering — the user already has candidates or a clear ask ("order \
   affinity tests on my best designs against EGFR"). Validate, price, confirm, \
   and order.

2. Autonomous discovery — the user gives a goal ("find me an EGFR binder"). \
   Then run the full loop yourself:
     a. design_candidates to generate a batch,
     b. score_candidates to cheaply pre-screen (each real assay costs money, so \
        only send the most promising ones),
     c. estimate_cost, state the price, and — unless the user pre-authorised — \
        confirm before ordering,
     d. order_assay for the top candidates,
     e. get_results (in this demo, call fast_forward_experiment first to skip the \
        ~3-week wait — always tell the user you are simulating the turnaround),
     f. feed the winners back into design_candidates as `parents` to start a \
        better round. Report the K_D improvement between rounds.

Rules of thumb:
- Work with candidate ids (cand_xxx), not raw sequences.
- Always pre-screen in silico before ordering — it is the whole point of not \
  wasting assay budget.
- 'affinity' and 'screening' assays need a target_id; get it from list_targets.
- Be transparent about cost and about what is real vs. simulated. This backend \
  is a faithful mock of the real Adaptyv API; make that clear if asked.
- Lead with the outcome. Keep messages tight; surface the numbers that matter \
  (K_D in nM, price, how many candidates), not a play-by-play.
"""


def build_client() -> anthropic.Anthropic:
    # Reads ANTHROPIC_API_KEY (or an `ant auth login` profile) from the env.
    return anthropic.Anthropic()


def run_turn(
    client: anthropic.Anthropic,
    ctx: AgentContext,
    messages: list[dict],
    max_steps: int = 12,
) -> Iterator[dict]:
    """Run one user turn to completion.

    Mutates ``messages`` in place (appending assistant + tool-result turns) and
    yields event dicts:
      {"type": "thinking", "text": ...}
      {"type": "text", "text": ...}
      {"type": "tool_use", "name": ..., "input": ...}
      {"type": "tool_result", "name": ..., "result": ...}
      {"type": "error", "text": ...}
    """
    for _ in range(max_steps):
        try:
            resp = client.messages.create(
                model=MODEL,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                thinking={"type": "adaptive", "display": "summarized"},
                tools=TOOLS,
                messages=messages,
            )
        except anthropic.APIError as exc:
            yield {"type": "error", "text": f"API error: {exc}"}
            return

        messages.append({"role": "assistant", "content": resp.content})

        for block in resp.content:
            if block.type == "thinking":
                if block.thinking:
                    yield {"type": "thinking", "text": block.thinking}
            elif block.type == "text":
                yield {"type": "text", "text": block.text}

        if resp.stop_reason != "tool_use":
            return

        tool_results = []
        for block in resp.content:
            if block.type != "tool_use":
                continue
            yield {"type": "tool_use", "name": block.name, "input": block.input}
            result = dispatch(ctx, block.name, block.input)
            yield {"type": "tool_result", "name": block.name, "result": result}
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result),
                }
            )
        messages.append({"role": "user", "content": tool_results})

    yield {"type": "error", "text": f"Stopped after {max_steps} tool steps."}
