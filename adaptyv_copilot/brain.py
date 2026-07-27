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
     e. get_results (against the simulator, call fast_forward_experiment first to \
        skip the ~3-week wait — always tell the user you are simulating the \
        turnaround),
     f. feed the winners back into design_candidates as `parents` to start a \
        better round. Report the K_D improvement between rounds.

Rules of thumb:
- Work with candidate ids (cand_xxx), not raw sequences.
- Always pre-screen in silico before ordering — it is the whole point of not \
  wasting assay budget.
- 'affinity', 'screening' and 'epitope_binning' need a target_id; get it from \
  list_targets. 'affinity' and 'screening' also take a method: 'bli' (default) \
  or 'spr'. Other assay types reject both.
- Be transparent about cost and about what is real vs. simulated.
- Lead with the outcome. Keep messages tight; surface the numbers that matter \
  (K_D in nM, price, how many candidates), not a play-by-play.
"""

_SIMULATED_NOTE = """\

BACKEND: SIMULATED. The Foundry behind your tools is a faithful mock — same \
assay types, same status lifecycle, same pricing shape as the real API — but no \
robot builds anything and no money is spent. Results are generated from the \
sequence, not measured. Never present them as real measurements; say they are \
simulated whenever you report them, and whenever the user asks what is real.
"""

_REAL_READONLY_NOTE = """\

BACKEND: REAL — READ-ONLY. Your tools talk to the live Adaptyv Foundry. Reads \
(list_targets, estimate_cost, check_experiment, get_results) return genuine data \
from Adaptyv. Ordering is currently disabled by configuration, so order_assay \
will fail — that is expected and safe, not a bug. If the user wants to actually \
order, tell them to set ADAPTYV_ALLOW_ORDERING=true, and warn them it places a \
real, paid lab order. fast_forward_experiment does not exist against the real \
lab: real assays take about three weeks. Target ids are catalog UUIDs from \
list_targets — never invent one.
"""

_REAL_ORDERING_NOTE = """\

BACKEND: REAL — ORDERING ENABLED. Your tools talk to the live Adaptyv Foundry \
and order_assay places a GENUINE, PAID lab order that a human will be invoiced \
for. Always call estimate_cost and get explicit confirmation of the actual \
dollar amount before ordering — no exceptions, and never treat an earlier \
general 'go ahead' as authorisation for a specific spend. fast_forward_experiment \
does not exist here: real assays take about three weeks, so after ordering, tell \
the user results will not be available today. Target ids are catalog UUIDs from \
list_targets — never invent one.
"""


def build_system_prompt(ctx: AgentContext) -> str:
    """System prompt tailored to the backend actually in use."""
    from .foundry import ordering_enabled

    if getattr(ctx.foundry, "mode", "simulated") != "real":
        return SYSTEM_PROMPT + _SIMULATED_NOTE
    return SYSTEM_PROMPT + (
        _REAL_ORDERING_NOTE if ordering_enabled() else _REAL_READONLY_NOTE
    )


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
    system = build_system_prompt(ctx)
    for _ in range(max_steps):
        try:
            resp = client.messages.create(
                model=MODEL,
                max_tokens=4096,
                system=system,
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
