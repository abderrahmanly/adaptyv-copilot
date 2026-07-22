"""Adaptyv Copilot — Streamlit chat UI.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import json
import os

import streamlit as st
from dotenv import load_dotenv

from adaptyv_copilot.brain import build_client, run_turn
from adaptyv_copilot.tools import AgentContext

load_dotenv()

st.set_page_config(page_title="Adaptyv Copilot", page_icon="🧬", layout="centered")

# On Streamlit Community Cloud, config lives in the hosted "Secrets" store
# (st.secrets), not in a .env file. Bridge it to environment variables so the
# rest of the app (anthropic.Anthropic(), ADAPTYV_MODE, the password gate) works
# identically whether running locally or deployed.
try:  # st.secrets raises if no secrets are configured at all
    for _k in ("ANTHROPIC_API_KEY", "ADAPTYV_MODE", "APP_PASSWORD"):
        if _k in st.secrets and not os.getenv(_k):
            os.environ[_k] = str(st.secrets[_k])
except Exception:
    pass


def password_ok() -> bool:
    """Optional access gate. If APP_PASSWORD is set (recommended for a public
    URL), require it before showing the chat. If it's unset, no gate."""
    expected = os.getenv("APP_PASSWORD")
    if not expected:
        return True
    if st.session_state.get("_authed"):
        return True
    st.title("🧬 Adaptyv Copilot")
    st.caption("This demo is password-protected.")
    entered = st.text_input("Access password", type="password")
    if entered and entered == expected:
        st.session_state["_authed"] = True
        st.rerun()
    elif entered:
        st.error("Incorrect password.")
    return False


if not password_ok():
    st.stop()

TOOL_LABELS = {
    "list_targets": "🔎 Browsing Foundry targets",
    "design_candidates": "🧬 Designing candidate binders",
    "score_candidates": "📊 Pre-screening in silico",
    "estimate_cost": "💵 Estimating assay cost",
    "order_assay": "🧪 Ordering a wet-lab assay",
    "check_experiment": "⏳ Checking experiment status",
    "fast_forward_experiment": "⏩ Simulating the ~3-week turnaround",
    "get_results": "📥 Fetching lab results",
}


def render_event(ev: dict) -> None:
    """Render one agent event inside the current chat message."""
    kind = ev["type"]
    if kind == "text":
        st.markdown(ev["text"])
    elif kind == "thinking":
        with st.expander("💭 reasoning", expanded=False):
            st.markdown(ev["text"])
    elif kind == "tool_use":
        label = TOOL_LABELS.get(ev["name"], f"🛠 {ev['name']}")
        with st.expander(label, expanded=False):
            st.code(json.dumps(ev["input"], indent=2), language="json")
    elif kind == "tool_result":
        with st.expander(f"↳ result · {ev['name']}", expanded=False):
            st.code(json.dumps(ev["result"], indent=2), language="json")
    elif kind == "error":
        st.error(ev["text"])


# --- session state ---------------------------------------------------------- #
if "messages" not in st.session_state:
    st.session_state.messages = []          # Anthropic-format history
if "transcript" not in st.session_state:
    st.session_state.transcript = []        # render-friendly log of events
if "ctx" not in st.session_state:
    st.session_state.ctx = AgentContext(mode=os.getenv("ADAPTYV_MODE", "simulated"))

# --- sidebar ---------------------------------------------------------------- #
with st.sidebar:
    st.title("🧬 Adaptyv Copilot")
    st.caption("An autonomous Design → Test → Learn agent for the Adaptyv Foundry.")
    st.markdown(f"**Backend:** `{st.session_state.ctx.foundry.mode}`")
    st.markdown(f"**Candidates in memory:** {len(st.session_state.ctx.candidates)}")
    if not os.getenv("ANTHROPIC_API_KEY"):
        st.error("Set ANTHROPIC_API_KEY in a .env file to run the agent.")
    st.divider()
    st.markdown("**Try:**")
    st.code("Find me a strong EGFR binder.", language=None)
    st.code("Design 12 candidates, test the\nbest 4, then improve them.", language=None)
    if st.button("🔄 Reset conversation"):
        for k in ("messages", "transcript", "ctx"):
            st.session_state.pop(k, None)
        st.rerun()

# --- replay existing transcript --------------------------------------------- #
for item in st.session_state.transcript:
    with st.chat_message(item["role"]):
        for ev in item["events"]:
            render_event(ev)

# --- handle new input ------------------------------------------------------- #
prompt = st.chat_input("Ask the copilot to design, test, or order proteins…")

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.transcript.append(
        {"role": "user", "events": [{"type": "text", "text": prompt}]}
    )
    st.session_state.messages.append({"role": "user", "content": prompt})

    client = build_client()
    events: list[dict] = []
    with st.chat_message("assistant"):
        for ev in run_turn(client, st.session_state.ctx, st.session_state.messages):
            events.append(ev)
            render_event(ev)
    st.session_state.transcript.append({"role": "assistant", "events": events})
