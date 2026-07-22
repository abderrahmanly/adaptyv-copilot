# 🧬 Adaptyv Copilot

**An AI agent that runs Adaptyv Bio's Design → Build → Test → Learn loop for you — from plain English.**

Adaptyv is a *cloud lab for protein designers*: you upload AI-designed protein
sequences, their robots physically build and test them (binding affinity,
expression, thermostability), and you get clean, ML-ready data back in ~3 weeks.
Adaptyv even ships an ["API for AI agents"](https://docs.adaptyvbio.com/api-reference)
so the loop can run without a human clicking buttons.

**Adaptyv Copilot is that agent.** Tell it a goal and it will design candidate
binders, cheaply pre-screen them in silico, order the promising ones from the
Foundry, read the measured results, and design a *better* batch — closing the
loop. Or just tell it to order assays on sequences you already have.

```
        You, in plain English
  ("order affinity tests for these"   OR   "find me a strong EGFR binder")
                 │
                 ▼
        ┌───────────────────┐
        │  Agent (Claude)   │  decides what to do, talks back to you
        └─────────┬─────────┘
                  │ tool calls
   ┌──────────────┼───────────────────────────────┐
   ▼              ▼               ▼                 ▼
design_        score_         estimate_        order_assay / check /
candidates     candidates     cost             get_results
(generate)    (pre-screen)   (price)          (Adaptyv Foundry API)
                                                     │
             ┌───────────────────────────────────────┘
             ▼
       LEARN: feed winners back → next, better round
```

---

## Why this is useful to Adaptyv

- It uses **their actual product surface** (the Foundry ordering API), not a generic side project.
- It's the exact thing their ["API for AI agents"](https://agents.adaptyvbio.com) pitch invites, and which almost nobody has built end-to-end.
- The **in-silico pre-screen** is a real money-saver: in Adaptyv's public EGFR
  competition only ~600 of ~1,800 designs were worth lab-testing. The agent
  spends assay budget only on candidates likely to succeed.
- It demonstrates the whole loop their business is built on — generation,
  model-based scoring, agentic ordering, and *learning from wet-lab data*.

---

## Quick start

Requires **Python 3.11+** and an **Anthropic API key**.

```bash
# 1. install
cd adaptyv-copilot
python -m venv .venv && . .venv/Scripts/activate      # Windows
# python -m venv .venv && source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt

# 2. configure
cp .env.example .env         # then put your ANTHROPIC_API_KEY in .env

# 3a. web chat (recommended for the demo)
streamlit run app.py

# 3b. or terminal chat
python cli.py

# 3c. or the LLM-free mechanics check (no API key needed)
python demo_loop.py
```

Then try: **“Find me a strong EGFR binder — design a batch, test the best few, and improve them.”**

---

## What's real vs. simulated (read this)

The **agent, the design, and the scoring are real**. The **Foundry backend is a
faithful simulation** by default (`ADAPTYV_MODE=simulated`), because real assays
cost money and take ~3 weeks — not suitable for a demo you can run in a minute.

- The simulated Foundry mirrors the real API's shape (`list_targets`,
  `cost_estimate`, `create_experiment`, `submit`, `get_results`, the
  Draft→…→Done lifecycle, the five assay types, per-sequence pricing).
- A hidden "ground-truth" fitness landscape stands in for the wet lab; the
  agent's in-silico score is a deliberately *imperfect* predictor of it — which
  is exactly why pre-screening saves money but the lab still surprises you.
- `fast_forward_experiment` skips the ~3-week wait so the demo can show the
  Learn step; the agent always tells you when it does this.
- Flip `ADAPTYV_MODE=real` (plus a Foundry token) to route through the official
  [`adaptyv-sdk`](https://github.com/adaptyvbio/adaptyv-sdk). `RealFoundry` in
  `foundry.py` marks exactly where each real SDK call plugs in.

Nothing here fabricates results presented as real: simulation is labelled
everywhere, in the UI, the tools, and the agent's own words.

---

## How it works

| File | Role |
|------|------|
| `adaptyv_copilot/brain.py` | The agent: a manual Claude (Opus 4.8) tool-use loop with adaptive thinking; streams events for the UI. |
| `adaptyv_copilot/tools.py` | Tool schemas + dispatch; an ID-based candidate store so the model handles `cand_xxx` ids, not raw sequences. |
| `adaptyv_copilot/foundry.py` | `SimulatedFoundry` (mirrors the real API) and `RealFoundry` (swap-in via `adaptyv-sdk`). |
| `adaptyv_copilot/science.py` | Sequence generation, the cheap in-silico predictor, and the hidden lab-truth landscape. |
| `adaptyv_copilot/catalog.py` | Targets (EGFR, PD-L1) and seed binders. |
| `app.py` / `cli.py` | Streamlit and terminal front-ends. |
| `demo_loop.py` | LLM-free run of the full loop — a mechanics test and reproducible learning-curve demo. |

**The agent's tools**

`list_targets` · `design_candidates` · `score_candidates` · `estimate_cost` ·
`order_assay` · `check_experiment` · `fast_forward_experiment` · `get_results`

---

## Suggested 3-minute demo (Loom)

1. Open the Streamlit app; show the sidebar backend = `simulated`.
2. Type: *"Find me a strong EGFR binder. Design 12, pre-screen them, test the best 4, then run one improvement round."*
3. Narrate as the agent: designs candidates → pre-screens in silico → prices the
   assay → orders it → (simulated turnaround) → reads K_D results → carries the
   winners into a better round → reports the K_D improvement.
4. Show a direct-ordering ask too: *"Order an expression assay on cand_003 and cand_007."*
5. Point at `foundry.py`'s `RealFoundry` and note the one-env-var switch to the live Adaptyv API.

---

## Extending toward production

- Swap the mutation-based generator for **ProteinMPNN/RFdiffusion/ESM** output.
- Replace the heuristic pre-screen with **ESMFold/AlphaFold** confidence (pLDDT/ipTM).
- Wire `RealFoundry` to the live `adaptyv-sdk` and add **webhook** handling so
  results notify the agent when they're ready (no polling).
- Package the tool layer as an **MCP server** so any assistant can order assays.

---

*Built as a take-home for Adaptyv Bio. Backend simulated by default; the agent, design, and scoring are real.*
