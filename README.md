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

Two backends sit behind one interface, chosen by `ADAPTYV_MODE`.

**`real`** — a working client for the live Adaptyv Foundry REST API
(`adaptyv_copilot/api.py`), built against the public
[OpenAPI schema](https://devs.adaptyvbio.com/api/v1/openapi.json). It speaks the
real contract: catalog-UUID targets, the `bli`/`spr` method required on binding
assays, all seven experiment types, the lowercase status enum, live pricing in
cents, and K_D returned in molar (converted to nM at the boundary).

> **Real mode is read-only unless you opt in.** Creating and submitting an
> experiment places a genuine, paid lab order, so both are refused unless
> `ADAPTYV_ALLOW_ORDERING=true`. Reads — targets, status, results, cost
> estimates — always work and cost nothing. A **Viewer** token is enough for
> them.

**`simulated`** (default) — free, offline, deterministic, and deliberately
*vocabulary-identical* to the real backend, so agent behaviour doesn't change
when you switch. A hidden "ground-truth" fitness landscape stands in for the wet
lab; the agent's in-silico score is an intentionally *imperfect* predictor of it
— which is exactly why pre-screening saves money but the lab still surprises
you. `fast_forward_experiment` skips the ~3-week wait so a demo can reach the
Learn step; it exists only here, and the agent always says when it uses it.

The agent's system prompt changes with the backend, so it never describes
simulated numbers as measurements — and in real mode it knows ordering may be
blocked and that results take three weeks.

Verify the integration yourself:

```bash
python verify_real_api.py
```

17 checks run with no credentials — request-payload construction against the
published contract, the per-assay-type validation matrix, cents→USD and
molar→nM conversion, error surfacing, the ordering guard — plus a live ping of
Adaptyv's health endpoint. Add `ADAPTYV_API_TOKEN` to `.env` and it additionally
performs real authenticated reads (`whoami`, the live target catalog). It never
creates or submits anything, so it cannot spend money.

Nothing here fabricates results presented as real: simulation is labelled
everywhere — in the UI, in the tool output (`"source"`), and in the agent's own
words.

---

## How it works

| File | Role |
|------|------|
| `adaptyv_copilot/brain.py` | The agent: a manual Claude (Opus 4.8) tool-use loop with adaptive thinking; streams events for the UI. |
| `adaptyv_copilot/tools.py` | Tool schemas + dispatch; an ID-based candidate store so the model handles `cand_xxx` ids, not raw sequences. |
| `adaptyv_copilot/api.py` | HTTP client for the live Foundry REST API — auth, pagination, error surfacing. |
| `adaptyv_copilot/foundry.py` | `SimulatedFoundry` (mirrors the real API) and `RealFoundry` (talks to it), behind one interface. |
| `adaptyv_copilot/science.py` | Sequence generation, the cheap in-silico predictor, and the hidden lab-truth landscape. |
| `adaptyv_copilot/catalog.py` | Targets (EGFR, PD-L1) and seed binders. |
| `app.py` / `cli.py` | Streamlit and terminal front-ends. |
| `demo_loop.py` | LLM-free run of the full loop — a mechanics test and reproducible learning-curve demo. |
| `verify_real_api.py` | Checks the real-API integration against the published contract; live reads if a token is set. |

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
5. Run `python verify_real_api.py` on camera — it proves the real-API client
   matches Adaptyv's published contract and pings their live health endpoint.
6. Point at `api.py` / `RealFoundry`, and at the `ADAPTYV_ALLOW_ORDERING` guard:
   real mode is read-only until you deliberately opt in to spending money.

---

## Extending toward production

- Swap the mutation-based generator for **ProteinMPNN/RFdiffusion/ESM** output.
- Replace the heuristic pre-screen with **ESMFold/AlphaFold** confidence (pLDDT/ipTM).
- Add **webhook** handling (`webhook_url` is already supported at experiment
  creation) so results notify the agent when ready, instead of polling. The
  signature check is `X-Adaptyv-Signature: sha256=<HMAC over the raw body>`.
- Package the tool layer as an **MCP server** so any assistant can order assays.
  (Adaptyv also hosts one at `https://mcp.adaptyvbio.com/mcp/`.)

---

*Built as a take-home for Adaptyv Bio. Backend simulated by default; the agent, design, and scoring are real.*
