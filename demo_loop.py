"""Deterministic Design -> Test -> Learn loop with NO LLM and NO network.

Exercises the same tools the agent calls, so you can verify the whole loop
(design -> pre-screen -> order -> results -> learn) works before wiring up the
Anthropic API. Run:  python demo_loop.py
"""

from __future__ import annotations

from adaptyv_copilot.tools import AgentContext, dispatch

TARGET = "egfr"
ROUNDS = 3
BATCH = 12
TEST_TOP = 4


def best_kd(results: list[dict]) -> float:
    kds = [r["kd_nM"] for r in results if "kd_nM" in r]
    return min(kds) if kds else float("inf")


def main() -> None:
    ctx = AgentContext(mode="simulated")
    parents: list[str] = []          # candidate ids carried between rounds
    history: list[float] = []        # best-so-far K_D after each round
    champions: list[tuple[float, str]] = []  # (kd_nM, clone_id) tested so far

    for rnd in range(1, ROUNDS + 1):
        print(f"\n=== Round {rnd} ===")

        # 1. Design
        design = dispatch(
            ctx, "design_candidates",
            {"target_id": TARGET, "n": BATCH, "parents": parents},
        )
        ids = [c["id"] for c in design["candidates"]]
        print(f"Designed {len(ids)} candidates ({design['origin']}).")

        # 2. Pre-screen in silico, keep the top few
        ranked = dispatch(
            ctx, "score_candidates",
            {"candidate_ids": ids, "top_k": TEST_TOP},
        )["ranked"]
        top_ids = [r["id"] for r in ranked]
        print("Top by in-silico score:",
              ", ".join(f"{r['id']}={r['insilico_score']}" for r in ranked))

        # 3. Price + order an affinity assay on the winners
        cost = dispatch(
            ctx, "estimate_cost",
            {"experiment_type": "affinity", "candidate_ids": top_ids},
        )
        print(f"Ordering affinity assay on {len(top_ids)} candidates "
              f"(${cost['total_usd']}).")
        order = dispatch(
            ctx, "order_assay",
            {
                "name": f"EGFR round {rnd}",
                "experiment_type": "affinity",
                "candidate_ids": top_ids,
                "target_id": TARGET,
            },
        )
        exp_id = order["experiment"]["id"]

        # 4. Skip the ~3-week wait (demo only) and read measured results
        dispatch(ctx, "fast_forward_experiment", {"experiment_id": exp_id})
        results = dispatch(ctx, "get_results", {"experiment_id": exp_id})["results"]
        for r in results:
            print(f"   {r['clone_id']}: K_D = {r.get('kd_nM')} nM, "
                  f"expr = {r.get('expression_mg_per_L')} mg/L")

        # 5. Learn (elitism): keep the best binders SO FAR as parents, so a
        #    noisy round can't throw away a champion — exactly how real
        #    directed-evolution campaigns hold onto their best hit.
        for r in results:
            if "kd_nM" in r:
                champions.append((r["kd_nM"], r["clone_id"]))
        champions.sort(key=lambda c: c[0])
        parents = [cid for _, cid in champions[:2]]
        history.append(champions[0][0])
        print(f"Best K_D this round: {best_kd(results)} nM  |  "
              f"best so far: {history[-1]} nM  (parents -> {parents})")

    print("\n=== Learning curve (best K_D so far, per round) ===")
    for i, kd in enumerate(history, 1):
        print(f"  Round {i}: {kd} nM")
    if history[-1] < history[0]:
        factor = round(history[0] / history[-1], 1)
        print(f"\n✅ Improved best binder {factor}x over {ROUNDS} rounds.")
    else:
        print("\n(No improvement this run — try more rounds or a larger batch.)")


if __name__ == "__main__":
    main()
