"""Tool definitions and dispatch for the agent.

The agent works with short candidate IDs (e.g. ``cand_003``) rather than raw
amino-acid strings — this keeps the model's context compact and stops it from
corrupting long sequences. The full sequences live in a server-side
``AgentContext`` and are resolved back only when an assay is ordered.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import catalog, science
from .foundry import EXPERIMENT_TYPES, FoundryError, get_foundry


@dataclass
class AgentContext:
    """Per-conversation state shared across tool calls."""

    mode: str = "simulated"
    foundry: object = None
    candidates: dict[str, str] = field(default_factory=dict)  # id -> sequence
    target_names: dict[str, str] = field(default_factory=dict)  # target id -> name
    _counter: int = 0

    def __post_init__(self):
        if self.foundry is None:
            self.foundry = get_foundry(self.mode)

    def add_candidates(self, sequences: list[str]) -> list[str]:
        ids = []
        for seq in sequences:
            self._counter += 1
            cid = f"cand_{self._counter:03d}"
            self.candidates[cid] = seq
            ids.append(cid)
        return ids

    def resolve(self, ids: list[str]) -> dict[str, str]:
        missing = [i for i in ids if i not in self.candidates]
        if missing:
            raise FoundryError(f"Unknown candidate ids: {missing}")
        return {i: self.candidates[i] for i in ids}


# --------------------------------------------------------------------------- #
# Tool schemas (Anthropic tool-use format)
# --------------------------------------------------------------------------- #

TOOLS = [
    {
        "name": "list_targets",
        "description": "List protein targets available in the Adaptyv Foundry "
        "(e.g. EGFR). Call this first when the user names or asks about a target.",
        "input_schema": {
            "type": "object",
            "properties": {
                "search": {"type": "string", "description": "Optional name filter."}
            },
        },
    },
    {
        "name": "design_candidates",
        "description": "Generate new binder candidate sequences for a target. "
        "With no parents, mutates the target's seed binder; with parents (winning "
        "candidate ids from a previous round) it optimises around them. Returns "
        "compact candidate ids and biophysical properties — not raw sequences.",
        "input_schema": {
            "type": "object",
            "properties": {
                "target_id": {"type": "string"},
                "n": {"type": "integer", "description": "How many to generate (1-24)."},
                "parents": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional candidate ids to optimise around (the Learn step).",
                },
                "mutation_rate": {
                    "type": "number",
                    "description": "Fraction of residues to change per variant (0.02-0.20). Default 0.06.",
                },
            },
            "required": ["target_id", "n"],
        },
    },
    {
        "name": "score_candidates",
        "description": "Cheaply predict which candidates are worth testing, using "
        "an in-silico score (0-1, higher is better) and a predicted K_D (nM, lower "
        "is tighter). Use this to pre-screen before paying for lab assays.",
        "input_schema": {
            "type": "object",
            "properties": {
                "candidate_ids": {"type": "array", "items": {"type": "string"}},
                "top_k": {
                    "type": "integer",
                    "description": "Optionally return only the best K after ranking.",
                },
            },
            "required": ["candidate_ids"],
        },
    },
    {
        "name": "estimate_cost",
        "description": "Estimate the price of an assay before ordering it. Against "
        "the real Foundry this returns live pricing; pass target_id for binding "
        "assays so antigen costs resolve.",
        "input_schema": {
            "type": "object",
            "properties": {
                "experiment_type": {"type": "string", "enum": list(EXPERIMENT_TYPES)},
                "candidate_ids": {"type": "array", "items": {"type": "string"}},
                "target_id": {"type": "string"},
                "method": {
                    "type": "string",
                    "enum": ["bli", "spr"],
                    "description": "Measurement method. Required for affinity and "
                    "screening (defaults to bli); rejected for other types.",
                },
                "n_replicates": {
                    "type": "integer",
                    "description": "Technical replicates. Default 1; the lab recommends 3+.",
                },
            },
            "required": ["experiment_type", "candidate_ids"],
        },
    },
    {
        "name": "order_assay",
        "description": "Order a wet-lab assay from the Adaptyv Foundry for the given "
        "candidates: creates the experiment and submits it. Confirm cost with the "
        "user before calling this. 'affinity', 'screening' and 'epitope_binning' "
        "require a target_id; 'affinity' and 'screening' also take a method "
        "(bli or spr). Against the real Foundry this places a paid order.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "A human-readable experiment name."},
                "experiment_type": {"type": "string", "enum": list(EXPERIMENT_TYPES)},
                "candidate_ids": {"type": "array", "items": {"type": "string"}},
                "target_id": {
                    "type": "string",
                    "description": "Target id from list_targets (a catalog UUID "
                    "against the real Foundry).",
                },
                "method": {"type": "string", "enum": ["bli", "spr"]},
                "n_replicates": {"type": "integer"},
            },
            "required": ["name", "experiment_type", "candidate_ids"],
        },
    },
    {
        "name": "check_experiment",
        "description": "Check the status of an ordered experiment.",
        "input_schema": {
            "type": "object",
            "properties": {"experiment_id": {"type": "string"}},
            "required": ["experiment_id"],
        },
    },
    {
        "name": "fast_forward_experiment",
        "description": "DEMO ONLY: skip the ~3-week wet-lab turnaround so results "
        "are ready immediately. Use this to demonstrate the Learn step end-to-end.",
        "input_schema": {
            "type": "object",
            "properties": {"experiment_id": {"type": "string"}},
            "required": ["experiment_id"],
        },
    },
    {
        "name": "get_results",
        "description": "Fetch measured lab results for a completed experiment "
        "(per-candidate K_D, expression, etc.). Feed these back into design_candidates "
        "(as parents) to close the Design-Test-Learn loop.",
        "input_schema": {
            "type": "object",
            "properties": {"experiment_id": {"type": "string"}},
            "required": ["experiment_id"],
        },
    },
]


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #

def dispatch(ctx: AgentContext, name: str, args: dict) -> dict:
    """Execute a tool call. Always returns a JSON-serialisable dict."""
    try:
        return _DISPATCH[name](ctx, args)
    except KeyError:
        return {"error": f"Unknown tool '{name}'."}
    except (FoundryError, Exception) as exc:  # surface errors to the model
        return {"error": str(exc)}


def _list_targets(ctx, args):
    targets = ctx.foundry.list_targets(args.get("search"))
    # Remember id -> name so design_candidates can pick a starting scaffold even
    # when the id is an opaque Foundry catalog UUID.
    for t in targets:
        if t.get("id") and t.get("name"):
            ctx.target_names[str(t["id"])] = t["name"]
    return {"targets": targets}


def _design_candidates(ctx, args):
    target_id = args["target_id"]
    n = max(1, min(int(args["n"]), 24))
    mutation_rate = float(args.get("mutation_rate", 0.06))
    parent_ids = args.get("parents") or []
    if parent_ids:
        parents = list(ctx.resolve(parent_ids).values())
        origin = f"optimising around {len(parent_ids)} parent(s)"
    else:
        seed, provenance = catalog.resolve_seed(
            target_id, ctx.target_names.get(str(target_id))
        )
        parents = [seed]
        origin = f"mutating the {provenance}"
    # Seed generator on current candidate count for round-to-round variety.
    variants = science.generate_variants(
        parents, n, mutation_rate=mutation_rate, seed=len(ctx.candidates) + 1
    )
    ids = ctx.add_candidates(variants)
    return {
        "target_id": target_id,
        "origin": origin,
        "candidates": [
            {"id": cid, **science.biophysical_props(ctx.candidates[cid])} for cid in ids
        ],
    }


def _score_candidates(ctx, args):
    ids = args["candidate_ids"]
    seqs = ctx.resolve(ids)
    scored = [
        {
            "id": cid,
            "insilico_score": round(science.insilico_score(seq), 3),
            "predicted_kd_nM": science.predicted_kd_nm(seq),
        }
        for cid, seq in seqs.items()
    ]
    scored.sort(key=lambda r: r["insilico_score"], reverse=True)
    top_k = args.get("top_k")
    if top_k:
        scored = scored[: int(top_k)]
    return {"ranked": scored, "note": "insilico_score is a cheap prediction, not lab truth."}


def _estimate_cost(ctx, args):
    seqs = ctx.resolve(args["candidate_ids"])
    kwargs = {
        "method": args.get("method"),
        "n_replicates": int(args.get("n_replicates", 1)),
    }
    if ctx.foundry.mode == "real":
        # The live pricing endpoint needs the target to resolve antigen costs.
        kwargs["target_id"] = args.get("target_id")
    return ctx.foundry.cost_estimate(args["experiment_type"], seqs, **kwargs)


def _order_assay(ctx, args):
    seqs = ctx.resolve(args["candidate_ids"])
    created = ctx.foundry.create_experiment(
        name=args["name"],
        experiment_type=args["experiment_type"],
        sequences=seqs,
        target_id=args.get("target_id"),
        method=args.get("method"),
        n_replicates=int(args.get("n_replicates", 1)),
    )
    submitted = ctx.foundry.submit_experiment(created["id"])
    return {"experiment": submitted, "ordered_candidate_ids": list(seqs)}


def _check_experiment(ctx, args):
    return ctx.foundry.get_experiment(args["experiment_id"])


def _fast_forward(ctx, args):
    return ctx.foundry.fast_forward_experiment(args["experiment_id"])


def _get_results(ctx, args):
    return ctx.foundry.get_results(args["experiment_id"])


_DISPATCH = {
    "list_targets": _list_targets,
    "design_candidates": _design_candidates,
    "score_candidates": _score_candidates,
    "estimate_cost": _estimate_cost,
    "order_assay": _order_assay,
    "check_experiment": _check_experiment,
    "fast_forward_experiment": _fast_forward,
    "get_results": _get_results,
}
