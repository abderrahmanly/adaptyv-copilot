"""Target catalog and seed binders.

In the real product these come from `GET /targets` on the Adaptyv Foundry API.
Here we ship a small offline catalog so the simulated backend behaves like the
real one and the demo runs with zero network access.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Target:
    id: str
    name: str
    full_name: str
    # A known, expressible binder scaffold to optimise from. For a real
    # de-novo campaign the agent would start from RFdiffusion/ProteinMPNN
    # output; here a curated ~63-residue three-helix miniprotein stands in.
    seed_binder: str
    # Context the agent can quote to the user.
    note: str


# A plausible de-novo three-helix miniprotein scaffold (standard amino acids
# only). Used as the starting point the agent mutates around.
_EGFR_SEED = (
    "MGSEIEELRKRAEELAKKNPSPEVLKLLQEAQKLLKENPSDPELLQLAKKVAELAKKLGGSGE"
)
_PDL1_SEED = (
    "MKKEELLKKAEELAKKNPSDEVLKLNLEAQKLLKEHPSDKELLELAKKVAELLKKLGSNGES"
)


TARGETS: dict[str, Target] = {
    t.id: t
    for t in [
        Target(
            id="egfr",
            name="EGFR",
            full_name="Epidermal Growth Factor Receptor (ectodomain)",
            seed_binder=_EGFR_SEED,
            note=(
                "The target from Adaptyv's public binder-design competition. "
                "Best community de-novo binder reached ~82 nM; optimised binders "
                "reached ~1.2 nM (8x stronger than the antibody Cetuximab)."
            ),
        ),
        Target(
            id="pdl1",
            name="PD-L1",
            full_name="Programmed death-ligand 1",
            seed_binder=_PDL1_SEED,
            note="Immuno-oncology checkpoint target; blocking it can re-activate T cells.",
        ),
    ]
}


def list_targets(search: str | None = None) -> list[dict]:
    out = []
    for t in TARGETS.values():
        if search and search.lower() not in (t.name + t.full_name).lower():
            continue
        out.append(
            {
                "id": t.id,
                "name": t.name,
                "full_name": t.full_name,
                "seed_binder_length": len(t.seed_binder),
                "note": t.note,
            }
        )
    return out


def get_target(target_id: str) -> Target:
    key = (target_id or "").lower().strip()
    if key not in TARGETS:
        raise KeyError(
            f"Unknown target '{target_id}'. Known targets: {', '.join(TARGETS)}"
        )
    return TARGETS[key]
