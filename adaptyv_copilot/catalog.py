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
                "The target from Adaptyv's public binder-design competition "
                "(130 teams, ~1,800 designs, 601 selected for wet-lab testing). "
                "Best de-novo binder reached K_D 82 nM; the winning optimised "
                "binder reached 1.21 nM — 8x tighter than an scFv-format of "
                "Cetuximab (9.94 nM). Measured on Adaptyv's automated BLI pipeline."
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


# A generic de-novo three-helix miniprotein scaffold, used when designing
# against a target we have no curated starting binder for (every target in the
# real Foundry catalog, which is keyed by UUID rather than by our short ids).
GENERIC_SCAFFOLD = (
    "MSEEELKKLAEELVKKNPSDEVLKLLQEAEKLLKEHPSDPELLELAKKVAELLKKLGSGSE"
)


def resolve_seed(target_id: str, target_name: str | None = None) -> tuple[str, str]:
    """Find a starting binder for ``target_id``.

    Returns ``(sequence, provenance)``. Falls back to a generic scaffold for
    targets outside the local catalog — real Foundry targets are catalog UUIDs,
    and Adaptyv supplies the antigen, not a starting binder for it.
    """
    key = (target_id or "").lower().strip()
    if key in TARGETS:
        t = TARGETS[key]
        return t.seed_binder, f"curated seed binder for {t.name}"

    if target_name:
        name = target_name.lower()
        for t in TARGETS.values():
            if t.name.lower() in name or name in t.full_name.lower():
                return t.seed_binder, f"curated seed binder for {t.name}"

    label = target_name or target_id
    return GENERIC_SCAFFOLD, (
        f"generic de-novo miniprotein scaffold (no curated starting binder for "
        f"'{label}')"
    )
