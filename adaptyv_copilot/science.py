"""Protein biophysics: sequence generation, in-silico scoring, and the
hidden 'ground-truth' landscape the simulated lab measures.

Design intent
-------------
A protein is a string over the 20 amino acids. Two functions matter here:

* ``latent_fitness``  -> the *true* quality of a binder. The simulated Foundry
  turns this into a measured binding affinity (K_D). This is the "wet-lab
  truth" the agent cannot see directly.
* ``insilico_score``  -> a cheap *prediction* the agent computes before paying
  for an assay. It is correlated with ``latent_fitness`` but imperfect (it only
  sees observable biophysical features), which is exactly why pre-screening
  saves money but the lab still surprises you.

Everything is deterministic (seeded by the sequence itself), so a Design ->
Test -> Learn loop actually climbs the landscape instead of drifting on noise.
No heavy models, no GPU, no network.
"""

from __future__ import annotations

import hashlib
import random

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

# Kyte-Doolittle hydropathy (higher = more hydrophobic).
_HYDROPATHY = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5,
    "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9,
    "M": 1.9, "F": 2.8, "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9,
    "Y": -1.3, "V": 4.2,
}
_CHARGE = {"D": -1, "E": -1, "K": 1, "R": 1, "H": 0.1}

# A hidden "epitope-complementarity" motif: residues that (secretly) help the
# binder grip the target when they appear anywhere in the interface-facing set.
# The agent never sees this — the lab implicitly rewards it.
_FAVOURABLE = set("WYFRH")
_UNFAVOURABLE = set("PG")


def _stable_unit(seq: str, salt: str = "") -> float:
    """Deterministic pseudo-random value in [0, 1) keyed by the sequence."""
    h = hashlib.sha256((salt + "|" + seq).encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def biophysical_props(seq: str) -> dict:
    """Cheap observable features a designer could compute in seconds."""
    n = max(len(seq), 1)
    gravy = sum(_HYDROPATHY.get(a, 0.0) for a in seq) / n
    net_charge = sum(_CHARGE.get(a, 0.0) for a in seq)
    aromatic = sum(seq.count(a) for a in "WYF") / n
    return {
        "length": len(seq),
        "gravy": round(gravy, 3),          # grand average of hydropathy
        "net_charge": round(net_charge, 1),
        "aromatic_fraction": round(aromatic, 3),
    }


def _biophysical_heuristic(seq: str) -> float:
    """A designer's cheap guess at 'foldability/expressibility', in [0, 1].

    Rewards a balanced, slightly-hydrophilic surface and a length near the
    scaffold's sweet spot; penalises extremes that tend not to express.
    """
    p = biophysical_props(seq)
    # Prefer gravy near -0.3 (soluble but not too polar).
    gravy_term = max(0.0, 1.0 - abs(p["gravy"] + 0.3) / 2.5)
    # Prefer modest net charge magnitude.
    charge_term = max(0.0, 1.0 - abs(p["net_charge"]) / 12.0)
    # Prefer length in a plausible mini-binder window.
    length_term = max(0.0, 1.0 - abs(p["length"] - 62) / 40.0)
    return max(0.0, min(1.0, 0.4 * gravy_term + 0.3 * charge_term + 0.3 * length_term))


def latent_fitness(seq: str) -> float:
    """The TRUE binder quality in [0, 1] — the lab's hidden ground truth.

    Combines a hidden interface motif with a rugged deterministic component so
    the landscape is learnable but not trivial.
    """
    if not seq:
        return 0.0
    fav = sum(1 for a in seq if a in _FAVOURABLE) / len(seq)
    unfav = sum(1 for a in seq if a in _UNFAVOURABLE) / len(seq)
    motif = max(0.0, min(1.0, 0.5 + 3.0 * fav - 2.5 * unfav))
    # A little non-heritable ruggedness keeps the landscape from being trivial,
    # but the heritable motif dominates so directed evolution actually climbs.
    ruggedness = _stable_unit(seq, salt="latent")
    fold = _biophysical_heuristic(seq)
    fit = 0.58 * motif + 0.15 * ruggedness + 0.27 * fold
    return max(0.0, min(1.0, fit))


def insilico_score(seq: str) -> float:
    """The agent's CHEAP prediction in [0, 1], computed before ordering assays.

    Correlated with ``latent_fitness`` but only from observable features plus a
    little model noise — so it is a useful, imperfect pre-screen.
    """
    heuristic = _biophysical_heuristic(seq)
    fav = sum(1 for a in seq if a in _FAVOURABLE) / max(len(seq), 1)
    predicted_interface = max(0.0, min(1.0, 0.5 + 2.5 * fav))
    model_noise = _stable_unit(seq, salt="insilico")
    score = 0.45 * predicted_interface + 0.35 * heuristic + 0.20 * model_noise
    return max(0.0, min(1.0, score))


def predicted_kd_nm(seq: str) -> float:
    """Map an in-silico score to a predicted K_D in nanomolar (lower = tighter)."""
    return _fitness_to_kd(insilico_score(seq))


def _fitness_to_kd(fitness: float) -> float:
    # fitness 1.0 -> ~3 nM (very tight); fitness 0.0 -> ~5000 nM (weak).
    exponent = 0.5 + (1.0 - fitness) * 3.2
    return round(10.0 ** exponent, 2)


# --------------------------------------------------------------------------- #
# Sequence generation (Design step)
# --------------------------------------------------------------------------- #

def mutate(seq: str, n_mutations: int, rng: random.Random) -> str:
    """Apply ``n_mutations`` random point substitutions to ``seq``."""
    chars = list(seq)
    # Never touch the first residue (Met start codon), keep it realistic.
    positions = list(range(1, len(chars)))
    rng.shuffle(positions)
    for pos in positions[:n_mutations]:
        chars[pos] = rng.choice(AMINO_ACIDS)
    return "".join(chars)


def generate_variants(
    parents: list[str],
    n: int,
    mutation_rate: float = 0.06,
    seed: int = 0,
) -> list[str]:
    """Generate ``n`` unique variants by mutating around the ``parents``.

    ``mutation_rate`` is the expected fraction of residues changed per variant.
    This stands in for a generative design model (ProteinMPNN/ESM); it is
    intentionally simple so the demo is fast and reproducible.
    """
    rng = random.Random(seed)
    out: list[str] = []
    seen: set[str] = set(parents)
    attempts = 0
    while len(out) < n and attempts < n * 50:
        attempts += 1
        parent = rng.choice(parents)
        k = max(1, round(len(parent) * mutation_rate))
        variant = mutate(parent, k, rng)
        if variant not in seen:
            seen.add(variant)
            out.append(variant)
    return out
