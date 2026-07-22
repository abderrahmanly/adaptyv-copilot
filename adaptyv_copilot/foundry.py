"""Adaptyv Foundry backend.

`SimulatedFoundry` mirrors the shape of the real Adaptyv Foundry API/SDK
(https://docs.adaptyvbio.com/api-reference) so the whole Design->Test->Learn
loop runs offline and for free. `RealFoundry` is a thin swap-in that delegates
to the official `adaptyv-sdk` when ADAPTYV_MODE=real.

Experiment lifecycle (matches the real API):
    Draft -> WaitingForConfirmation -> ... -> InProduction -> DataAnalysis -> Done

Experiment types (matches the real API):
    screening | affinity | expression | thermostability | fluorescence
"""

from __future__ import annotations

import os
import time
import uuid

from . import catalog, science

# Per-sequence prices (USD), aligned with Adaptyv's public "$149 / protein"
# starting point. Purely illustrative in simulated mode.
PRICE_PER_SEQUENCE = {
    "screening": 149.0,
    "affinity": 249.0,
    "expression": 99.0,
    "thermostability": 129.0,
    "fluorescence": 129.0,
}

_TARGET_REQUIRED = {"screening", "affinity"}


class FoundryError(Exception):
    pass


class SimulatedFoundry:
    """In-memory mock of the Adaptyv Foundry. Deterministic and free."""

    mode = "simulated"

    def __init__(self) -> None:
        self._experiments: dict[str, dict] = {}

    # -- catalog ---------------------------------------------------------- #
    def list_targets(self, search: str | None = None) -> list[dict]:
        return catalog.list_targets(search)

    # -- pricing ---------------------------------------------------------- #
    def cost_estimate(self, experiment_type: str, sequences: dict[str, str]) -> dict:
        self._check_type(experiment_type)
        unit = PRICE_PER_SEQUENCE[experiment_type]
        n = len(sequences)
        return {
            "experiment_type": experiment_type,
            "n_sequences": n,
            "unit_price_usd": unit,
            "total_usd": round(unit * n, 2),
            "currency": "USD",
        }

    # -- experiments ------------------------------------------------------ #
    def create_experiment(
        self,
        name: str,
        experiment_type: str,
        sequences: dict[str, str],
        target_id: str | None = None,
    ) -> dict:
        self._check_type(experiment_type)
        if experiment_type in _TARGET_REQUIRED and not target_id:
            raise FoundryError(
                f"experiment_type '{experiment_type}' requires a target_id"
            )
        if not sequences:
            raise FoundryError("At least one sequence is required.")
        for cid, seq in sequences.items():
            bad = set(seq) - set(science.AMINO_ACIDS)
            if bad:
                raise FoundryError(
                    f"Sequence '{cid}' has invalid residues: {sorted(bad)}"
                )

        exp_id = "exp_" + uuid.uuid4().hex[:12]
        self._experiments[exp_id] = {
            "id": exp_id,
            "name": name,
            "experiment_type": experiment_type,
            "target_id": target_id,
            "sequences": dict(sequences),
            "status": "Draft",
            "results_status": "none",
            "created_at": time.time(),
        }
        return self._public(exp_id)

    def submit_experiment(self, experiment_id: str) -> dict:
        exp = self._get(experiment_id)
        if exp["status"] != "Draft":
            raise FoundryError(
                f"Only Draft experiments can be submitted (is '{exp['status']}')."
            )
        exp["status"] = "InProduction"
        return self._public(experiment_id)

    def get_experiment(self, experiment_id: str) -> dict:
        return self._public(experiment_id)

    def get_results(self, experiment_id: str) -> dict:
        exp = self._get(experiment_id)
        if exp["status"] != "Done":
            return {
                "experiment_id": experiment_id,
                "status": exp["status"],
                "results_status": exp["results_status"],
                "message": (
                    "Results are not ready yet. Real assays take ~3 weeks; in "
                    "this demo call fast_forward_experiment to simulate the wait."
                ),
                "results": [],
            }
        return {
            "experiment_id": experiment_id,
            "status": "Done",
            "results_status": "all",
            "results": self._measure(exp),
        }

    def fast_forward_experiment(self, experiment_id: str) -> dict:
        """DEMO ONLY: skip the ~3-week wet-lab turnaround and mark results ready.

        This has no equivalent in the real API — it exists so a live demo can
        show the 'Learn' step without waiting for real assays.
        """
        exp = self._get(experiment_id)
        if exp["status"] == "Draft":
            raise FoundryError("Submit the experiment before fast-forwarding.")
        exp["status"] = "Done"
        exp["results_status"] = "all"
        return {
            "experiment_id": experiment_id,
            "status": "Done",
            "note": "Simulated ~3-week turnaround skipped for the demo.",
        }

    # -- internals -------------------------------------------------------- #
    def _measure(self, exp: dict) -> list[dict]:
        rows = []
        etype = exp["experiment_type"]
        for cid, seq in exp["sequences"].items():
            fit = science.latent_fitness(seq)
            row = {"clone_id": cid}
            if etype in ("screening", "affinity"):
                kd = science._fitness_to_kd(fit)
                row["binds"] = kd < 2000  # screening-style yes/no
                if etype == "affinity":
                    row["kd_nM"] = kd
                    row["signal_confidence"] = round(0.5 + 0.5 * fit, 2)
            if etype in ("expression", "affinity", "screening"):
                # Expression yield (mg/L-ish), correlated with foldability.
                yield_mg_l = round(
                    max(0.0, 120 * science._biophysical_heuristic(seq) - 10), 1
                )
                row["expression_mg_per_L"] = yield_mg_l
                row["expressed"] = yield_mg_l > 5
            if etype == "thermostability":
                row["tm_celsius"] = round(45 + 30 * fit, 1)
            rows.append(row)
        # Sort best-first for affinity so the agent gets a clean ranking.
        if etype == "affinity":
            rows.sort(key=lambda r: r.get("kd_nM", 1e9))
        return rows

    def _check_type(self, experiment_type: str) -> None:
        if experiment_type not in PRICE_PER_SEQUENCE:
            raise FoundryError(
                f"Unknown experiment_type '{experiment_type}'. "
                f"Valid: {', '.join(PRICE_PER_SEQUENCE)}"
            )

    def _get(self, experiment_id: str) -> dict:
        if experiment_id not in self._experiments:
            raise FoundryError(f"No experiment '{experiment_id}'.")
        return self._experiments[experiment_id]

    def _public(self, experiment_id: str) -> dict:
        exp = self._get(experiment_id)
        return {
            "id": exp["id"],
            "name": exp["name"],
            "experiment_type": exp["experiment_type"],
            "target_id": exp["target_id"],
            "n_sequences": len(exp["sequences"]),
            "status": exp["status"],
            "results_status": exp["results_status"],
        }


class RealFoundry:
    """Swap-in that talks to the live Foundry via the official adaptyv-sdk.

    Kept intentionally thin: it shows exactly where real ordering would plug in.
    Requires `pip install adaptyv-sdk` and ADAPTYV_* env vars. Because real
    assays cost money and take weeks, the demo defaults to SimulatedFoundry.
    """

    mode = "real"

    def __init__(self) -> None:
        try:
            from adaptyv import lab  # type: ignore  # noqa: F401
        except Exception as exc:  # pragma: no cover - optional path
            raise FoundryError(
                "ADAPTYV_MODE=real needs the official SDK: `pip install adaptyv-sdk` "
                "and set ADAPTYV_API_KEY / ADAPTYV_API_URL / ADAPTYV_ORGANIZATION_ID. "
                f"(import failed: {exc})"
            )
        self._lab = __import__("adaptyv", fromlist=["lab"]).lab

    # These delegate to the SDK; signatures mirror SimulatedFoundry so the
    # agent's tool layer is backend-agnostic. Left as documented stubs because
    # they hit a paid, live service.
    def list_targets(self, search=None):
        raise FoundryError("RealFoundry.list_targets: wire up client.targets.list().")

    def cost_estimate(self, experiment_type, sequences):
        raise FoundryError("RealFoundry.cost_estimate: wire up experiments.cost_estimate().")

    def create_experiment(self, name, experiment_type, sequences, target_id=None):
        raise FoundryError("RealFoundry.create_experiment: wire up lab.create_experiment().")

    def submit_experiment(self, experiment_id):
        raise FoundryError("RealFoundry.submit_experiment: wire up experiments.submit().")

    def get_experiment(self, experiment_id):
        raise FoundryError("RealFoundry.get_experiment: wire up lab.get_experiment().")

    def get_results(self, experiment_id):
        raise FoundryError("RealFoundry.get_results: wire up experiments.get_results().")

    def fast_forward_experiment(self, experiment_id):
        raise FoundryError("Not available against the real lab — assays take ~3 weeks.")


def get_foundry(mode: str | None = None):
    mode = (mode or os.getenv("ADAPTYV_MODE") or "simulated").lower()
    return RealFoundry() if mode == "real" else SimulatedFoundry()
