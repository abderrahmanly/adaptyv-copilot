"""Adaptyv Foundry backends.

Two implementations behind one interface, so the agent's tool layer never has
to know which is in play:

* ``SimulatedFoundry`` — in-memory, deterministic, free. Mirrors the real API's
  vocabulary exactly (same experiment types, same lowercase status enum, same
  BLI/SPR methods) so nothing about the agent's behaviour changes when you
  switch backends.
* ``RealFoundry`` — talks to the live Foundry REST API via ``api.AdaptyvClient``.

Selected by ``ADAPTYV_MODE`` (``simulated`` by default).

Experiment lifecycle (from the public OpenAPI schema, read 2026-07-27):
    draft -> waiting_for_confirmation -> quote_sent -> waiting_for_materials
          -> in_queue -> in_production -> data_analysis -> in_review -> done
    (``canceled`` from draft or waiting_for_confirmation)

Experiment types:
    screening | affinity | thermostability | fluorescence | expression
    | epitope_binning | enzyme_activity

MONEY WARNING
-------------
Against the real backend, ``create_experiment`` + ``submit_experiment`` place a
genuine lab order. Real mode is therefore **read-only unless you opt in** with
ADAPTYV_ALLOW_ORDERING=true. Reads (targets, status, results, cost estimates)
are always allowed and cost nothing.
"""

from __future__ import annotations

import os
import time
import uuid

from . import catalog, science
from .api import AdaptyvAPIError, AdaptyvClient

# Assay types the real API accepts.
EXPERIMENT_TYPES = (
    "screening",
    "affinity",
    "thermostability",
    "fluorescence",
    "expression",
    "epitope_binning",
    "enzyme_activity",
)

# Binding assays require a target and a measurement method (BLI or SPR).
_TARGET_REQUIRED = {"screening", "affinity", "epitope_binning"}
_METHOD_REQUIRED = {"screening", "affinity"}
_METHODS = ("bli", "spr")

# Terminal / ready states from the real status enum.
STATUS_DONE = "done"
STATUS_DRAFT = "draft"

# Per-sequence prices (USD) used by the simulator only. Aligned with Adaptyv's
# public "from $149 / protein" starting point. The real backend returns actual
# pricing from POST /experiments/cost-estimate — these numbers are never used
# in real mode.
PRICE_PER_SEQUENCE = {
    "screening": 149.0,
    "affinity": 249.0,
    "expression": 99.0,
    "thermostability": 129.0,
    "fluorescence": 129.0,
    "epitope_binning": 199.0,
    "enzyme_activity": 179.0,
}


class FoundryError(Exception):
    pass


def _check_type(experiment_type: str) -> None:
    if experiment_type not in EXPERIMENT_TYPES:
        raise FoundryError(
            f"Unknown experiment_type '{experiment_type}'. "
            f"Valid: {', '.join(EXPERIMENT_TYPES)}"
        )


def _check_method(experiment_type: str, method: str | None) -> str | None:
    """Validate the BLI/SPR method against the real API's contract."""
    if experiment_type in _METHOD_REQUIRED:
        method = (method or "bli").lower()
        if method not in _METHODS:
            raise FoundryError(
                f"method must be one of {_METHODS} for '{experiment_type}'."
            )
        return method
    if method:
        raise FoundryError(
            f"experiment_type '{experiment_type}' does not accept a method "
            f"(only {sorted(_METHOD_REQUIRED)} do)."
        )
    return None


def _validate_sequences(sequences: dict[str, str]) -> None:
    if not sequences:
        raise FoundryError("At least one sequence is required.")
    for cid, seq in sequences.items():
        bad = set(seq.upper()) - set(science.AMINO_ACIDS)
        if bad:
            raise FoundryError(
                f"Sequence '{cid}' has invalid residues: {sorted(bad)}"
            )


# --------------------------------------------------------------------------- #
# Simulated backend
# --------------------------------------------------------------------------- #

class SimulatedFoundry:
    """In-memory mock of the Adaptyv Foundry. Deterministic and free."""

    mode = "simulated"

    def __init__(self) -> None:
        self._experiments: dict[str, dict] = {}

    # -- catalog ---------------------------------------------------------- #
    def list_targets(self, search: str | None = None) -> list[dict]:
        return catalog.list_targets(search)

    # -- pricing ---------------------------------------------------------- #
    def cost_estimate(self, experiment_type: str, sequences: dict[str, str],
                      method: str | None = None, n_replicates: int = 1) -> dict:
        _check_type(experiment_type)
        _check_method(experiment_type, method)
        unit = PRICE_PER_SEQUENCE[experiment_type]
        n = len(sequences)
        return {
            "experiment_type": experiment_type,
            "n_sequences": n,
            "n_replicates": n_replicates,
            "unit_price_usd": unit,
            "total_usd": round(unit * n * max(1, n_replicates), 2),
            "currency": "USD",
            "source": "simulated price list",
        }

    # -- experiments ------------------------------------------------------ #
    def create_experiment(self, name: str, experiment_type: str,
                          sequences: dict[str, str], target_id: str | None = None,
                          method: str | None = None, n_replicates: int = 1) -> dict:
        _check_type(experiment_type)
        method = _check_method(experiment_type, method)
        if experiment_type in _TARGET_REQUIRED and not target_id:
            raise FoundryError(
                f"experiment_type '{experiment_type}' requires a target_id"
            )
        _validate_sequences(sequences)

        exp_id = "exp_" + uuid.uuid4().hex[:12]
        self._experiments[exp_id] = {
            "id": exp_id,
            "name": name,
            "experiment_type": experiment_type,
            "method": method,
            "target_id": target_id,
            "n_replicates": n_replicates,
            "sequences": dict(sequences),
            "status": STATUS_DRAFT,
            "results_status": "none",
            "created_at": time.time(),
        }
        return self._public(exp_id)

    def submit_experiment(self, experiment_id: str) -> dict:
        exp = self._get(experiment_id)
        if exp["status"] != STATUS_DRAFT:
            raise FoundryError(
                f"Only draft experiments can be submitted (is '{exp['status']}')."
            )
        exp["status"] = "in_production"
        return self._public(experiment_id)

    def get_experiment(self, experiment_id: str) -> dict:
        return self._public(experiment_id)

    def get_results(self, experiment_id: str) -> dict:
        exp = self._get(experiment_id)
        if exp["status"] != STATUS_DONE:
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
            "status": STATUS_DONE,
            "results_status": "all",
            "results": self._measure(exp),
            "source": "SIMULATED — not measured in a real lab",
        }

    def fast_forward_experiment(self, experiment_id: str) -> dict:
        """DEMO ONLY: skip the ~3-week wet-lab turnaround and mark results ready.

        This has no equivalent in the real API — it exists so a live demo can
        show the 'Learn' step without waiting for real assays.
        """
        exp = self._get(experiment_id)
        if exp["status"] == STATUS_DRAFT:
            raise FoundryError("Submit the experiment before fast-forwarding.")
        exp["status"] = STATUS_DONE
        exp["results_status"] = "all"
        return {
            "experiment_id": experiment_id,
            "status": STATUS_DONE,
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
            "method": exp["method"],
            "target_id": exp["target_id"],
            "n_sequences": len(exp["sequences"]),
            "status": exp["status"],
            "results_status": exp["results_status"],
        }


# --------------------------------------------------------------------------- #
# Real backend
# --------------------------------------------------------------------------- #

def ordering_enabled() -> bool:
    """Whether real-mode ordering (which spends money) is opted into."""
    return (os.getenv("ADAPTYV_ALLOW_ORDERING") or "").strip().lower() in (
        "1", "true", "yes", "on"
    )


class RealFoundry:
    """Talks to the live Adaptyv Foundry over its REST API.

    Reads are always permitted. Writes that cost money (create + submit) are
    refused unless ADAPTYV_ALLOW_ORDERING=true, so pointing the demo at a real
    token cannot silently generate an invoice.
    """

    mode = "real"

    def __init__(self, client: AdaptyvClient | None = None) -> None:
        try:
            self._client = client or AdaptyvClient()
        except AdaptyvAPIError as exc:
            raise FoundryError(str(exc)) from exc

    # -- catalog ---------------------------------------------------------- #
    def list_targets(self, search: str | None = None) -> list[dict]:
        with _wrap():
            targets = self._client.list_targets(search=search)
        out = []
        for t in targets:
            note_bits = [b for b in (t.get("vendor_name"), t.get("catalog_number")) if b]
            out.append({
                "id": t.get("id"),
                "name": t.get("name"),
                "full_name": t.get("name"),
                "uniprot_id": t.get("uniprot_id"),
                "url": t.get("url"),
                "note": " · ".join(note_bits) or "From the live Adaptyv catalog.",
            })
        return out

    # -- pricing ---------------------------------------------------------- #
    def cost_estimate(self, experiment_type: str, sequences: dict[str, str],
                      method: str | None = None, n_replicates: int = 1,
                      target_id: str | None = None) -> dict:
        spec = self._spec(experiment_type, sequences, target_id, method, n_replicates)
        with _wrap():
            body = self._client.cost_estimate(spec)
        return _parse_cost(body, experiment_type, len(sequences), n_replicates)

    # -- experiments ------------------------------------------------------ #
    def create_experiment(self, name: str, experiment_type: str,
                          sequences: dict[str, str], target_id: str | None = None,
                          method: str | None = None, n_replicates: int = 1) -> dict:
        _require_ordering("create an experiment")
        spec = self._spec(experiment_type, sequences, target_id, method, n_replicates)
        with _wrap():
            body = self._client.create_experiment(name=name, experiment_spec=spec)
        return _public_experiment(body)

    def submit_experiment(self, experiment_id: str) -> dict:
        _require_ordering("submit an experiment for processing")
        with _wrap():
            body = self._client.submit_experiment(experiment_id)
        return _public_experiment(body)

    def get_experiment(self, experiment_id: str) -> dict:
        with _wrap():
            return _public_experiment(self._client.get_experiment(experiment_id))

    def get_results(self, experiment_id: str) -> dict:
        with _wrap():
            exp = self._client.get_experiment(experiment_id)
            status = (exp.get("status") or "").lower()
            if status != STATUS_DONE:
                return {
                    "experiment_id": experiment_id,
                    "status": status,
                    "results_status": exp.get("results_status", "none"),
                    "message": (
                        "Results are not ready yet. Real assays take ~3 weeks — "
                        "there is no way to skip the wait against the real lab. "
                        "Register a webhook_url at creation to be notified."
                    ),
                    "results": [],
                }
            items = self._client.get_results(experiment_id)
        return {
            "experiment_id": experiment_id,
            "status": STATUS_DONE,
            "results_status": exp.get("results_status", "all"),
            "results": _parse_results(items),
            "source": "REAL measurements from the Adaptyv Foundry",
        }

    def fast_forward_experiment(self, experiment_id: str) -> dict:
        raise FoundryError(
            "fast_forward_experiment is simulation-only — real assays take about "
            "three weeks and cannot be skipped. Poll check_experiment, or register "
            "a webhook when the experiment is created."
        )

    # -- internals -------------------------------------------------------- #
    def _spec(self, experiment_type: str, sequences: dict[str, str],
              target_id: str | None, method: str | None,
              n_replicates: int) -> dict:
        _check_type(experiment_type)
        method = _check_method(experiment_type, method)
        _validate_sequences(sequences)
        if experiment_type in _TARGET_REQUIRED and not target_id:
            raise FoundryError(
                f"experiment_type '{experiment_type}' requires a target_id (the "
                "catalog UUID from list_targets)."
            )
        spec: dict = {
            "experiment_type": experiment_type,
            # The API keys sequences by a human-readable name; we use the
            # candidate ids so results map straight back onto them.
            "sequences": {cid: seq.upper() for cid, seq in sequences.items()},
        }
        if method:
            spec["method"] = method
        if target_id and experiment_type in _TARGET_REQUIRED:
            spec["target_id"] = target_id
        if n_replicates and experiment_type != "epitope_binning":
            spec["n_replicates"] = max(1, int(n_replicates))
        return spec


class _wrap:
    """Translate API errors into FoundryError so the agent sees one error type."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type and issubclass(exc_type, AdaptyvAPIError):
            raise FoundryError(str(exc)) from exc
        return False


def _require_ordering(action: str) -> None:
    if not ordering_enabled():
        raise FoundryError(
            f"Refusing to {action} against the real Adaptyv Foundry: this places "
            "a paid lab order. Set ADAPTYV_ALLOW_ORDERING=true to allow it. "
            "Reads (targets, status, results, cost estimates) work without it."
        )


def _public_experiment(body: dict) -> dict:
    """Flatten the API's ExpInfo into the same shape SimulatedFoundry returns."""
    spec = body.get("experiment_spec") or {}
    target = spec.get("target") or {}
    sequences = spec.get("sequences") or {}
    return {
        "id": body.get("id"),
        "code": body.get("code"),
        "name": body.get("name"),
        "experiment_type": spec.get("experiment_type"),
        "method": spec.get("method"),
        "target_id": target.get("target_catalog_id"),
        "target_name": target.get("name"),
        "n_sequences": len(sequences) if isinstance(sequences, dict) else None,
        "status": (body.get("status") or "").lower(),
        "results_status": body.get("results_status"),
        "experiment_url": body.get("experiment_url"),
    }


def _parse_cost(body: dict, experiment_type: str, n_sequences: int,
                n_replicates: int) -> dict:
    """Convert CostEstimateResponse (cents) into the app's USD shape."""
    breakdown = body.get("breakdown") or {}
    assay = breakdown.get("assay") or {}
    total_cents = breakdown.get("total_cents")
    unit_cents = assay.get("unit_price_cents")
    out = {
        "experiment_type": assay.get("experiment_type", experiment_type),
        "n_sequences": assay.get("sequence_count", n_sequences),
        "n_replicates": assay.get("n_replicates", n_replicates),
        "unit_price_usd": round(unit_cents / 100, 2) if unit_cents is not None else None,
        "total_usd": round(total_cents / 100, 2) if total_cents is not None else None,
        "currency": "USD",
        "pricing_version": breakdown.get("pricing_version"),
        "source": "live Adaptyv pricing",
    }
    warnings = body.get("warnings") or []
    if body.get("incomplete"):
        warnings = list(warnings) + [
            "Estimate is incomplete — some pricing could not be resolved."
        ]
    if warnings:
        out["warnings"] = warnings
    return out


def _parse_results(items: list[dict]) -> list[dict]:
    """Flatten the API's per-readout results into the app's row shape.

    The API reports K_D in molar; everything else in this app speaks nanomolar,
    so convert (x 1e9) rather than silently mixing units.
    """
    rows: list[dict] = []
    for item in items:
        for summary in item.get("summary") or []:
            rtype = summary.get("result_type") or item.get("result_type")
            row: dict = {"result_type": rtype}

            seq = summary.get("sequence")
            if isinstance(seq, dict):
                row["clone_id"] = seq.get("name") or seq.get("id")
            elif summary.get("sequence_name"):
                row["clone_id"] = summary["sequence_name"]
            elif summary.get("sequence_id"):
                row["clone_id"] = summary["sequence_id"]

            if rtype == "affinity":
                kd_molar = summary.get("kd_mean")
                if kd_molar is not None:
                    row["kd_nM"] = round(kd_molar * 1e9, 3)
                row["binding_strength"] = summary.get("binding_strength")
                row["binds"] = summary.get("binding")
                row["fit_quality"] = summary.get("fit_quality")
                row["expression"] = summary.get("expression")
                if summary.get("positive_control"):
                    row["positive_control"] = True
            elif rtype == "thermostability":
                row["tm_celsius"] = summary.get("tm")

            rows.append({k: v for k, v in row.items() if v is not None})

    rows.sort(key=lambda r: r.get("kd_nM", float("inf")))
    return rows


def get_foundry(mode: str | None = None):
    mode = (mode or os.getenv("ADAPTYV_MODE") or "simulated").lower()
    return RealFoundry() if mode == "real" else SimulatedFoundry()
