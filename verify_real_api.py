"""Verify the RealFoundry integration against the live Adaptyv Foundry API.

Runs three tiers, stopping at whatever your credentials allow:

  1. No credentials  — payload construction, validation rules and the response
                       parsers are checked against the published contract, and
                       the live API's health endpoint is pinged.
  2. Viewer token    — real authenticated reads: whoami + the live target
                       catalog. Costs nothing.
  3. Member token    — still read-only here. This script never creates or
                       submits an experiment, so it can never spend money.

    python verify_real_api.py
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

from adaptyv_copilot import foundry
from adaptyv_copilot.api import AdaptyvClient, AdaptyvAPIError, get_token
from adaptyv_copilot.foundry import FoundryError, RealFoundry, SimulatedFoundry

load_dotenv()

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name: str, fn) -> None:
    try:
        detail = fn() or ""
        results.append((PASS, name, str(detail)))
    except Exception as exc:  # noqa: BLE001 - this is a test harness
        results.append((FAIL, name, f"{type(exc).__name__}: {exc}"))


SEQ = "MGSEIEELRKRAEELAKKNPSPEVLKLLQEAQKLLKENPSDPELLQLAKKVAELAKKLGGSGE"
UUID = "019a03da-b87f-7e15-8b02-cef171c9871d"


# --------------------------------------------------------------------------- #
# Tier 1 — offline contract checks (no token needed)
# --------------------------------------------------------------------------- #

class _SpecOnly(RealFoundry):
    """RealFoundry without a network client, to inspect built payloads."""

    def __init__(self):  # noqa: D107 - deliberately skips AdaptyvClient
        self._client = None


def tier1() -> None:
    spec_builder = _SpecOnly()

    def affinity_spec():
        spec = spec_builder._spec("affinity", {"cand_001": SEQ}, UUID, None, 3)
        assert spec["experiment_type"] == "affinity", spec
        assert spec["method"] == "bli", "method must default to bli"
        assert spec["target_id"] == UUID, spec
        assert spec["sequences"] == {"cand_001": SEQ}, spec
        assert spec["n_replicates"] == 3, spec
        return f"keys={sorted(spec)}"

    check("affinity spec matches the API contract", affinity_spec)

    def expression_spec():
        spec = spec_builder._spec("expression", {"cand_001": SEQ}, None, None, 1)
        assert "method" not in spec, "method is rejected for expression"
        assert "target_id" not in spec, "target_id is rejected for expression"
        return f"keys={sorted(spec)}"

    check("expression spec omits method and target_id", expression_spec)

    def method_rejected():
        try:
            spec_builder._spec("expression", {"c": SEQ}, None, "bli", 1)
        except FoundryError:
            return "correctly rejected"
        raise AssertionError("method should be rejected for expression")

    check("method rejected on non-binding assay", method_rejected)

    def target_required():
        try:
            spec_builder._spec("affinity", {"c": SEQ}, None, None, 1)
        except FoundryError:
            return "correctly rejected"
        raise AssertionError("affinity without target_id should fail")

    check("target_id required for affinity", target_required)

    def bad_method():
        try:
            spec_builder._spec("affinity", {"c": SEQ}, UUID, "elisa", 1)
        except FoundryError:
            return "correctly rejected"
        raise AssertionError("invalid method should fail")

    check("invalid method rejected", bad_method)

    def bad_residues():
        try:
            spec_builder._spec("expression", {"c": "MGSXZ"}, None, None, 1)
        except FoundryError:
            return "correctly rejected"
        raise AssertionError("invalid residues should fail")

    check("non-standard residues rejected", bad_residues)

    def all_types():
        for t in foundry.EXPERIMENT_TYPES:
            kwargs = {
                "target_id": UUID if t in foundry._TARGET_REQUIRED else None,
                "method": "bli" if t in foundry._METHOD_REQUIRED else None,
            }
            seqs = {f"cand_{i:03d}": SEQ for i in range(4)}
            spec_builder._spec(t, seqs, kwargs["target_id"], kwargs["method"], 1)
        return f"{len(foundry.EXPERIMENT_TYPES)} types build cleanly"

    check("every real experiment_type builds a valid spec", all_types)

    def cost_parse():
        body = {
            "breakdown": {
                "total_cents": 149700,
                "pricing_version": "v1_2026-01-20",
                "assay": {
                    "experiment_type": "affinity",
                    "sequence_count": 3,
                    "n_replicates": 2,
                    "unit_price_cents": 24950,
                    "subtotal_cents": 149700,
                    "replicate_price_cents": 0,
                },
            },
            "warnings": [],
        }
        out = foundry._parse_cost(body, "affinity", 3, 2)
        assert out["total_usd"] == 1497.0, out
        assert out["unit_price_usd"] == 249.5, out
        assert out["currency"] == "USD"
        return f"{out['total_usd']} USD from {body['breakdown']['total_cents']} cents"

    check("cost estimate converts cents -> USD", cost_parse)

    def results_parse():
        items = [{
            "id": "r1", "experiment_id": "e1", "result_type": "affinity",
            "summary": [
                {
                    "result_type": "affinity",
                    "sequence": {"name": "cand_002", "aa_string": SEQ},
                    "kd_mean": 4.2e-9,          # 4.2 nM expressed in molar
                    "kd_units": "M",
                    "binding_strength": "strong",
                    "binding": "true",
                    "fit_quality": "good",
                    "positive_control": False,
                },
                {
                    "result_type": "affinity",
                    "sequence": {"name": "cand_001", "aa_string": SEQ},
                    "kd_mean": 8.6e-7,          # 860 nM
                    "kd_units": "M",
                    "binding_strength": "weak",
                    "binding": "true",
                    "positive_control": False,
                },
            ],
        }]
        rows = foundry._parse_results(items)
        assert rows[0]["clone_id"] == "cand_002", rows
        assert rows[0]["kd_nM"] == 4.2, rows
        assert rows[1]["kd_nM"] == 860.0, rows
        return "molar -> nM correct, sorted best-first"

    check("affinity results convert molar -> nM", results_parse)

    def thermo_parse():
        items = [{
            "result_type": "thermostability",
            "summary": [{
                "result_type": "thermostability",
                "sequence_name": "cand_004",
                "tm": 68.4,
            }],
        }]
        rows = foundry._parse_results(items)
        assert rows[0]["clone_id"] == "cand_004" and rows[0]["tm_celsius"] == 68.4
        return "tm parsed"

    check("thermostability results parse", thermo_parse)

    def exp_parse():
        body = {
            "id": "exp-abc", "code": "EXP-2026-001", "name": "EGFR panel",
            "status": "in_production", "results_status": "none",
            "experiment_url": "https://foundry.adaptyvbio.com/experiments/exp-abc",
            "experiment_spec": {
                "experiment_type": "affinity", "method": "bli",
                "sequences": {"cand_001": SEQ, "cand_002": SEQ},
                "target": {"name": "EGFR", "target_catalog_id": UUID},
            },
        }
        out = foundry._public_experiment(body)
        assert out["n_sequences"] == 2 and out["status"] == "in_production", out
        assert out["target_name"] == "EGFR" and out["target_id"] == UUID, out
        return f"code={out['code']} status={out['status']}"

    check("experiment response flattens correctly", exp_parse)

    def status_parity():
        sim = SimulatedFoundry()
        exp = sim.create_experiment("t", "affinity", {"cand_001": SEQ},
                                    target_id="egfr", method="bli")
        assert exp["status"] == "draft", exp
        assert sim.submit_experiment(exp["id"])["status"] == "in_production"
        sim.fast_forward_experiment(exp["id"])
        assert sim.get_experiment(exp["id"])["status"] == "done"
        return "draft -> in_production -> done"

    check("simulator uses the real lowercase status enum", status_parity)

    def ordering_guard():
        os.environ.pop("ADAPTYV_ALLOW_ORDERING", None)
        assert not foundry.ordering_enabled()
        try:
            foundry._require_ordering("create an experiment")
        except FoundryError as exc:
            assert "ADAPTYV_ALLOW_ORDERING" in str(exc)
            os.environ["ADAPTYV_ALLOW_ORDERING"] = "true"
            assert foundry.ordering_enabled(), "opt-in should enable ordering"
            os.environ.pop("ADAPTYV_ALLOW_ORDERING")
            return "blocked by default, enabled on opt-in"
        raise AssertionError("ordering should be blocked by default")

    check("real ordering blocked unless opted in", ordering_guard)

    def no_fast_forward():
        try:
            _SpecOnly().fast_forward_experiment("exp-abc")
        except FoundryError as exc:
            assert "three weeks" in str(exc)
            return "correctly refused"
        raise AssertionError("fast-forward must not work against the real lab")

    check("fast-forward refused in real mode", no_fast_forward)

    def live_health():
        import httpx
        base = os.getenv("ADAPTYV_API_URL", "https://devs.adaptyvbio.com/api/v1")
        r = httpx.get(f"{base}/info/health", timeout=20)
        r.raise_for_status()
        return f"{base}/info/health -> {r.json()}"

    check("live Adaptyv API reachable", live_health)

    def auth_required():
        import httpx
        base = os.getenv("ADAPTYV_API_URL", "https://devs.adaptyvbio.com/api/v1")
        r = httpx.get(f"{base}/targets", params={"limit": 1}, timeout=20)
        assert r.status_code == 401, f"expected 401, got {r.status_code}"
        return "unauthenticated /targets -> 401 as expected"

    check("API rejects unauthenticated reads", auth_required)

    def error_surface():
        import httpx

        class _Boom:
            def request(self, *a, **k):
                return httpx.Response(
                    401, json={"error": "Authentication failed.",
                               "request_id": "abc-123"},
                    request=httpx.Request("GET", "https://x/api/v1/targets"),
                )

        c = AdaptyvClient.__new__(AdaptyvClient)
        c._client = _Boom()
        try:
            c._request("GET", "/targets")
        except AdaptyvAPIError as exc:
            assert "abc-123" in str(exc), exc
            assert exc.status == 401
            return "request_id surfaced in the error"
        raise AssertionError("should have raised")

    check("API errors surface status + request_id", error_surface)


# --------------------------------------------------------------------------- #
# Tier 2 — authenticated live reads (needs a token)
# --------------------------------------------------------------------------- #

def tier2() -> bool:
    if not get_token():
        return False
    client = AdaptyvClient()

    check("live whoami", lambda: str(client.whoami())[:160])

    def live_targets():
        real = RealFoundry(client=client)
        targets = real.list_targets()
        assert targets, "no targets returned"
        first = targets[0]
        assert first.get("id") and first.get("name"), first
        return f"{len(targets)} targets; first: {first['name']} ({first['id']})"

    check("live target catalog", live_targets)

    def live_search():
        real = RealFoundry(client=client)
        hits = real.list_targets(search="EGFR")
        return f"EGFR search -> {len(hits)} hit(s): " + ", ".join(
            h["name"] for h in hits[:3]
        ) if hits else "EGFR search -> 0 hits (catalog may not carry it)"

    check("live target search", live_search)
    return True


def main() -> int:
    print("=" * 74)
    print("TIER 1 — contract + parser checks (no credentials needed)")
    print("=" * 74)
    tier1()
    n1 = len(results)
    for status, name, detail in results:
        print(f"  [{status}] {name}" + (f"\n         {detail}" if detail else ""))

    print()
    print("=" * 74)
    if tier2():
        print("TIER 2 — authenticated live reads")
        print("=" * 74)
        for status, name, detail in results[n1:]:
            print(f"  [{status}] {name}" + (f"\n         {detail}" if detail else ""))
    else:
        print("TIER 2 — SKIPPED: no ADAPTYV_API_TOKEN set")
        print("=" * 74)
        print("  Add a Foundry token to .env to exercise real authenticated reads:")
        print("    ADAPTYV_API_TOKEN=...   (Viewer role is enough, and is free)")
        print("  Get one at https://foundry.adaptyvbio.com")
        print("    -> Organization -> Settings -> Tokens")

    failed = [r for r in results if r[0] == FAIL]
    print()
    print("=" * 74)
    print(f"{len(results) - len(failed)}/{len(results)} checks passed")
    if failed:
        print("FAILURES:")
        for _, name, detail in failed:
            print(f"  - {name}: {detail}")
    print("VERDICT:", "PASS" if not failed else "FAIL")
    print("=" * 74)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
