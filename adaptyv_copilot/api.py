"""Thin HTTP client for the real Adaptyv Foundry REST API.

Endpoints, payload shapes and enum values here are taken from the public
OpenAPI schema at https://devs.adaptyvbio.com/api/v1/openapi.json (read
2026-07-27). Adaptyv's Python SDK is not published to PyPI, so this talks to
the REST API directly — one less dependency, and the wire format is the
authoritative contract anyway.

Auth: `Authorization: Bearer <token>`. Tokens come from the Foundry portal
(https://foundry.adaptyvbio.com) under Organization -> Settings -> Tokens, and
are role-scoped: a **Viewer** token is read-only, a **Member** token can create
and submit experiments (i.e. spend money).
"""

from __future__ import annotations

import os

import httpx

DEFAULT_BASE_URL = "https://devs.adaptyvbio.com/api/v1"

# Environment variables read for credentials, in order of precedence.
_TOKEN_VARS = ("ADAPTYV_API_TOKEN", "FOUNDRY_API_TOKEN", "ADAPTYVBIO_API_TOKEN")


class AdaptyvAPIError(RuntimeError):
    """An error returned by the Foundry API, or a transport failure."""

    def __init__(self, message: str, *, status: int | None = None,
                 request_id: str | None = None):
        self.status = status
        self.request_id = request_id
        if request_id:
            message = f"{message} (Adaptyv request_id: {request_id})"
        super().__init__(message)


def get_token() -> str | None:
    for var in _TOKEN_VARS:
        token = os.getenv(var)
        if token and token.strip():
            return token.strip()
    return None


class AdaptyvClient:
    """Minimal client covering the endpoints the copilot needs."""

    def __init__(self, token: str | None = None, base_url: str | None = None,
                 timeout: float = 30.0):
        self.base_url = (base_url or os.getenv("ADAPTYV_API_URL")
                         or DEFAULT_BASE_URL).rstrip("/")
        self.token = token or get_token()
        if not self.token:
            raise AdaptyvAPIError(
                "No Adaptyv API token found. Set ADAPTYV_API_TOKEN in your .env "
                "(get one from https://foundry.adaptyvbio.com -> Organization -> "
                "Settings -> Tokens)."
            )
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Accept": "application/json",
                "User-Agent": "adaptyv-copilot/1.0",
            },
        )

    # -- plumbing --------------------------------------------------------- #
    def _request(self, method: str, path: str, **kwargs) -> dict:
        try:
            resp = self._client.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            raise AdaptyvAPIError(f"Could not reach the Adaptyv API: {exc}") from exc

        if resp.status_code >= 400:
            detail, request_id = self._extract_error(resp)
            raise AdaptyvAPIError(
                f"Adaptyv API {resp.status_code} on {method} {path}: {detail}",
                status=resp.status_code,
                request_id=request_id,
            )
        if not resp.content:
            return {}
        try:
            return resp.json()
        except ValueError as exc:
            raise AdaptyvAPIError(
                f"Adaptyv API returned non-JSON on {method} {path}."
            ) from exc

    @staticmethod
    def _extract_error(resp: httpx.Response) -> tuple[str, str | None]:
        try:
            body = resp.json()
        except ValueError:
            return resp.text[:300] or resp.reason_phrase, None
        if not isinstance(body, dict):
            return str(body)[:300], None
        request_id = body.get("request_id")
        # FastAPI validation errors arrive as {"detail": [...]}; the Foundry's
        # own errors use {"error": "..."}.
        detail = body.get("error") or body.get("detail") or body.get("message")
        if isinstance(detail, list):
            detail = "; ".join(
                f"{'.'.join(str(p) for p in item.get('loc', []))}: {item.get('msg')}"
                for item in detail if isinstance(item, dict)
            ) or str(body)[:300]
        return str(detail)[:500], request_id

    def close(self) -> None:
        self._client.close()

    # -- endpoints -------------------------------------------------------- #
    def whoami(self) -> dict:
        """Cheap authenticated call — use it to validate a token."""
        return self._request("GET", "/whoami")

    def list_targets(self, search: str | None = None, limit: int = 50,
                     selfservice_only: bool | None = None) -> list[dict]:
        params: dict = {"limit": max(1, min(limit, 100))}
        if search:
            params["search"] = search
        if selfservice_only is not None:
            params["selfservice_only"] = selfservice_only
        body = self._request("GET", "/targets", params=params)
        return _items(body)

    def get_target(self, target_id: str) -> dict:
        return self._request("GET", f"/targets/{target_id}")

    def cost_estimate(self, experiment_spec: dict) -> dict:
        return self._request(
            "POST", "/experiments/cost-estimate",
            json={"experiment_spec": experiment_spec},
        )

    def create_experiment(self, name: str, experiment_spec: dict,
                          webhook_url: str | None = None) -> dict:
        payload: dict = {"name": name, "experiment_spec": experiment_spec}
        if webhook_url:
            payload["webhook_url"] = webhook_url
        # skip_draft / auto_accept_quote are deliberately never set: the
        # experiment lands in Draft so a human still confirms the quote.
        return self._request("POST", "/experiments", json=payload)

    def submit_experiment(self, experiment_id: str) -> dict:
        return self._request("POST", f"/experiments/{experiment_id}/submit")

    def get_experiment(self, experiment_id: str) -> dict:
        return self._request("GET", f"/experiments/{experiment_id}")

    def get_results(self, experiment_id: str, limit: int = 100) -> list[dict]:
        body = self._request(
            "GET", f"/experiments/{experiment_id}/results",
            params={"limit": max(1, min(limit, 100))},
        )
        return _items(body)


def _items(body) -> list[dict]:
    """Unwrap the API's paginated list envelope: {items, total, count, offset}."""
    if isinstance(body, dict):
        return body.get("items", []) or []
    if isinstance(body, list):
        return body
    return []
