"""NABH hospital directory (spec §4.4.2).

``nabh.co.in`` is a parked domain; the current site is ``nabh.co`` and the
directory is a WordPress ``admin-ajax.php`` **POST** returning JSON
(``html``, ``pagination``, ``mapData``). No nonce or CAPTCHA is needed for the
search path. The directory is paginated, so the adapter emits one ref per page
with the POST method/body in ``metadata`` for the fetch layer.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any
from urllib.parse import urlencode

from fmd.adapters.base import BaseAdapter, SourceUnavailable, compact_record
from fmd.discover.base import DocumentRef


class NabhAdapter(BaseAdapter):
    name = "nabh"

    # ------------------------------------------------------------------ parsing
    def _payload(self, text: str) -> dict:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise SourceUnavailable("invalid_json", detail=str(exc)) from exc
        if not isinstance(payload, dict):
            raise SourceUnavailable("invalid_json", detail="expected a JSON object")
        return payload

    def parse_states(self, text: str) -> list[str]:
        """Strip the leading spaces NABH returns and drop empty entries."""
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise SourceUnavailable("invalid_json", detail=str(exc)) from exc
        if isinstance(payload, dict):
            items = payload.get("states", [])
        elif isinstance(payload, list):
            items = payload
        else:
            raise SourceUnavailable("invalid_json", detail="expected a JSON array")
        return [str(item).strip() for item in items if str(item).strip()]

    def parse_hospitals(self, text: str) -> list[dict[str, Any]]:
        payload = self._payload(text)
        rows: list[dict[str, Any]] = []
        for item in payload.get("mapData") or []:
            if not isinstance(item, dict):
                continue
            rows.append(
                compact_record(
                    item,
                    field_whitelist=("name", "address", "lat", "lng"),
                    pii_fields=self.pii_fields,
                )
            )
        return rows

    def parse_listing(self, text: str, *, state: str | None = None) -> list[DocumentRef]:
        """Emit one POST ref per result page (not per hospital row)."""
        payload = self._payload(text)
        pagination = payload.get("pagination") or {}
        if not isinstance(pagination, dict):
            pagination = {}
        total_results = int(pagination.get("total_results") or 0)
        if total_results <= 0:
            return []
        total_pages = int(pagination.get("total_pages") or 1)
        max_pages = int(self.param("max_pages", total_pages) or total_pages)
        page_param = self.param("page_param")
        state_param = self.param("state_param")
        action = self.param("action_hospitals")
        if not page_param or not action:
            raise SourceUnavailable(
                "missing_registry_param",
                detail="nabh needs params.page_param and params.action_hospitals",
            )
        body: dict[str, Any] = {"action": action}
        if state is not None and state_param:
            body[state_param] = state
        refs: list[DocumentRef] = []
        for page in range(1, min(total_pages, max_pages) + 1):
            page_body = {**body, page_param: page}
            refs.append(
                self.ref(
                    str(self.param("ajax_url") or getattr(self.source, "base_url", "")),
                    metadata={
                        "request_method": "POST",
                        "request_body": page_body,
                        "page": page,
                        "state": state,
                    },
                )
            )
        return refs

    # -------------------------------------------------------------------- fetch
    def _post_json(self, url: str, body: dict[str, Any]) -> str:
        """POST after the same robots/politeness checks the GET ladder applies."""
        robots = self.ladder.robots.decision(url)
        if not robots.allowed:
            raise SourceUnavailable("robots_disallow", url=url, detail=robots.reason)
        delay = self.ladder.politeness.note_request(url)
        if delay > 0:
            self.ladder.sleeper(delay)
        import httpx

        from fmd.settings import get_settings

        response = httpx.post(
            url,
            data=body,
            headers={"User-Agent": get_settings().user_agent},
            follow_redirects=True,
            timeout=30.0,
        )
        if response.status_code in (403, 429, 503):
            raise SourceUnavailable("blocked", url=url, detail=f"http_{response.status_code}")
        if response.status_code >= 400:
            raise SourceUnavailable(f"http_{response.status_code}", url=url)
        return response.text

    def fetch(self) -> Iterable[DocumentRef]:
        ajax_url = self.param("ajax_url") or getattr(self.source, "base_url", None)
        action_states = self.param("action_states")
        action = self.param("action_hospitals")
        state_param = self.param("state_param")
        page_param = self.param("page_param")
        if not ajax_url or not action or not state_param or not page_param:
            raise SourceUnavailable(
                "missing_registry_param",
                detail="nabh needs ajax_url/action_hospitals/state_param/page_param",
            )
        states = list(self.param("states") or [])
        if not states and action_states:
            states = self.parse_states(
                self.fetch_text(f"{ajax_url}?{urlencode({'action': action_states})}")
            )
        if not states:
            raise SourceUnavailable("no_states", detail="NABH state list is empty")
        refs: list[DocumentRef] = []
        for state in states:
            body = {"action": action, state_param: state, page_param: 1}
            text = self._post_json(ajax_url, body)
            refs.extend(self.parse_listing(text, state=state))
        return refs
