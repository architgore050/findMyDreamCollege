"""NABH hospital directory (spec §4.4.2).

``nabh.co.in`` is a parked domain; the current site is ``nabh.co`` and the
directory is a WordPress ``admin-ajax.php`` **POST** returning JSON
(``html``, ``pagination``, ``mapData``).

Verified against the live site on 2026-11-16:

* ``POST admin-ajax.php`` with an urlencoded body ``action=get_hospitals&page=1``
  returns HTTP 200 and a ~107 KB JSON document (866 pages / 21,637 results,
  25 rows per page). A ``GET`` with ``action`` in the query string also works,
  but only the **POST body** carries ``page``: a GET page=2 is silently served
  as page 1. The ref therefore records ``request_method=POST`` and the body in
  ``metadata``, and repeats the safe parameters in the URL so a method-blind
  GET still returns valid JSON instead of HTTP 400. Consequence: until the
  crawl layer honours ``request_method``/``request_body``, a page>1 ref
  degrades to page 1 (HTTP 200) rather than failing -- the POST body is the
  only exact pagination signal.
* A bare ``GET admin-ajax.php`` with no ``action`` is what WordPress answers
  with HTTP 400 and the body ``0``. That is the production failure this module
  now reports loudly instead of swallowing.
* No nonce, cookie, ``X-Requested-With`` or CAPTCHA is needed on this path; the
  honest User-Agent plus the site Referer is enough. robots.txt explicitly
  allows ``/wp-admin/admin-ajax.php``.
* ``selectState`` is accepted but ignored server-side (a state-filtered POST
  returns the same 21,637 global results), so the registry marks state
  filtering unsupported and the default sweep is the unfiltered listing. A
  registry-supplied ``states`` list is honoured only when
  ``state_filter_supported`` is true.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from typing import Any
from urllib.parse import urlencode

from fmd.adapters.base import BaseAdapter, SourceUnavailable, compact_record
from fmd.discover.base import DocumentRef

#: Row delimiter in the directory ``html`` fragment.
_ROW_MARKER = "hp-body-row"
#: Collapse all whitespace (NABH pads names with tabs and newlines).
_WS = re.compile(r"\s+")
_TAG = re.compile(r"<[^>]+>")
_COLUMN = r'hs-col hs-col-{index}\b[^"]*"[^>]*>(.*?)</div>'
_NAME_ANCHOR = re.compile(
    r'<a title="Download Certificate[^"]*"\s+href="([^"]*)"[^>]*>(.*?)</a>',
    re.DOTALL,
)
#: How much of an error body is safe to put in a ``SourceUnavailable``.
_ERROR_BODY_LIMIT = 400


def _text(fragment: str) -> str:
    """Visible text of an HTML fragment, whitespace-normalised."""
    return _WS.sub(" ", _TAG.sub(" ", fragment)).strip()


def _key(name: str) -> str:
    """Join key between ``html`` rows and ``mapData`` markers."""
    return _WS.sub(" ", str(name).replace("\t", " ")).strip().casefold()


def _truncate(body: str, limit: int = _ERROR_BODY_LIMIT) -> str:
    """One-line, bounded error body so a 400 is diagnosable from the log."""
    collapsed = _WS.sub(" ", body or "").strip()
    if not collapsed:
        return "<empty body>"
    return collapsed[:limit]


class NabhAdapter(BaseAdapter):
    name = "nabh"

    # ------------------------------------------------------------------ parsing
    def _payload(self, text: str) -> dict:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise SourceUnavailable("invalid_json", detail=_truncate(text)) from exc
        if not isinstance(payload, dict):
            raise SourceUnavailable("invalid_json", detail="expected a JSON object")
        return payload

    def parse_states(self, text: str) -> list[str]:
        """Strip the leading spaces NABH returns and drop empty entries."""
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise SourceUnavailable("invalid_json", detail=_truncate(text)) from exc
        if isinstance(payload, dict):
            items = payload.get("states", [])
        elif isinstance(payload, list):
            items = payload
        else:
            raise SourceUnavailable("invalid_json", detail="expected a JSON array")
        return [str(item).strip() for item in items if str(item).strip()]

    def _html_rows(self, html: str) -> list[dict[str, Any]]:
        """Rows from the directory ``html`` fragment (authoritative listing)."""
        rows: list[dict[str, Any]] = []
        for chunk in html.split(_ROW_MARKER)[1:]:
            match = _NAME_ANCHOR.search(chunk)
            name = _text(match.group(2)) if match else ""
            if not name:
                continue
            address = re.search(_COLUMN.format(index=2), chunk, re.DOTALL)
            acc_no = re.search(_COLUMN.format(index=4), chunk, re.DOTALL)
            rows.append(
                {
                    "name": name,
                    "address": _text(address.group(1)) if address else "",
                    "acc_no": _text(acc_no.group(1)) if acc_no else "",
                    "certificate_url": match.group(1) if match else "",
                }
            )
        return rows

    def parse_hospitals(self, text: str) -> list[dict[str, Any]]:
        """Parse a result page into hospital rows.

        ``html`` has one row per hospital (25/page); ``mapData`` only carries
        coordinates and covers a subset (23/25 on page 1), so the HTML rows are
        authoritative and coordinates are joined in by name. If the server ever
        returns ``mapData`` without ``html``, fall back to the markers.
        """
        payload = self._payload(text)
        markers: dict[str, dict[str, Any]] = {}
        for item in payload.get("mapData") or []:
            if isinstance(item, dict) and item.get("name"):
                markers[_key(item["name"])] = item

        rows = self._html_rows(str(payload.get("html") or ""))
        records: list[dict[str, Any]] = []
        if rows:
            for row in rows:
                marker = markers.get(_key(row["name"])) or {}
                merged = {
                    **row,
                    "lat": marker.get("lat"),
                    "lng": marker.get("lng"),
                }
                record = compact_record(
                    merged,
                    field_whitelist=(
                        "name",
                        "address",
                        "acc_no",
                        "certificate_url",
                        "lat",
                        "lng",
                    ),
                    pii_fields=self.pii_fields,
                )
                records.append(
                    {key: value for key, value in record.items() if value not in (None, "")}
                )
            return records

        for item in markers.values():
            record = compact_record(
                item,
                field_whitelist=("name", "address", "lat", "lng"),
                pii_fields=self.pii_fields,
            )
            records.append({key: value for key, value in record.items() if value not in (None, "")})
        return records

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
        ajax_url = self.param("ajax_url") or getattr(self.source, "base_url", "")
        if not page_param or not action or not ajax_url:
            raise SourceUnavailable(
                "missing_registry_param",
                detail="nabh needs params.ajax_url, params.page_param and params.action_hospitals",
            )
        body: dict[str, Any] = {"action": action}
        if state is not None and state_param:
            body[state_param] = state
        refs: list[DocumentRef] = []
        for page in range(1, min(total_pages, max_pages) + 1):
            page_body = {**body, page_param: page}
            refs.append(
                self.ref(
                    self._page_url(str(ajax_url), page_body),
                    metadata={
                        "request_method": "POST",
                        "request_body": page_body,
                        "request_headers": self._request_headers(),
                        "page": page,
                        "state": state,
                    },
                )
            )
        return refs

    # ------------------------------------------------------------------ request
    @staticmethod
    def _page_url(ajax_url: str, body: dict[str, Any]) -> str:
        """URL carrying the safe parameters, so a method-blind GET is not a 400.

        WordPress reads ``action`` from the query string, but ``page`` only from
        the POST body; the body in ``metadata`` is what makes pagination exact.
        """
        separator = "&" if "?" in ajax_url else "?"
        return f"{ajax_url}{separator}{urlencode(body)}"

    def _request_headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        listing_url = self.param("listing_url")
        if listing_url:
            headers["Referer"] = str(listing_url)
        return headers

    def _checked_text(self, url: str, response: Any) -> str:
        """Never turn a failure into an empty success; keep the body as evidence."""
        if response.status_code in (403, 429, 503):
            raise SourceUnavailable(
                "blocked",
                url=url,
                detail=f"http_{response.status_code}: {_truncate(response.text)}",
            )
        if response.status_code >= 400:
            raise SourceUnavailable(
                f"http_{response.status_code}",
                url=url,
                detail=_truncate(response.text),
            )
        if not (response.content or b""):
            raise SourceUnavailable("empty_response", url=url)
        return response.text

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

        headers = {"User-Agent": get_settings().user_agent, **self._request_headers()}
        response = httpx.post(
            url,
            data=body,
            headers=headers,
            follow_redirects=True,
            timeout=30.0,
        )
        return self._checked_text(url, response)

    def fetch(self) -> Iterable[DocumentRef]:
        ajax_url = self.param("ajax_url") or getattr(self.source, "base_url", None)
        action_states = self.param("action_states")
        action = self.param("action_hospitals")
        state_param = self.param("state_param")
        page_param = self.param("page_param")
        if not ajax_url or not action or not page_param:
            raise SourceUnavailable(
                "missing_registry_param",
                detail="nabh needs ajax_url/action_hospitals/page_param",
            )
        states: list[str] = []
        # The live endpoint ignores selectState; only sweep per state when the
        # registry says the server actually supports it. No state list is
        # hardcoded here, and filters come from the registry only.
        if bool(self.param("state_filter_supported", False)):
            states = list(self.param("states") or [])
            if not states and action_states:
                states = self.parse_states(
                    self._post_json(str(ajax_url), {"action": action_states})
                )
            if not states:
                raise SourceUnavailable("no_states", detail="NABH state list is empty")
        refs: list[DocumentRef] = []
        for state in states or [None]:
            body: dict[str, Any] = {"action": action}
            if state is not None and state_param:
                body[state_param] = state
            body[page_param] = 1
            text = self._post_json(str(ajax_url), body)
            refs.extend(self.parse_listing(text, state=state))
        return refs
