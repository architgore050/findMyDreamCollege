"""Registry source -> adapter instantiation (brief §8, §21).

``config/sources.yaml`` names an adapter; this module is the only place that
turns that name into an object. Adapters are imported lazily so a run without
the heavy crawl extra still fails per-source instead of failing to start.

A source whose adapter has no implementation yet (or whose optional dependency
is missing) raises :class:`AdapterUnavailable`. The orchestrator records that as
one source's error and keeps going: one unimplemented source must not abort the
crawl.
"""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable, Iterable, Mapping
from typing import Any

# Implemented adapters. Keep alphabetical; discovery adapters (searxng,
# common_crawl) are not crawl sources and deliberately absent.
ADAPTER_CLASSES: dict[str, tuple[str, str]] = {
    "aishe": ("fmd.adapters.aishe", "AisheAdapter"),
    "fee_regulator": ("fmd.adapters.fee_regulator", "FeeRegulatorAdapter"),
    "health_university": ("fmd.adapters.health_university", "HealthUniversityAdapter"),
    "institution_site": ("fmd.adapters.institution_site", "InstitutionSiteAdapter"),
    "naac": ("fmd.adapters.naac", "NaacAdapter"),
    "nabh": ("fmd.adapters.nabh", "NabhAdapter"),
    "nirf": ("fmd.adapters.nirf", "NirfAdapter"),
    "pay_matrix": ("fmd.adapters.pay_matrix", "PayMatrixAdapter"),
    "state_council": ("fmd.adapters.state_council", "StateCouncilAdapter"),
    "state_counselling": ("fmd.adapters.state_counselling", "StateCounsellingAdapter"),
    "ugc": ("fmd.adapters.ugc", "UgcAdapter"),
}

AdapterFactory = Callable[[Any], Any]


class AdapterUnavailable(RuntimeError):
    """No usable implementation for the adapter name a source declares."""

    def __init__(self, adapter: str, reason: str) -> None:
        self.adapter = adapter
        self.reason = reason
        super().__init__(f"adapter {adapter!r} unavailable: {reason}")


def implemented_adapters() -> frozenset[str]:
    return frozenset(ADAPTER_CLASSES)


def _construct(
    cls: type,
    source: Any,
    *,
    ladder: Any,
    denylisted_domains: Iterable[str] | None,
) -> Any:
    """Pass only the keyword arguments this adapter actually declares."""
    try:
        signature = inspect.signature(cls)
        accepted = set(signature.parameters)
    except (TypeError, ValueError):  # pragma: no cover - builtins only
        accepted = {"source", "ladder", "denylisted_domains"}
    kwargs: dict[str, Any] = {}
    if "ladder" in accepted:
        kwargs["ladder"] = ladder
    if "denylisted_domains" in accepted and denylisted_domains is not None:
        kwargs["denylisted_domains"] = tuple(denylisted_domains)
    return cls(source, **kwargs)


def build_adapter(
    source: Any,
    *,
    ladder: Any = None,
    denylisted_domains: Iterable[str] | None = None,
    overrides: Mapping[str, Callable[[Any], Any]] | None = None,
) -> Any:
    """Instantiate the adapter for one registry source.

    ``overrides`` is the test/CLI seam: it maps an adapter name (or source id) to
    a factory, so an offline test never needs a real adapter.
    """
    adapter_name = getattr(source, "adapter", None)
    if not adapter_name:
        raise AdapterUnavailable(str(adapter_name), "source declares no adapter")
    if overrides:
        factory = overrides.get(getattr(source, "id", "")) or overrides.get(adapter_name)
        if factory is not None:
            return factory(source)
    entry = ADAPTER_CLASSES.get(str(adapter_name))
    if entry is None:
        raise AdapterUnavailable(str(adapter_name), "no implementation in this build")
    module_name, class_name = entry
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:  # optional crawl extra is absent
        raise AdapterUnavailable(
            str(adapter_name), f"missing optional dependency: {exc.name}"
        ) from exc
    cls = getattr(module, class_name)
    return _construct(
        cls,
        source,
        ladder=ladder,
        denylisted_domains=denylisted_domains,
    )


def call_fetch(adapter: Any, *, domains: Iterable[str] | None = None) -> list[Any]:
    """Call ``adapter.fetch()``, supplying ``domains`` only when declared.

    ``InstitutionSiteAdapter.fetch(domains=...)`` is the one adapter that needs a
    target list; every other adapter reads its targets from the registry.
    """
    fetch = adapter.fetch
    try:
        parameters = inspect.signature(fetch).parameters
    except (TypeError, ValueError):  # pragma: no cover - builtins only
        parameters = {}
    if "domains" in parameters:
        return list(fetch(domains=domains))
    return list(fetch())
