"""Ingestion of *inventory* documents into entities and crawl seeds.

An inventory payload is not evidence: it names institutions and sometimes the
website they publish. Sending it to extraction is what made the first live crawl
produce zero claims. This package parses the payload into rows a human can check,
and exposes the website domains as institution-site crawl seeds.

Everything is registry-driven:

* the field names live in the source entry (``params.inventory.fields``);
* the country and the minimum fields live in ``params.inventory``;
* no institution name, state, host or field name is hardcoded here.

Only :mod:`fmd.ingest.inventory` (pure parsing) and
:mod:`fmd.ingest.targeting` (pure priority scoring) live here. Database writes
are orchestrated by :mod:`fmd.orchestrate.ingest`.
"""

from fmd.ingest.inventory import (
    InventoryFieldMap,
    InventoryRecord,
    domain_from_website,
    parse_inventory,
    records_from_payload,
)
from fmd.ingest.targeting import (
    TargetWeights,
    priority_for,
    select_targets,
)

__all__ = [
    "InventoryFieldMap",
    "InventoryRecord",
    "TargetWeights",
    "domain_from_website",
    "parse_inventory",
    "priority_for",
    "records_from_payload",
    "select_targets",
]
