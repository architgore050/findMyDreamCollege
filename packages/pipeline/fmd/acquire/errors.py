"""Shared acquisition errors.

Acquisition never returns an empty document to pretend a tier ran. A missing
optional dependency, an unconfigured endpoint or an exhausted model host is a
:class:`TierUnavailable`, and the caller decides whether a cheaper tier can
still answer.
"""

from __future__ import annotations


class TierUnavailable(RuntimeError):
    """A tier cannot run on this install/host. Never a silent empty result."""
