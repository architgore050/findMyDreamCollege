"""Shared acquisition errors.

Acquisition never returns an empty document to pretend a tier ran. A missing
optional dependency, an unconfigured endpoint or an exhausted model host is a
:class:`TierUnavailable`, and the caller decides whether a cheaper tier can
still answer.
"""

from __future__ import annotations


class TierUnavailable(RuntimeError):
    """A tier cannot run on this install/host. Never a silent empty result."""


class ArtifactsUnavailable(TierUnavailable):
    """A model-backed tier has no local artifact cache and cannot download.

    Offline mode must never block on Hugging Face or any other artifact host:
    the tier is unavailable *now*, and the reason names the exact cache path and
    the one-time online prefetch command. The dispatcher routes this to T3
    rather than pretending the cheaper tier answered the same question.
    """
