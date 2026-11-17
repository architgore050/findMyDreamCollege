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


class AcquisitionFailed(TierUnavailable):
    """Acquisition could not run for a reason the caller must report.

    Unlike :class:`TierUnavailable` (a cheaper tier may still answer the same
    question), this is a *resource or runtime* failure — a CUDA OOM, a starved
    GPU preflight — where falling back to flat T0 text would manufacture a
    successful extraction with no evidence. The reason is a stable code with
    the ``acquisition_failed:`` prefix; the orchestration layer persists it on
    the document and marks the run failed.
    """

    def __init__(self, reason: str, detail: str | None = None) -> None:
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason}: {detail}" if detail else reason)


#: Prefix shared by every acquisition failure reason (run ledger + document row).
ACQUISITION_FAILED_PREFIX = "acquisition_failed:"
