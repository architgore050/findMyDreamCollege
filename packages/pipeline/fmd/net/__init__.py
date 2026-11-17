"""Outbound network policy (timeouts, retries, error taxonomy).

Single source of truth for every HTTP interaction in the pipeline. Import
:mod:`fmd.net.policy` from a call site rather than constructing a bare
``httpx.Client``: one place defines connect/read/total timeouts, retry budgets,
backoff with jitter, ``Retry-After`` handling and the outage circuit breaker.

Nothing here makes a request at import time, and every policy value can be
overridden per call so offline tests stay deterministic.
"""

from __future__ import annotations

from fmd.net.policy import (
    BROWSER_ENV_DEFAULTS,
    DEFAULT_RETRY,
    DEFAULT_TIMEOUTS,
    NETWORK_UNAVAILABLE,
    OUTAGE_KINDS,
    RETRYABLE_STATUSES,
    BrowserPolicy,
    NetworkFault,
    NetworkGuard,
    NetworkUnavailable,
    RetryPolicy,
    Timeouts,
    backoff_delay,
    classify_browser_error,
    classify_exception,
    classify_status,
    hf_download_env,
    is_local_endpoint,
    llm_timeouts,
    make_client,
    parse_retry_after,
    request_with_retries,
)

__all__ = [
    "BROWSER_ENV_DEFAULTS",
    "DEFAULT_RETRY",
    "DEFAULT_TIMEOUTS",
    "NETWORK_UNAVAILABLE",
    "OUTAGE_KINDS",
    "RETRYABLE_STATUSES",
    "BrowserPolicy",
    "NetworkFault",
    "NetworkGuard",
    "NetworkUnavailable",
    "RetryPolicy",
    "Timeouts",
    "backoff_delay",
    "classify_browser_error",
    "classify_exception",
    "classify_status",
    "hf_download_env",
    "is_local_endpoint",
    "llm_timeouts",
    "make_client",
    "parse_retry_after",
    "request_with_retries",
]
