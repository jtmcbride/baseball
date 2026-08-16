"""Polite HTTP: token-bucket rate limiting plus bounded retry with jitter.

Every source client goes through `RateLimitedClient`. The limits themselves come
from settings, not from call sites, so slowing the crawler down is an env change.

This matters beyond etiquette: Savant and FanGraphs will start returning 429s or
empty bodies under load, and an unthrottled backfill that half-succeeds produces a
silently incomplete dataset — the worst possible failure mode for a project whose
whole output is downstream statistics.
"""

from __future__ import annotations

import random
import threading
import time
from typing import Any

import httpx

from bbcore.logging import get_logger

log = get_logger(__name__)

RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}


class TokenBucket:
    """Simple thread-safe bucket. `rps` may be < 1 for very slow crawls."""

    def __init__(self, rps: float, burst: int = 1) -> None:
        if rps <= 0:
            raise ValueError("rps must be positive")
        self.rps = rps
        self.capacity = max(1, burst)
        self._tokens = float(self.capacity)
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                self._tokens = min(self.capacity, self._tokens + (now - self._last) * self.rps)
                self._last = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                deficit = (1.0 - self._tokens) / self.rps
            time.sleep(deficit)


class RateLimitedClient:
    """httpx client with a bucket in front and retry/backoff behind."""

    def __init__(
        self,
        *,
        rps: float,
        user_agent: str,
        timeout_s: float = 60.0,
        max_retries: int = 5,
        base_url: str = "",
        headers: dict[str, str] | None = None,
    ) -> None:
        self.bucket = TokenBucket(rps)
        self.max_retries = max_retries
        self._client = httpx.Client(
            base_url=base_url,
            timeout=httpx.Timeout(timeout_s),
            follow_redirects=True,
            headers={"User-Agent": user_agent, **(headers or {})},
        )

    def get(self, url: str, params: dict[str, Any] | None = None) -> httpx.Response:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            self.bucket.acquire()
            try:
                resp = self._client.get(url, params=params)
            except (httpx.TransportError, httpx.TimeoutException) as exc:
                last_exc = exc
                if attempt == self.max_retries:
                    break
                self._sleep_backoff(attempt, reason=type(exc).__name__)
                continue

            if resp.status_code in RETRYABLE_STATUS:
                if attempt == self.max_retries:
                    resp.raise_for_status()
                # Honor Retry-After when the server tells us how long to wait.
                retry_after = _parse_retry_after(resp)
                self._sleep_backoff(
                    attempt, reason=f"HTTP {resp.status_code}", override=retry_after
                )
                continue

            resp.raise_for_status()
            return resp

        assert last_exc is not None
        raise last_exc

    def _sleep_backoff(self, attempt: int, *, reason: str, override: float | None = None) -> None:
        delay = override if override is not None else min(60.0, 2.0**attempt) + random.uniform(0, 1)
        log.warning(
            "retry %d/%d after %s — sleeping %.1fs", attempt + 1, self.max_retries, reason, delay
        )
        time.sleep(delay)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> RateLimitedClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _parse_retry_after(resp: httpx.Response) -> float | None:
    raw = resp.headers.get("Retry-After")
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        return None
