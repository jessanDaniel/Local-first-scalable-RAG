"""One HTTP POST helper shared by the providers: timeouts, retries, backoff.

Retried: connection failures, 429, and 5xx. Not retried: read timeouts (a
model that took too long will usually take too long again) and other 4xx
(bad request, wrong model name, bad key).
"""

import time

import requests

from rag.llm.interface import LLMError

CONNECT_TIMEOUT_SECONDS = 5.0
RETRY_STATUS_CODES = {429, 500, 502, 503, 504}
MAX_BACKOFF_SECONDS = 30.0


def backoff_seconds(attempt: int, retry_after: str | None) -> float:
    if retry_after is not None:
        try:
            return min(float(retry_after), MAX_BACKOFF_SECONDS)
        except ValueError:
            pass
    return min(2.0**attempt, MAX_BACKOFF_SECONDS)


def post_json(
    url: str,
    payload: dict,
    *,
    timeout_seconds: float,
    max_retries: int,
    headers: dict | None = None,
) -> dict:
    for attempt in range(max_retries + 1):
        is_last = attempt == max_retries
        try:
            response = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=(CONNECT_TIMEOUT_SECONDS, timeout_seconds),
            )
        except requests.ConnectionError as exc:
            if is_last:
                raise LLMError(f"POST {url}: connection failed after {attempt + 1} attempts: {exc}") from exc
            time.sleep(backoff_seconds(attempt, None))
            continue
        except requests.Timeout as exc:
            raise LLMError(f"POST {url}: no response within {timeout_seconds}s") from exc

        if response.status_code in RETRY_STATUS_CODES and not is_last:
            time.sleep(backoff_seconds(attempt, response.headers.get("retry-after")))
            continue
        if not response.ok:
            raise LLMError(f"POST {url}: HTTP {response.status_code}: {response.text[:500]}")
        return response.json()

    raise AssertionError("unreachable: the last attempt always returns or raises")
