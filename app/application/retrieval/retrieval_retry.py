from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol, TypeVar


@dataclass(frozen=True, slots=True)
class RetrievalRetryPolicy:
    max_attempts: int = 3
    base_delay_seconds: float = 0.25
    max_delay_seconds: float = 2.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        if self.base_delay_seconds <= 0:
            raise ValueError("base_delay_seconds must be positive")
        if self.max_delay_seconds < self.base_delay_seconds:
            raise ValueError(
                "max_delay_seconds must be at least base_delay_seconds"
            )

    def next_delay(
        self,
        *,
        status_code: int,
        failed_attempt: int,
        retry_after: str | None = None,
    ) -> float | None:
        if status_code not in (429, 503):
            return None
        if failed_attempt < 1:
            raise ValueError("failed_attempt must be positive")
        if failed_attempt >= self.max_attempts:
            return None

        exponential_delay = min(
            self.base_delay_seconds * (2 ** (failed_attempt - 1)),
            self.max_delay_seconds,
        )
        server_delay = _parse_retry_after_seconds(retry_after)
        if server_delay is None:
            return exponential_delay
        return max(exponential_delay, server_delay)


def _parse_retry_after_seconds(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        seconds = float(value.strip())
    except ValueError:
        return None
    if seconds < 0:
        return None
    return seconds


class RetryableHttpResponse(Protocol):
    status_code: int
    headers: Mapping[str, str]


ResponseT = TypeVar("ResponseT", bound=RetryableHttpResponse)


@dataclass(frozen=True, slots=True)
class RetrievalRetryExecution:
    response: RetryableHttpResponse
    attempt_count: int
    delays_seconds: tuple[float, ...]


class RetrievalRetryExecutor:
    def __init__(
        self,
        policy: RetrievalRetryPolicy,
        *,
        sleep: Callable[[float], None],
        random_unit: Callable[[], float],
        jitter_ratio: float = 0.2,
    ) -> None:
        if not 0 <= jitter_ratio <= 1:
            raise ValueError("jitter_ratio must be between zero and one")
        self._policy = policy
        self._sleep = sleep
        self._random_unit = random_unit
        self._jitter_ratio = jitter_ratio

    def execute(
        self,
        operation: Callable[[], ResponseT],
    ) -> RetrievalRetryExecution:
        delays: list[float] = []
        for attempt in range(1, self._policy.max_attempts + 1):
            response = operation()
            delay = self._policy.next_delay(
                status_code=response.status_code,
                failed_attempt=attempt,
                retry_after=_header_value(
                    response.headers,
                    "retry-after",
                ),
            )
            if delay is None:
                return RetrievalRetryExecution(
                    response=response,
                    attempt_count=attempt,
                    delays_seconds=tuple(delays),
                )
            jittered_delay = delay * (
                1 + self._jitter_ratio * self._random_unit()
            )
            delays.append(jittered_delay)
            self._sleep(jittered_delay)
        raise AssertionError("retry loop must always return")


def _header_value(
    headers: Mapping[str, str],
    name: str,
) -> str | None:
    normalized_name = name.lower()
    return next(
        (
            value
            for key, value in headers.items()
            if key.lower() == normalized_name
        ),
        None,
    )
