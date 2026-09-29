import unittest

from dataclasses import dataclass, field

from app.application.retrieval.retrieval_retry import (
    RetrievalRetryExecutor,
    RetrievalRetryPolicy,
)


@dataclass
class _Response:
    status_code: int
    headers: dict[str, str] = field(default_factory=dict)


class RetrievalRetryPolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = RetrievalRetryPolicy(
            max_attempts=3,
            base_delay_seconds=0.25,
            max_delay_seconds=2.0,
        )

    def test_retries_429_with_server_retry_after(self) -> None:
        delay = self.policy.next_delay(
            status_code=429,
            failed_attempt=1,
            retry_after="1",
        )

        self.assertEqual(1.0, delay)

    def test_does_not_shorten_longer_server_retry_after(self) -> None:
        delay = self.policy.next_delay(
            status_code=429,
            failed_attempt=1,
            retry_after="10",
        )

        self.assertEqual(10.0, delay)

    def test_uses_bounded_exponential_delay_without_header(self) -> None:
        first = self.policy.next_delay(
            status_code=503,
            failed_attempt=1,
        )
        second = self.policy.next_delay(
            status_code=503,
            failed_attempt=2,
        )

        self.assertEqual(0.25, first)
        self.assertEqual(0.5, second)

    def test_stops_after_max_attempts_and_does_not_retry_422(self) -> None:
        self.assertIsNone(
            self.policy.next_delay(
                status_code=429,
                failed_attempt=3,
                retry_after="1",
            )
        )

    def test_executor_retries_then_returns_success_with_jitter(self) -> None:
        responses = iter(
            [
                _Response(429, {"Retry-After": "1"}),
                _Response(503),
                _Response(200),
            ]
        )
        sleeps: list[float] = []
        executor = RetrievalRetryExecutor(
            self.policy,
            sleep=sleeps.append,
            random_unit=lambda: 0.5,
            jitter_ratio=0.2,
        )

        execution = executor.execute(lambda: next(responses))

        self.assertEqual(200, execution.response.status_code)
        self.assertEqual(3, execution.attempt_count)
        self.assertEqual([1.1, 0.55], sleeps)
        self.assertEqual((1.1, 0.55), execution.delays_seconds)

    def test_executor_does_not_retry_non_transient_status(self) -> None:
        calls = 0

        def operation() -> _Response:
            nonlocal calls
            calls += 1
            return _Response(422)

        execution = RetrievalRetryExecutor(
            self.policy,
            sleep=lambda _: self.fail("must not sleep"),
            random_unit=lambda: 0.0,
        ).execute(operation)

        self.assertEqual(1, calls)
        self.assertEqual(1, execution.attempt_count)
        self.assertIsNone(
            self.policy.next_delay(
                status_code=422,
                failed_attempt=1,
            )
        )


if __name__ == "__main__":
    unittest.main()
