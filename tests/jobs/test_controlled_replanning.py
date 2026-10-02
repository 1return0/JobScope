import unittest

from app.application.jobs.controlled_replanning import ControlledReplanPolicy


class ControlledReplanPolicyTest(unittest.TestCase):
    def test_allows_one_read_only_evidence_recovery_attempt(self) -> None:
        decision = ControlledReplanPolicy().assess(
            result_code="insufficient_evidence",
            replan_count=0,
            operation_kind="read",
        )
        self.assertEqual("replan_once", decision.decision)

    def test_never_replans_write_or_second_attempt(self) -> None:
        policy = ControlledReplanPolicy()
        self.assertEqual(
            "stop",
            policy.assess(
                result_code="insufficient_evidence",
                replan_count=0,
                operation_kind="write",
            ).decision,
        )
        self.assertEqual(
            "stop",
            policy.assess(
                result_code="no_matches",
                replan_count=1,
                operation_kind="read",
            ).decision,
        )

    def test_audit_preserves_original_constraints_in_stable_order(self) -> None:
        audit = ControlledReplanPolicy().assess_with_audit(
            original_query="找上海校招岗位",
            original_constraints={"recruitment_type": "校园招聘", "location": "上海"},
            result_code="no_matches",
            replan_count=0,
            operation_kind="read",
        )
        self.assertEqual("replan_once", audit.assessment.decision)
        self.assertEqual(
            (("location", "上海"), ("recruitment_type", "校园招聘")),
            audit.original_constraints,
        )
