from unittest import TestCase

from app.application.memory.safe_conversation import (
    merge_safe_conversation_turns,
    safe_agent_summary,
    safe_conversation_turn,
)


class SafeConversationTest(TestCase):
    def test_redacts_bearer_tokens_and_serializes_plain_data(self) -> None:
        turn = safe_conversation_turn(
            role="user",
            content="Bearer abc.def-123 find jobs",
        )

        self.assertEqual(
            {"role": "user", "content": "[redacted bearer token] find jobs"},
            turn.to_checkpoint(),
        )

    def test_keeps_only_six_complete_rounds(self) -> None:
        turns = [
            safe_conversation_turn(role="user", content=f"question {index}").to_checkpoint()
            for index in range(13)
        ]

        merged = merge_safe_conversation_turns([], turns)

        self.assertEqual(12, len(merged))
        self.assertEqual("question 1", merged[0]["content"])

    def test_agent_turn_is_a_summary_not_raw_tool_payload(self) -> None:
        summary = safe_agent_summary(
            outcome="tool_selected",
            reason="current job records are required",
        )

        self.assertEqual("assistant", summary.role)
        self.assertNotIn("tool_result", summary.content)
