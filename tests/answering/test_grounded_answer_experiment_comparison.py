import unittest

from app.application.answering.grounded_answer_experiment_comparison import (
    GroundedAnswerExperimentIdentity,
    classify_grounded_answer_experiment,
)


class GroundedAnswerExperimentComparisonTest(unittest.TestCase):
    def test_classifies_prompt_only_experiment(self) -> None:
        comparison = classify_grounded_answer_experiment(
            self._identity(prompt_version="prompt-v1"),
            self._identity(prompt_version="prompt-v2"),
        )

        self.assertEqual("prompt", comparison.axis)
        self.assertEqual(
            ("prompt_version",),
            comparison.changed_fields,
        )

    def test_classifies_same_identity_as_repeatability_experiment(
        self,
    ) -> None:
        comparison = classify_grounded_answer_experiment(
            self._identity(),
            self._identity(),
        )

        self.assertEqual("repeatability", comparison.axis)
        self.assertEqual((), comparison.changed_fields)

    def test_rejects_different_dataset_hashes(self) -> None:
        with self.assertRaisesRegex(ValueError, "different.*datasets"):
            classify_grounded_answer_experiment(
                self._identity(dataset_sha256="a" * 64),
                self._identity(dataset_sha256="b" * 64),
            )

    def test_rejects_model_and_prompt_change_together(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "more than one independent variable",
        ):
            classify_grounded_answer_experiment(
                self._identity(
                    model_name="model-a",
                    prompt_version="prompt-v1",
                ),
                self._identity(
                    model_name="model-b",
                    prompt_version="prompt-v2",
                ),
            )

    @staticmethod
    def _identity(
        *,
        dataset_sha256: str = "a" * 64,
        model_name: str = "model-a",
        prompt_version: str = "prompt-v1",
    ) -> GroundedAnswerExperimentIdentity:
        return GroundedAnswerExperimentIdentity(
            dataset_id="answer-evaluation-v1",
            dataset_version=1,
            dataset_sha256=dataset_sha256,
            model_name=model_name,
            prompt_version=prompt_version,
        )


if __name__ == "__main__":
    unittest.main()
