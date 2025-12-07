"""Optimized answerability metric using dspy."""

import dspy

from ragalyst.metrics.base import BaseMetric


class AnswerabilityDspy(dspy.Signature):
    """You will be given a student answer and a ground truth.

    Your task is to evaluate the student answer by comparing it with the ground truth.
    Give your evaluation on a scale of 0.0 to 1.0, where 0.0 means that the answer is completely unrelated to the ground truth, and 1.0 means that the answer is completely accurate and aligns perfectly with the ground truth.

    For instance,
    correctness_score: 0.0: The answer is completely unrelated to the ground truth.
    correctness_score: 0.3: The answer has minor relevance but does not align with the ground truth.
    correctness_score: 0.5: The answer has moderate relevance but contains inaccuracies.
    correctness_score: 0.7: The answer aligns with the reference but has minor errors or omissions.
    correctness_score: 1.0: The answer is completely accurate and aligns perfectly with the ground truth.

    You MUST provide values for 'correctness_score:' in your answer.
    """

    question: str = dspy.InputField(description="question")
    context: str = dspy.InputField(description="context")
    answerability: float = dspy.OutputField(
        description="A float of either 0.0 or 1.0 that reflects if the question is answerable given the context."
    )


class AnswerabilityOptimizedMetric(BaseMetric):
    """Optimized answerability metric using dspy."""

    def __init__(self, cfg):
        """Initialize an AnswerabilityOptimizedMetric instance."""
        self.cfg = cfg

        assert cfg.metrics.llm_model_name == "gpt-4o-mini", (
            "AnswerabilityOptimizedMetric is designed to work with gpt-4o-mini. "
            "Please set metrics.llm_model_name=gpt-4o-mini in your config."
        )
        self.answerability = dspy.Predict(AnswerabilityDspy)
        self.llm = dspy.LM(cfg.metrics.llm_model_name)
        dspy.configure(lm=self.llm)

    def evaluate(
        self,
        question=None,
        answer=None,
        response=None,
        ground_truth=None,
        context=None,
        max_retries=3,
    ) -> float:
        """Evaluate the answerability of a question given the context."""
        assert question is not None, "Question cannot be None"
        assert context is not None, "Context cannot be None"

        result = self.answerability(question=question, context=context)
        return result.answerability
