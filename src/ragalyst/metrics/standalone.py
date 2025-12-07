"""Standalone Metric Implementation."""

from ragalyst.metrics.base import BaseMetric


class StandaloneMetric(BaseMetric):
    """Standalone Metric to evaluate how context-independent a question is."""

    def evaluate(
        self,
        question=None,
        answer=None,
        response=None,
        ground_truth=None,
        context=None,
        max_retries=3,
    ) -> float:
        """Evaluate how context-independent a question is."""
        assert question is not None, "Question cannot be None"

        template = f"""
            You will be given a question.
            Your task is to provide a 'total rating' representing how context-independant this question is.
            Give your answer on a scale of 0.0 to 1.0, where 0.0 means that the question depends on additional information to be understood, and 1.0 means that the question makes sense by itself.
            For instance, if the question refers to a particular setting, like 'in the context' or 'in the document', the rating must be 0.0.
            The questions can contain obscure technical nouns or acronyms and still be a 1.0.
            For example military technical term term/noun like MEDEVAC, UH-47, ... do not reduce the score of the question.
            We assume that the operator has access to documentation and can look up the meaning of these terms.
            However, refer to a specific context (i.e in the context, as described in the context) should reduce the score.
            Specific scenarios contexts, roles, concepts related to the military do not reduce the score of the question.
            Asume military knowledge is universal knowledge.

            You MUST provide values for 'standalone_score:' in your answer.

            Now here is the question.

            Question: {question}\n

            Format the output as a single number such as standalone_score: 0.5 for example. Do not produce any other output.
            standalone_score: your rating, as a number between 0.0 and 1.0
        """
        for attempt in range(max_retries + 1):
            try:
                return self.extract_score(template, "standalone_score:")
            except Exception as e:
                print(
                    f"[Retry {attempt}/{max_retries}] Error occurred with standalone metric: {e}"
                )
                continue

        return 0.0
