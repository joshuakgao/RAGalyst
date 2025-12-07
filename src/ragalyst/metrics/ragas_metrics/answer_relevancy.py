"""Ragas Answer Relevancy Metric Implementation."""

import asyncio

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import AnswerRelevancy

from ragalyst.metrics.base import BaseMetric


class RagasAnswerRelevancyMetric(BaseMetric):
    """Ragas Answer Relevancy Metric Implementation."""

    def evaluate(
        self,
        question=None,
        answer=None,
        response=None,
        ground_truth=None,
        context=None,
        max_retries=10,
    ) -> float:
        """Evaluate the relevancy of an answer given the context."""
        assert question is not None, "Question cannot be None"
        assert answer is not None, "Answer cannot be None"
        assert context is not None, "Context cannot be None"

        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=[context],
        )
        scorer = AnswerRelevancy(
            llm=self.llm.get_ragas_wrapper(),
            embeddings=self.embedder.get_ragas_wrapper(),
        )
        for attempt in range(max_retries + 1):
            try:
                loop = asyncio.get_event_loop()
                result = loop.run_until_complete(
                    asyncio.wait_for(
                        scorer.single_turn_ascore(sample), timeout=(attempt + 1) * 60
                    )
                )
                return result
            except (asyncio.TimeoutError, Exception) as e:
                print(
                    f"[Retry {attempt}/{max_retries}] Error occurred with ragas_answer_relevancy metric: {e}"
                )
                continue

        print("Max retries exceeded, returning 0.0")
        return 0.0
