"""Ragas Answer Correctness Metric Implementation."""

import asyncio
import json
from datetime import datetime

import pandas as pd
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import AnswerCorrectness, answer_similarity
from tqdm import tqdm

from ragalyst.metrics.base import BaseMetric


class RagasAnswerCorrectnessMetric(BaseMetric):
    """Ragas Answer Correctness Metric Implementation."""

    def evaluate(
        self,
        question=None,
        answer=None,
        response=None,
        ground_truth=None,
        context=None,
        max_retries=10,
    ) -> float:
        """Evaluate the correctness of an answer given the ground truth."""
        assert question is not None, "Question cannot be None"
        assert answer is not None, "Answer cannot be None"
        assert ground_truth is not None, "Ground truth cannot be None"

        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            reference=ground_truth,
        )
        answer_similarity.embeddings = self.embedder.get_ragas_wrapper()
        scorer = AnswerCorrectness(
            llm=self.llm.get_ragas_wrapper(),
            answer_similarity=answer_similarity,
        )
        loop = asyncio.get_event_loop()

        for attempt in range(max_retries + 1):
            try:
                result = loop.run_until_complete(
                    asyncio.wait_for(
                        scorer.single_turn_ascore(sample), timeout=(attempt + 1) * 60
                    )
                )
                return result
            except (asyncio.TimeoutError, Exception) as e:
                print(
                    f"[Retry {attempt}/{max_retries}] Error occurred with ragas_answer_correctness metric: {e}"
                )
                continue

        print("Max retries exceeded, returning 0.0")
        return 0.0

    def validate_metric(self):
        """Validate the metric prompt on the STS-B (semantic textual similarity benchmark) dataset.

        https://gluebenchmark.com/tasks/
        This shows that our LLM based metric and prompt align with human judgment.
        """
        self._load_datasets()

        preds = []
        for item in tqdm(
            self.sts_test,
            total=len(self.sts_test),
            desc="Validating Answer Correctness Metric",
        ):
            pred = self.evaluate(
                question="",
                answer=item["sentence1"],
                ground_truth=item["sentence2"],
            )
            preds.append(pred)

        # Compare preds with ground_truths
        results = []
        for pred, sample in zip(preds, self.sts_test):
            results.append(
                {
                    "sentence1": sample["sentence1"],
                    "sentence2": sample["sentence2"],
                    "predicted": pred,
                    "ground_truth": sample["similarity"],
                }
            )

        df = pd.DataFrame(results)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"evaluations/metrics/answer_correctness/ragas/{self.cfg.metrics.llm_model_name}_{timestamp}.json"

        spearman_correlation, se = self._spearman_correlation_report(
            df, "predicted", "ground_truth", path
        )

        return spearman_correlation, se

    def _load_datasets(self):
        with open("src/ragalyst/metrics/datasets/sts_train.json", "r") as f:
            self.sts_train = json.load(f)

        with open("src/ragalyst/metrics/datasets/sts_test.json", "r") as f:
            self.sts_test = json.load(f)
