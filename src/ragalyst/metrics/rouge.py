"""ROUGE-L metric implementation."""

from rouge_score import rouge_scorer

from ragalyst.metrics.base import BaseMetric


class RougeLMetric(BaseMetric):
    """Implementation of ROUGE-L metric."""

    def __init__(self, cfg):
        """Init ROUGE-L metric."""
        super().__init__(cfg)
        self.scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    def evaluate(self, text1: str, text2: str) -> float | None:
        """Calculates the ROUGE-L F1 score between two sentences."""
        try:
            score = self.scorer.score(text2, text1)["rougeL"].fmeasure
            return score
        except Exception as e:
            print(e)
            return None
