"""BLEU metric implementation."""

from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

from ragalyst.metrics.base import BaseMetric


class BleuMetric(BaseMetric):
    """Implementation of BLEU metric."""

    def evaluate(self, text1: str, text2: str) -> float | None:
        """Calculates the sentence-level BLEU score between two sentences."""
        try:
            smoothing = SmoothingFunction().method1

            # Tokenize both sentences
            ref_tokens: list[str] = text2.split() if isinstance(text2, str) else text2
            pred_tokens: list[str] = text1.split() if isinstance(text1, str) else text1

            # sentence_bleu expects a list of reference token lists
            score = sentence_bleu(
                [ref_tokens], pred_tokens, smoothing_function=smoothing
            )
            score = float(score)  # type: ignore
            assert 0.0 <= score <= 1.0

            return score
        except Exception as e:
            print(e)
            return None
