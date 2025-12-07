"""Base class for all metrics."""

import json
import warnings
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from omegaconf import OmegaConf
from scipy import stats
from scipy.stats import spearmanr
from tqdm import tqdm


class BaseMetric(ABC):
    """Base class for all metrics."""

    def __init__(self, cfg):
        """Initialize the BaseMetric class.

        Args:
            cfg (DictConfig): Configuration settings.
        """
        # Imported here to avoid circular imports
        from ragalyst.module_registry import get_metrics_embedder, get_metrics_llm

        self.cfg = cfg

        if cfg.metrics.get("llm_device", "auto") != "no_load":
            self.llm = get_metrics_llm(cfg)

        # Load embedder conditionally
        if cfg.metrics.get("embedder_device", "auto") != "no_load":
            self.embedder = get_metrics_embedder(cfg)

    def extract_score(self, template, named_entity):
        """Extract score from LLM evaluation response."""
        evaluation = self.llm.chat("", template)
        return float(evaluation.replace(named_entity, "").strip())

    @abstractmethod
    def evaluate(
        self,
        question=None,
        answer=None,
        response=None,
        ground_truth=None,
        context=None,
        max_retries=10,
    ) -> float:
        """Evaluate a metric."""
        pass

    def evaluate_all(
        self,
        questions=None,
        answers=None,
        responses=None,
        ground_truths=None,
        contexts=None,
    ) -> list:
        """Evaluate a list of questions and answers.

        Args:
            questions (List[str]): List of questions.
            answers (List[str]): List of answers.
            responses (List[str]): List of responses.
            ground_truths (List[str]): List of ground truths.
            contexts (List[str]): List of contexts.

        Returns:
            List[float]: List of evaluation scores.
        """
        if self.cfg.metrics.llm_type == "ollama":
            warnings.warn(
                "\033[33mDue to Python's Global Interpreter Lock (GIL), multithreading in evaluate_all() will not improve performance for compute-intensive tasks, such as inference with Ollama models.\033[0m"
            )

        question_len = len(questions) if questions is not None else 0
        answer_len = len(answers) if answers is not None else 0
        response_len = len(responses) if responses is not None else 0
        ground_truth_len = len(ground_truths) if ground_truths is not None else 0
        context_len = len(contexts) if contexts is not None else 0

        max_len = max(
            question_len, answer_len, response_len, ground_truth_len, context_len
        )
        if questions is None:
            questions = [None] * max_len
        if answers is None:
            answers = [None] * max_len
        if responses is None:
            responses = [None] * max_len
        if ground_truths is None:
            ground_truths = [None] * max_len
        if contexts is None:
            contexts = [None] * max_len

        if self.cfg.metrics.llm_type in ["huggingface", "ollama"]:
            max_workers = 1
        else:
            max_workers = 16

        results = [0.0] * max_len
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    self.evaluate,
                    questions[i],
                    answers[i],
                    responses[i],
                    ground_truths[i],
                    contexts[i],
                    max_retries=10,
                )
                for i in range(max_len)
            ]
            for i, future in enumerate(tqdm(futures, desc="Evaluating", total=max_len)):
                results[i] = future.result()
        return results

    def evaluate_average(
        self,
        questions=None,
        answers=None,
        responses=None,
        ground_truths=None,
        contexts=None,
    ) -> float:
        """Evaluate a list of questions and answers and return the average score.

        Args:
            questions (List[str]): List of questions.
            answers (List[str]): List of answers.
            responses (List[str]): List of responses.
            ground_truths (List[str]): List of ground truths.
            contexts (List[str]): List of contexts.

        Returns:
            float: Average evaluation score.
        """
        results = self.evaluate_all(
            questions, answers, responses, ground_truths, contexts
        )
        return sum(results) / len(results)

    def _spearman_correlation_report(self, df, column1_name, column2_name, path=None):
        corr, p_value = spearmanr(df[column1_name], df[column2_name])
        corr, p_value = float(corr), float(p_value)  # type: ignore
        plain_dict = OmegaConf.to_container(self.cfg)

        se = (
            (1 + (corr**2 / 2)) / (len(df) - 3)
        ) ** 0.5  # Bonnet and Wright's SE approximation

        confidence_intervals = {}
        confidence_levels = [0.90, 0.95, 0.997]
        for level in confidence_levels:
            # Use the t-distribution for small samples
            t_score = stats.t.ppf(1 - (1 - level) / 2, df=len(df) - 1)
            margin_of_error = t_score * se
            confidence_intervals[f"{int(level * 100)}%"] = margin_of_error

        report = {
            "config": plain_dict,
            "stats": {
                "spearman_correlation": corr,
                "p_value": p_value,
                "standard_error_approx": se,
                "confidence_intervals": confidence_intervals,
            },
            "data": df.to_dict(orient="records"),
        }

        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                json.dump(report, f, indent=4)

        return corr, se
