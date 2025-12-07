"""Base class for experiments in RAG evaluation."""

import math
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from scipy import stats


class BaseExperiment(ABC):
    """Abstract base class for experiments in RAG evaluation."""

    def __init__(self, cfg):
        """Initialize the BaseExperiment instance."""
        self.cfg = cfg
        self.output_path: Path = Path()

    @abstractmethod
    def run(self) -> Any:
        """Run the experiment."""
        pass

    def get_stats(self, values: list[float]) -> dict[str, float | dict[str, float]]:
        """Get statistics from a list of values, including multiple confidence intervals.

        Args:
            values (list[float]): List of values to compute statistics on.

        Returns:
            dict[str, float | dict[str, float]]: Dictionary containing
            mean, standard deviation, and confidence intervals for 90%, 95%, and 99% levels.
        """
        if not values:
            return {
                "mean": 0.0,
                "std": 0.0,
                "confidence_intervals": {
                    "90%": 0.0,
                    "95%": 0.0,
                    "99.7%": 0.0,
                },
                "standard_error": 0.0,
            }

        n = len(values)
        mean = sum(values) / n

        # We use sample standard deviation (unbiased) for confidence interval calculation
        # The original formula calculates population standard deviation
        std = (sum((x - mean) ** 2 for x in values) / (n - 1)) ** 0.5 if n > 1 else 0.0

        # Calculate confidence intervals for 90%, 95%, and 99%
        confidence_intervals = {}
        confidence_levels = [0.90, 0.95, 0.997]

        standard_error = std / math.sqrt(n)
        for level in confidence_levels:
            # Use the t-distribution for small samples
            t_score = stats.t.ppf(1 - (1 - level) / 2, df=n - 1)
            margin_of_error = t_score * standard_error
            confidence_intervals[f"{int(level * 100)}%"] = margin_of_error

        return {
            "mean": mean,
            "std": std,
            "confidence_intervals": confidence_intervals,
            "standard_error": standard_error,
        }
