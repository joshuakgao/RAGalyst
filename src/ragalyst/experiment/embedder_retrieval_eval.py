"""Experiment for evaluating embedder retrieval capabilities on QCA dataset."""

import json
from pathlib import Path

from omegaconf import OmegaConf
from tqdm import tqdm

from ragalyst.experiment.base import BaseExperiment


class EmbedderRetrievalEvaluation(BaseExperiment):
    """Experiment to evaluate embedder retrieval capabilities on QCA dataset."""

    def __init__(self, cfg):
        """Initialize an EmbedderRetrievalEvaluation instance."""
        # Imported here to avoid circular imports
        from ragalyst.module_registry import get_embedder, get_metrics, get_rag

        super().__init__(cfg)

        self.cfg = cfg
        self.embedder = get_embedder(cfg)
        self.metrics = get_metrics(cfg)
        self.rag = get_rag(cfg)

    def run(self):
        """Evaluate the embedder's retrieval capabilities on the QCA dataset."""
        # Set output path with unique filename per run (using timestamp)
        self.output_path = (
            Path(f"{self.cfg.domain}")
            / "embedder_eval"
            / f"{self.cfg.embedder.model_name.replace('/', '_')}.json"
        )
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Read dataset
        # TODO: this dataset_path shouldn't be hardcoded to qca. This function should also work with ragas_qca
        dataset_path = Path(self.cfg.domain) / "qca" / "data.json"
        with open(dataset_path, "r") as f:
            qac_triples = json.load(f)

        # Get qca's into their own lists
        questions: list[str] = [item["question"] for item in qac_triples]
        gt_contexts: list[str] = [item["context"] for item in qac_triples]

        assert len(questions) == len(gt_contexts), (
            "Questions, answers and contexts must have the same length."
        )
        assert len(questions) > 0 and len(gt_contexts) > 0, "No data to evaluate."

        # Use embedder to retrieve similar contexts
        contexts: list[list[str]] = []
        for q in tqdm(questions, desc="Retrieving contexts", total=len(questions)):
            retrieved_contexts = self.rag.search(q)
            contexts.append(retrieved_contexts)

        # Evaluate retreived contexts
        hit_rate_scores = self.metrics.hit_rate.evaluate_all(
            contexts=contexts, ground_truths=gt_contexts
        )
        reciprocal_rank_scores = self.metrics.rank.evaluate_all(
            contexts=contexts, ground_truths=gt_contexts
        )

        # Store the evals for each qca
        qca_evals = [
            {
                "question": questions[i],
                "context": gt_contexts[i],
                "retrieved_contexts": contexts[i],
                "hit_rate": hit_rate_scores[i],
                "reciprocal_rank": reciprocal_rank_scores[i],
            }
            for i in range(len(questions))
        ]

        # Build final report
        plain_dict = OmegaConf.to_container(self.cfg)
        report = {
            "config": plain_dict,
            "stats": {
                "hit_rate": self.get_stats(hit_rate_scores),
                "reciprocal_rank": self.get_stats(reciprocal_rank_scores),
            },
            "qca_evaluations": qca_evals,
        }

        # Save evaluation to file
        with open(self.output_path, "w") as f:
            json.dump(report, f, indent=4)

        return report
