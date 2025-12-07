"""Experiment for evaluating LLM with RAG on QCA dataset."""

import json
from pathlib import Path

from omegaconf import OmegaConf
from tqdm import tqdm

from ragalyst.experiment.base import BaseExperiment


class LlmWithRagEvaluation(BaseExperiment):
    """Experiment to evaluate LLM with RAG on QCA dataset."""

    def __init__(self, cfg):
        """Initialize an LlmWithRagEvaluation instance."""
        # Imported here to avoid circular imports
        from ragalyst.module_registry import get_embedder, get_metrics, get_rag

        super().__init__(cfg)

        self.cfg = cfg
        self.embedder = get_embedder(cfg)
        self.metrics = get_metrics(cfg)
        self.rag = get_rag(cfg)

    def run(self):
        """Evaluate the LLM on the QCA dataset."""
        # Set output path with unique filename per run (using timestamp)
        self.output_path = (
            Path(self.cfg.domain)
            / "llm_with_rag"
            / self.cfg.embedder.model_name.replace("/", "_")
            / f"{self.cfg.llm.model_name.replace('/', '_')}_k{self.cfg.rag.top_k}.json"
        )
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Read dataset
        dataset_path = Path(self.cfg.domain) / "qca" / "data.json"
        with open(dataset_path, "r") as f:
            qac_triples = json.load(f)

        # Get questions and answers
        questions: list[str] = [item["question"] for item in qac_triples]
        contexts: list[str] = [item["context"] for item in qac_triples]
        gt_answers: list[str] = [item["answer"] for item in qac_triples]

        assert len(questions) == len(gt_answers), (
            "Questions and answers must have the same length."
        )
        assert len(questions) > 0 and len(gt_answers) > 0, "No data to evaluate."

        # Ask questions to the LLM
        answers: list[str] = []
        retrieved_contexts: list[list[str]] = []
        for q in tqdm(questions, desc="Asking questions", total=len(questions)):
            for attempt in range(5):
                try:
                    answer, retrieved_context = self.rag.query_with_rag(q)
                    answers.append(answer)
                    retrieved_contexts.append(retrieved_context)
                    break
                except Exception:
                    if attempt == 4:
                        raise
        retrieved_contexts_flat = ["\n".join(ctxs) for ctxs in retrieved_contexts]

        # Evaluate answers
        faithfulness_scores = self.metrics.faithfulness.evaluate_all(
            questions=questions, answers=answers, contexts=retrieved_contexts_flat
        )
        answer_relevancy_scores = self.metrics.answer_relevancy.evaluate_all(
            questions=questions, answers=answers, contexts=retrieved_contexts_flat
        )
        answer_correctness_scores = self.metrics.answer_correctness.evaluate_all(
            questions=questions, answers=answers, ground_truths=gt_answers
        )

        # Store the evals for each qca
        qca_evals = [
            {
                "question": questions[i],
                "context": contexts[i],
                "answer": gt_answers[i],
                "llm_answer": answers[i],
                "faithfulness": faithfulness_scores[i],
                "answer_relevancy": answer_relevancy_scores[i],
                "answer_correctness": answer_correctness_scores[i],
                "retrieved_contexts": retrieved_contexts[i],
            }
            for i in range(len(questions))
        ]

        # Build final report
        plain_dict = OmegaConf.to_container(self.cfg)
        report = {
            "config": plain_dict,
            "stats": {
                "faithfulness": self.get_stats(faithfulness_scores),
                "answer_relevancy": self.get_stats(answer_relevancy_scores),
                "answer_correctness": self.get_stats(answer_correctness_scores),
            },
            "qca_evaluations": qca_evals,
        }

        # Save evaluation to file
        with open(self.output_path, "w") as f:
            json.dump(report, f, indent=4)

        return report
