"""Ragas QCA Dataset Manager for generating and evaluating QAC triples using Ragas framework."""

import json
from pathlib import Path

from ragas.executor import RunConfig
from ragas.testset import TestsetGenerator
from ragas.testset.synthesizers import (
    QueryDistribution,
    SingleHopSpecificQuerySynthesizer,
)

from ragalyst.data.dataset_manager.base import BaseDatasetManager


class RagasQcaDatasetManager(BaseDatasetManager):
    """Dataset manager for generating and evaluating QAC triples using Ragas framework."""

    def __init__(self, cfg):
        """Initialize a RagasQcaDatasetManager instance."""
        # Imported here to avoid circular imports
        from ragalyst.module_registry import get_embedder, get_llm, get_metrics

        super().__init__(cfg)

        self.llm = get_llm(cfg)
        self.embedder = get_embedder(cfg)
        self.metrics = get_metrics(cfg)

        self.output_dir = Path("src/ragalyst/data/datasets/ragas_qca") / cfg.domain
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Multithreading settings
        self.num_workers: int = cfg.data.dataset_manager.num_workers

        assert cfg.data.text_processor.type == "knowledge_graph", (
            "Ragas QCA dataset generator only supports knowledge_graph text processor type."
        )

    def generate(self) -> list[dict]:
        """Generate a dataset of QAC triples using Ragas framework."""
        self.text_processor.process()

        assert self.text_processor.kg is not None, "Knowledge graph is not initialized."

        # Init query synthesizer
        query_distribution: QueryDistribution = [
            (SingleHopSpecificQuerySynthesizer(llm=self.llm.get_ragas_wrapper()), 1.0)
        ]

        # Init testset generator
        generator = TestsetGenerator(
            llm=self.llm.get_ragas_wrapper(),
            embedding_model=self.embedder.get_ragas_wrapper(),
            knowledge_graph=self.text_processor.kg,
        )

        # Generate dataset
        dataset = generator.generate(
            testset_size=self.num_samples,
            query_distribution=query_distribution,
            with_debugging_logs=True,
            run_config=RunConfig(max_workers=self.num_workers, max_retries=1),
        )

        # Convert to a list of dictionaries
        dataset = dataset.to_list()

        # Write to output file
        with open(self.output_dir / "data.json", "w") as f:
            json.dump(dataset, f, indent=4)

        return dataset

    def evaluate(self) -> dict:
        """Evaluate the generated QAC triples using Ragas metrics."""
        assert self.output_dir.exists(), (
            "Output directory does not exist. Please generate the dataset first."
        )

        with open(self.output_dir / "data.json", "r") as f:
            qac_triples = json.load(f)

        # Get qca's into their own lists
        questions = [item["user_input"] for item in qac_triples]
        answers = [item["reference"] for item in qac_triples]
        contexts = [item["reference_contexts"][0] for item in qac_triples]

        assert len(questions) == len(answers) == len(contexts), (
            "Questions, answers and contexts must have the same length."
        )
        assert len(questions) > 0 and len(answers) > 0 and len(contexts) > 0, (
            "No data to evaluate."
        )

        # Evaluate metrics with mulithreading
        faithfulness_scores = self.metrics.faithfulness.evaluate_all(
            questions=questions, answers=answers, contexts=contexts
        )
        answerability_scores = self.metrics.answerability.evaluate_all(
            questions=questions, contexts=contexts
        )
        answer_relevancy_scores = self.metrics.answer_relevancy.evaluate_all(
            questions=questions, answers=answers, contexts=contexts
        )

        # Calculate averages
        avg_faithfulness = sum(faithfulness_scores) / len(faithfulness_scores)
        avg_answerability = sum(answerability_scores) / len(answerability_scores)
        avg_answer_relevancy = sum(answer_relevancy_scores) / len(
            answer_relevancy_scores
        )

        # Store the evals for each qca
        qca_evals = [
            {
                "user_input": questions[i],
                "reference": answers[i],
                "reference_contexts": [contexts[i]],
                "faithfulness": faithfulness_scores[i],
                "answerability": answerability_scores[i],
                "answer_relevancy": answer_relevancy_scores[i],
            }
            for i in range(len(questions))
        ]

        # Build final report
        report = {
            "average_faithfulness": avg_faithfulness,
            "average_answerability": avg_answerability,
            "average_answer_relevancy": avg_answer_relevancy,
            "qca_evaluations": qca_evals,
        }

        # Write to eval output path
        with open(self.output_dir / "eval.json", "w") as f:
            json.dump(report, f, indent=4)

        return report
