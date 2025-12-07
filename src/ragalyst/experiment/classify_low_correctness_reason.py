"""Discover why LLMs with RAG perform poorly on Answer Correctness."""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch
from pydantic import BaseModel, Field
from tqdm import tqdm

from ragalyst.experiment.base import BaseExperiment
from ragalyst.module_registry import get_llm


# Pydantic Models
class FailureReason(BaseModel):
    """Model for individual failure reason."""

    applies: bool
    explanation: str = ""


class HallucinationReasons(BaseModel):
    """Model for all hallucination/failure reasons."""

    missed_top_ranked: FailureReason = Field(
        default_factory=lambda: FailureReason(applies=False)
    )
    not_extracted: FailureReason = Field(
        default_factory=lambda: FailureReason(applies=False)
    )
    wrong_format: FailureReason = Field(
        default_factory=lambda: FailureReason(applies=False)
    )
    under_specificity: FailureReason = Field(
        default_factory=lambda: FailureReason(applies=False)
    )
    over_specificity: FailureReason = Field(
        default_factory=lambda: FailureReason(applies=False)
    )
    factual_contradiction: FailureReason = Field(
        default_factory=lambda: FailureReason(applies=False)
    )
    factual_fabrication: FailureReason = Field(
        default_factory=lambda: FailureReason(applies=False)
    )
    instruction_inconsistency: FailureReason = Field(
        default_factory=lambda: FailureReason(applies=False)
    )
    context_inconsistency: FailureReason = Field(
        default_factory=lambda: FailureReason(applies=False)
    )
    logical_inconsistency: FailureReason = Field(
        default_factory=lambda: FailureReason(applies=False)
    )
    other: FailureReason = Field(default_factory=lambda: FailureReason(applies=False))


class QCAEvaluation(BaseModel):
    """Model for Question-Context-Answer evaluation."""

    question: str
    answer: str
    context: str
    llm_answer: str
    faithfulness: float
    answer_correctness: float
    answer_relevancy: float
    retrieved_contexts: List[str]
    hallucination_reasons: Optional[HallucinationReasons] = None


class EvaluationData(BaseModel):
    """Model for the entire evaluation dataset."""

    qca_evaluations: List[QCAEvaluation]


class FailureGroup(BaseModel):
    """Model for failure type groupings."""

    rag: List[str] = [
        "missed_top_ranked",
        "not_extracted",
        "wrong_format",
        "under_specificity",
        "over_specificity",
    ]
    llm: List[str] = [
        "factual_contradiction",
        "factual_fabrication",
        "instruction_inconsistency",
        "context_inconsistency",
        "logical_inconsistency",
    ]
    other: List[str] = ["other"]


failure_types = [
    "missed_top_ranked",
    "not_extracted",
    "wrong_format",
    "under_specificity",
    "over_specificity",
    "factual_contradiction",
    "factual_fabrication",
    "instruction_inconsistency",
    "context_inconsistency",
    "logical_inconsistency",
]

group = FailureGroup()


class ClassifyLowCorrectnessReason(BaseExperiment):
    """Experiment to classify low answer correctness QCAs."""

    def __init__(self, cfg) -> None:
        """Initialize the experiment."""
        super().__init__(cfg)
        self.llm = get_llm(cfg)
        self.dataset_path: Path = Path(cfg.experiment.evaluated_qa_dataset)

    def run(self) -> None:
        """Run the experiment."""
        with open(self.dataset_path, "r") as f:
            data = json.load(f)

        eval_data = EvaluationData(**data)

        for i, qca in tqdm(
            enumerate(eval_data.qca_evaluations), total=len(eval_data.qca_evaluations)
        ):
            try:
                prompt = self.create_evaluation_prompt(qca)
                classifications = (
                    self.llm.chat(context="", question=prompt)
                    .replace("```json", "")
                    .replace("```", "")
                    .strip()
                )
                result = json.loads(classifications)

                retrieved_contexts = qca.retrieved_contexts
                gt_context = qca.context
                if len(retrieved_contexts) > 0 and gt_context not in retrieved_contexts:
                    result["missed_top_ranked"] = {
                        "applies": True,
                        "explanation": "The ground truth context was not among the retrieved contexts.",
                    }
                else:
                    result["missed_top_ranked"] = {"applies": False, "explanation": ""}

                eval_data.qca_evaluations[
                    i
                ].hallucination_reasons = HallucinationReasons(**result)
            except Exception as e:
                print(e)
                print(f"Error processing QCA index {i}, skipping...")

        # Save back to file
        out_path = Path(
            str(self.dataset_path).replace("llm_with_rag", "low_correctness_analysis")
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(eval_data.model_dump(), f, indent=2)

    def create_evaluation_prompt(self, qca: QCAEvaluation) -> str:
        """Create the evaluation prompt for LLM."""
        return f"""
        Question: {qca.question}
        Ground Truth Answer: {qca.answer}
        LLM Answer: {qca.llm_answer}
        Ground Truth Context: {qca.context}
        Retrieved Contexts: {qca.retrieved_contexts}
        Correctness: {qca.answer_correctness}

        RAG QA Evaluation Prompt (Unified Failure + Hallucination Framework)
        You are an expert evaluator analyzing a Question–Context–Answer (QCA) triplet produced by a Retrieval-Augmented Generation (RAG) system.
        Your goal is to determine how and why the model's response may have failed, covering both retrieval-related issues and generation-related issues such as hallucinations, factuality, or instruction violations.
        Read the Question, Retrieved Context, and Answer carefully.
        Then evaluate whether any of the failure types below apply.
        Provide short explanations (1–2 sentences) for each type that applies.

        Retrieval & Contextual Failures (RAG-specific)
        FP1. Not Extracted
        The correct answer is present in the retrieved context, but the model failed to extract or focus on it, possibly due to distractions or contradictions.
        FP2. Wrong Format
        The question asked for a specific structure (e.g., list, table, or translation), but the answer failed to match the required format.
        FP3. Under Specificity
        The answer is too vague or general, missing the expected granularity of the question.
        FP4. Over Specificity
        The answer is too specific, providing excessive detail that is not necessary for the question.

        Generation & Reasoning Failures (Hallucination + Faithfulness)
        F1. Factual Contradiction
        The answer directly contradicts known facts or the retrieved context.
        F2. Factual Fabrication
        The answer introduces entirely fabricated information not supported anywhere in the retrieved context or known facts.
        Example: mentioning nonexistent entities, events, or quotes.
        F3. Instruction Inconsistency
        The model ignores explicit task instructions.
        Example: being told to translate or summarize but instead answers the question.
        F4. Context Inconsistency
        The summary or generated text distorts details from the retrieved context (e.g., misstates names, numbers, or causal relationships).
        The model's statements are inconsistent with the evidence it was given.
        F5. Logical Inconsistency
        The reasoning or steps in the answer contradict each other logically or mathematically.
        Example: producing inconsistent intermediate steps in an equation or argument.

        If none of the above failure types apply, and answer correctness is less than 1.0 evaluate as "other" failure and explain why.

        Output Format
        Output a structured JSON object with each failure type as a key:
        {{
            "not_extracted": {{"applies": true/false, "explanation": "..."}},
            "wrong_format": {{"applies": true/false, "explanation": "..."}},
            "under_specificity": {{"applies": true/false, "explanation": "..."}},
            "over_specificity": {{"applies": true/false, "explanation": "..."}},
            "factual_contradiction": {{"applies": true/false, "explanation": "..."}},
            "factual_fabrication": {{"applies": true/false, "explanation": "..."}},
            "instruction_inconsistency": {{"applies": true/false, "explanation": "..."}},
            "context_inconsistency": {{"applies": true/false, "explanation": "..."}},
            "logical_inconsistency": {{"applies": true/false, "explanation": "..."}},
            "other": {{"applies": true/false, "explanation": "..."}}
        }}
        Evaluation Guidelines
        Multiple failure types may apply simultaneously.
        A perfect answer should have all categories marked "applies": false. If the answer has correctness < 1.0, at least one failure type should apply.
        When unsure between two related categories (e.g., Factual Fabrication vs. Missing Content), favor the more specific one (Fabrication).
        Keep explanations concise and evidence-based, referring only to the given context and question.
        """

    def count_failures(
        self, eval_data: EvaluationData, correctness_threshold: Optional[float] = None
    ) -> Dict[str, int]:
        """Count failures by type. If threshold is None, count all QCAs. Otherwise, count only QCAs with AC < threshold."""
        failure_counts = defaultdict(int)

        for qca in eval_data.qca_evaluations:
            # Apply threshold filter if provided
            if (
                correctness_threshold is not None
                and qca.answer_correctness >= correctness_threshold
            ):
                continue

            if qca.hallucination_reasons is None:
                continue

            reasons_dict = qca.hallucination_reasons.model_dump()
            for ft in failure_types:
                if reasons_dict.get(ft, {}).get("applies", False):
                    failure_counts[ft] += 1

        return failure_counts

    def plot_failures(
        self,
        failure_counts: Dict[str, int],
        domain: str,
        group: FailureGroup,
        filter_type: str = "all",
    ):
        """Plot failure counts with group coloring.

        Args:
            failure_counts: Dictionary of failure type counts
            domain: Domain name
            group: FailureGroup instance
            filter_type: "all" for all QCAs or "filtered" for AC < 0.9
        """
        if not failure_counts or sum(failure_counts.values()) == 0:
            print(f"No failures to plot for {domain} ({filter_type})")
            return

        failure_series = pd.Series(failure_counts).sort_values(ascending=False)

        group_colors = {"rag": "tab:blue", "llm": "tab:orange", "other": "tab:green"}

        # Create color map
        color_map = {}
        for group_name in ["rag", "llm", "other"]:
            fts = getattr(group, group_name)
            for ft in fts:
                color_map[ft] = group_colors[group_name]

        bar_colors = [color_map.get(ft, "gray") for ft in failure_series.index]

        plt.figure(figsize=(10, 5))
        failure_series = failure_series.rename(
            index=lambda ft: ft.replace("_", " ").title()
        )
        failure_series.plot(kind="bar", color=bar_colors)

        real_domain = {
            "army": "Military Operations",
            "cybersecurity": "Cybersecurity",
            "engineering": "Bridge Engineering",
        }

        # Set title based on filter type
        if filter_type == "all":
            filter_label = " (All QCAs)"
        else:
            filter_label = " (Answer Correctness < 0.9)"

        if domain == "All Domains":
            plt.title(f"Overall Failure Counts Across All Domains{filter_label}")
        else:
            plt.title(
                f"Failure Counts for {real_domain.get(domain, domain).capitalize()} Domain{filter_label}"
            )

        plt.xlabel("Failure Type")
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha="right")

        label = ["RAG", "LLM"]
        handles = [
            Patch(color=c, label=lab) for lab, c in zip(label, group_colors.values())
        ]
        plt.legend(handles=handles, title="Failure Group")

        plt.tight_layout()

        # Save with appropriate filename
        if filter_type == "all":
            filename = f"outputs/failure_counts_{domain}_all.png"
        else:
            filename = f"outputs/failure_counts_{domain}_ac_lt_0.9.png"

        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_comparison(
        self,
        all_counts: Dict[str, int],
        filtered_counts: Dict[str, int],
        domain: str,
        group: FailureGroup,
    ):
        """Plot side-by-side comparison of all QCAs vs filtered QCAs."""
        # Get union of all failure types
        all_types = set(all_counts.keys()) | set(filtered_counts.keys())

        if not all_types:
            print(f"No data to plot comparison for {domain}")
            return

        # Sort by total count (all + filtered)
        sorted_types = sorted(
            all_types,
            key=lambda x: all_counts.get(x, 0) + filtered_counts.get(x, 0),
            reverse=True,
        )

        all_series = pd.Series({ft: all_counts.get(ft, 0) for ft in sorted_types})
        filtered_series = pd.Series(
            {ft: filtered_counts.get(ft, 0) for ft in sorted_types}
        )

        # Rename indices
        display_names = [ft.replace("_", " ").title() for ft in sorted_types]

        fig, ax = plt.subplots(figsize=(12, 6))

        x = range(len(sorted_types))
        width = 0.35

        ax.bar(
            [i - width / 2 for i in x],
            all_series.to_numpy(),
            width,
            label="All QCAs",
            color="tab:blue",
            alpha=0.8,
        )
        ax.bar(
            [i + width / 2 for i in x],
            filtered_series.to_numpy(),
            width,
            label="AC < 0.9",
            color="tab:red",
            alpha=0.8,
        )

        real_domain = {
            "army": "Military Operations",
            "cybersecurity": "Cybersecurity",
            "engineering": "Bridge Engineering",
        }

        if domain == "All Domains":
            title = "Comparison: All QCAs vs AC < 0.9 (All Domains)"
        else:
            title = (
                f"Comparison: All QCAs vs AC < 0.9 ({real_domain.get(domain, domain)})"
            )

        ax.set_title(title)
        ax.set_xlabel("Failure Type")
        ax.set_ylabel("Count")
        ax.set_xticks(x)
        ax.set_xticklabels(display_names, rotation=45, ha="right")
        ax.legend()

        plt.tight_layout()
        plt.savefig(f"failure_comparison_{domain}.png", dpi=300, bbox_inches="tight")
        plt.close()
