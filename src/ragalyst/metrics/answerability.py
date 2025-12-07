"""Answerability metric for evaluating if a question is answerable given a context."""

import json
import os
from datetime import datetime
from pathlib import Path

import dspy
import pandas as pd
from dspy.teleprompt import COPRO, LabeledFewShot, MIPROv2
from matplotlib import pyplot as plt
from tqdm import tqdm

from ragalyst.metrics.base import BaseMetric


class AnswerabilityMetric(BaseMetric):
    """Answerability metric for evaluating if a question is answerable given a context."""

    def __init__(self, cfg):
        """Initialize an AnswerabilityMetric instance."""
        super().__init__(cfg)
        self._dspy_initialized = False
        self.train_examples = None
        self.answerability = None

    def _load_datasets(self):
        with open("src/ragalyst/metrics/datasets/squadv2_train.json", "r") as f:
            self.squad_train = json.load(f)

        with open("src/ragalyst/metrics/datasets/squadv2_test.json", "r") as f:
            self.squad_test = json.load(f)

    def _build_signature(self):
        class AnswerabilityDspy(dspy.Signature):
            """You will be given a context and a question.

            Your task is to determine if the question is clearly and unambiguously answerable using only the given context.
            - If the context contains **all** the necessary information to answer the question **without making assumptions** or using **any external knowledge**, then the groundedness is 1.
            - Otherwise, if any key information is **missing**, ambiguous, or requires inference beyond what is stated, then the groundedness is 0.

            You MUST provide values for 'answerability_flag:' in your answer.

            Use only the provided context. Do not use prior knowledge, common sense, or information not explicitly contained in the context.
            """

            question: str = dspy.InputField(description="question")
            context: str = dspy.InputField(description="context")
            answerability: float = dspy.OutputField(
                description="A float of either 0.0 or 1.0 that reflects if the question is answerable given the context."
            )

        return AnswerabilityDspy

    def _init_dspy(self):
        if self._dspy_initialized:
            return  # already initialized

        # Step 1: Load datasets
        self._load_datasets()

        # Step 2: Convert STS-B to DSPy examples
        def convert_to_example(data):
            return [
                dspy.Example(
                    question=sample["question"],
                    context=sample["context"],
                    answerability=sample["answerability"],
                ).with_inputs("question", "context")
                for sample in tqdm(data)
            ]

        self.train_examples = convert_to_example(self.squad_train)

        # Step 3: Configure LLM and signature
        signature_cls = self._build_signature()
        self.answerability = dspy.Predict(signature_cls)
        lm = dspy.LM("openai/gpt-4o-mini")
        dspy.configure(lm=lm)

        self._dspy_initialized = True

    def evaluate(
        self,
        question=None,
        answer=None,
        response=None,
        ground_truth=None,
        context=None,
        max_retries=3,
    ) -> float:
        """Evaluate the answerability of a question given the context."""
        assert question is not None, "Question cannot be None"
        assert context is not None, "Context cannot be None"

        template = f"""
            You will be given a context and a question.
            Your task is to determine if the question is clearly and unambiguously answerable using only the given context.
            - If the context contains **all** the necessary information to answer the question **without making assumptions** or using **any external knowledge**, then the groundedness is 1.
            - Otherwise, if any key information is **missing**, ambiguous, or requires inference beyond what is stated, then the groundedness is 0.

            You MUST provide values for 'answerability_flag:' in your answer.

            Use only the provided context. Do not use prior knowledge, common sense, or information not explicitly contained in the context. Here is the question and context:

            Question: {question}
            Context: {context}

            Format the output as a single number such as answerability_flag: 1 for example. Do not produce any other output.
            answerability_flag: boolean value (1 or 0)
        """
        for attempt in range(max_retries + 1):
            try:
                return self.extract_score(template, "answerability_flag:")
            except Exception as e:
                print(
                    f"[Retry {attempt}/{max_retries}] Error occurred with answerability metric: {e}"
                )
                continue

        return 0.0

    def validate_metric(self, dspy_metric=None):
        """Validate the metric prompt on the SQuAD 2.0 dataset.

        https://rajpurkar.github.io/SQuAD-explorer/
        This shows that our LLM based metric and prompt align with human judgment.
        """
        self._init_dspy()

        preds = []
        for item in tqdm(
            self.squad_test,
            total=len(self.squad_test),
            desc="Validating Answerability Metric",
        ):
            if dspy_metric:
                pred = dspy_metric(
                    question=item["question"],
                    context=item["context"],
                ).answerability
            else:
                pred = self.evaluate(
                    question=item["question"],
                    context=item["context"],
                )
            preds.append(pred)

        # Compare preds with ground_truths
        results = []
        for pred, sample in zip(preds, self.squad_test):
            results.append(
                {
                    "question": sample["question"],
                    "context": sample["context"],
                    "predicted": pred,
                    "ground_truth": sample["answerability"],
                }
            )

        df = pd.DataFrame(results)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if dspy_metric:
            path = None
        else:
            path = f"evaluations/metrics/answerability/base/{self.cfg.metrics.llm_model_name}_{timestamp}.json"
        spearman_correlation, se = self._spearman_correlation_report(
            df, "predicted", "ground_truth", path
        )

        return spearman_correlation, se

    def optimize_with_copro(self):
        """Optimize the metric prompt using DSPy's COPRO."""
        self._init_dspy()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"evaluations/metrics/answerability/copro/{timestamp}")
        output_dir.mkdir(parents=True, exist_ok=True)

        copro_optimizer = COPRO(
            metric=self._compare_metric,
            prompt_model=dspy.LM("openai/gpt-4o-mini"),
        )
        copro_metric = copro_optimizer.compile(
            self.answerability,
            trainset=self.train_examples,
            eval_kwargs={"num_threads": 10, "display_progress": True},
        )
        spearman_correlation, se = self.validate_metric(dspy_metric=copro_metric)
        copro_metric.save(output_dir / "prompt.json")

        with open(output_dir / "performance.json", "w") as f:
            json.dump(
                {
                    "corr": spearman_correlation,
                    "se_approx": se,
                },
                f,
                indent=4,
            )

    def optimize_with_miprov2(self):
        """Optimize the metric prompt using DSPy's MIPROv2."""
        self._init_dspy()
        assert self.answerability is not None, (
            "Answerability metric must be initialized."
        )
        assert self.train_examples is not None, "Train examples must be initialized."

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"evaluations/metrics/answerability/miprov2/{timestamp}")
        output_dir.mkdir(parents=True, exist_ok=True)

        miprov2_optimizer = MIPROv2(
            metric=self._compare_metric,
            prompt_model=dspy.LM("openai/gpt-4o-mini"),
        )
        miprov2_metric = miprov2_optimizer.compile(
            self.answerability,
            trainset=self.train_examples,
        )
        spearman_correlation, se = self.validate_metric(miprov2_metric)
        miprov2_metric.save(output_dir / "prompt.json")

        with open(output_dir / "performance.json", "w") as f:
            json.dump(
                {
                    "corr": spearman_correlation,
                    "se_approx": se,
                },
                f,
                indent=4,
            )

    def optimize_with_labeled_few_shot(self):
        """Optimize the metric prompt using DSPy's Labeled Few-Shot."""
        self._init_dspy()
        assert self.answerability is not None, (
            "Answerability metric must be initialized."
        )
        assert self.train_examples is not None, "Train examples must be initialized."

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for k in [1, 2, 4, 8, 16, 32, 64, 128]:
            output_dir = Path(
                f"evaluations/metrics/answerability/labeledfewshot/{timestamp}/k={k}"
            )
            output_dir.mkdir(parents=True, exist_ok=True)

            lfs_optimizer = LabeledFewShot(k=k)
            lfs_metric = lfs_optimizer.compile(
                self.answerability,
                trainset=self.train_examples,
            )
            spearman_correlation, se = self.validate_metric(lfs_metric)
            lfs_metric.save(output_dir / "prompt.json")

            with open(output_dir / "performance.json", "w") as f:
                json.dump(
                    {
                        "corr": spearman_correlation,
                        "se_approx": se,
                    },
                    f,
                    indent=4,
                )

    def _compare_metric(self, pred_score, human_score, trace=None):
        return pred_score.answerability == human_score.answerability

    def plot(self):
        """Plot the Spearman correlation of different models."""
        # Optional: assign colors to models if you want consistent coloring
        model_colors = {
            "gemini-2.5-flash-lite": "royalblue",
            "gemini-2.5-flash": "royalblue",
            "gemini-2.5-pro": "royalblue",
            "gemma-3-27b-it": "gray",
            "Qwen3-30B-A3B-Instruct-2507": "gray",
            "gpt-4.1": "darkgreen",
            "gpt-4.1-mini": "darkgreen",
            "gpt-4.1-nano": "darkgreen",
            "gpt-4o-mini": "darkgreen",
        }

        # Collect data
        model_data = []
        for file in os.listdir("evaluations/metrics/answerability/base/"):
            if file.endswith(".json"):
                file_path = os.path.join(
                    "evaluations/metrics/answerability/base/", file
                )
                model_name = file.split("_")[0]
                with open(file_path, "r") as f:
                    data = json.load(f)["stats"]
                model_data.append(
                    (
                        model_name,
                        data["spearman_correlation"],
                        data["standard_error_approx"],
                    )
                )

        # Sort by Spearman correlation (descending)
        model_data.sort(key=lambda x: x[1], reverse=True)

        # Unpack for plotting
        models = [x[0] for x in model_data]
        answerability_scores = [x[1] for x in model_data]
        # standard_errors = [x[2] for x in model_data]
        colors = [model_colors.get(model, "blue") for model in models]

        # Plot
        plt.figure(figsize=(8, 6))
        bars = plt.bar(
            range(len(answerability_scores)),
            answerability_scores,
            # yerr=standard_errors,
            capsize=5,
            color=colors,
        )
        # Add value labels above bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.005,  # Slightly above the bar
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        plt.xlabel("Model")
        plt.ylabel("Spearman Correlation")
        plt.title("Answerability: Spearman Correlation by Model")
        plt.xticks(range(len(models)), models, rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig("outputs/answerability_plot.png")
