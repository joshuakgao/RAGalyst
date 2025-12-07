"""QCA Dataset Manager for generating and evaluating Question-Answer-Context triples."""

import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Set

from tqdm import tqdm

from ragalyst.data.dataset_manager.base import BaseDatasetManager


class QcaDatasetManager(BaseDatasetManager):
    """Dataset manager for generating and evaluating QAC triples from documents."""

    def __init__(self, cfg):
        """Initialize a QcaDatasetManager instance."""
        # Imported here to avoid circular imports
        from ragalyst.module_registry import get_embedder, get_llm, get_metrics

        super().__init__(cfg)

        self.llm = get_llm(cfg)
        self.embedder = get_embedder(cfg)
        self.metrics = get_metrics(cfg)

        self.output_dir = Path(cfg.domain) / "qca"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Multithreading settings
        self.num_workers: int = cfg.data.dataset_manager.num_workers
        self.batch_size: int = cfg.data.dataset_manager.batch_size

        assert self.num_workers <= self.batch_size, (
            "Number of workers should not exceed batch size for multithreading."
        )

        assert cfg.data.text_processor.type == "chunk", (
            "QCA dataset generator only supports chunk text processor type."
        )

    def generate(self) -> List[dict]:
        """Generate Question-Answer-Context triples from a PDF using multithreading.

        Returns:
            List of generated QAC triples
        """
        self.text_processor.process()
        chunks: list | None = self.text_processor.chunks
        assert chunks is not None, "No document chunks found in PDF."
        total_chunks = len(chunks)

        # Calculate number of chunks to retrieve based on max context length and chunk size
        num_chunks_retrieved = max(
            1,
            int(
                self.cfg.data.dataset_manager.max_context_length
                / self.cfg.data.text_processor.chunk_size
            ),
        )

        qac_triples = []
        generated_questions = set()
        failed_chunks = set()

        # Initialize master progress bar
        master_pbar = tqdm(
            total=self.num_samples,
            desc="QAC GENERATION",
            position=0,
            leave=True,
            postfix={"chunks_left": total_chunks},
        )

        # Load existing data if output_dir exists
        if (self.output_dir / "data.json").exists():
            with open(self.output_dir / "data.json", "r") as f:
                existing_data = json.load(f)
                for item in existing_data:
                    qac_triples.append(item)
                    generated_questions.add(item["question"])

                master_pbar.update(len(qac_triples))
                master_pbar.refresh()

        if len(qac_triples) >= self.num_samples:
            print(
                f"Already generated {len(qac_triples)} samples, no need to generate more."
            )
            master_pbar.close()
            return qac_triples

        def generate_qca_worker(
            chunk_id: int,
            context: str,
            generated_questions: Set[dict],
            pbar: tqdm,
        ) -> dict | None:
            """Worker function to generate a single QAC triple.

            Args:
                chunk_id: ID of the chunk being processed
                context: Text context for QAC generation
                generated_questions: Set of already generated questions
                worker_position: Position for this worker's progress bar
                pbar: Progress bar for this worker

            Returns:
                Dictionary containing QAC triple or None if failed
            """
            import asyncio

            try:
                # Check if an event loop exists, if not create one
                loop = asyncio.get_event_loop()
            except RuntimeError:
                # Create a new event loop for this thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            try:
                # Generate question
                pbar.set_postfix({"status": "Generating question"})
                question = self._generate_question(context=context)
                if question == "None":
                    raise ValueError("Question generation failed")
                pbar.update(1)

                # Check answerability
                pbar.set_postfix({"status": "Checking answerability"})
                answerable = self.metrics.answerability.evaluate(
                    question=question, context=context, max_retries=3
                )

                if not answerable:
                    raise ValueError(f"Answerability failed: {answerable}")

                # if not answerable:
                #     # Try with expanded context
                #     prev_chunk = (
                #         chunks[chunk_id - 1].page_content if chunk_id > 0 else ""
                #     )
                #     next_chunk = (
                #         chunks[chunk_id + 1].page_content
                #         if chunk_id < len(chunks) - 1
                #         else ""
                #     )
                #     expanded_context = "\n".join(
                #         filter(None, [prev_chunk, context, next_chunk])
                #     )

                #     pbar.set_postfix({"status": f"Retrying with expanded context"})
                #     answerable = self.metrics.answerability.evaluate(
                #         question=question, context=expanded_context
                #     )
                #     if not answerable:
                #         raise ValueError(f"Answerability failed: {answerable}")
                #     context = expanded_context
                pbar.update(1)

                # Generate answer
                pbar.set_postfix({"status": "Generating answer"})
                answer = self._generate_answer(question, context)
                if answer == "None":
                    raise ValueError("Answer generation failed")
                pbar.update(1)

                # Check faithfulness
                pbar.set_postfix({"status": "Checking faithfulness"})
                fscore = self.metrics.faithfulness.evaluate(
                    question=question, answer=answer, context=context, max_retries=3
                )
                if fscore < 0.9:
                    raise ValueError(f"Faithfulness too low: {fscore:.2f}")
                pbar.update(1)

                # Check answer relevancy
                pbar.set_postfix({"status": "Checking relevancy"})
                ascore = self.metrics.answer_relevancy.evaluate(
                    question=question,
                    answer=answer,
                    context=context,
                    max_retries=3,
                )
                if ascore < 0.9:
                    raise ValueError(f"Answer relevancy too low: {ascore:.2f}")
                pbar.update(1)

                # Check for duplicates
                pbar.set_postfix({"status": "Checking duplicates"})
                if question in generated_questions:
                    raise ValueError("Duplicate question detected")
                pbar.update(1)

                pbar.set_postfix({"status": "✅"})
                pbar.update(1)

                return {
                    "context": context,
                    "question": question,
                    "answer": answer,
                }

            except Exception as e:
                failed_chunks.add(chunk_id)
                pbar.n = 6
                pbar.set_postfix({"status": f"❌ {str(e).replace(chr(10), ' ')}"})
                pbar.refresh()
                return None

        # Attempt 5 times the number of batches it takes to generate the samples
        # If we need 100 samples, we will attempt to generate a max of 500 samples
        # If batch size is 20, we will attempt 25 batches (20*25=500)
        # This is to ensure we don't have an infinite loop, while still being lenient to qac generation failures
        while len(qac_triples) < self.num_samples:
            available_chunks = [
                _ for _ in range(total_chunks) if _ not in failed_chunks
            ]
            master_pbar.set_postfix({"chunks_left": len(available_chunks)})

            if not available_chunks:
                print("All chunks exhausted")
                break

            # Create progress bars for each worker
            worker_pbars = []
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                # We schedule 20 task at a time so that the documents have a chance to pop
                # Once the documents are popped, the next 50 tasks will be scheduled
                futures = []
                # Generate more than needed to ensure enough valid samples
                for i in range(self.batch_size):
                    random_chunk_id = random.choice(
                        [i for i in range(total_chunks) if i not in failed_chunks]
                    )
                    begin_idx = max(0, random_chunk_id)
                    end_idx = min(total_chunks, random_chunk_id + num_chunks_retrieved)
                    context = "\n".join(
                        doc.page_content for doc in chunks[begin_idx:end_idx]
                    )

                    worker_pbars.append(
                        tqdm(
                            total=6,
                            desc=f"{i + 1}".zfill(2),
                            position=i + 2,
                            leave=False,
                            postfix={"status": "idle"},
                        )
                    )

                    futures.append(
                        executor.submit(
                            generate_qca_worker,
                            random_chunk_id,
                            context,
                            generated_questions,
                            worker_pbars[i],
                        )
                    )

                for future in as_completed(futures):
                    result = future.result()

                    # Continue if no qa is generated
                    if not result:
                        continue

                    # Add question to dataset if num_samples is not reached
                    if len(qac_triples) < self.num_samples:
                        qac_triples.append(result)
                        generated_questions.add(result["question"])
                        master_pbar.update(1)
                        with open(self.output_dir / "data.json", "w") as f:
                            json.dump(qac_triples, f, indent=4)

            # Check if we have enough samples
            assert len(qac_triples) <= self.num_samples, (
                f"Generated more samples than expected: {len(qac_triples)} > {self.num_samples}"
            )
            if len(qac_triples) == self.num_samples:
                master_pbar.write(
                    f"Generated {len(qac_triples)} samples, stopping generation."
                )
                master_pbar.close()
                return qac_triples

            master_pbar.write("Reloading chunks...")

        return qac_triples

    def _generate_question(self, context) -> str:
        prompt = f"""\
        You will be given a context.
        Create a question that is specific to the context. Avoid creating generic or general questions.
        The question should be answerable based on the information in the context only.
        Avoid using terms that refer to the context, such as "As mentioned in the text, as described in the context, ...".
        The question should be clear and unambiguous to understand without the context.
        If the context does not contain enough information to create answerable question, return "None" as the question.

        Here is the context.
        context: {context}

        Only provide the question as the output.
        """

        assert self.llm.model is not None, "LLM model is not initialized."
        question = self.llm.chat(context="", question=prompt)

        return question

    def _generate_answer(self, question, context) -> str:
        answer_template = f"""\
        You will be given a question and a context, answer the question using the information from the context only.
        The answer as close to the context as possible. However, picking information from different parts of the context is allowed.
        Do not paraphrase the context.
        Do not make up information that is not in the context.
        If the context does not contain enough information to answer the question, return "None" as the answer.

        Here is the question and context.
        question: {question}
        context: {context}

        Only provide the answer as the output.
        """

        assert self.llm.model is not None, "LLM model is not initialized."
        answer = self.llm.chat(context="", question=answer_template)
        return answer

    def evaluate(self) -> dict:
        """Evaluate the generated QAC triples using Ragas metrics."""
        assert self.output_dir.exists(), (
            "Output path does not exist. Please generate the dataset first."
        )

        with open(self.output_dir / "data.json", "r") as f:
            qac_triples = json.load(f)

        # Get qca's into their own lists
        questions = [item["question"] for item in qac_triples]
        answers = [item["answer"] for item in qac_triples]
        contexts = [item["context"] for item in qac_triples]

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
                "question": questions[i],
                "answer": answers[i],
                "context": contexts[i],
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
