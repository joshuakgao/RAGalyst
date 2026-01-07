# RAGalyst

A Retrieval-Augmented Generation (RAG) Evaluation Framework by the Structure and Artificial Intelligence Lab

Please contact Dr. Vedhus Hoskere (vhoskere@central.uh.edu) and Joshua Gao (jkgao@cougarnet.uh.edu) for any inquiries.

## Overview

RAGalyst is a fully automated end-to-end RAG evaluation framework that employs novel LLM-as-a-Judge metrcs: J-Answerability for improved QA synthesis, and J-Correctness for more accurate evaluation of RAG answer generation.

- Paper: https://arxiv.org/abs/2511.04502
- Data Repository (includes synthesized QA benchmarks, and source documents): https://huggingface.co/datasets/hoskerelab/ragalyst-qac

## Installation and Usage

1. Clone the repository

   ```bash
   git clone https://github.com/joshuakgao/RAGalyst.git
   cd RAGalyst
   ```

2. Setup Python Environment

   ```bash
   uv sync
   source .venv/bin/activate
   ```

3. Setup API Keys

   Create .env file in root dir. Follow .env.example and add your api keys.

   ```bash
   # Example
   OPENAI_API_KEY="sk-proj-abcdefghijklmnopqrstuvwxyz1234567890"
   GOOGLE_API_KEY="abcdefghijklmnopqrstuvwxyz1234567890"
   HF_TOKEN="hf_abcdefghijklmnopqrstuvwxyz1234567890"
   ```

4. Upload Domain-Specific Documents in root dir

   ```plaintext
   RAGalyst/
   ├── examples/
   ├── src/rageval/           # Main library
   ├── tests/
   └── electrician_manuals/        # Folder consisting of your domain-specific documents
       ├── changing_lightbulbs_for_dummies.pdf
       ├── ladder_safety_manual.txt
       └── clockwise_and_counterclockwise_difference.md

   ```

5. Generate QA Benchmark

   ```bash
   ragalyst electrician_manuals
   # Select: "Generate domain specific dataset"
   # Select embedder then LLM to be used for QA generation
   # Input how many QAs should be in your benchmark
   # Benchmark will be found in "electrician_manuals/qca/data.json"
   ```

6. Evaluate Embedding Models Retrieval Quality

   ```bash
   ragalyst electrician_manuals
   # Select: "Run experiment"
   # Select: "Embedder evaluation"
   # Select which embedders to evaluate
   # Evaluation results will be found in "electrician_manuals/embedder_eval"
   ```

7. Evaluate LLM Model Answer Generation Performance

   ```bash
   ragalyst electrician_manuals
   # Select: "Run experiment"
   # Select: "RAG evaluation"
   # Select which LLMs to evaluate
   # Select embedding model for retrieval
   # Evaluation results will be found in "electrician_manuals/llm_with_rag"
   ```

8. Evaluate LLM Model Answer Generation Performance with Different Number of Retrieved Text Chunks

   ```bash
   ragalyst electrician_manuals
   # Select: "Run experiment"
   # Select: "K chunks retrieved evaluation"
   # Select which LLM to evaluate
   # Select which embedding model for retrieval
   # Input number of chunks retrieved to evaluate
   # Evaluation results will be found in "electrician_manuals/llm_with_rag"
   ```

9. Perform Low J-Correctness Analysis

   ```bash
   ragalyst electrician_manuals
   # Select: "Run experiment"
   # Select: "Low answer correctness analysis"
   # Select which QA dataset you would like to analyze
   # Analysis will be found in "electrician_manuals/low_correctness_analysis"
   ```
