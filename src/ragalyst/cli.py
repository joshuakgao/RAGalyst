"""Entry point for the ragalyst command-line interface."""

import argparse
import glob
import os
from pathlib import Path

import inquirer

from ragalyst.load_config import load_config
from ragalyst.module_registry import get_dataset_manager, get_experiment


def parse_model_type_and_name(model: str) -> tuple[str, str]:
    """Parses type and model name from model string. "huggingface:gemma-1b-it" -> ("huggingface", "gemma-1b-it")."""
    split = model.split(":")
    model_type = split[0]
    model_name = ":".join(split[1:])
    return (model_type, model_name)


def prompt_model_selection(model_type: str, choices: list[str]) -> str:
    """Prompt user to select a model or enter a custom one."""
    q = [
        inquirer.List(
            "selection",
            message=f"What {model_type} would you like to use?",
            choices=choices + ["Other Model"],
        ),
    ]
    result = inquirer.prompt(q)
    assert result is not None

    if result["selection"] == "Other Model":
        print(
            """Format: (provider:model-name)
Examples: 
- openai:gpt-4o-mini
- gemini:gemini-2.5-flash
- ollama:gpt-oss:20b
- huggingface:google/gemma-3-27b-it)"""
        )
        q_custom = [
            inquirer.Text(
                "custom_model",
                message=f"Enter custom {model_type}",
            ),
        ]
        custom_result = inquirer.prompt(q_custom)
        assert custom_result is not None
        return custom_result["custom_model"]

    return result["selection"]


def generate_dataset_cfg() -> dict:
    """Set up configuration by prompting the user for inputs."""
    embedder_choices = [
        "huggingface:Qwen/Qwen3-Embedding-0.6B",
        "huggingface:Qwen/Qwen3-Embedding-8B",
    ]
    llm_choices = [
        "openai:gpt-4o-mini",
        "huggingface:google/gemma-3-1b-it",
    ]

    embedder = prompt_model_selection("embedder", embedder_choices)
    llm = prompt_model_selection("LLM", llm_choices)

    q = [
        inquirer.Text(
            "num_samples",
            message="How many samples would you like to generate for the domain?",
            default="100",
        ),
    ]
    result = inquirer.prompt(q)
    assert result is not None

    return {
        "embedder": embedder,
        "llm": llm,
        "num_samples": result["num_samples"],
    }


def generate_embedder_experiment_cfg() -> dict:
    """Set up configuration by prompting the user for inputs."""
    embedder_choices = [
        "huggingface:Qwen/Qwen3-Embedding-0.6B",
        "huggingface:Qwen/Qwen3-Embedding-8B",
    ]

    embedder = prompt_model_selection("embedder", embedder_choices)

    return {"embedder": embedder}


def generate_llm_with_rag_experiment_cfg() -> dict:
    """Set up configuration by prompting the user for inputs."""
    llm_choices = [
        "openai:gpt-4o-mini",
        "ollama:gemma3:27b",
        "huggingface:google/gemma-3-1b-it",
        "huggingface:Qwen/Qwen3-30B-A3B-Instruct-2507",
        "huggingface:openai/gpt-oss-20b",
        "huggingface:meta-llama/Llama-3.1-8B-Instruct",
    ]
    embedder_choices = [
        "huggingface:Qwen/Qwen3-Embedding-0.6B",
        "huggingface:Qwen/Qwen3-Embedding-8B",
    ]

    llm = prompt_model_selection("LLM", llm_choices)
    embedder = prompt_model_selection("embedder", embedder_choices)

    return {
        "llm": llm,
        "embedder": embedder,
    }


def generate_k_chunks_retrieved_experiment_cfg() -> dict:
    """Set up configuration by prompting the user for inputs."""
    llm_choices = [
        "openai:gpt-4o-mini",
        "ollama:gemma3:27b",
        "huggingface:google/gemma-3-1b-it",
        "huggingface:Qwen/Qwen3-30B-A3B-Instruct-2507",
        "huggingface:openai/gpt-oss-20b",
        "huggingface:meta-llama/Llama-3.1-8B-Instruct",
    ]
    embedder_choices = [
        "huggingface:Qwen/Qwen3-Embedding-0.6B",
        "huggingface:Qwen/Qwen3-Embedding-8B",
    ]

    llm = prompt_model_selection("LLM", llm_choices)
    embedder = prompt_model_selection("embedder", embedder_choices)

    q = [
        inquirer.Text(
            "k",
            message="Which k's do you want to test? Example: 1,2,3,4,5,10,27,99,100",
        ),
    ]
    result = inquirer.prompt(q)
    assert result is not None

    return {
        "llm": llm,
        "embedder": embedder,
        "k": result["k"],
    }


def generate_classify_low_correctness_reason_cfg(folder: str) -> dict:
    """Set up configuration by prompting the user for inputs."""
    search_pattern = os.path.join(f"{folder}/llm_with_rag", "**", "*.json")
    json_paths = glob.glob(search_pattern, recursive=True)
    q = [
        inquirer.List(
            "path",
            message="Which QA dataset would you like run a Low Answer Correctness analysis?",
            choices=json_paths,
        ),
    ]
    cfg = inquirer.prompt(q)
    assert cfg is not None
    return cfg


def main():
    """Main function for the ragalyst CLI."""
    print("Welcome to RAGalyst CLI!")

    parser = argparse.ArgumentParser(description="Process a folder of documents")
    parser.add_argument("folder", type=Path, help="Folder containing PDF/MD/TXT files")
    args = parser.parse_args()

    # Collect all files with the given extensions
    folder = args.folder
    exts = ["*.pdf", "*.md", "*.txt"]
    doc_files = []
    for pattern in exts:
        doc_files.extend(folder.glob(pattern))

    q = [
        inquirer.List(
            "action",
            message="What would you like to do?",
            choices=["Generate domain specific dataset", "Run experiment"],
        )
    ]
    action = inquirer.prompt(q)
    assert action is not None

    if action["action"] == "Generate domain specific dataset":
        input_cfg = generate_dataset_cfg()
        embedder = input_cfg["embedder"]
        embedder_type, embedder_name = parse_model_type_and_name(embedder)
        llm = input_cfg["llm"]
        llm_type, llm_name = parse_model_type_and_name(llm)
        num_samples = input_cfg["num_samples"]

        cfg = load_config(
            overrides=[
                f"domain={folder.name}",
                f"embedder.type={embedder_type}",
                f"embedder.model_name={embedder_name}",
                f"llm={llm_type}",
                f"llm.model_name={llm_name}",
                "metrics=openai",
                "metrics.llm_type=openai",
                "metrics.llm_model_name=gpt-4o-mini",
                "metrics.embedder_type=huggingface",
                "metrics.embedder_model_name=Qwen/Qwen3-Embedding-0.6B",
                "data.text_processor.type=chunk",
                "data.dataset_manager.type=qca",
                f"data.dataset_manager.num_samples={num_samples}",
                "data.dataset_manager.num_workers=2",
                "data.dataset_manager.batch_size=10",
            ]
        )
        manager = get_dataset_manager(cfg=cfg)
        manager.generate()
    elif action["action"] == "Run experiment":
        assert (Path(args.folder) / "qca" / "data.json").exists(), (
            "Dataset not found in the specified folder. Please generate the dataset first."
        )

        q = [
            inquirer.List(
                "experiment",
                message="What experiment would you like to run?",
                choices=[
                    "Embedder evaluation",
                    "RAG evaluation",
                    "K chunks retrieved evaluation",
                    "Low answer correctness analysis",
                ],
            ),
        ]
        experiment = inquirer.prompt(q)
        assert experiment is not None
        experiment = experiment["experiment"]

        if experiment == "Embedder evaluation":
            input_cfg = generate_embedder_experiment_cfg()
            embedder = input_cfg["embedder"]
            embedder_type, embedder_name = parse_model_type_and_name(embedder)

            cfg = load_config(
                overrides=[
                    "experiment=embedder_retrieval_eval",
                    f"domain={folder}",
                    f"embedder.type={embedder_type}",
                    f"embedder.model_name={embedder_name}",
                    "rag=vector",
                    "rag.top_k=10",
                ]
            )

            experiment = get_experiment(cfg)
            experiment.run()
        elif experiment == "RAG evaluation":
            input_cfg = generate_llm_with_rag_experiment_cfg()
            llm = input_cfg["llm"]
            llm_type, llm_name = parse_model_type_and_name(llm)
            embedder = input_cfg["embedder"]
            embedder_type, embedder_name = parse_model_type_and_name(embedder)

            cfg = load_config(
                overrides=[
                    "experiment=llm_with_rag_eval",
                    f"domain={folder}",
                    f"llm={llm_type}",
                    f"llm.model_name={llm_name}",
                    f"embedder={embedder_type}",
                    f"embedder.model_name={embedder_name}",
                    "metrics=openai",
                    "metrics.llm_type=openai",
                    "metrics.llm_model_name=gpt-4o-mini",
                    "metrics.embedder_type=huggingface",
                    "metrics.embedder_model_name=Qwen/Qwen3-Embedding-0.6B",
                    "rag=vector",
                    "rag.top_k=10",
                ]
            )
            experiment = get_experiment(cfg)
            experiment.run()
        elif experiment == "K chunks retrieved evaluation":
            input_cfg = generate_k_chunks_retrieved_experiment_cfg()
            k: str = input_cfg["k"]
            k_list = k.split(",")
            llm_type, llm_name = parse_model_type_and_name(input_cfg["llm"])
            embedder_type, embedder_name = parse_model_type_and_name(
                input_cfg["embedder"]
            )

            for k in k_list:
                cfg = load_config(
                    overrides=[
                        "experiment=llm_with_rag_eval",
                        f"domain={folder}",
                        f"llm={llm_type}",
                        f"llm.model_name={llm_name}",
                        f"embedder={embedder_type}",
                        f"embedder.model_name={embedder_name}",
                        "metrics=openai",
                        "metrics.llm_type=openai",
                        "metrics.llm_model_name=gpt-4o-mini",
                        "metrics.embedder_type=huggingface",
                        "metrics.embedder_model_name=Qwen/Qwen3-Embedding-0.6B",
                        "rag=vector",
                        f"rag.top_k={k}",
                    ]
                )
                experiment = get_experiment(cfg)
                experiment.run()
        elif experiment == "Low answer correctness analysis":
            input_cfg = generate_classify_low_correctness_reason_cfg(folder)
            cfg = load_config(
                overrides=[
                    "experiment=classify_low_correctness_reason",
                    f"experiment.evaluated_qa_dataset={input_cfg['path']}",
                    f"domain={folder}",
                    "llm=openai",
                    "llm.model_name=gpt-4o-mini",
                    # following cfgs are set to openai so no model is loaded into memory
                    "embedder=openai",
                    "metrics=openai",
                    "metrics.embedder_type=openai",
                    "metrics.embedder_model_name=text-embedding-3-small",
                ]
            )
            experiment = get_experiment(cfg)
            experiment.run()
