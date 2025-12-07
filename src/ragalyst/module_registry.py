"""Module registry for managing and retrieving instances of components in RAGalyst with lazy loading.

We load modules only when they are needed during experiment time rather than on CLI startup.
We do this since CLI startup time was too slow for good user experience.
Unfortunately, this does not support intellisense well in IDEs.
"""

import importlib

from omegaconf import DictConfig, open_dict


def lazy_import_class(path: str):
    """Lazy import a class from a module."""
    module, cls = path.rsplit(".", 1)
    return getattr(importlib.import_module(module), cls)


LLM_REGISTRY = {
    "ollama": "ragalyst.llm.OllamaLlm",
    "openai": "ragalyst.llm.OpenAiLlm",
    "gemini": "ragalyst.llm.GeminiLlm",
    "huggingface": "ragalyst.llm.HuggingfaceLlm",
}

TEXT_PROCESSOR_REGISTRY = {
    "chunk": "ragalyst.data.text_processor.ChunkTextProcessor",
    "knowledge_graph": "ragalyst.data.text_processor.KnowledgeGraphTextProcessor",
}

DATASET_MANAGER_REGISTRY = {
    "qca": "ragalyst.data.dataset_manager.QcaDatasetManager",
    "ragas_qca": "ragalyst.data.dataset_manager.RagasQcaDatasetManager",
}

RAG_REGISTRY = {
    "vector": "ragalyst.rag.VectorRag",
    "light": "ragalyst.rag.LightRag",
}

EMBEDDER_REGISTRY = {
    "gemini": "ragalyst.embedder.GeminiEmbedder",
    "huggingface": "ragalyst.embedder.HuggingfaceEmbedder",
    "ollama": "ragalyst.embedder.OllamaEmbedder",
    "openai": "ragalyst.embedder.OpenAiEmbedder",
}

FAISS_INDEX_REGISTRY = {
    "IndexFlatIP": "faiss.IndexFlatIP",
    "IndexFlatL2": "faiss.IndexFlatL2",
}

EXPERIMENT_REGISTRY = {
    "embedder_retrieval_eval": "ragalyst.experiment.EmbedderRetrievalEvaluation",
    "llm_with_rag_eval": "ragalyst.experiment.LlmWithragalystuation",
    "classify_low_correctness_reason": "ragalyst.experiment.ClassifyLowCorrectnessReason",
}


llm_instance = None
embedder_instance = None
text_processor_instance = None
dataset_manager_instance = None
metrics_instance = None
metrics_llm_instance = None
metrics_embedder_instance = None
rag_instance = None
faiss_index_instance = None
experiment_instance = None


def get_llm(cfg: DictConfig):
    """Get or create a singleton LLM instance based on the configuration."""
    global llm_instance
    if llm_instance is None:
        cls = lazy_import_class(LLM_REGISTRY[cfg.llm.type])
        llm_instance = cls(cfg)
    return llm_instance


def get_embedder(cfg: DictConfig):
    """Get or create a singleton Embedder instance based on the configuration."""
    global embedder_instance, metrics_embedder_instance

    embedder_type = cfg.embedder.type
    embedder_model = cfg.embedder.model_name
    embedder_device = cfg.embedder.get("device", "auto")

    metrics_type = cfg.metrics.get("embedder_type", None)
    metrics_model = cfg.metrics.get("embedder_model_name", None)
    metrics_device = cfg.metrics.get("embedder_device", "auto")

    same_as_metrics = (
        embedder_type == metrics_type
        and embedder_model == metrics_model
        and embedder_device == metrics_device
    )

    if same_as_metrics and metrics_embedder_instance is not None:
        embedder_instance = metrics_embedder_instance
        return embedder_instance

    if embedder_instance is None:
        cls = lazy_import_class(EMBEDDER_REGISTRY[embedder_type])
        embedder_instance = cls(cfg)

    return embedder_instance


def get_text_processor(cfg: DictConfig):
    """Get or create a singleton TextProcessor instance based on the configuration."""
    global text_processor_instance
    cls = lazy_import_class(TEXT_PROCESSOR_REGISTRY[cfg.data.text_processor.type])
    text_processor_instance = cls(cfg)
    return text_processor_instance


def get_dataset_manager(cfg: DictConfig):
    """Get or create a singleton DatasetManager instance based on the configuration."""
    global dataset_manager_instance
    cls = lazy_import_class(DATASET_MANAGER_REGISTRY[cfg.data.dataset_manager.type])
    dataset_manager_instance = cls(cfg)
    return dataset_manager_instance


def get_metrics(cfg: DictConfig):
    """Get or create a singleton Metrics instance based on the configuration."""

    class Metrics:
        def __init__(self, cfg):
            self.groundedness = lazy_import_class(
                "ragalyst.metrics.GroundednessMetric"
            )(cfg)
            self.faithfulness = lazy_import_class(
                "ragalyst.metrics.RagasFaithfulnessMetric"
            )(cfg)
            self.response_relevancy = lazy_import_class(
                "ragalyst.metrics.RagasResponseRelevancyMetric"
            )(cfg)
            self.relevance = lazy_import_class("ragalyst.metrics.RelevanceMetric")(cfg)
            self.correctness = lazy_import_class("ragalyst.metrics.CorrectnessMetric")(
                cfg
            )
            self.answer_correctness = lazy_import_class(
                "ragalyst.metrics.RagasAnswerCorrectnessMetric"
            )(cfg)
            self.answer_relevancy = lazy_import_class(
                "ragalyst.metrics.RagasAnswerRelevancyMetric"
            )(cfg)
            self.answerability = lazy_import_class(
                "ragalyst.metrics.AnswerabilityMetric"
            )(cfg)
            self.hit_rate = lazy_import_class("ragalyst.metrics.HitRateMetric")(cfg)
            self.rank = lazy_import_class("ragalyst.metrics.RankMetric")(cfg)
            self.standalone = lazy_import_class("ragalyst.metrics.StandaloneMetric")(
                cfg
            )

    global metrics_instance
    metrics_instance = Metrics(cfg)
    return metrics_instance


def get_metrics_llm(cfg: DictConfig):
    """Get or create a singleton LLM instance for metrics based on the configuration."""
    global metrics_llm_instance
    _cfg = cfg.copy()
    with open_dict(_cfg):
        _cfg.llm.type = cfg.metrics.llm_type
        _cfg.llm.model_name = cfg.metrics.llm_model_name
        _cfg.llm.device = cfg.metrics.get("llm_device", "auto")
        _cfg.llm.temperature = 0.0

    if metrics_llm_instance is None:
        cls = lazy_import_class(LLM_REGISTRY[cfg.metrics.llm_type])
        metrics_llm_instance = cls(_cfg)
    return metrics_llm_instance


def get_metrics_embedder(cfg: DictConfig):
    """Get or create a singleton Embedder instance for metrics based on the configuration."""
    global metrics_embedder_instance, embedder_instance

    _cfg = cfg.copy()
    with open_dict(_cfg):
        _cfg.embedder.type = cfg.metrics.embedder_type
        _cfg.embedder.model_name = cfg.metrics.embedder_model_name
        _cfg.embedder.device = cfg.metrics.get("embedder_device", "auto")

    same = (
        cfg.embedder.type == _cfg.embedder.type
        and cfg.embedder.model_name == _cfg.embedder.model_name
        and cfg.embedder.get("device", "auto") == _cfg.embedder.device
    )

    if same:
        if embedder_instance is None:
            cls = lazy_import_class(EMBEDDER_REGISTRY[cfg.embedder.type])
            embedder_instance = cls(cfg)
        metrics_embedder_instance = embedder_instance
    else:
        if metrics_embedder_instance is None:
            cls = lazy_import_class(EMBEDDER_REGISTRY[_cfg.embedder.type])
            metrics_embedder_instance = cls(_cfg)

    return metrics_embedder_instance


def get_rag(cfg: DictConfig):
    """Get or create a singleton RAG instance based on the configuration."""
    global rag_instance
    cls = lazy_import_class(RAG_REGISTRY[cfg.rag.type])
    rag_instance = cls(cfg)
    return rag_instance


def get_faiss_index(cfg: DictConfig, dim=None):
    """Get or create a singleton FAISS index instance based on the configuration."""
    global faiss_index_instance
    cls = lazy_import_class(FAISS_INDEX_REGISTRY[cfg.rag.index_type])
    faiss_index_instance = cls(dim)
    return faiss_index_instance


def get_experiment(cfg: DictConfig):
    """Get or create a singleton Experiment instance based on the configuration."""
    global experiment_instance
    cls = lazy_import_class(EXPERIMENT_REGISTRY[cfg.experiment.type])
    experiment_instance = cls(cfg)
    return experiment_instance
