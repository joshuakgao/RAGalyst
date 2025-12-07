"""Knowledge graph text processor for RAG."""

from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.transforms import apply_transforms, default_transforms

from ragalyst.data.text_processor.base import BaseTextProcessor


class KnowledgeGraphTextProcessor(BaseTextProcessor):
    """Knowledge graph text processor for RAG."""

    def __init__(self, cfg):
        """Initialize a knowledge graph text processor instance."""
        # Imported here to avoid circular import issues
        from ragalyst.module_registry import get_embedder, get_llm

        super().__init__(cfg)

        self.llm = get_llm(cfg)
        self.embedder = get_embedder(cfg)

    def process(self) -> KnowledgeGraph:
        """Generate knowledge graph from documents."""
        kg = KnowledgeGraph()
        for doc in self.documents:
            kg.nodes.append(
                Node(
                    type=NodeType.DOCUMENT,
                    properties={
                        "page_content": doc.page_content,
                        "document_metadata": doc.metadata,
                    },
                )
            )

        trans = default_transforms(
            documents=self.documents,
            llm=self.llm.get_ragas_wrapper(),
            embedding_model=self.embedder.get_ragas_wrapper(),
        )
        apply_transforms(kg, trans)

        self.kg = kg
        return kg
