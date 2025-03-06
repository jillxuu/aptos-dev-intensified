"""Example custom RAG provider implementation."""

from typing import List, Dict, Any
import logging
import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from app.rag_providers import RAGProvider, RAGProviderRegistry

logger = logging.getLogger(__name__)


class CustomRAGProvider(RAGProvider):
    """
    Example custom RAG provider that can be used as a template.

    This provider demonstrates how to implement a custom RAG provider
    using a FAISS vector store with OpenAI embeddings.
    """

    def __init__(self):
        """Initialize the custom RAG provider."""
        self._initialized = False
        self._vector_store = None
        self._embeddings = None
        self._config = {}

    @property
    def name(self) -> str:
        """Get the name of the RAG provider."""
        return "custom"

    @property
    def description(self) -> str:
        """Get the description of the RAG provider."""
        return "Custom RAG provider using your own knowledge base"

    async def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the RAG provider with the given configuration.

        Args:
            config: Configuration dictionary for the RAG provider with the following keys:
                - vector_store_path: Path to the FAISS vector store
                - openai_api_key: OpenAI API key (optional, defaults to environment variable)
        """
        try:
            # Get configuration
            vector_store_path = config.get("vector_store_path")
            openai_api_key = config.get("openai_api_key", os.getenv("OPENAI_API_KEY"))

            if not vector_store_path:
                raise ValueError("vector_store_path is required")

            if not openai_api_key:
                raise ValueError("OpenAI API key is required")

            # Initialize embeddings
            self._embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

            # Load vector store
            if os.path.exists(vector_store_path):
                self._vector_store = FAISS.load_local(
                    vector_store_path, self._embeddings
                )
                logger.info(f"Loaded vector store from {vector_store_path}")
            else:
                raise ValueError(f"Vector store not found at {vector_store_path}")

            self._config = config
            self._initialized = True
            logger.info("Initialized custom RAG provider")
        except Exception as e:
            logger.error(f"Failed to initialize custom RAG provider: {str(e)}")
            raise

    async def get_relevant_context(
        self, query: str, k: int = 5, include_series: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get relevant context from the custom knowledge base.

        Args:
            query: The user query
            k: Number of top documents to return
            include_series: Whether to include related documents from the same series

        Returns:
            List of dictionaries containing content, section, source, and summary information
        """
        if not self._initialized or not self._vector_store:
            logger.warning("Custom RAG provider not initialized")
            return []

        try:
            # Perform similarity search
            docs_with_scores = self._vector_store.similarity_search_with_score(
                query, k=k
            )

            # Format results
            results = []
            for doc, score in docs_with_scores:
                # Extract metadata
                metadata = doc.metadata

                # Create result dictionary
                result = {
                    "content": doc.page_content,
                    "score": float(score),
                    "source": metadata.get("source", "Unknown"),
                    "section": metadata.get("section", ""),
                    "summary": metadata.get("summary", ""),
                }

                results.append(result)

            return results
        except Exception as e:
            logger.error(f"Error in custom RAG provider: {str(e)}")
            return []


# Register the custom RAG provider (but not as default)
custom_provider = CustomRAGProvider()
RAGProviderRegistry.register(custom_provider)
