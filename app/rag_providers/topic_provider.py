"""
Topic-based RAG provider for the Aptos Developer Assistant.

This provider uses topic-based chunking to improve retrieval by considering
document relationships and topic coherence.
"""

import logging
from typing import Dict, Any, List, Optional

from app.rag_providers import RAGProvider, RAGProviderRegistry
from app.utils.topic_chunks import (
    load_enhanced_chunks,
    initialize_vector_store,
    get_topic_aware_context,
)

logger = logging.getLogger(__name__)


class TopicRAGProvider(RAGProvider):
    """
    RAG provider that uses topic-based chunking for improved retrieval.
    """

    def __init__(self):
        """Initialize the provider's state."""
        self.initialized = False
        self.enhanced_chunks = []
        self.vector_store = None

    @property
    def name(self) -> str:
        """Get the provider's name."""
        return "topic"

    @property
    def description(self) -> str:
        """Get the provider's description."""
        return "Topic-based RAG provider for Aptos documentation"

    async def initialize(self, config: Dict[str, Any] = None) -> None:
        """
        Initialize the provider by loading enhanced chunks and setting up the vector store.

        Args:
            config: Configuration dictionary (not used in this provider)
        """
        try:
            logger.info("Initializing topic-based RAG provider")

            # Load enhanced chunks
            self.enhanced_chunks = await load_enhanced_chunks()
            if not self.enhanced_chunks:
                logger.warning("No enhanced chunks loaded")
                self.initialized = False
                return

            # Initialize vector store
            self.vector_store = await initialize_vector_store(self.enhanced_chunks)
            if not self.vector_store:
                logger.warning("Failed to initialize vector store")
                self.initialized = False
                return

            self.initialized = True
            logger.info("Topic-based RAG provider initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing topic-based RAG provider: {e}")
            self.initialized = False

    async def get_relevant_context(
        self, query: str, k: int = 5, include_series: bool = True, **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Get relevant context for a query using topic-based retrieval.

        Args:
            query: The user query
            k: Number of top documents to return
            include_series: Whether to include related documents (repurposed for topic relationships)
            **kwargs: Additional arguments

        Returns:
            List of dictionaries containing content, section, source, and metadata
        """
        if not self.initialized:
            logger.warning("Topic-based RAG provider not initialized")
            return []

        try:
            # Get relevant context
            results = await get_topic_aware_context(
                query=query,
                vector_store=self.vector_store,
                enhanced_chunks=self.enhanced_chunks,
                k=k,
            )

            # Format results
            formatted_results = []
            for result in results:
                formatted_result = {
                    "content": result["content"],
                    "section": result["section"],
                    "source": result["source"],
                    "summary": result["summary"],
                    "score": result["score"],
                    "metadata": {
                        "related_documents": result["related_documents"],
                        "is_priority": result["is_priority"],
                    },
                }
                formatted_results.append(formatted_result)

            logger.info(f"Retrieved {len(formatted_results)} relevant chunks for query")
            return formatted_results

        except Exception as e:
            logger.error(f"Error retrieving relevant context: {e}")
            return []


# Register the provider as the default
topic_provider = TopicRAGProvider()
RAGProviderRegistry.register(topic_provider, is_default=True)
