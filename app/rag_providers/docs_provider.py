"""Docs RAG provider that handles different documentation paths with topic-based chunking."""

from typing import List, Dict, Any, Optional
import logging
import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from app.rag_providers import RAGProvider, RAGProviderRegistry
from app.utils.topic_chunks import (
    load_enhanced_chunks,
    initialize_vector_store,
    get_topic_aware_context,
)
from app.config import (
    PROVIDER_TYPES,
    DEFAULT_PROVIDER,
    get_vector_store_path,
    get_content_path,
    get_docs_url,
)

logger = logging.getLogger(__name__)


class DocsRAGProvider(RAGProvider):
    """RAG provider that handles different documentation paths with topic-based chunking."""

    def __init__(self):
        """Initialize the docs RAG provider."""
        self._initialized = False
        self.enhanced_chunks = []
        self.vector_store = None
        self._current_path = None

        # Map of initialized providers to avoid reinitializing
        self._initialized_providers: Dict[PROVIDER_TYPES, bool] = {
            provider_type: False for provider_type in PROVIDER_TYPES.__args__
        }
        # Store vector stores for each provider
        self._vector_stores: Dict[PROVIDER_TYPES, FAISS] = {}
        self._enhanced_chunks_map: Dict[PROVIDER_TYPES, List[Any]] = {}

    @property
    def name(self) -> str:
        """Get the name of the RAG provider."""
        return "docs"

    @property
    def description(self) -> str:
        """Get the description of the RAG provider."""
        return "RAG provider for Aptos documentation with topic-based chunking"

    async def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the RAG provider with the given configuration.

        Args:
            config: Configuration dictionary with:
                - docs_path: Path to documentation (one of PROVIDER_TYPES)
        """
        try:
            # Get configuration
            docs_path = config.get("docs_path", DEFAULT_PROVIDER)
            if docs_path not in PROVIDER_TYPES.__args__:
                raise ValueError(f"docs_path must be one of {PROVIDER_TYPES.__args__}")

            # Skip if already initialized for this path
            if self._initialized_providers.get(docs_path, False):
                logger.info(f"Provider already initialized for {docs_path}")
                return

            logger.info(f"Initializing docs RAG provider with path: {docs_path}")

            # Load enhanced chunks for the specified docs path
            enhanced_chunks = await load_enhanced_chunks(
                docs_path=get_content_path(docs_path)
            )
            if not enhanced_chunks:
                logger.warning(f"No enhanced chunks loaded for {docs_path}")
                return

            # Initialize vector store with topic-based chunks
            vector_store = await initialize_vector_store(
                enhanced_chunks, vector_store_path=get_vector_store_path(docs_path)
            )
            if not vector_store:
                logger.warning(f"Failed to initialize vector store for {docs_path}")
                return

            # Store the initialized components
            self._vector_stores[docs_path] = vector_store
            self._enhanced_chunks_map[docs_path] = enhanced_chunks
            self._initialized_providers[docs_path] = True

            # Set as current if not already set
            if not self._current_path:
                self._current_path = docs_path
                self.vector_store = vector_store
                self.enhanced_chunks = enhanced_chunks

            self._initialized = True
            logger.info(
                f"Initialized docs RAG provider with topic-based chunking for {docs_path}"
            )

        except Exception as e:
            logger.error(
                f"Failed to initialize docs RAG provider for {docs_path}: {str(e)}"
            )
            raise

    async def switch_provider(self, provider_type: PROVIDER_TYPES) -> None:
        """
        Switch to a different provider type.

        Args:
            provider_type: The provider type to switch to
        """
        # Initialize if not already initialized
        if not self._initialized_providers.get(provider_type, False):
            await self.initialize({"docs_path": provider_type})

        # Switch to the provider
        if self._initialized_providers.get(provider_type, False):
            self._current_path = provider_type
            self.vector_store = self._vector_stores[provider_type]
            self.enhanced_chunks = self._enhanced_chunks_map[provider_type]
            logger.info(f"Switched to {provider_type} provider")
        else:
            raise ValueError(f"Provider {provider_type} not initialized")

    async def get_relevant_context(
        self,
        query: str,
        k: int = 5,
        include_series: bool = True,
        provider_type: Optional[PROVIDER_TYPES] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get relevant context from the documentation using topic-based retrieval.

        Args:
            query: The user query
            k: Number of top documents to return
            include_series: Whether to include related documents from the same topic
            provider_type: Optional provider type to use for this query

        Returns:
            List of dictionaries containing content, section, source, and metadata
        """
        if not self._initialized:
            logger.warning("Docs RAG provider not initialized")
            return []

        try:
            # Switch provider if requested
            if provider_type and provider_type != self._current_path:
                await self.switch_provider(provider_type)

            # Get relevant context using topic-aware retrieval
            results = await get_topic_aware_context(
                query=query,
                vector_store=self.vector_store,
                enhanced_chunks=self.enhanced_chunks,
                k=k,
            )

            # Format results
            formatted_results = []
            for result in results:
                # Convert source path to URL using the current provider type
                source_url = (
                    get_docs_url(result["source"], self._current_path)
                    if result.get("source")
                    else ""
                )

                formatted_result = {
                    "content": result["content"],
                    "section": result["section"],
                    "source": source_url,  # Use the URL instead of the path
                    "source_path": result["source"],  # Keep the original path
                    "summary": result["summary"],
                    "score": result["score"],
                    "metadata": {
                        "related_documents": result["related_documents"],
                        "is_priority": result["is_priority"],
                        "docs_path": self._current_path,
                    },
                }
                formatted_results.append(formatted_result)

            logger.info(
                f"Retrieved {len(formatted_results)} relevant chunks for query using {self._current_path}"
            )
            return formatted_results

        except Exception as e:
            logger.error(f"Error retrieving relevant context: {e}")
            return []


# Register the docs RAG provider as the default
docs_provider = DocsRAGProvider()
RAGProviderRegistry.register(docs_provider, is_default=True)
