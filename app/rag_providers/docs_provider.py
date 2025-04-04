"""Docs RAG provider that handles different documentation paths with topic-based chunking."""

from typing import List, Dict, Any, Optional
import logging
import os
import json
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from app.rag_providers import RAGProvider, RAGProviderRegistry
from app.utils.topic_chunks import (
    load_enhanced_chunks,
    initialize_vector_store,
    get_topic_aware_context,
)
from app.path_registry import path_registry
from app.config import (
    PROVIDER_TYPES,
    DEFAULT_PROVIDER,
    get_vector_store_path,
    get_content_path,
    get_generated_data_path,
)

logger = logging.getLogger(__name__)


class DocsRAGProvider(RAGProvider):
    """RAG provider that handles different documentation paths with topic-based chunking."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DocsRAGProvider, cls).__new__(cls)
            cls._instance._initialized = False
            cls._instance.enhanced_chunks = []
            cls._instance.vector_store = None
            cls._instance._current_path = None
            cls._instance.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
            cls._instance._initialized_providers = {
                provider_type: False for provider_type in PROVIDER_TYPES.__args__
            }
            cls._instance._vector_stores = {}
            cls._instance._enhanced_chunks_map = {}
        return cls._instance

    def __init__(self):
        """Initialize the docs RAG provider."""
        # Skip initialization if already done
        if hasattr(self, "_initialized"):
            return

        self._initialized = False
        self.enhanced_chunks = []
        self.vector_store = None
        self._current_path = None
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

        # Map of initialized providers to avoid reinitializing
        self._initialized_providers = {
            provider_type: False for provider_type in PROVIDER_TYPES.__args__
        }
        # Store vector stores for each provider
        self._vector_stores = {}
        self._enhanced_chunks_map = {}

    @property
    def name(self) -> str:
        """Get the name of the RAG provider."""
        return "docs"

    @property
    def description(self) -> str:
        """Get the description of the RAG provider."""
        return "RAG provider for Aptos documentation with topic-based chunking"

    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the provider with configuration."""
        try:
            # Get the provider type from config
            provider_type = config.get("docs_path", DEFAULT_PROVIDER)

            # Initialize path registry
            url_mappings_file = os.path.join(
                get_generated_data_path(provider_type), "url_mappings.yaml"
            )
            await path_registry.initialize_from_config(url_mappings_file)

            # Load enhanced chunks
            enhanced_chunks_file = os.path.join(
                get_generated_data_path(provider_type), "enhanced_chunks.json"
            )
            logger.info(f"Looking for enhanced chunks at: {enhanced_chunks_file}")

            if not os.path.exists(enhanced_chunks_file):
                logger.error(
                    f"Enhanced chunks file not found at: {enhanced_chunks_file}"
                )
                raise FileNotFoundError(
                    f"Enhanced chunks file not found at: {enhanced_chunks_file}"
                )

            file_size = os.path.getsize(enhanced_chunks_file) / (
                1024 * 1024
            )  # Convert to MB
            logger.info(f"Found enhanced chunks file (size: {file_size:.2f}MB)")

            with open(enhanced_chunks_file, "r") as f:
                self.enhanced_chunks = json.load(f)
            logger.info(
                f"Successfully loaded {len(self.enhanced_chunks)} chunks from file"
            )

            # Initialize vector store
            vector_store_path = get_vector_store_path(provider_type)
            self.vector_store = FAISS.load_local(
                vector_store_path,
                self.embeddings,
                allow_dangerous_deserialization=True,  # Safe since we generate these files ourselves
            )

            # Store the initialized data for this provider
            self._vector_stores[provider_type] = self.vector_store
            self._enhanced_chunks_map[provider_type] = self.enhanced_chunks
            self._current_path = provider_type

            # Mark as initialized
            self._initialized = True
            self._initialized_providers[provider_type] = True

            logger.info(
                f"Initialized docs provider with {len(self.enhanced_chunks)} chunks"
            )

        except Exception as e:
            logger.error(f"Error initializing docs provider: {e}")
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
        logger.info(f"[DOCS-RAG] Starting context retrieval for query: {query}")
        logger.info(
            f"[DOCS-RAG] Parameters: k={k}, include_series={include_series}, provider_type={provider_type}"
        )

        if not self._initialized:
            logger.warning("[DOCS-RAG] Provider not initialized - cannot proceed")
            return []

        try:
            logger.info(f"[DOCS-RAG] Current provider path: {self._current_path}")

            # Switch provider if requested
            if provider_type and provider_type != self._current_path:
                logger.info(
                    f"[DOCS-RAG] Switching provider from {self._current_path} to {provider_type}"
                )
                await self.switch_provider(provider_type)
                logger.info(
                    f"[DOCS-RAG] Successfully switched to {provider_type} provider"
                )

            # Validate current state
            if not self.vector_store:
                logger.error("[DOCS-RAG] Vector store is None")
                return []
            if not self.enhanced_chunks:
                logger.error("[DOCS-RAG] No enhanced chunks available")
                return []

            logger.info(
                f"[DOCS-RAG] Using vector store with {len(self.enhanced_chunks)} chunks"
            )

            # Get relevant context using topic-aware retrieval
            logger.info("[DOCS-RAG] Calling get_topic_aware_context")
            results = await get_topic_aware_context(
                query=query,
                vector_store=self.vector_store,
                enhanced_chunks=self.enhanced_chunks,
                k=k,
            )
            logger.info(
                f"[DOCS-RAG] Retrieved {len(results)} results from topic-aware context"
            )

            # Format results
            formatted_results = []
            logger.info("[DOCS-RAG] Starting result formatting")

            for i, result in enumerate(results):
                source_path = result.get("source")
                source_url = path_registry.get_url(source_path) if source_path else None

                logger.debug(f"[DOCS-RAG] Processing result {i+1}/{len(results)}")
                logger.debug(f"[DOCS-RAG] Source path: {source_path}")
                logger.debug(f"[DOCS-RAG] Source URL: {source_url}")

                formatted_result = {
                    "content": result["content"],
                    "section": result["section"],
                    "source": source_url or "",
                    "source_path": source_path,
                    "summary": result["summary"],
                    "score": result["score"],
                    "metadata": {
                        "related_documents": result["related_documents"],
                        "is_priority": result["is_priority"],
                        "docs_path": self._current_path,
                    },
                }
                formatted_results.append(formatted_result)

                # Log details about each result
                logger.debug(f"[DOCS-RAG] Result {i+1} details:")
                logger.debug(f"[DOCS-RAG]   Section: {formatted_result['section']}")
                logger.debug(f"[DOCS-RAG]   Score: {formatted_result['score']:.4f}")
                logger.debug(
                    f"[DOCS-RAG]   Priority: {formatted_result['metadata']['is_priority']}"
                )
                logger.debug(
                    f"[DOCS-RAG]   Related docs: {len(formatted_result['metadata']['related_documents'])}"
                )

            # Log summary of results
            if formatted_results:
                logger.info("[DOCS-RAG] Top 3 results by score:")
                sorted_results = sorted(
                    formatted_results, key=lambda x: x["score"], reverse=True
                )
                for i, result in enumerate(sorted_results[:3]):
                    logger.info(
                        f"[DOCS-RAG] {i+1}. {result['section']} (score: {result['score']:.4f})"
                    )

            logger.info(
                f"[DOCS-RAG] Successfully formatted {len(formatted_results)} results"
            )
            return formatted_results

        except Exception as e:
            logger.error(f"[DOCS-RAG] Error retrieving relevant context: {str(e)}")
            logger.error("[DOCS-RAG] Full traceback:", exc_info=True)
            return []


# Register the docs RAG provider as the default
docs_provider = DocsRAGProvider()
RAGProviderRegistry.register(docs_provider, is_default=True)
