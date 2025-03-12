"""Default Aptos RAG provider implementation."""

from typing import List, Dict, Any
import logging
from app.rag_providers import RAGProvider, RAGProviderRegistry
from app.models import get_relevant_context as original_get_relevant_context

logger = logging.getLogger(__name__)


class AptosRAGProvider(RAGProvider):
    """Default RAG provider that uses the Aptos documentation."""

    def __init__(self):
        """Initialize the Aptos RAG provider."""
        self._initialized = False

    @property
    def name(self) -> str:
        """Get the name of the RAG provider."""
        return "aptos"

    @property
    def description(self) -> str:
        """Get the description of the RAG provider."""
        return "RAG provider using Aptos documentation"

    async def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the RAG provider with the given configuration.

        Args:
            config: Configuration dictionary for the RAG provider
        """
        # The default provider doesn't need additional configuration
        # as it's already initialized in the app.models module
        self._initialized = True
        logger.info("Initialized Aptos RAG provider")

    async def get_relevant_context(
        self, query: str, k: int = 5, include_series: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get relevant context from the Aptos documentation.

        Args:
            query: The user query
            k: Number of top documents to return
            include_series: Whether to include related documents from the same series

        Returns:
            List of dictionaries containing content, section, source, and summary information
        """
        if not self._initialized:
            logger.warning("Aptos RAG provider not initialized")
            return []

        # Use the existing implementation
        return original_get_relevant_context(query, k, include_series)


# Register the Aptos RAG provider as the default
aptos_provider = AptosRAGProvider()
RAGProviderRegistry.register(aptos_provider, is_default=True)
