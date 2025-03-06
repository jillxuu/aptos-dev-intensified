"""RAG Provider Interface for the Aptos Chatbot Plugin."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class RAGProvider(ABC):
    """Abstract base class for RAG providers."""

    @abstractmethod
    async def get_relevant_context(
        self, query: str, k: int = 5, include_series: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get relevant context from the knowledge base.

        Args:
            query: The user query
            k: Number of top documents to return
            include_series: Whether to include related documents from the same series

        Returns:
            List of dictionaries containing content, section, source, and summary information
        """
        pass

    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the RAG provider with the given configuration.

        Args:
            config: Configuration dictionary for the RAG provider
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the name of the RAG provider."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Get the description of the RAG provider."""
        pass


class RAGProviderRegistry:
    """Registry for RAG providers."""

    _providers: Dict[str, RAGProvider] = {}
    _default_provider: Optional[str] = None

    @classmethod
    def register(cls, provider: RAGProvider, is_default: bool = False) -> None:
        """
        Register a RAG provider.

        Args:
            provider: The RAG provider to register
            is_default: Whether this provider should be the default
        """
        cls._providers[provider.name] = provider
        if is_default:
            cls._default_provider = provider.name

    @classmethod
    def get_provider(cls, name: Optional[str] = None) -> RAGProvider:
        """
        Get a RAG provider by name.

        Args:
            name: The name of the provider to get, or None for the default

        Returns:
            The requested RAG provider

        Raises:
            ValueError: If the provider is not found
        """
        if name is None:
            if cls._default_provider is None:
                raise ValueError("No default RAG provider registered")
            return cls._providers[cls._default_provider]

        if name not in cls._providers:
            raise ValueError(f"RAG provider '{name}' not found")

        return cls._providers[name]

    @classmethod
    def list_providers(cls) -> List[Dict[str, str]]:
        """
        List all registered RAG providers.

        Returns:
            List of dictionaries containing name and description of each provider
        """
        return [
            {"name": provider.name, "description": provider.description}
            for provider in cls._providers.values()
        ]
