"""Configuration settings for the application."""

import os
from typing import Dict, Literal

# Documentation provider types
PROVIDER_TYPES = Literal["developer-docs", "aptos-learn"]
DEFAULT_PROVIDER = os.getenv("DEFAULT_PROVIDER", "developer-docs")

# Base paths for different documentation sources
DOC_BASE_PATHS: Dict[str, str] = {
    "developer-docs": os.getenv("DEVELOPER_DOCS_PATH", "data/developer-docs"),
    "aptos-learn": os.getenv("APTOS_LEARN_PATH", "data/aptos-learn"),
}

# Generated data paths
GENERATED_DATA_PATHS: Dict[str, str] = {
    "developer-docs": "data/generated/developer-docs",
    "aptos-learn": "data/generated/aptos-learn",
}

# Documentation URL settings
DOCS_BASE_URLS: Dict[str, str] = {
    "developer-docs": os.getenv("DEVELOPER_DOCS_URL", "https://aptos.dev"),
    "aptos-learn": os.getenv("APTOS_LEARN_URL", "https://learn.aptoslabs.com"),
}
DOCS_LANG_PREFIX = os.getenv("DOCS_LANG_PREFIX", "en")

# Vector store paths
VECTOR_STORE_PATHS: Dict[str, str] = {
    "developer-docs": os.path.join(
        GENERATED_DATA_PATHS["developer-docs"], "vector_store"
    ),
    "aptos-learn": os.path.join(GENERATED_DATA_PATHS["aptos-learn"], "vector_store"),
}

# Documentation content paths
CONTENT_PATHS: Dict[str, str] = {
    "developer-docs": DOC_BASE_PATHS["developer-docs"],
    "aptos-learn": os.path.join(DOC_BASE_PATHS["aptos-learn"], "src", "content"),
}


def get_generated_data_path(provider: PROVIDER_TYPES) -> str:
    """Get the generated data path for a provider."""
    return GENERATED_DATA_PATHS[provider]


def get_vector_store_path(provider: PROVIDER_TYPES) -> str:
    """Get the vector store path for a provider."""
    return VECTOR_STORE_PATHS[provider]


def get_content_path(provider: PROVIDER_TYPES) -> str:
    """Get the content path for a provider."""
    return CONTENT_PATHS[provider]


def get_docs_url(path: str, provider: PROVIDER_TYPES = DEFAULT_PROVIDER) -> str:
    """
    Convert a documentation path to a URL.

    Args:
        path: The documentation path to convert
        provider: The provider type to use for the base URL

    Returns:
        The full documentation URL
    """
    # Remove file extension and handle index files
    if path.endswith((".md", ".mdx")):
        path = os.path.splitext(path)[0]
    if path.endswith("/index") or path == "index":
        path = path[:-6] if path.endswith("/index") else ""

    # Format the URL path
    url_path = path.replace(os.path.sep, "/")
    if not url_path.startswith(f"{DOCS_LANG_PREFIX}/"):
        url_path = f"{DOCS_LANG_PREFIX}/{url_path}"

    # Get the base URL for the provider
    base_url = DOCS_BASE_URLS[provider]

    return f"{base_url}/{url_path}"
