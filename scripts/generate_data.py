#!/usr/bin/env python3
"""Script to generate all necessary data for the chatbot."""

import os
import sys
import asyncio
import logging
import yaml
from typing import List, Dict, Any
from pathlib import Path

# Add the app directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import (
    PROVIDER_TYPES,
    get_generated_data_path,
    get_content_path,
    get_vector_store_path,
)
from app.utils.generate_url_mappings import generate_mappings
from scripts.preprocess_topic_chunks import process_documentation
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def initialize_vector_store(
    enhanced_chunks: List[Dict[str, Any]], vector_store_path: str
) -> None:
    """Initialize and save the vector store."""
    try:
        # Convert enhanced chunks to Documents
        documents = []
        for chunk in enhanced_chunks:
            doc = Document(
                page_content=chunk["content"],
                metadata={**chunk.get("metadata", {}), "id": chunk["id"]},
            )
            documents.append(doc)

        if not documents:
            logger.error(f"No documents to create vector store at {vector_store_path}")
            # Create an empty directory to indicate initialization was attempted
            os.makedirs(os.path.dirname(vector_store_path), exist_ok=True)
            return

        # Initialize embeddings and vector store
        embeddings = OpenAIEmbeddings("text-embedding-3-large")
        vector_store = FAISS.from_documents(documents, embeddings)

        # Save vector store
        os.makedirs(os.path.dirname(vector_store_path), exist_ok=True)
        vector_store.save_local(vector_store_path)

    except Exception as e:
        logger.error(f"Error initializing vector store: {e}")
        raise


async def generate_provider_data(provider: str) -> None:
    """Generate all necessary data for a specific provider."""
    try:
        # Create generated data directory
        generated_dir = get_generated_data_path(provider)
        os.makedirs(generated_dir, exist_ok=True)

        # 1. Generate URL mappings
        docs_dir = os.path.join(get_content_path(provider), "apps/nextra/pages")
        if not os.path.exists(docs_dir):
            docs_dir = get_content_path(provider)  # Fallback to base content path
        url_mappings_file = os.path.join(generated_dir, "url_mappings.yaml")

        mappings, redirects = await generate_mappings(docs_dir)

        # Save URL mappings
        os.makedirs(os.path.dirname(url_mappings_file), exist_ok=True)
        with open(url_mappings_file, "w") as f:
            yaml.safe_dump(
                {"mappings": mappings, "redirects": redirects},
                f,
                default_flow_style=False,
            )

        # 2. Generate enhanced chunks
        enhanced_chunks_file = os.path.join(generated_dir, "enhanced_chunks.json")
        enhanced_chunks = process_documentation(
            docs_dir=docs_dir,  # Use the same docs_dir as URL mappings
            output_path=enhanced_chunks_file,
        )

        # 3. Initialize and save vector store
        vector_store_path = get_vector_store_path(provider)
        initialize_vector_store(enhanced_chunks, vector_store_path)

    except Exception as e:
        logger.error(f"Error generating data for {provider}: {e}")
        raise


async def generate_all_data() -> None:
    """Generate all necessary data for all providers."""
    for provider in PROVIDER_TYPES.__args__:
        await generate_provider_data(provider)


def main():
    """Main entry point."""
    try:
        # Create necessary directories
        for provider in PROVIDER_TYPES.__args__:
            os.makedirs(get_generated_data_path(provider), exist_ok=True)

        # Run data generation
        asyncio.run(generate_all_data())

    except Exception as e:
        logger.error(f"Error during data generation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
