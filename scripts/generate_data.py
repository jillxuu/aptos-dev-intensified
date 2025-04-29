#!/usr/bin/env python3
"""Script to generate all necessary data for the chatbot."""

import os
import sys
import asyncio
import logging
import yaml
from typing import List, Dict, Any
from pathlib import Path
import time

# Add the app directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import (
    PROVIDER_TYPES,
    provider_types_list,
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
        logger.info(f"Starting data generation for provider: {provider}")
        logger.info(f"Generated data directory: {generated_dir}")

        # 1. Generate URL mappings
        docs_dir = os.path.join(get_content_path(provider), "apps/nextra/pages")
        if not os.path.exists(docs_dir):
            docs_dir = get_content_path(provider)  # Fallback to base content path
        url_mappings_file = os.path.join(generated_dir, "url_mappings.yaml")

        logger.info(f"Generating URL mappings from docs directory: {docs_dir}")
        mappings, redirects = await generate_mappings(docs_dir)

        # Save URL mappings
        os.makedirs(os.path.dirname(url_mappings_file), exist_ok=True)
        logger.info(f"Saving URL mappings to: {url_mappings_file}")
        with open(url_mappings_file, "w") as f:
            yaml.safe_dump(
                {"mappings": mappings, "redirects": redirects},
                f,
                default_flow_style=False,
            )
        logger.info(f"Saved {len(mappings)} URL mappings")

        # 2. Generate enhanced chunks
        enhanced_chunks_file = os.path.join(generated_dir, "enhanced_chunks.json")
        logger.info(f"Generating enhanced chunks from docs directory: {docs_dir}")
        logger.info(f"Enhanced chunks will be saved to: {enhanced_chunks_file}")

        start_time = time.time()
        enhanced_chunks = process_documentation(
            docs_dir=docs_dir,  # Use the same docs_dir as URL mappings
            output_path=enhanced_chunks_file,
        )
        end_time = time.time()
        logger.info(
            f"Enhanced chunks generation completed in {end_time - start_time:.2f} seconds"
        )
        logger.info(f"Generated {len(enhanced_chunks)} enhanced chunks")

        # 3. Initialize and save vector store
        vector_store_path = get_vector_store_path(provider)
        logger.info(f"Initializing vector store at: {vector_store_path}")
        start_time = time.time()
        initialize_vector_store(enhanced_chunks, vector_store_path)
        end_time = time.time()
        logger.info(
            f"Vector store creation completed in {end_time - start_time:.2f} seconds"
        )
        logger.info(f"Data generation for provider {provider} completed successfully")

    except Exception as e:
        logger.error(f"Error generating data for {provider}: {e}")
        raise


async def generate_all_data() -> None:
    """Generate all necessary data for all providers."""
    for provider in provider_types_list:
        await generate_provider_data(provider)


def main():
    """Main entry point."""
    try:
        # Create necessary directories
        logger.info("Starting data generation process")
        start_time = time.time()

        for provider in provider_types_list:
            os.makedirs(get_generated_data_path(provider), exist_ok=True)

        # Run data generation
        asyncio.run(generate_all_data())

        end_time = time.time()
        logger.info(
            f"All data generation completed in {end_time - start_time:.2f} seconds"
        )

    except Exception as e:
        logger.error(f"Error during data generation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
