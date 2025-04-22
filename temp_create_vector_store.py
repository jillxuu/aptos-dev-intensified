#!/usr/bin/env python3
"""Script to create the vector store from enhanced chunks."""

import os
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any

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
    enhanced_chunks_file: str,
    vector_store_path: str
) -> None:
    """Initialize and save the vector store."""
    try:
        # Load enhanced chunks
        logger.info(f"Loading enhanced chunks from {enhanced_chunks_file}")
        with open(enhanced_chunks_file, "r") as f:
            enhanced_chunks = json.load(f)
        
        logger.info(f"Loaded {len(enhanced_chunks)} enhanced chunks")
        
        # Convert enhanced chunks to Documents
        documents = []
        for chunk in enhanced_chunks:
            metadata = chunk.get("metadata", {})
            metadata["id"] = chunk["id"]
            metadata["chunk_id"] = chunk["id"]  # Ensure chunk_id is set
            
            doc = Document(
                page_content=chunk["content"],
                metadata=metadata,
            )
            documents.append(doc)

        if not documents:
            logger.error(f"No documents to create vector store at {vector_store_path}")
            return

        # Initialize embeddings and vector store
        logger.info("Initializing OpenAI embeddings")
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        
        logger.info(f"Creating vector store with {len(documents)} documents")
        vector_store = FAISS.from_documents(documents, embeddings)

        # Save vector store
        os.makedirs(os.path.dirname(vector_store_path), exist_ok=True)
        logger.info(f"Saving vector store to {vector_store_path}")
        vector_store.save_local(vector_store_path)
        logger.info("Vector store saved successfully")

    except Exception as e:
        logger.error(f"Error initializing vector store: {e}")
        raise

def main():
    """Main entry point."""
    try:
        # Path settings
        enhanced_chunks_file = "data/generated/developer-docs/enhanced_chunks.json"
        vector_store_path = "data/generated/developer-docs/vector_store"
        
        # Check if enhanced chunks file exists
        if not os.path.exists(enhanced_chunks_file):
            logger.error(f"Enhanced chunks file not found: {enhanced_chunks_file}")
            sys.exit(1)
            
        # Initialize vector store
        initialize_vector_store(enhanced_chunks_file, vector_store_path)
        
    except Exception as e:
        logger.error(f"Error during vector store creation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 