"""
Utility functions for working with topic-based chunks.

This module provides functions for loading enhanced chunks, initializing a vector store,
and retrieving topic-aware context.
"""

import os
import json
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)

# Configuration
ENHANCED_CHUNKS_PATH = "data/enhanced_chunks.json"
SIMILARITY_THRESHOLD = 0.7  # Minimum similarity score to include in results
MAX_CHUNKS_TO_RETURN = 10  # Maximum number of chunks to return


async def load_enhanced_chunks(
    file_path: str = ENHANCED_CHUNKS_PATH,
    docs_path: str = None,
) -> List[Dict[str, Any]]:
    """
    Load enhanced chunks from the specified file.

    Args:
        file_path: Path to the enhanced chunks file
        docs_path: Optional path to the documentation directory

    Returns:
        List of enhanced chunks
    """
    try:
        # If docs_path is provided, construct the file path
        if docs_path:
            file_path = os.path.join(docs_path, "enhanced_chunks.json")

        if not os.path.exists(file_path):
            logger.error(f"Enhanced chunks file not found: {file_path}")
            return []

        with open(file_path, "r") as f:
            enhanced_chunks = json.load(f)

        return enhanced_chunks
    except Exception as e:
        logger.error(f"Error loading enhanced chunks: {e}")
        return []


async def initialize_vector_store(
    enhanced_chunks: List[Dict[str, Any]],
    vector_store_path: str = None,
) -> Optional[FAISS]:
    """
    Initialize a vector store with enhanced chunks.

    Args:
        enhanced_chunks: List of enhanced chunks
        vector_store_path: Optional path to save/load the vector store

    Returns:
        Initialized vector store or None if initialization fails
    """
    try:
        if not enhanced_chunks:
            logger.warning(
                "No enhanced chunks provided for vector store initialization"
            )
            return None

        # Convert enhanced chunks to Documents with enhanced embedding text
        documents = []
        chunks_missing_summaries = 0

        for chunk in enhanced_chunks:
            # Check if this is a code block - either by contains_code flag or by checking for code markers
            is_code_block = (
                chunk["metadata"].get("contains_code", False)
                or "```" in chunk["content"]
            )

            # For code blocks, create a combined representation that includes the summary
            if is_code_block:
                # Get metadata for context
                code_summary = chunk["metadata"].get("code_summary", "")

                # If no code summary exists or it's empty, log a warning
                if not code_summary:
                    chunks_missing_summaries += 1
                    logger.warning(
                        f"Code block chunk {chunk['id']} is missing a code summary"
                    )

                parent_title = chunk["metadata"].get("parent_title", "")
                title = chunk["metadata"].get("title", "")
                code_languages = ", ".join(chunk["metadata"].get("code_languages", []))

                # Create enhanced embedding text that combines summary and code
                embedding_text = f"""
                Section: {parent_title or title}
                
                Purpose: {code_summary}
                
                Code{' (' + code_languages + ')' if code_languages else ''}:
                {chunk["content"]}
                """

                # Store original content in metadata and use enhanced text for embedding
                doc = Document(
                    page_content=embedding_text.strip(),  # Use combined text for embedding
                    metadata={
                        **chunk["metadata"],
                        "chunk_id": chunk["id"],  # Add chunk_id to metadata
                        "id": chunk[
                            "id"
                        ],  # Keep original id for backward compatibility
                        "original_content": chunk["content"],
                        "is_enhanced_embedding": True,
                        "contains_code": True,  # Ensure this is marked as a code block
                    },
                )
            else:
                # For non-code blocks, use regular content
                doc = Document(
                    page_content=chunk["content"],
                    metadata={
                        **chunk["metadata"],
                        "chunk_id": chunk["id"],  # Add chunk_id to metadata
                        "id": chunk[
                            "id"
                        ],  # Keep original id for backward compatibility
                    },
                )

            documents.append(doc)

        if chunks_missing_summaries > 0:
            logger.warning(
                f"Found {chunks_missing_summaries} code blocks missing summaries out of {len(enhanced_chunks)} total chunks"
            )

        # Initialize embeddings with text-embedding-3-large model
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large", openai_api_key=os.getenv("OPENAI_API_KEY")
        )

        # Create new vector store
        vector_store = FAISS.from_documents(documents, embeddings)

        # Save vector store if path is provided
        if vector_store_path:
            os.makedirs(os.path.dirname(vector_store_path), exist_ok=True)
            vector_store.save_local(vector_store_path)

        return vector_store
    except Exception as e:
        logger.error(f"Error initializing vector store: {e}")
        return None


async def get_topic_aware_context(
    query: str,
    vector_store: FAISS,
    enhanced_chunks: List[Dict[str, Any]],
    k: int = MAX_CHUNKS_TO_RETURN,
) -> List[Dict[str, Any]]:
    """
    Get topic-aware context for a query.

    Args:
        query: User query
        vector_store: Initialized vector store
        enhanced_chunks: List of enhanced chunks
        k: Maximum number of chunks to return

    Returns:
        List of relevant chunks with metadata
    """
    # Start timing the entire function
    start_time = time.time()
    logger.info(f"[TOPIC-AWARE] Starting retrieval for query: '{query}'")

    try:
        # Validate inputs - Step 1: Input validation
        validation_start = time.time()
        if not vector_store:
            logger.error("[TOPIC-AWARE] Vector store is None - cannot proceed")
            return []

        if not enhanced_chunks:
            logger.error("[TOPIC-AWARE] No enhanced chunks provided - cannot proceed")
            return []
        validation_time = time.time() - validation_start
        logger.info(f"[TOPIC-PERF] Input validation took {validation_time:.4f}s")

        # Step 2: Create chunk map
        chunk_map_start = time.time()
        # Create a mapping of chunk IDs to enhanced chunks for quick lookup
        chunk_map = {chunk["id"]: chunk for chunk in enhanced_chunks}

        # Log the number of items in the chunk map for debugging
        logger.info(f"[TOPIC-AWARE] Query: '{query}', chunk map size: {len(chunk_map)}")
        chunk_map_time = time.time() - chunk_map_start
        logger.info(f"[TOPIC-PERF] Chunk map creation took {chunk_map_time:.4f}s")

        # Step 3: Vector search
        vector_search_start = time.time()
        try:
            docs_with_scores = vector_store.similarity_search_with_score(query, k=k)
            logger.info(
                f"[TOPIC-AWARE] Found {len(docs_with_scores)} documents for query: {query}"
            )

            # Log a few result metadata samples for debugging
            if docs_with_scores:
                logger.info(
                    f"[TOPIC-AWARE] First document metadata keys: {list(docs_with_scores[0][0].metadata.keys())}"
                )
                logger.info(
                    f"[TOPIC-AWARE] First document content preview: {docs_with_scores[0][0].page_content[:100]}..."
                )
        except Exception as e:
            logger.error(f"[TOPIC-AWARE] Error during similarity search: {str(e)}")
            return []
        vector_search_time = time.time() - vector_search_start
        logger.info(f"[TOPIC-PERF] Vector search took {vector_search_time:.4f}s")

        # Step 4: Process results
        process_start = time.time()
        results = []
        skipped_doc_count = 0
        processed_doc_count = 0

        # Step 4a: Initial result extraction
        initial_processing_start = time.time()
        for i, (doc, score) in enumerate(docs_with_scores):
            # Try both chunk_id and id in metadata
            chunk_id = doc.metadata.get("chunk_id") or doc.metadata.get("id")

            if not chunk_id:
                skipped_doc_count += 1
                logger.warning(
                    f"[TOPIC-AWARE] Document {i+1} has no chunk_id or id in metadata"
                )
                # Add basic result even without chunk ID
                result = {
                    "content": doc.page_content,
                    "section": doc.metadata.get("section", ""),
                    "source": doc.metadata.get("source", ""),
                    "summary": doc.metadata.get("summary", ""),
                    "score": float(score),
                    "is_priority": doc.metadata.get("is_priority", False),
                    "related_documents": [],
                    "is_code_block": doc.metadata.get("contains_code", False),
                    "code_summary": None,
                    "code_languages": doc.metadata.get("code_languages", []),
                    "parent_info": None,
                }
                results.append(result)
                processed_doc_count += 1
                continue

            if chunk_id not in chunk_map:
                logger.warning(
                    f"[TOPIC-AWARE] Chunk ID {chunk_id} not found in chunk map"
                )
                continue

            # Get the enhanced chunk
            enhanced_chunk = chunk_map[chunk_id]

            # If this was an enhanced embedding, use the original content for display
            content = doc.metadata.get("original_content", doc.page_content)

            # Get code summary if available
            code_summary = None
            if doc.metadata.get("contains_code") and doc.metadata.get("code_summary"):
                code_summary = doc.metadata.get("code_summary")

            # Step 4b: Process related documents - most expensive part
            related_start = time.time()
            # Get related chunks
            related_ids = enhanced_chunk["metadata"].get("related_topics", [])
            related_chunks = []

            for related_id in related_ids:
                if related_id in chunk_map:
                    related_chunk = chunk_map[related_id]
                    similarity = (
                        enhanced_chunk["metadata"]
                        .get("topic_similarity_scores", {})
                        .get(related_id, 0)
                    )

                    if similarity >= SIMILARITY_THRESHOLD:
                        related_chunks.append(
                            {
                                "id": related_id,
                                "title": related_chunk["metadata"].get("title", ""),
                                "summary": related_chunk["metadata"].get("summary", ""),
                                "similarity": similarity,
                            }
                        )

            # Sort related chunks by similarity
            related_chunks.sort(key=lambda x: x["similarity"], reverse=True)
            related_time = time.time() - related_start

            # Step 4c: Process parent information
            parent_start = time.time()
            # Get hierarchical relationships if available
            parent_id = doc.metadata.get("parent_id")
            parent_info = None

            if parent_id:
                for chunk in enhanced_chunks:
                    if chunk["id"] == parent_id:
                        parent_info = {
                            "id": parent_id,
                            "title": chunk["metadata"].get("title", ""),
                            "summary": chunk["metadata"].get("summary", ""),
                        }
                        break
            parent_time = time.time() - parent_start

            # Format result
            result = {
                "content": content,
                "section": doc.metadata.get("section", ""),
                "source": doc.metadata.get("source", ""),
                "summary": doc.metadata.get("summary", ""),
                "score": float(score),
                "is_priority": doc.metadata.get("is_priority", False),
                "related_documents": related_chunks,
                "is_code_block": doc.metadata.get("contains_code", False),
                "code_summary": code_summary,
                "code_languages": doc.metadata.get("code_languages", []),
                "parent_info": parent_info,
            }

            results.append(result)
            processed_doc_count += 1

            # For the first document only, log detailed timing
            if i == 0:
                logger.info(
                    f"[TOPIC-PERF] First document: Related processing took {related_time:.4f}s, Parent processing took {parent_time:.4f}s"
                )

        initial_processing_time = time.time() - initial_processing_start
        logger.info(
            f"[TOPIC-PERF] Initial result processing took {initial_processing_time:.4f}s"
        )

        # Step 4d: Sort results
        sort_start = time.time()
        # Sort results by score and priority status
        results.sort(key=lambda x: (not x["is_priority"], -x["score"]))
        sort_time = time.time() - sort_start
        logger.info(f"[TOPIC-PERF] Sorting results took {sort_time:.4f}s")

        process_time = time.time() - process_start
        logger.info(f"[TOPIC-PERF] Total result processing took {process_time:.4f}s")

        # Log summary statistics
        total_time = time.time() - start_time
        logger.info(
            f"[TOPIC-AWARE] Results summary: {processed_doc_count} processed, {skipped_doc_count} skipped due to missing IDs"
        )
        logger.info(f"[TOPIC-PERF] PERFORMANCE SUMMARY:")
        logger.info(
            f"[TOPIC-PERF] - Input validation: {validation_time:.4f}s ({(validation_time/total_time)*100:.1f}%)"
        )
        logger.info(
            f"[TOPIC-PERF] - Chunk map creation: {chunk_map_time:.4f}s ({(chunk_map_time/total_time)*100:.1f}%)"
        )
        logger.info(
            f"[TOPIC-PERF] - Vector search: {vector_search_time:.4f}s ({(vector_search_time/total_time)*100:.1f}%)"
        )
        logger.info(
            f"[TOPIC-PERF] - Result processing: {process_time:.4f}s ({(process_time/total_time)*100:.1f}%)"
        )
        logger.info(f"[TOPIC-PERF] - Total time: {total_time:.4f}s")

        return results
    except Exception as e:
        total_time = time.time() - start_time
        logger.error(
            f"[TOPIC-AWARE] Error retrieving topic-aware context after {total_time:.4f}s: {str(e)}"
        )
        return []
