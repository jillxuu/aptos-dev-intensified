#!/usr/bin/env python3
"""
Preprocess Aptos documentation with topic-based chunking.

This script analyzes the Aptos developer documentation, identifies topic relationships
between document chunks, and stores enhanced metadata for use by the RAG system.
"""

import os
import json
import sys
import logging
import argparse
import pickle
from typing import List, Dict, Any, Set, Tuple
from pathlib import Path
import numpy as np
from collections import defaultdict
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import hashlib
from app.config import CONTENT_PATHS, DEFAULT_PROVIDER, DOC_BASE_PATHS
from app.path_registry import path_registry

# Add the project root to the path so we can import from app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import functions from the app
from langchain_core.documents import Document

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("logs/topic_chunking.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Ensure NLTK resources are downloaded
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

# Define priority topics (similar to aptos-qa-dataset)
PRIORITY_TOPICS = [
    "wallet",
    "wallets",
    "petra",
    "martian",
    "api",
    "apis",
    "rest api",
    "fullnode api",
    "indexer api",
    "move",
    "smart contract",
    "smart contracts",
    "module",
    "modules",
    "sdk",
    "sdks",
    "typescript",
    "python",
    "rust",
    "cli",
    "command line",
    "aptos cli",
    "faucet",
    "testnet",
    "devnet",
    "testnet tokens",
    "getting started",
    "tutorial",
    "quickstart",
    "transaction",
    "transactions",
    "account",
    "accounts",
    "token",
    "tokens",
    "nft",
    "nfts",
    "digital assets",
]

# Configuration
SIMILARITY_THRESHOLD = 0.2  # Minimum similarity score to consider chunks related
MAX_RELATED_CHUNKS = 10  # Maximum number of related chunks to store
MIN_KEYWORD_LENGTH = 4  # Minimum length for a keyword to be considered significant
MAX_KEYWORDS_PER_CHUNK = 20  # Maximum number of keywords to extract per chunk

path_registry = PathRegistry()


def extract_significant_keywords(title: str, content: str) -> Set[str]:
    """
    Extract significant keywords from title and content.

    Args:
        title: The title of the document chunk
        content: The content of the document chunk

    Returns:
        A set of significant keywords
    """
    # Get stopwords
    stop_words = set(stopwords.words("english"))

    # Add common technical stopwords
    additional_stopwords = {
        "use",
        "using",
        "used",
        "example",
        "examples",
        "can",
        "will",
        "may",
        "also",
        "one",
        "two",
        "three",
        "first",
        "second",
        "third",
        "following",
        "follow",
        "follows",
        "see",
        "look",
        "looking",
        "note",
        "notes",
        "important",
        "step",
        "steps",
        "guide",
        "guides",
        "tutorial",
        "tutorials",
        "documentation",
        "docs",
        "document",
        "documents",
        "file",
        "files",
        "code",
        "codes",
        "function",
        "functions",
        "method",
        "methods",
        "class",
        "classes",
        "object",
        "objects",
        "return",
        "returns",
        "value",
        "values",
        "parameter",
        "parameters",
        "argument",
        "arguments",
        "type",
        "types",
        "string",
        "number",
        "boolean",
        "array",
        "object",
        "null",
        "undefined",
        "true",
        "false",
    }
    stop_words.update(additional_stopwords)

    # Extract words from title (with higher weight)
    title_words = set()
    if title:
        try:
            # Tokenize and filter title words
            title_tokens = word_tokenize(title.lower())
            title_words = {
                word
                for word in title_tokens
                if word.isalnum()
                and len(word) >= MIN_KEYWORD_LENGTH
                and word not in stop_words
            }
        except Exception as e:
            logger.warning(f"Error tokenizing title: {e}")
            # Fallback to simple splitting
            title_words = {
                word
                for word in title.lower().split()
                if word.isalnum()
                and len(word) >= MIN_KEYWORD_LENGTH
                and word not in stop_words
            }

    # Extract words from content
    content_words = set()
    if content:
        try:
            # Use only the first 1000 characters of content for efficiency
            content_preview = content[:1000].lower()
            content_tokens = word_tokenize(content_preview)
            content_words = {
                word
                for word in content_tokens
                if word.isalnum()
                and len(word) >= MIN_KEYWORD_LENGTH
                and word not in stop_words
            }
        except Exception as e:
            logger.warning(f"Error tokenizing content: {e}")
            # Fallback to simple splitting
            content_words = {
                word
                for word in content_preview.split()
                if word.isalnum()
                and len(word) >= MIN_KEYWORD_LENGTH
                and word not in stop_words
            }

    # Combine words, prioritizing title words
    all_words = title_words.union(content_words)

    # Check for priority topics
    priority_keywords = set()
    for topic in PRIORITY_TOPICS:
        if topic.lower() in title.lower() or topic.lower() in content.lower():
            # Add the topic as a keyword
            priority_keywords.add(topic.lower())

    # Combine all keywords
    all_keywords = priority_keywords.union(all_words)

    # Limit the number of keywords
    if len(all_keywords) > MAX_KEYWORDS_PER_CHUNK:
        # Prioritize keeping title words and priority keywords
        priority_set = priority_keywords.union(title_words)
        remaining_slots = MAX_KEYWORDS_PER_CHUNK - len(priority_set)

        if remaining_slots > 0:
            # Add some content words
            content_words_list = list(content_words - priority_set)
            selected_content_words = set(content_words_list[:remaining_slots])
            return priority_set.union(selected_content_words)
        else:
            # Just use priority words and title words
            return set(list(priority_set)[:MAX_KEYWORDS_PER_CHUNK])

    return all_keywords


def calculate_similarity(keywords1: Set[str], keywords2: Set[str]) -> float:
    """
    Calculate similarity between two sets of keywords using Jaccard similarity.

    Args:
        keywords1: First set of keywords
        keywords2: Second set of keywords

    Returns:
        Similarity score between 0 and 1
    """
    if not keywords1 or not keywords2:
        return 0.0

    # Calculate Jaccard similarity: intersection / union
    intersection = len(keywords1.intersection(keywords2))
    union = len(keywords1.union(keywords2))

    return intersection / union if union > 0 else 0.0


def generate_chunk_id(chunk) -> str:
    """
    Generate a unique ID for a document chunk.

    Args:
        chunk: Document chunk

    Returns:
        A unique ID string
    """
    # Create a unique identifier based on content and metadata
    content_hash = hashlib.md5(chunk.page_content.encode("utf-8")).hexdigest()

    # Include source in the ID if available
    source = chunk.metadata.get("source", "")
    if source:
        return f"{source}:{content_hash[:10]}"

    return content_hash


def analyze_topic_relationships(chunks: List[Any]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Analyze topic relationships between document chunks.

    Args:
        chunks: List of document chunks

    Returns:
        Dictionary mapping chunk IDs to lists of related chunks with similarity scores
    """
    logger.info(f"Analyzing topic relationships for {len(chunks)} chunks")

    # First pass: Extract topics from each chunk and assign IDs
    chunk_to_topics = {}
    chunk_ids = {}

    for i, chunk in enumerate(chunks):
        # Generate a unique ID for this chunk
        chunk_id = generate_chunk_id(chunk)
        chunk_ids[i] = chunk_id

        # Extract title from metadata
        title = ""
        for header_level in ["Header 1", "Header 2", "Header 3", "Header 4"]:
            if chunk.metadata.get(header_level):
                title += chunk.metadata.get(header_level) + " "
        title = title.strip()

        # Extract keywords
        keywords = extract_significant_keywords(title, chunk.page_content)

        # Store keywords for this chunk
        chunk_to_topics[chunk_id] = keywords

        # Add chunk ID to the metadata
        chunk.metadata["chunk_id"] = chunk_id

    logger.info(f"Extracted topics for {len(chunk_to_topics)} chunks")

    # Second pass: Calculate similarity between chunks
    chunk_relationships = {}

    for i, chunk in enumerate(chunks):
        chunk_id = chunk_ids[i]
        topics = chunk_to_topics[chunk_id]
        related_chunks = []

        # Compare with all other chunks
        for j, other_chunk in enumerate(chunks):
            if i == j:  # Skip self-comparison
                continue

            other_id = chunk_ids[j]
            other_topics = chunk_to_topics[other_id]

            # Calculate similarity
            similarity = calculate_similarity(topics, other_topics)

            # Store if above threshold
            if similarity > SIMILARITY_THRESHOLD:
                related_chunks.append({"chunk_id": other_id, "similarity": similarity})

        # Sort by similarity (highest first) and limit
        related_chunks.sort(key=lambda x: x["similarity"], reverse=True)
        chunk_relationships[chunk_id] = related_chunks[:MAX_RELATED_CHUNKS]

        # Log progress periodically
        if (i + 1) % 100 == 0 or i == len(chunks) - 1:
            logger.info(f"Processed relationships for {i+1}/{len(chunks)} chunks")

    return chunk_relationships


def mark_priority_chunks(chunks: List[Any]) -> None:
    """
    Mark chunks that contain priority topics.

    Args:
        chunks: List of document chunks
    """
    logger.info("Marking priority chunks")
    priority_count = 0

    for chunk in chunks:
        # Check if content contains any priority topics
        content = chunk.page_content.lower()
        is_priority = any(topic.lower() in content for topic in PRIORITY_TOPICS)

        # Mark in metadata
        chunk.metadata["is_priority"] = is_priority

        if is_priority:
            priority_count += 1

    logger.info(f"Marked {priority_count} priority chunks out of {len(chunks)}")


def store_enhanced_metadata(
    chunks: List[Any], relationships: Dict[str, List[Dict[str, Any]]], output_path: str
) -> None:
    """
    Store chunks with enhanced topic relationship metadata.

    Args:
        chunks: List of document chunks
        relationships: Dictionary of chunk relationships
        output_path: Path to store the enhanced data
    """
    logger.info(f"Storing enhanced metadata to {output_path}")

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    enhanced_data = []

    for chunk in chunks:
        chunk_id = chunk.metadata.get("chunk_id")
        if not chunk_id:
            continue

        # Get related chunks for this chunk
        related = relationships.get(chunk_id, [])

        # Add relationship metadata
        enhanced_chunk = {
            "id": chunk_id,
            "content": chunk.page_content,
            "metadata": {
                **chunk.metadata,
                "related_topics": [r["chunk_id"] for r in related],
                "topic_similarity_scores": {
                    r["chunk_id"]: r["similarity"] for r in related
                },
            },
        }

        enhanced_data.append(enhanced_chunk)

    # Save to disk
    with open(output_path, "w") as f:
        json.dump(enhanced_data, f)

    logger.info(f"Stored {len(enhanced_data)} enhanced chunks")


def process_markdown_document(content: str, file_path: str = "") -> List[Document]:
    """Process a markdown document into chunks with metadata."""
    try:
        # Get normalized path from registry
        normalized_path = path_registry.register_file(file_path)

        # Get URL for the path
        url = path_registry.get_url(normalized_path)

        # Base metadata
        metadata = {"source": normalized_path, "url": url}

        # Process the document into chunks
        chunks = []
        current_section = ""
        current_content = []

        for line in content.split("\n"):
            if line.startswith("#"):
                # If we have content, create a chunk
                if current_content:
                    chunk_content = "\n".join(current_content).strip()
                    if chunk_content:
                        chunk_metadata = {**metadata, "section": current_section}
                        chunks.append(
                            Document(
                                page_content=chunk_content, metadata=chunk_metadata
                            )
                        )
                # Update section title
                current_section = line.lstrip("#").strip()
                current_content = []
            else:
                current_content.append(line)

        # Handle the last chunk
        if current_content:
            chunk_content = "\n".join(current_content).strip()
            if chunk_content:
                chunk_metadata = {**metadata, "section": current_section}
                chunks.append(
                    Document(page_content=chunk_content, metadata=chunk_metadata)
                )

        return chunks

    except Exception as e:
        logger.error(f"Error processing markdown document: {e}")
        return []


def process_docs_with_topics(docs_dir: str) -> List[Dict[str, Any]]:
    """Process documentation with topic-based chunking."""
    logger.info(f"Processing documentation from {docs_dir}")

    # 1. Load and process documents
    chunks = []
    file_count = 0

    for root, _, files in os.walk(docs_dir):
        for file in files:
            if file.endswith((".md", ".mdx")) and not file.endswith("_meta.ts"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()

                    # Extract section info from file path
                    relative_path = os.path.relpath(file_path, docs_dir)
                    section_path = os.path.dirname(relative_path)
                    section = (
                        section_path.replace(os.path.sep, "/")
                        if section_path != "."
                        else "root"
                    )

                    # Process the document using our custom function
                    doc_sections = process_markdown_document(content, relative_path)

                    # Add each section to chunks with metadata
                    for doc_section in doc_sections:
                        if doc_section:
                            # Update metadata with file information
                            doc_section.metadata.update(
                                {
                                    "source": relative_path,
                                    "section": section,
                                    "file_type": (
                                        "mdx" if file.endswith(".mdx") else "md"
                                    ),
                                }
                            )
                            chunks.append(doc_section)

                    file_count += 1

                    # Log progress periodically
                    if file_count % 50 == 0:
                        logger.info(
                            f"Processed {file_count} files, found {len(chunks)} chunks"
                        )

                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {e}")

    logger.info(f"Processed {file_count} files, found {len(chunks)} chunks")

    # 2. Analyze topic relationships
    relationships = analyze_topic_relationships(chunks)

    # 3. Mark priority chunks
    mark_priority_chunks(chunks)

    # 4. Store enhanced data in memory
    enhanced_data = []
    for chunk in chunks:
        chunk_id = chunk.metadata.get("chunk_id")
        if not chunk_id:
            continue

        # Get related chunks for this chunk
        related = relationships.get(chunk_id, [])

        # Add relationship metadata
        enhanced_chunk = {
            "id": chunk_id,
            "content": chunk.page_content,
            "metadata": {
                **chunk.metadata,
                "related_topics": [r["chunk_id"] for r in related],
                "topic_similarity_scores": {
                    r["chunk_id"]: r["similarity"] for r in related
                },
            },
        }
        enhanced_data.append(enhanced_chunk)

    logger.info(f"Created {len(enhanced_data)} enhanced chunks")
    return enhanced_data


def process_documentation(docs_dir: str, output_path: str) -> List[Dict[str, Any]]:
    """Process documentation and save enhanced chunks."""
    try:
        # Process documentation and get enhanced chunks
        enhanced_chunks = process_docs_with_topics(docs_dir)

        # Save enhanced chunks
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(enhanced_chunks, f, indent=2)
        logger.info(f"Saved {len(enhanced_chunks)} enhanced chunks to {output_path}")

        return enhanced_chunks

    except Exception as e:
        logger.error(f"Error processing documentation: {e}")
        raise


def save_enhanced_chunks(
    chunks: List[Dict[str, Any]], provider: str = DEFAULT_PROVIDER
) -> None:
    """Save enhanced chunks to a JSON file."""
    output_dir = f"data/generated/{provider}"
    output_file = os.path.join(output_dir, "enhanced_chunks.json")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save enhanced chunks
    with open(output_file, "w") as f:
        json.dump(chunks, f, indent=2)
    logger.info(f"Saved {len(chunks)} enhanced chunks to {output_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Process Aptos documentation")
    parser.add_argument(
        "--docs-dir",
        default=CONTENT_PATHS[DEFAULT_PROVIDER],
        help="Path to the documentation directory",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(DOC_BASE_PATHS[DEFAULT_PROVIDER], "enhanced_chunks.json"),
        help="Path to store the enhanced data",
    )
    args = parser.parse_args()

    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    try:
        process_documentation(args.docs_dir, args.output)
        logger.info("Preprocessing completed successfully")
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
