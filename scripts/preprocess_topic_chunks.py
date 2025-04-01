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
import asyncio
from typing import List, Dict, Any, Set, Tuple, Optional
from pathlib import Path
import numpy as np
from collections import defaultdict
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import hashlib
from app.config import CONTENT_PATHS, DEFAULT_PROVIDER, DOC_BASE_PATHS
from app.path_registry import PathRegistry

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

# Define TEST_MODE from environment to prevent the error
TEST_MODE = os.getenv("CHAT_TEST_MODE", "false").lower() == "true"

# Initialize path registry
path_registry = PathRegistry()

def process_markdown_document(content: str, file_path: str = "") -> List[Document]:
    """
    Process markdown content with enhanced chunking that preserves code blocks with their context.

    Args:
        content: The markdown content to process
        file_path: The path to the file (for metadata)

    Returns:
        List of Document objects with hierarchical metadata
    """
    try:
        # Get normalized path and URL from registry
        normalized_path = path_registry.register_file(file_path)
        url = path_registry.get_url(normalized_path)

        # Split content into lines for processing
        lines = content.split("\n")
        chunks = []
        current_chunk = []
        current_title = ""
        current_header_level = 0
        in_code_block = False
        code_block_buffer = []
        code_language = ""

        for line in lines:
            # Check if line starts or ends a code block
            if line.startswith("```"):
                # Toggle code block state
                in_code_block = not in_code_block
                
                # If starting a code block, extract language if present
                if in_code_block:
                    code_language = line.replace("```", "").strip()
                
                # Add line to current chunk
                current_chunk.append(line)
                continue
                
            # If we're in a code block, add line and continue (no header processing)
            if in_code_block:
                current_chunk.append(line)
                continue
                
            # Process headers (only when not in code block)
            header_match = None
            new_header_level = 0
            
            if line.startswith("# "):
                header_match = line.replace("# ", "").strip()
                new_header_level = 1
            elif line.startswith("## "):
                header_match = line.replace("## ", "").strip()
                new_header_level = 2
            elif line.startswith("### "):
                header_match = line.replace("### ", "").strip()
                new_header_level = 3
                
            # If we found a header, create a new chunk
            if header_match:
                # Save previous chunk if it exists
                if current_chunk and current_title:
                    chunks.append({
                        "title": current_title,
                        "content": "\n".join(current_chunk),
                        "header_level": current_header_level
                    })
                    
                # Start new chunk
                current_title = header_match
                current_header_level = new_header_level
                current_chunk = [line]  # Include the header in the chunk
            else:
                # Add line to current chunk
                current_chunk.append(line)

        # Save the last chunk
        if current_chunk and current_title:
            chunks.append({
                "title": current_title,
                "content": "\n".join(current_chunk),
                "header_level": current_header_level
            })

        # If no chunks were created (no headers), create one chunk with the whole content
        if not chunks:
            title = os.path.basename(file_path) if file_path else "Untitled"
            chunks.append({
                "title": title,
                "content": content,
                "header_level": 0
            })
        
        # Process chunks to create hierarchical structure
        hierarchical_chunks = create_hierarchical_structure(chunks, file_path)
        
        # Create overlapping chunks to maintain context
        processed_chunks = create_overlapping_chunks(hierarchical_chunks)
        
        # Convert to Document objects with enhanced metadata
        documents = []
        for chunk in processed_chunks:
            if len(chunk["content"].strip()) > 0:  # Only include non-empty chunks
                # Extract code blocks for additional metadata
                contains_code = "```" in chunk["content"]
                code_languages = detect_code_languages(chunk["content"]) if contains_code else []
                
                # Generate summary
                summary = generate_chunk_summary(chunk["title"], chunk["content"][:250])
                
                # Create document with enhanced metadata
                doc = Document(
                    page_content=chunk["content"],
                    metadata={
                        "title": chunk["title"],
                        "source": normalized_path,
                        "url": url,
                        "section": os.path.dirname(normalized_path) if normalized_path else "",
                        "header_level": chunk["header_level"],
                        "parent_id": chunk.get("parent_id"),
                        "parent_title": chunk.get("parent_title", ""),
                        "children": chunk.get("children", []),
                        "summary": summary,
                        "contains_code": contains_code,
                        "code_languages": code_languages,
                        "has_overlap": chunk.get("has_overlap", False)
                    },
                )
                documents.append(doc)

        return documents

    except Exception as e:
        logger.error(f"Error processing markdown document: {e}")
        return []

def create_hierarchical_structure(chunks: List[Dict], file_path: str) -> List[Dict]:
    """
    Create hierarchical structure from chunks, establishing parent-child relationships.
    
    Args:
        chunks: List of chunks with title, content, and header level
        file_path: The source file path
        
    Returns:
        List of chunks with added hierarchical metadata
    """
    if not chunks:
        return []
        
    # Sort chunks by their appearance in the document
    sorted_chunks = list(enumerate(chunks))
    
    # Initialize hierarchy tracking
    hierarchy_stack = []
    hierarchical_chunks = []
    
    for i, chunk in sorted_chunks:
        current_level = chunk["header_level"]
        
        # Add chunk ID
        chunk_id = f"{file_path}:{i}"
        chunk["id"] = chunk_id
        
        # Pop stack until we find a parent (lower header level)
        while hierarchy_stack and hierarchy_stack[-1]["header_level"] >= current_level:
            hierarchy_stack.pop()
            
        # Set parent information if we have one
        if hierarchy_stack:
            parent = hierarchy_stack[-1]
            chunk["parent_id"] = parent["id"]
            chunk["parent_title"] = parent["title"]
            
            # Add this chunk as a child of its parent
            if "children" not in parent:
                parent["children"] = []
            parent["children"].append(chunk_id)
        
        # Push current chunk to the stack
        hierarchy_stack.append(chunk)
        hierarchical_chunks.append(chunk)
    
    return hierarchical_chunks


def create_overlapping_chunks(chunks: List[Dict], overlap_percentage: float = 0.15) -> List[Dict]:
    """
    Create overlapping chunks to maintain context between sections.
    
    Args:
        chunks: List of chunks with hierarchical structure
        overlap_percentage: Percentage of content to overlap
        
    Returns:
        List of chunks with overlap between adjacent chunks
    """
    if not chunks:
        return []
        
    # Sort chunks by their order of appearance
    sorted_chunks = sorted(chunks, key=lambda x: x.get("id", ""))
    overlapped_chunks = []
    
    for i, chunk in enumerate(sorted_chunks):
        current_chunk = dict(chunk)  # Create a copy to avoid modifying original
        
        # If not first chunk and has the same parent as previous, add overlap
        if i > 0:
            prev_chunk = sorted_chunks[i-1]
            
            # Only add overlap between chunks with the same parent or when current is a child of previous
            if (prev_chunk.get("parent_id") == current_chunk.get("parent_id") or 
                prev_chunk.get("id") == current_chunk.get("parent_id")):
                # Calculate overlap size (in characters)
                prev_content = prev_chunk["content"]
                overlap_size = int(len(prev_content) * overlap_percentage)
                
                # Get last part of previous chunk for overlap
                if overlap_size > 0:
                    overlap_text = prev_content[-overlap_size:]
                    
                    # Add overlap to beginning of current chunk
                    current_chunk["content"] = overlap_text + "\n\n" + current_chunk["content"]
                    current_chunk["has_overlap"] = True
        
        overlapped_chunks.append(current_chunk)
    
    return overlapped_chunks


def detect_code_languages(content: str) -> List[str]:
    """
    Detect programming languages in code blocks.
    
    Args:
        content: Chunk content possibly containing code blocks
        
    Returns:
        List of detected programming languages
    """
    languages = []
    code_block_pattern = r"```(\w*)"
    
    # Find all code blocks and extract language
    matches = re.findall(code_block_pattern, content)
    
    for match in matches:
        # Clean and normalize language name
        lang = match.strip().lower()
        if lang and lang not in languages:
            languages.append(lang)
    
    return languages


def generate_chunk_summary(title: str, content_preview: str) -> str:
    """
    Generate a simple summary of the chunk content.
    
    Args:
        title: Chunk title
        content_preview: Preview of chunk content
        
    Returns:
        A brief summary of the chunk
    """
    # For now, we'll use a simple combination of title and first bit of content
    # This would be a good place to use an LLM to generate better summaries
    if not content_preview:
        return title
        
    # Clean up the content preview
    preview = re.sub(r'\s+', ' ', content_preview).strip()
    
    # Truncate if needed
    if len(preview) > 150:
        preview = preview[:147] + "..."
        
    return f"{title}: {preview}"

def save_enhanced_chunks(chunks: List[Dict[str, Any]], provider: str = DEFAULT_PROVIDER) -> None:
    """Save enhanced chunks to a JSON file."""
    output_dir = f"data/generated/{provider}"
    output_file = os.path.join(output_dir, "enhanced_chunks.json")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save enhanced chunks
    with open(output_file, "w") as f:
        json.dump(chunks, f, indent=2)
    logger.info(f"Saved {len(chunks)} enhanced chunks to {output_file}")

def process_documentation(docs_dir: str) -> Tuple[List[Document], Dict[str, List[Dict[str, Any]]]]:
    """
    Process documentation files and extract chunks with relationships.
    
    Args:
        docs_dir: Path to the documentation directory
        
    Returns:
        Tuple of (list of chunks, dictionary of relationships)
    """
    logger.info(f"Processing documentation from {docs_dir}")
    chunks = []
    relationships = defaultdict(list)
    
    try:
        # Walk through the documentation directory
        for root, _, files in os.walk(docs_dir):
            for file in files:
                if file.endswith('.mdx') or file.endswith('.md'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            
                        # Process the markdown document
                        doc_chunks = process_markdown_document(content, file_path)
                        chunks.extend(doc_chunks)
                        
                        # Log progress
                        logger.info(f"Processed {file_path}: {len(doc_chunks)} chunks")
                    except Exception as e:
                        logger.error(f"Error processing file {file_path}: {e}")
                        continue
        
        # Create relationships between chunks
        for i, chunk in enumerate(chunks):
            chunk_id = chunk.metadata.get("chunk_id", str(i))
            chunk_content = chunk.page_content.lower()
            
            # Compare with other chunks
            for j, other_chunk in enumerate(chunks):
                if i == j:
                    continue
                    
                other_id = other_chunk.metadata.get("chunk_id", str(j))
                other_content = other_chunk.page_content.lower()
                
                # Calculate similarity (simple keyword overlap for now)
                chunk_words = set(word_tokenize(chunk_content))
                other_words = set(word_tokenize(other_content))
                common_words = chunk_words.intersection(other_words)
                
                if len(common_words) > 0:
                    similarity = len(common_words) / max(len(chunk_words), len(other_words))
                    
                    if similarity >= SIMILARITY_THRESHOLD:
                        relationships[chunk_id].append({
                            "chunk_id": other_id,
                            "similarity": similarity
                        })
        
        logger.info(f"Processed {len(chunks)} total chunks with {len(relationships)} relationships")
        return chunks, relationships
        
    except Exception as e:
        logger.error(f"Error processing documentation: {e}")
        return [], {}

async def main_async():
    """Async version of the main function."""
    parser = argparse.ArgumentParser(
        description="Preprocess Aptos documentation with topic-based chunking"
    )
    parser.add_argument(
        "--docs_dir",
        type=str,
        default=CONTENT_PATHS[DEFAULT_PROVIDER],
        help="Path to the documentation directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(DOC_BASE_PATHS[DEFAULT_PROVIDER], "enhanced_chunks.json"),
        help="Path to store the enhanced data",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=200,
        help="Number of code chunks to process in parallel (default: 200)"
    )
    args = parser.parse_args()
    
    # Process documentation
    chunks, relationships = process_documentation(args.docs_dir)
    
    # Store enhanced metadata with parallel processing
    await store_enhanced_metadata_async(
        chunks, 
        relationships, 
        args.output,
        batch_size=args.batch_size
    )
    
    # Also save in the provider-specific location
    save_enhanced_chunks(chunks)
    
    logger.info("Preprocessing completed successfully")

def main():
    """Main entry point for the script."""
    # Run the async main function
    asyncio.run(main_async())

async def generate_code_summary(code_content: str, parent_title: str = "", block_title: str = "") -> str:
    """
    Asynchronously generate a natural language summary of what a code block does.
    
    Args:
        code_content: The code content to summarize
        parent_title: The title of the parent section
        block_title: The title of the current block
        
    Returns:
        A natural language summary of the code
    """
    # Check if we're in test mode - if so, generate a simple summary
    if os.getenv("OPENAI_API_KEY") is None or TEST_MODE:
        return f"Code example for {block_title or parent_title or 'this section'}"
    
    try:
        import openai
        
        # Set API key
        openai.api_key = os.getenv("OPENAI_API_KEY")
        
        # Create a clean version of the code for the prompt
        # Extract code block content
        code_blocks = []
        in_code_block = False
        current_block = []
        
        for line in code_content.split("\n"):
            if line.startswith("```"):
                if in_code_block:
                    code_blocks.append("\n".join(current_block))
                    current_block = []
                in_code_block = not in_code_block
            elif in_code_block:
                current_block.append(line)
        
        # Use the first code block found
        code_to_summarize = code_blocks[0] if code_blocks else code_content
        
        # Limit code size for the API call
        if len(code_to_summarize) > 2000:
            code_to_summarize = code_to_summarize[:2000] + "..."
        
        # Create prompt
        prompt = f"""
        This is a code block from the Aptos documentation section "{parent_title or block_title}".
        
        ```
        {code_to_summarize}
        ```
        
        Provide a brief, clear explanation of what this code does in a single paragraph.
        Focus on explaining the purpose and functionality, not the syntax.
        Limit your response to 100 words or less.
        """
        
        # Use asyncio to prevent blocking
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: openai.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a technical documentation assistant that specializes in explaining code examples clearly and concisely."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0  # Keep it factual and consistent
            )
        )
        
        # Extract and return the summary
        summary = response.choices[0].message.content.strip()
        return summary
        
    except Exception as e:
        logger.error(f"Error generating code summary with OpenAI: {e}")
        # Fallback to simple summary
        return f"Code example for {block_title or parent_title or 'this section'}"


async def process_all_chunks_with_metadata(chunks: List, batch_size: int = 200) -> List[Dict]:
    """
    Process all document chunks and enhance with metadata, processing code chunks in parallel batches.
    
    Args:
        chunks: List of document chunks
        batch_size: Number of code chunks to process in parallel
        
    Returns:
        List of enhanced chunks with metadata (code summaries added only for code chunks)
    """
    # Process all chunks to the enhanced format first
    enhanced_chunks = [{
        "id": chunk.metadata.get("chunk_id", ""),
        "content": chunk.page_content,
        "metadata": {**chunk.metadata}
    } for chunk in chunks]
    
    # Identify which chunks contain code
    code_chunks = []
    for i, chunk in enumerate(chunks):
        if chunk.metadata.get("contains_code", False):
            code_chunks.append({
                "index": i,
                "chunk": chunk
            })
    
    total_code_chunks = len(code_chunks)
    logger.info(f"Found {total_code_chunks} code chunks to summarize")
    
    # If there are no code chunks, return early
    if not code_chunks:
        return enhanced_chunks
    
    # Process code chunks in batches
    for i in range(0, len(code_chunks), batch_size):
        batch = code_chunks[i:i+batch_size]
        batch_num = i//batch_size + 1
        total_batches = (len(code_chunks) + batch_size - 1) // batch_size
        progress = min(i+batch_size, len(code_chunks))
        
        logger.info(f"Processing batch {batch_num}/{total_batches} ({progress}/{total_code_chunks} chunks, {progress/total_code_chunks:.1%} complete)")
        
        # Create tasks for each code chunk
        tasks = []
        for item in batch:
            chunk = item["chunk"]
            tasks.append(
                generate_code_summary(
                    chunk.page_content,
                    chunk.metadata.get("parent_title", ""),
                    chunk.metadata.get("title", "")
                )
            )
        
        # Process batch in parallel
        summaries = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Update the enhanced chunks with summaries
        for j, summary in enumerate(summaries):
            chunk_index = batch[j]["index"]
            
            if isinstance(summary, Exception):
                logger.error(f"Error generating summary: {summary}")
                enhanced_chunks[chunk_index]["metadata"]["code_summary"] = f"Code example for {chunks[chunk_index].metadata.get('title', 'this section')}"
            else:
                enhanced_chunks[chunk_index]["metadata"]["code_summary"] = summary
        
        # Add a small delay to avoid hitting API rate limits
        await asyncio.sleep(1.0)  # Increased delay for larger batch sizes
    
    return enhanced_chunks


async def store_enhanced_metadata_async(
    chunks: List[Any], 
    relationships: Dict[str, List[Dict[str, Any]]], 
    output_path: str,
    batch_size: int = 10
) -> None:
    """
    Store chunks with enhanced metadata, processing code summaries in parallel.

    Args:
        chunks: List of document chunks
        relationships: Dictionary of chunk relationships
        output_path: Path to store the enhanced data
        batch_size: Number of code chunks to process in parallel
    """
    logger.info(f"Storing enhanced metadata to {output_path}")

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Process all chunks with metadata (including code summaries for code chunks)
    enhanced_chunks = await process_all_chunks_with_metadata(chunks, batch_size)

    # Add relationship data to each chunk
    for enhanced_chunk in enhanced_chunks:
        chunk_id = enhanced_chunk["id"]
        if not chunk_id:
            continue

        # Get related chunks for this chunk
        related = relationships.get(chunk_id, [])

        # Add relationship metadata
        enhanced_chunk["metadata"]["related_topics"] = [r["chunk_id"] for r in related]
        enhanced_chunk["metadata"]["topic_similarity_scores"] = {
                    r["chunk_id"]: r["similarity"] for r in related
        }

    # Save to disk
    with open(output_path, "w") as f:
        json.dump(enhanced_chunks, f)
        
    logger.info(f"Stored {len(enhanced_chunks)} enhanced chunks")

if __name__ == "__main__":
    main()