"""
Adaptive multi-step retrieval functions for the RAG system.

This module provides functions for adaptive multi-step retrieval, including
query complexity analysis and iterative retrieval with follow-up queries.
"""

import json
import logging
import hashlib
import asyncio
import time
from typing import Dict, List, Any
import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAI, ChatOpenAI
from app.utils.topic_chunks import get_topic_aware_context
from app.utils.vector_store_utils import batch_similarity_search_with_score, similarity_search_with_batch_support

logger = logging.getLogger(__name__)

# Use the OPENAI_API_KEY from environment for API calls
openai_client = None
try:
    from openai import OpenAI as OpenAIClient
    openai_client = OpenAIClient()
except ImportError:
    logger.error("OpenAI Python package not found. Install with 'pip install openai'")
    openai_client = None


async def generate_retrieval_queries(query: str) -> Dict[str, Any]:
    """
    Generate targeted retrieval queries to break down the original question.
    
    Args:
        query: The user query
        
    Returns:
        Dictionary containing retrieval queries
    """
    start_time = time.time()
    logger.info(f"[ADAPTIVE-RAG] Starting retrieval query generation")
    
    prompt = f"""
    You are an expert in Aptos blockchain development and the Move programming language. 
    Analyze this developer question:
    "{query}"
    
    Break down this query into targeted retrieval questions that will gather all the necessary technical information to provide a complete answer. Focus specifically on Aptos and Move concepts.
    
    Consider these question types:
    1. Core concepts - What fundamental blockchain or Move concepts need explanation?
    2. Implementation patterns - What code patterns or implementation approaches are relevant?
    3. API usage - What specific API calls, functions, or modules are needed?
    4. Error handling - What potential errors or edge cases should be addressed?
    5. Debugging - What debugging information would help solve the problem?
    
    For each aspect of the original question, create a targeted retrieval query that will fetch the most relevant documentation or examples from a vector database.
    
    Guidelines for effective retrieval queries:
    - Use technical terminology specific to Aptos and Move
    - Include relevant function names, module names, or error codes mentioned in the query
    - Create queries for both high-level concepts and specific implementation details
    - Phrase queries as direct questions or information requests
    - Include all relevant context from the original question
    
    Return as JSON with this field:
    - retrieval_queries: list[string] (3-5 targeted questions to retrieve relevant information)
    """
    
    messages = [{"role": "system", "content": prompt}]
    try:
        # Use synchronous approach instead of await
        response = openai_client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.05,
            max_tokens=350  # Limit tokens for faster response
        )
        
        # Parse the result
        result = json.loads(response.choices[0].message.content)
        
        # Ensure we have retrieval_queries field with default empty list
        if "retrieval_queries" not in result or not isinstance(result["retrieval_queries"], list):
            result["retrieval_queries"] = []
            
        # Add the original query to ensure we always include it
        if query not in result["retrieval_queries"]:
            result["retrieval_queries"].insert(0, query)
        
        # Limit to 5 queries max to reduce processing time
        if len(result["retrieval_queries"]) > 5:
            result["retrieval_queries"] = result["retrieval_queries"][:5]
            
        elapsed = time.time() - start_time
        logger.info(f"[ADAPTIVE-RAG] Generated {len(result['retrieval_queries'])} retrieval queries in {elapsed:.2f}s")
        return result
    except Exception as e:
        logger.error(f"Error generating retrieval queries: {e}")
        # Return a default result with just the original query
        elapsed = time.time() - start_time
        logger.info(f"[ADAPTIVE-RAG] Query generation failed after {elapsed:.2f}s")
        return {
            "retrieval_queries": [query]
        }


async def analyze_follow_up_needs(query: str, initial_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze if follow-up queries are needed based on initial results.
    
    Args:
        query: The original user query
        initial_results: The initial retrieval results
        
    Returns:
        Dictionary with follow-up queries if needed
    """
    start_time = time.time()
    
    # Skip follow-up analysis for shorter, simpler queries
    if len(query.split()) < 12 and len(initial_results) >= 3:
        logger.info(f"[ADAPTIVE-RAG] Simple query with sufficient results, skipping follow-up analysis")
        return {"is_complete": True, "follow_up_queries": []}
    
    # Format current results for analysis with fallbacks for missing fields
    if not initial_results or len(initial_results) < 2:
        logger.warning("[ADAPTIVE-RAG] Insufficient initial results, generating generic follow-ups")
        return {
            "is_complete": False, 
            "follow_up_queries": [
                f"implementation example for {query}",
                f"code sample for {query}"
            ]
        }
    
    # Only analyze if we have enough context to make a decision
    top_results = initial_results[:5]
    
    # Use shorter context format to reduce tokens
    context_preview = "\n".join([
        f"Doc {i+1}: {result.get('section', '')[:50]}... | Summary: {result.get('summary', '')[:50]}..."
        for i, result in enumerate(top_results)
    ])
    
    analysis_prompt = f"""
    You are an expert in Aptos blockchain development and the Move programming language.
    
    Assess whether these retrieved documents provide sufficient information to answer this developer question:
    "{query}"
    
    Retrieved documents:
    {context_preview}
    
    Analyze what's missing from these documents, if anything, to provide a complete, accurate answer:
    
    1. Missing technical concepts: Are there fundamental Aptos or Move concepts not covered?
    2. Missing implementation details: Are code examples or implementation patterns needed?
    3. Missing API reference: Are specific function signatures, parameters, or return types needed?
    4. Missing error handling: Are error cases or debugging approaches needed?
    5. Missing context: Are prerequisites or related modules/components needed?
    6. Prefer modern and latest features or code examples over deprecated or legacy code.
    
    For any identified gaps, create specific follow-up queries to retrieve that information.
    Your follow-up queries should be precise, technical, and focused on retrieving exactly what's missing.
    
    Return as JSON with these fields:
    - is_complete: boolean (true if results are sufficient for a complete answer)
    - follow_up_queries: list[string] (1-5 targeted queries to retrieve missing information)
    """
    
    # Get analysis
    messages = [{"role": "system", "content": analysis_prompt}]
    try:
        # Use synchronous approach instead of await
        response = openai_client.chat.completions.create(
            model="gpt-4.1-mini", 
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.05,
            max_tokens=350  # Limit tokens for faster response
        )
        
        # Parse result
        result = json.loads(response.choices[0].message.content)
        
        # Ensure required fields
        if "is_complete" not in result:
            result["is_complete"] = True
        if "follow_up_queries" not in result or not isinstance(result["follow_up_queries"], list):
            result["follow_up_queries"] = []
            
        elapsed = time.time() - start_time
        query_count = len(result.get("follow_up_queries", []))
        logger.info(f"[ADAPTIVE-RAG] Follow-up analysis completed in {elapsed:.2f}s, generated {query_count} follow-up queries")
        return result
    except Exception as e:
        logger.error(f"Error analyzing follow-up needs: {e}")
        # Default to complete in case of error
        return {"is_complete": True, "follow_up_queries": []}


async def execute_query(
    query: str,
    vector_store: FAISS,
    enhanced_chunks: List[Dict[str, Any]],
    retrieval_k: int,
    query_type: str = "primary"
) -> List[Dict[str, Any]]:
    """
    Execute a single retrieval query.
    
    Args:
        query: The query to execute
        vector_store: Initialized vector store
        enhanced_chunks: List of enhanced chunks
        retrieval_k: Number of chunks to retrieve
        query_type: Type of query ("primary" or "follow_up")
        
    Returns:
        List of relevant chunks with metadata
    """
    start_time = time.time()
    logger.info(f"[ADAPTIVE-RAG] Starting {query_type} query: {query}")
    
    results = await get_topic_aware_context(
        query=query,
        vector_store=vector_store,
        enhanced_chunks=enhanced_chunks,
        k=int(retrieval_k)
    )
    
    # Process results
    processed_results = []
    for result in results:
        # Generate a content hash as a fallback ID
        content = result.get("content", "")
        content_hash = hashlib.md5(content.encode()).hexdigest()
        
        # Try to use existing ID first, fall back to content hash
        result_id = result.get("id") or content_hash
        
        # Add the ID to the result for future reference
        if "id" not in result:
            result["id"] = result_id
            
        # Add retrieval metadata
        result["retrieval_query"] = query
        result["retrieval_type"] = query_type
        processed_results.append((result_id, result))
    
    elapsed = time.time() - start_time
    logger.info(f"[ADAPTIVE-RAG] {query_type.capitalize()} query completed in {elapsed:.2f}s, found {len(processed_results)} results")
    return processed_results


async def execute_batch_queries(
    queries: List[str],
    vector_store: FAISS,
    enhanced_chunks: List[Dict[str, Any]],
    retrieval_k: int,
    query_type: str = "primary"
) -> Dict[str, Dict[str, Any]]:
    """
    Execute multiple retrieval queries as a batch for improved performance.
    
    Args:
        queries: List of queries to execute
        vector_store: Initialized vector store
        enhanced_chunks: List of enhanced chunks
        retrieval_k: Number of chunks to retrieve per query
        query_type: Type of query ("primary" or "follow_up")
        
    Returns:
        Dictionary mapping result IDs to results
    """
    start_time = time.time()
    logger.info(f"[ADAPTIVE-RAG] Starting batch {query_type} queries: {len(queries)} queries")
    
    # Step 1: Get the batch similarity search results
    batch_search_start = time.time()
    
    # Use batch similarity search for all queries at once
    all_docs_with_scores = await similarity_search_with_batch_support(
        vector_store=vector_store,
        query="",  # Empty, using batch mode
        queries=queries,
        k=int(retrieval_k)
    )
    
    batch_search_time = time.time() - batch_search_start
    logger.info(f"[ADAPTIVE-RAG] Batch similarity search completed in {batch_search_time:.2f}s")
    
    # Step 2: Process documents from batch search
    process_start = time.time()
    
    # Create chunk map for quick lookups (done once for all results)
    chunk_map = {chunk["id"]: chunk for chunk in enhanced_chunks}
    
    # Initialize results dictionary to store deduplicated results
    all_results = {}
    
    # Process each document
    for doc, score in all_docs_with_scores:
        # Try both chunk_id and id in metadata
        chunk_id = doc.metadata.get("chunk_id") or doc.metadata.get("id")

        # For documents without chunk ID, generate hash from content
        if not chunk_id:
            content = doc.page_content
            content_hash = hashlib.md5(content.encode()).hexdigest()
            result_id = content_hash
            
            # Create a basic result
            result = {
                "id": result_id,
                "content": content,
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
                "retrieval_type": query_type,
                "retrieval_query": "batch query",  # No easy way to know which query matched
            }
            
            all_results[result_id] = result
            continue
            
        # Skip if chunk not in map
        if chunk_id not in chunk_map:
            logger.warning(f"[ADAPTIVE-RAG] Chunk ID {chunk_id} not found in chunk map")
            continue
            
        # Get the enhanced chunk
        enhanced_chunk = chunk_map[chunk_id]
        
        # If this was an enhanced embedding, use the original content for display
        content = doc.metadata.get("original_content", doc.page_content)
        
        # Get code summary if available
        code_summary = None
        if doc.metadata.get("contains_code") and doc.metadata.get("code_summary"):
            code_summary = doc.metadata.get("code_summary")
            
        # Process related chunks
        related_chunks = []
        related_ids = enhanced_chunk["metadata"].get("related_topics", [])
        
        for related_id in related_ids:
            if related_id in chunk_map:
                related_chunk = chunk_map[related_id]
                similarity = (
                    enhanced_chunk["metadata"]
                    .get("topic_similarity_scores", {})
                    .get(related_id, 0)
                )
                
                if similarity >= 0.7:  # Using SIMILARITY_THRESHOLD value from topic_chunks
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
        
        # Get parent info if available
        parent_info = None
        parent_id = doc.metadata.get("parent_id")
        
        if parent_id:
            if parent_id in chunk_map:
                parent_chunk = chunk_map[parent_id]
                parent_info = {
                    "id": parent_id,
                    "title": parent_chunk["metadata"].get("title", ""),
                    "summary": parent_chunk["metadata"].get("summary", ""),
                }
        
        # Create final result
        result = {
            "id": chunk_id,
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
            "retrieval_type": query_type,
            "retrieval_query": "batch query",  # No easy way to know which query matched
        }
        
        # If we already have this result, keep the one with higher score
        if chunk_id in all_results:
            if score > all_results[chunk_id]["score"]:
                all_results[chunk_id] = result
        else:
            all_results[chunk_id] = result
            
    process_time = time.time() - process_start
    total_time = time.time() - start_time
    result_count = len(all_results)
    
    logger.info(f"[ADAPTIVE-RAG] Batch {query_type} queries completed in {total_time:.2f}s")
    logger.info(f"[ADAPTIVE-RAG] PERFORMANCE BREAKDOWN:")
    logger.info(f"[ADAPTIVE-RAG] - Batch similarity search: {batch_search_time:.2f}s ({(batch_search_time/total_time)*100:.1f}%)")
    logger.info(f"[ADAPTIVE-RAG] - Results processing: {process_time:.2f}s ({(process_time/total_time)*100:.1f}%)")
    logger.info(f"[ADAPTIVE-RAG] Found {result_count} unique results across {len(queries)} queries")
    
    return all_results


async def adaptive_multi_step_retrieval(
    query: str,
    vector_store: FAISS,
    enhanced_chunks: List[Dict[str, Any]],
    k: int = 4,
    max_iterations: int = 2
) -> List[Dict[str, Any]]:
    """
    Perform multi-step retrieval with adaptive depth based on query complexity.
    
    Args:
        query: Initial user query
        vector_store: Initialized vector store
        enhanced_chunks: List of enhanced chunks
        k: Base number of chunks to retrieve per query
        max_iterations: Maximum number of retrieval iterations
        
    Returns:
        Combined list of relevant chunks
    """
    total_start_time = time.time()
    logger.info(f"[ADAPTIVE-RAG] Starting multi-step retrieval process for query: {query}")
    
    # Step 1: Generate retrieval queries to break down the original question
    queries_start_time = time.time()
    query_analysis = await generate_retrieval_queries(query)
    
    # Get all retrieval queries (original + generated)
    all_queries = query_analysis["retrieval_queries"]
    
    queries_elapsed = time.time() - queries_start_time
    logger.info(f"[ADAPTIVE-RAG] Prepared {len(all_queries)} queries in {queries_elapsed:.2f}s")
    
    # Track all retrieved chunks and their scores
    all_results = {}
    
    # Step 2: Execute primary retrieval using batched approach
    primary_start_time = time.time()
    
    if len(all_queries) > 1:
        # Use new batch processing for multiple queries
        logger.info(f"[ADAPTIVE-RAG] Executing primary retrieval using batched search for {len(all_queries)} queries")
        primary_results = await execute_batch_queries(
            all_queries, 
            vector_store, 
            enhanced_chunks, 
            k, 
            "primary"
        )
        all_results.update(primary_results)
    else:
        # Use traditional approach for single query
        logger.info(f"[ADAPTIVE-RAG] Executing primary retrieval using standard search")
        primary_result_pairs = await execute_query(
            all_queries[0], 
            vector_store, 
            enhanced_chunks, 
            k, 
            "primary"
        )
        for result_id, result in primary_result_pairs:
            all_results[result_id] = result
    
    primary_elapsed = time.time() - primary_start_time
    logger.info(f"[ADAPTIVE-RAG] Primary retrieval completed in {primary_elapsed:.2f}s, found {len(all_results)} unique chunks")
    
    # Step 3: Analyze if follow-up queries are needed
    follow_up_start_time = time.time()
    
    # Convert dictionary to list for analysis
    result_list = list(all_results.values())
    follow_up_analysis = await analyze_follow_up_needs(query, result_list)
    follow_up_queries = follow_up_analysis.get("follow_up_queries", [])
    
    # Step 4: Execute follow-up queries if needed
    if not follow_up_analysis.get("is_complete", True) and follow_up_queries:
        follow_up_queries_start_time = time.time()
        
        if follow_up_queries:
            # Use batched retrieval if multiple follow-up queries
            if len(follow_up_queries) > 1:
                logger.info(f"[ADAPTIVE-RAG] Executing follow-up retrieval using batched search for {len(follow_up_queries)} queries")
                follow_up_results = await execute_batch_queries(
                    follow_up_queries, 
                    vector_store, 
                    enhanced_chunks, 
                    k, 
                    "follow_up"
                )
                
                # Track new chunks
                new_chunks_count = 0
                for result_id, result in follow_up_results.items():
                    if result_id not in all_results:
                        all_results[result_id] = result
                        new_chunks_count += 1
                    else:
                        # Keep the highest score
                        all_results[result_id]["score"] = max(
                            all_results[result_id].get("score", 0),
                            result.get("score", 0)
                        )
            else:
                # Use traditional approach for single follow-up query
                logger.info(f"[ADAPTIVE-RAG] Executing single follow-up query using standard search")
                follow_up_result_pairs = await execute_query(
                    follow_up_queries[0], 
                    vector_store, 
                    enhanced_chunks, 
                    k, 
                    "follow_up"
                )
                
                # Track new chunks
                new_chunks_count = 0
                for result_id, result in follow_up_result_pairs:
                    if result_id not in all_results:
                        all_results[result_id] = result
                        new_chunks_count += 1
                    else:
                        # Keep the highest score
                        all_results[result_id]["score"] = max(
                            all_results[result_id].get("score", 0),
                            result.get("score", 0)
                        )
            
            follow_up_queries_elapsed = time.time() - follow_up_queries_start_time
            logger.info(f"[ADAPTIVE-RAG] Follow-up queries executed in {follow_up_queries_elapsed:.2f}s, found {new_chunks_count} new chunks")
    
    follow_up_elapsed = time.time() - follow_up_start_time
    logger.info(f"[ADAPTIVE-RAG] Total follow-up processing completed in {follow_up_elapsed:.2f}s")
    
    # Step 5: Return all combined results, sorted by score
    final_start_time = time.time()
    combined_results = list(all_results.values())
    
    # Ensure all results have a score for sorting
    for result in combined_results:
        if "score" not in result:
            result["score"] = 0
    
    combined_results.sort(key=lambda x: x.get("score", 0), reverse=True)
    
    # Limit results to top k*2 results
    top_limit = min(k * 2, len(combined_results))
    final_results = combined_results[:top_limit]
    final_elapsed = time.time() - final_start_time
    
    total_elapsed = time.time() - total_start_time
    logger.info(f"[ADAPTIVE-RAG] PERFORMANCE SUMMARY:")
    logger.info(f"[ADAPTIVE-RAG] - Query generation: {queries_elapsed:.2f}s")
    logger.info(f"[ADAPTIVE-RAG] - Primary retrieval ({len(all_queries)} queries): {primary_elapsed:.2f}s")
    logger.info(f"[ADAPTIVE-RAG] - Follow-up processing: {follow_up_elapsed:.2f}s")
    logger.info(f"[ADAPTIVE-RAG] - Final processing: {final_elapsed:.2f}s")
    logger.info(f"[ADAPTIVE-RAG] - Total elapsed time: {total_elapsed:.2f}s")
    logger.info(f"[ADAPTIVE-RAG] Retrieved {len(combined_results)} unique chunks across all queries")
    logger.info(f"[ADAPTIVE-RAG] Returning top {len(final_results)} chunks")
    
    return final_results 