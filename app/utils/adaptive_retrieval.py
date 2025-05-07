"""
Adaptive multi-step retrieval functions for the RAG system.

This module provides functions for adaptive multi-step retrieval, including
query complexity analysis and iterative retrieval with follow-up queries.
"""

import json
import logging
import hashlib
from typing import Dict, List, Any
import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAI, ChatOpenAI
from app.utils.topic_chunks import get_topic_aware_context

logger = logging.getLogger(__name__)

# Use the OPENAI_API_KEY from environment for API calls
openai_client = None
try:
    from openai import OpenAI as OpenAIClient
    openai_client = OpenAIClient()
except ImportError:
    logger.error("OpenAI Python package not found. Install with 'pip install openai'")
    openai_client = None


async def analyze_query_complexity(query: str) -> Dict[str, Any]:
    """
    Analyze the query to determine complexity and generate retrieval strategies.
    
    Args:
        query: The user query
        
    Returns:
        Dictionary containing query analysis and retrieval strategies
    """
    prompt = f"""
    Analyze this development query:
    "{query}"
    
    First, determine if this is a simple or complex question:
    - Simple: Can be answered with a single concept or documentation lookup
    - Complex: Requires understanding multiple components, examples, or implementation details
    
    Then, generate effective retrieval strategies:
    1. What specific components, functions, or concepts are mentioned?
    2. Generate 3-4 reformulated queries that would gather necessary information
    3. What additional information might be needed for a complete answer?
    
    Return as JSON with these fields:
    - complexity: string (either "simple" or "complex")
    - key_components: list[string] (relevant components, classes, functions, etc.)
    - reformulated_queries: list[string] (3-4 retrieval queries to gather information)
    - potential_follow_ups: list[string] (types of additional information that might be needed)
    """
    
    messages = [{"role": "system", "content": prompt}]
    try:
        # Use synchronous approach instead of await
        response = openai_client.chat.completions.create(
            model="gpt-4.1",
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.1
        )
        
        # Parse the result
        result = json.loads(response.choices[0].message.content)
        
        # Add default values if missing
        result.setdefault("complexity", "simple")
        result.setdefault("key_components", [])
        result.setdefault("reformulated_queries", [])
        result.setdefault("potential_follow_ups", [])
        
        return result
    except Exception as e:
        logger.error(f"Error analyzing query complexity: {e}")
        # Return a default analysis in case of error
        return {
            "complexity": "simple",
            "key_components": [],
            "reformulated_queries": [],
            "potential_follow_ups": []
        }


async def adaptive_multi_step_retrieval(
    query: str,
    vector_store: FAISS,
    enhanced_chunks: List[Dict[str, Any]],
    k: int = 7,
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
    # Step 1: Analyze query to understand complexity
    query_analysis = await analyze_query_complexity(query)
    is_complex = query_analysis["complexity"] == "complex"
    
    # Adjust retrieval parameters based on complexity
    # More chunks and iterations for complex queries
    retrieval_k = k * 1.5 if is_complex else k
    retrieval_iterations = max_iterations if is_complex else min(max_iterations, 2)
    
    logger.info(f"[ADAPTIVE-RAG] Query complexity: {query_analysis['complexity']}")
    logger.info(f"[ADAPTIVE-RAG] Using parameters: k={retrieval_k}, max_iterations={retrieval_iterations}")
    
    # Step 2: Prepare queries (original + reformulations)
    all_queries = [query] + query_analysis["reformulated_queries"]
    
    # Add targeted queries for key components if complex
    if is_complex and query_analysis["key_components"]:
        for component in query_analysis["key_components"]:
            if len(component) > 2:  # Avoid very short identifiers
                all_queries.append(f"documentation for {component}")
                all_queries.append(f"implementation of {component}")
    
    # Track all retrieved chunks and their scores
    all_results = {}
    iteration_results = []
    
    # Step 3: Perform initial retrieval with all queries
    for q in all_queries:
        logger.info(f"[ADAPTIVE-RAG] Executing query: {q}")
        results = await get_topic_aware_context(
            query=q,
            vector_store=vector_store,
            enhanced_chunks=enhanced_chunks,
            k=int(retrieval_k)
        )
        
        # Add to results tracking
        for result in results:
            # Generate a content hash as a fallback ID
            content = result.get("content", "")
            content_hash = hashlib.md5(content.encode()).hexdigest()
            
            # Try to use existing ID first, fall back to content hash
            result_id = result.get("id") or content_hash
            
            # Add the ID to the result for future reference
            if "id" not in result:
                result["id"] = result_id
                
            if result_id not in all_results:
                all_results[result_id] = result
                # Add retrieval source
                result["retrieval_query"] = q
                result["retrieval_type"] = "primary"
            else:
                # Keep the highest score if chunk was retrieved multiple times
                all_results[result_id]["score"] = max(
                    all_results[result_id].get("score", 0), 
                    result.get("score", 0)
                )
        
        # Track results from this iteration
        iteration_results.append({
            "query": q,
            "results_count": len(results)
        })
    
    # Step 4: Analyze if follow-up queries are needed (for complex queries or if follow-ups suggested)
    if retrieval_iterations > 1 and (is_complex or query_analysis["potential_follow_ups"]) and all_results:
        # Format current results for analysis with fallbacks for missing fields
        context_preview = "\n".join([
            f"Document {i+1}: {result.get('section', 'Unknown section')}\nSummary: {result.get('summary', 'No summary available')}"
            for i, result in enumerate(list(all_results.values())[:5])
        ])
        
        analysis_prompt = f"""
        Based on the user query: "{query}"
        
        I have retrieved these documents:
        {context_preview}
        
        The query was analyzed to be: {query_analysis['complexity']}
        Key components: {", ".join(query_analysis['key_components'])}
        
        Analyze if these results fully address the user's question:
        1. Is any important information missing?
        2. Are there specific aspects not covered by these documents?
        
        Return as JSON with fields:
        - is_complete: boolean (true if results are sufficient)
        - missing_information: string (what's missing, if anything)
        - follow_up_queries: list[string] (specific queries to find missing information)
        """
        
        # Get analysis
        messages = [{"role": "system", "content": analysis_prompt}]
        try:
            # Use synchronous approach instead of await
            response = openai_client.chat.completions.create(
                model="gpt-4.1", 
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.1
            )
            
            # Parse result
            follow_up_analysis = json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"Error analyzing follow-up needs: {e}")
            # Default to complete in case of error
            follow_up_analysis = {"is_complete": True, "follow_up_queries": []}
        
        # Step 5: Execute follow-up queries if needed
        if not follow_up_analysis.get("is_complete", True) and follow_up_analysis.get("follow_up_queries"):
            logger.info(f"[ADAPTIVE-RAG] Initial results incomplete. Running follow-up queries.")
            
            for iteration in range(int(retrieval_iterations) - 1):
                follow_up_queries = follow_up_analysis.get("follow_up_queries", [])
                if not follow_up_queries:
                    break
                    
                # Execute each follow-up query
                for q in follow_up_queries:
                    logger.info(f"[ADAPTIVE-RAG] Executing follow-up query: {q}")
                    results = await get_topic_aware_context(
                        query=q,
                        vector_store=vector_store,
                        enhanced_chunks=enhanced_chunks,
                        k=int(retrieval_k)
                    )
                    
                    # Add new results
                    for result in results:
                        # Generate a content hash as a fallback ID
                        content = result.get("content", "")
                        content_hash = hashlib.md5(content.encode()).hexdigest()
                        
                        # Try to use existing ID first, fall back to content hash
                        result_id = result.get("id") or content_hash
                        
                        # Add the ID to the result for future reference
                        if "id" not in result:
                            result["id"] = result_id
                        
                        if result_id not in all_results:
                            all_results[result_id] = result
                            # Tag this as a follow-up result
                            result["retrieval_type"] = "follow_up"
                            result["retrieval_query"] = q
                
                # Only do multiple iterations for complex queries
                if not is_complex:
                    break
    
    # Step 6: Return all combined results, sorted by score
    combined_results = list(all_results.values())
    
    # Ensure all results have a score for sorting
    for result in combined_results:
        if "score" not in result:
            result["score"] = 0
    
    combined_results.sort(key=lambda x: x.get("score", 0), reverse=True)
    
    # Limit results based on complexity
    # Return more results for complex queries, no extra for simple ones
    top_limit = min(int(retrieval_k) * 2 if is_complex else int(retrieval_k), len(combined_results))
    
    logger.info(f"[ADAPTIVE-RAG] Retrieved {len(combined_results)} unique chunks across all queries")
    logger.info(f"[ADAPTIVE-RAG] Returning top {top_limit} chunks")
    
    return combined_results[:top_limit] 