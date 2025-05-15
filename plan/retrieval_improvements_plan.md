# Multi-Step Retrieval Improvement Plan for RAG System

## Problem Statement

The current RAG system has a critical limitation: it fetches data based on the initial user question and produces a response, but lacks the intelligence to retrieve additional relevant information when needed. If the LLM determines that the retrieved chunks don't contain the complete answer but knows where the information might be found, it cannot fetch this additional data.

## Proposed Solution: Multi-Step Retrieval

Implement an agent-based retrieval system that allows the LLM to:

1. Analyze initial retrieval results
2. Determine if more information is needed
3. Formulate new retrieval queries
4. Get additional context when necessary

## Architectural Approaches Comparison

There are two main architectural approaches to implementing multi-step RAG capabilities:

### 1. Pre-Retrieval Multi-Step RAG (Currently Proposed)

**Description:**

- Initial retrieval layer performs multi-step retrieval before passing to LLM
- RAG layer handles all query refinement and iterative retrieval
- LLM receives comprehensive context as a single input

**Benefits:**

- Clear separation of concerns (retrieval system handles all retrieval)
- Potentially more efficient with batch processing of multiple queries
- Simpler prompt engineering for the LLM (just answer the question using provided context)
- Easier to debug and optimize the retrieval process independently
- Lower token usage for the more expensive reasoning LLM

**Drawbacks:**

- Less adaptive during response generation
- May retrieve unnecessary information that increases token usage
- Cannot decide mid-generation to fetch additional information

### 2. LLM-as-Agent with RAG Tool Approach

**Description:**

- LLM acts as an agent with access to RAG as a tool
- LLM can dynamically decide when to call for more information during response generation
- Uses a tool-calling framework to request specific information

**Benefits:**

- More dynamic and adaptive as generation proceeds
- Can make retrieval decisions based on its current reasoning state
- May use fewer total tokens by only retrieving exactly what's needed
- Better handles complex multi-part questions requiring different retrieval strategies
- Follows emerging best practices in agent-based LLM applications

**Drawbacks:**

- More complex implementation (requires agent framework, tool definitions)
- Potentially higher latency due to multiple back-and-forth calls
- More expensive due to higher usage of reasoning LLM
- More complex to debug and optimize
- May require more sophisticated prompt engineering

### Recommendation

Both approaches have merit, and the choice depends on priorities:

**Short-term solution:**
Implement the pre-retrieval multi-step approach (as outlined in this plan) as it's more straightforward to integrate with the existing architecture and provides immediate benefits.

**Long-term evolution:**
Consider evolving toward an LLM-as-agent approach as the system matures, especially if:

1. The pre-retrieval approach shows limitations in complex scenarios
2. Users frequently need follow-up questions for comprehensive answers
3. The technology stack evolves to better support agent-based LLM applications

For the agent-based approach, we would need to:

1. Define a RAG retrieval tool API for the LLM to call
2. Implement a function-calling LLM framework
3. Create agent execution logic with state management
4. Design prompts that instruct the LLM on when and how to use retrieval tools

## Implementation Plan

### 1. Enhanced Query Analysis Function

```python
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
    response = await openai_client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0
    )

    # Parse the result
    result = json.loads(response.choices[0].message.content)

    # Add default values if missing
    result.setdefault("complexity", "simple")
    result.setdefault("key_components", [])
    result.setdefault("reformulated_queries", [])
    result.setdefault("potential_follow_ups", [])

    return result
```

### 2. Adaptive Multi-Step Retrieval Function

```python
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
            chunk_id = result["id"]
            if chunk_id not in all_results:
                all_results[chunk_id] = result
                # Add retrieval source
                result["retrieval_query"] = q
            else:
                # Keep the highest score if chunk was retrieved multiple times
                all_results[chunk_id]["score"] = max(
                    all_results[chunk_id]["score"],
                    result["score"]
                )

        # Track results from this iteration
        iteration_results.append({
            "query": q,
            "results": [r["id"] for r in results]
        })

    # Step 4: Analyze if follow-up queries are needed (for complex queries or if follow-ups suggested)
    if retrieval_iterations > 1 and (is_complex or query_analysis["potential_follow_ups"]):
        # Format current results for analysis
        context_preview = "\n".join([
            f"Document {i+1}: {result['section']}\nSummary: {result['summary']}"
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
        response = await openai_client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0
        )

        # Parse result
        follow_up_analysis = json.loads(response.choices[0].message.content)

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
                        chunk_id = result["id"]
                        if chunk_id not in all_results:
                            all_results[chunk_id] = result
                            # Tag this as a follow-up result
                            result["retrieval_type"] = "follow_up"
                            result["retrieval_query"] = q

                # Only do multiple iterations for complex queries
                if not is_complex:
                    break

    # Step 6: Return all combined results, sorted by score
    combined_results = list(all_results.values())
    combined_results.sort(key=lambda x: x.get("score", 0), reverse=True)

    # Limit results based on complexity
    # Return more results for complex queries
    top_limit = min(int(retrieval_k) * 2 if is_complex else int(retrieval_k) * 1.5, len(combined_results))

    logger.info(f"[ADAPTIVE-RAG] Retrieved {len(combined_results)} unique chunks across all queries")
    logger.info(f"[ADAPTIVE-RAG] Returning top {top_limit} chunks")

    return combined_results[:top_limit]
```

### 3. Modify the DocsRAGProvider Class

Update the `get_relevant_context` method in `app/rag_providers/docs_provider.py`:

```python
async def get_relevant_context(
    self,
    query: str,
    k: int = 5,
    include_series: bool = True,
    provider_type: Optional[PROVIDER_TYPES] = None,
    use_multi_step: bool = True,
) -> List[Dict[str, Any]]:
    """
    Get relevant context from the documentation using topic-based retrieval.
    Now supports adaptive multi-step retrieval for improved results.

    Args:
        query: The user query
        k: Number of top documents to return
        include_series: Whether to include related documents from the same topic
        provider_type: Optional provider type to use for this query
        use_multi_step: Whether to use multi-step retrieval

    Returns:
        List of dictionaries containing content, section, source, and metadata
    """
    logger.info(f"[DOCS-RAG] Starting context retrieval for query: {query}")
    logger.info(
        f"[DOCS-RAG] Parameters: k={k}, include_series={include_series}, provider_type={provider_type}, use_multi_step={use_multi_step}"
    )

    if not self._initialized:
        logger.warning("[DOCS-RAG] Provider not initialized - cannot proceed")
        return []

    try:
        logger.info(f"[DOCS-RAG] Current provider path: {self._current_path}")

        # Switch provider if requested
        if provider_type and provider_type != self._current_path:
            logger.info(
                f"[DOCS-RAG] Switching provider from {self._current_path} to {provider_type}"
            )
            await self.switch_provider(provider_type)
            logger.info(
                f"[DOCS-RAG] Successfully switched to {provider_type} provider"
            )

        # Validate current state
        if not self.vector_store:
            logger.error("[DOCS-RAG] Vector store is None")
            return []
        if not self.enhanced_chunks:
            logger.error("[DOCS-RAG] No enhanced chunks available")
            return []

        logger.info(
            f"[DOCS-RAG] Using vector store with {len(self.enhanced_chunks)} chunks"
        )

        # Retrieve context using either adaptive multi-step or standard approach
        if use_multi_step:
            logger.info("[DOCS-RAG] Using adaptive multi-step retrieval")
            results = await adaptive_multi_step_retrieval(
                query=query,
                vector_store=self.vector_store,
                enhanced_chunks=self.enhanced_chunks,
                k=k
            )
        else:
            # Use the existing topic-aware retrieval
            logger.info("[DOCS-RAG] Using standard topic-aware retrieval")
            results = await get_topic_aware_context(
                query=query,
                vector_store=self.vector_store,
                enhanced_chunks=self.enhanced_chunks,
                k=k
            )

        logger.info(
            f"[DOCS-RAG] Retrieved {len(results)} results"
        )

        # Format results
        formatted_results = []
        logger.info("[DOCS-RAG] Starting result formatting")

        for i, result in enumerate(results):
            source_path = result.get("source")
            source_url = path_registry.get_url(source_path) if source_path else None

            logger.debug(f"[DOCS-RAG] Processing result {i+1}/{len(results)}")
            logger.debug(f"[DOCS-RAG] Source path: {source_path}")
            logger.debug(f"[DOCS-RAG] Source URL: {source_url}")

            formatted_result = {
                "content": result["content"],
                "section": result["section"],
                "source": source_url or "",
                "source_path": source_path,
                "summary": result["summary"],
                "score": result["score"],
                "metadata": {
                    "related_documents": result.get("related_documents", []),
                    "is_priority": result.get("is_priority", False),
                    "docs_path": self._current_path,
                    "retrieval_type": result.get("retrieval_type", "primary"),
                    "retrieval_query": result.get("retrieval_query", query),
                },
            }
            formatted_results.append(formatted_result)

        return formatted_results

    except Exception as e:
        logger.error(f"[DOCS-RAG] Error during retrieval: {e}")
        return []
```

### 4. Update Chat Response Generation

Modify `generate_ai_response` in `app/routes/chat.py` to use adaptive multi-step retrieval:

```python
# In generate_ai_response function
context_retrieval_start = datetime.now()
logger.info("[RAG] Retrieving relevant context...")
context_chunks = await rag_provider_obj.get_relevant_context(
    message,
    k=7,
    include_series=is_process_query,
    provider_type=provider_name,
    use_multi_step=True  # Enable adaptive multi-step retrieval
)
context_retrieval_time = (
    datetime.now() - context_retrieval_start
).total_seconds()
```

## Benefits of the Adaptive Multi-Step Retrieval Approach

1. **Complexity-Based Retrieval**: Dynamically adjusts retrieval depth and breadth based on the inherent complexity of the query.

2. **More Complete Answers**: Identifies when initial results are insufficient and intelligently retrieves additional context.

3. **Resource Efficiency**: Uses fewer resources for simple questions, more for complex ones - optimizing token usage.

4. **Follow-up Intelligence**: Analyzes retrieval results to determine what information is still missing and formulates targeted follow-up queries.

5. **Better User Experience**: Provides more comprehensive answers without requiring users to rephrase or break down complex questions.

## Implementation Notes

1. Add necessary imports in each file:

   ```python
   import json
   from typing import Dict, List, Any, Optional
   from langchain_community.vectorstores import FAISS
   ```

2. The implementation builds on the existing RAG architecture without requiring a complete redesign.

3. Performance considerations:

   - Multi-step retrieval will increase latency due to additional LLM calls and retrieval operations
   - Set appropriate timeout limits and consider async execution to mitigate impact
   - Monitor token usage and runtime metrics to optimize performance

4. Testing the implementation:
   - Create a set of complex test queries that require multi-step retrieval
   - Compare results with and without multi-step retrieval
   - Track retrieval quality metrics: relevance, completeness, and hallucination rate
