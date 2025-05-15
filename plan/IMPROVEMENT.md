# Potential Improvements for Aptos Developer RAG System

- Tune the similarity threshold (try increasing from 0.2 to 0.3-0.4)
- Enhance keyword extraction by adding more Aptos-specific technical terms to priority vocabulary
- Expand the priority topics list with more Aptos technical concepts
- Add metadata tagging to explicitly mark content by SDK type (TypeScript, Python, Rust)
- Update chunking strategy to preserve code block integrity with their explanations
- Implement query expansion specifically for Aptos-related technical terms
- Try newer embedding models like OpenAI's `text-embedding-3-small` or `text-embedding-3-large`
- Adjust the MAX_CHUNKS_TO_RETURN value (try more than 5 for complex queries)
- Implement dynamic k values based on query complexity
- Add post-retrieval re-ranking using a more sophisticated relevance model
- Enhance citation mechanism to clearly mark sources in responses
- Add confidence scoring to indicate when system is uncertain about source citations
- Implement SDK-specific intent detection in the query processing
- Try embedding-based similarity instead of just keyword overlap
- Add hierarchical chunking (both specific and contextual chunks)
- Improve preprocessing to identify code blocks and keep them intact
- Add verification step to check if response actually aligns with retrieved sources
- Experiment with domain-specific embedding models for blockchain/Aptos content
- Add query classification to determine if question is about specific SDK or feature
- Use weighted retrieval to prioritize official documentation over community content

## Notes from rag_evaluation_results.json Analysis

- Looks like semantic similarity is not the right metric- answers are semantically similar with incorrect citation, could there be multiple right sources?
- URL citations are consistently wrong with ground truth sources
- High semantic similarity scores (avg ~0.85) suggest content is relevant, but low ROUGE/BLEU scores indicate different phrasing/structure from ground truth
- No retrieval metrics are being captured in evaluation (all zeros), suggesting the RAG retrieval component isn't being properly evaluated or logged or always incorrect
- In code-heavy questions (e.g., TodoList resource checking), AI gives more implementation details than ground truth but misses citing official tutorials (We need to program it to stay away from coding questions for now, and just fetch the relevant sources, not code synthesis)
- Documentation structure mismatch - responses cite URLs that don't match the current documentation organization (e.g., references to /build/indexer/indexer-api instead of /build/en/api/reference)
- Responses about wallets and SDKs use outdated package names and APIs (e.g., "aptos-wallet-adapter" vs "@aptos-labs/wallet-adapter-react")?
- For complex topics (validator nodes, coin minting), AI tends to provide step-by-step instructions but misses citing specific guides
- Semantic vs. structural differences - responses for technical topics like fungible assets have good semantic similarity (~0.90) but structured differently from documentation
- Missing specific context retrieval - the core documentation chunks with accurate information don't seem to be retrieved properly
- Responses to SDK-specific questions don't differentiate between SDKs (Python, TypeScript, etc.) in their citations
- Simplified questions about APIs (e.g., Indexer filtering) have better performance, suggesting retrieval works better for focused topics

## Root Cause Analysis and Solutions

The issues in our system can be categorized into three main areas:

### 1. Chunking Strategy Issues

**Current Issues:**

- Documentation is likely being chunked in a way that breaks semantic units (splitting code examples from explanations)
- SDK-specific content may be mixed or not properly labeled
- Chunks may be too small or lack critical context for proper understanding
- Document structure and hierarchy information is being lost during chunking

**Solutions:**

- **Implement Semantic Chunking**: Rather than fixed-size chunks, use semantic boundaries like headers, paragraphs, and code blocks to define chunk boundaries
- **Preserve Code-Documentation Relationships**: Keep code examples together with their explanatory text in the same chunk
- **Hierarchical Chunking**: Create multi-level chunks - small specific chunks for precise retrieval, and larger parent chunks to maintain context
- **Document Structure Metadata**: Attach metadata to each chunk including:
  - Document path/URL
  - Section hierarchy (parent/child relationships)
  - Document type (tutorial, reference, guide)
  - SDK type (TypeScript, Python, Rust)
  - Last updated timestamp
- **Overlap Strategy**: Use sliding windows with 10-20% overlap between chunks to prevent context loss at boundaries

### 2. Retrieval System Issues

**Current Issues:**

- Retrieved results may not include the most relevant documentation pages
- The system appears to be retrieving semantically related but contextually incorrect information
- No differentiation between different SDKs or documentation types in retrieval
- URL paths in responses suggest retrieval is using outdated document mappings
- Zero retrieval metrics indicate possible evaluation/logging issues or complete retrieval failure

**Solutions:**

- **Hybrid Retrieval**: Implement a combination of:
  - Dense retrieval (vector embeddings) for semantic understanding
  - Sparse retrieval (BM25/keyword) for terminology matching
  - Graph-based retrieval to leverage document relationships
- **Query Enhancement**:
  - Implement automatic query expansion for Aptos-specific terminology
  - Use query classification to identify SDK intent (Python vs. TypeScript)
  - Generate multiple query variations to increase retrieval coverage
- **Re-ranking Pipeline**:
  - Implement a two-stage retrieval system: broad retrieval followed by precise re-ranking
  - Use a specialized re-ranker model trained on technical documentation
  - Score factors should include: semantic similarity, code presence, documentation recency
- **URL Mapping Database**: Create and maintain a database mapping old doc URLs to new ones to handle documentation reorganization
- **Instrumentation**: Add comprehensive logging to track which documents are being retrieved and why

### 3. Citation and Response Generation Issues

**Current Issues:**

- Responses fail to properly cite the specific sources of information
- Generated answers include syntactically correct but outdated information (old package names, APIs)
- The model may be relying on its parametric knowledge rather than retrieved documents
- Cited URLs don't match the ground truth documentation pages

**Solutions:**

- **Citation-Focused Prompting**: Explicitly instruct the model to cite sources for each claim or code snippet
- **Tool-Based Citation**: Use structured tool calling to force the model to link each part of its response to specific source documents
- **Post-Generation Verification**: Implement a verification step to check if the response aligns with retrieved documents
- **Source Tracing**: For each chunk used in generation, maintain a link to its original document and include in response
- **Documentation Version Control**: Include document version/last updated timestamp in context to avoid outdated information
- **Inline Citation Format**: Adopt a standard format for inline citations that makes it clear which parts of the response come from which sources
- **Citation Confidence Scoring**: Add confidence scores to citations, indicating how strongly the source supports the claim

### Implementation Plan

1. **First Phase (chunking):**

   - Redesign the chunking strategy to preserve semantic units and document structure
   - Implement metadata tagging for SDK types and document categories
   - Add comprehensive logging to the retrieval process for better diagnostics

2. **Second Phase (Retrieval Enhancement):**

   - Implement hybrid retrieval (dense + sparse + Graph)
   - Add query classification and expansion
   - Develop a re-ranking system optimized for technical documentation

3. **Third Phase (Response and Citation Improvement):**
   - Implement citation-focused prompting and inline citation format
   - Add post-generation verification
   - Develop confidence scoring for citations

## Chunking Implementation - Current Approach vs. Improvements

### Current Chunking Implementation Analysis

The current implementation in `scripts/preprocess_topic_chunks.py` has the following characteristics:

1. **Header-Based Chunking**:

   - Documents are split strictly at markdown headers (H1, H2, H3)
   - Each header section becomes an independent chunk
   - If no headers exist, the entire document becomes one chunk

2. **Simplistic Topic Relationships**:

   - Keywords are extracted from chunk titles and content (max 20 keywords per chunk)
   - Relationships between chunks are calculated based on keyword overlap
   - A low similarity threshold (0.2) is used to relate chunks

3. **Metadata Limitations**:

   - Basic metadata includes title, source path, and section
   - No explicit SDK tagging or document type classification
   - No hierarchical relationships between chunks are tracked

4. **Code Block Handling Issues**:
   - No special handling for code blocks
   - Code examples can be separated from their explanations if they cross header boundaries

### Chunking Improvement Implementation Plan

#### 1. Enhance Document Parsing and Segmentation

````python
def process_markdown_document(content: str, file_path: str = "") -> List[Document]:
    """Enhanced document processing that preserves code blocks with their context."""
    # Identify code blocks and their surrounding context
    code_blocks = extract_code_blocks_with_context(content)

    # Split by headers while preserving code block integrity
    chunks = []
    current_chunk = []
    current_title = ""
    in_code_block = False
    code_block_buffer = []

    lines = content.split("\n")

    for line in lines:
        # Check if line is part of a code block
        if line.startswith("```"):
            in_code_block = not in_code_block
            code_block_buffer.append(line)
            continue

        if in_code_block:
            code_block_buffer.append(line)
            continue

        # Handle headers only when not in code block
        if line.startswith("# ") and not in_code_block:
            # Save previous chunk if it exists
            if current_chunk and current_title:
                chunks.append((current_title, "\n".join(current_chunk)))
            # Start new chunk
            current_title = line.replace("# ", "").strip()
            current_chunk = []
        # Similar handling for H2 and H3 headers

        # Add line to current chunk
        current_chunk.append(line)

    # Process chunks to ensure code blocks stay with explanations
    enhanced_chunks = ensure_code_block_integrity(chunks)

    # Convert to Document objects with enhanced metadata
    return create_document_objects(enhanced_chunks, file_path)
````

#### 2. Create Hierarchical Chunk Structure

```python
def create_hierarchical_chunks(chunks: List[Document]) -> List[Document]:
    """Create parent-child relationships between chunks."""
    hierarchical_chunks = []
    current_parent = None

    # Sort chunks by their header level and document position
    sorted_chunks = sort_chunks_by_hierarchy(chunks)

    for chunk in sorted_chunks:
        header_level = detect_header_level(chunk.metadata.get("title", ""))

        if header_level == 1:  # H1 header is a parent
            current_parent = chunk
            current_parent.metadata["children"] = []
            hierarchical_chunks.append(current_parent)
        elif header_level > 1 and current_parent:  # Child chunk
            # Add parent reference
            chunk.metadata["parent_id"] = current_parent.metadata.get("chunk_id")
            # Add to parent's children list
            current_parent.metadata["children"].append(chunk.metadata.get("chunk_id"))
            hierarchical_chunks.append(chunk)

    return hierarchical_chunks
```

#### 3. Implement Chunk Overlap Strategy

```python
def create_overlapping_chunks(chunks: List[Document], overlap_percentage: float = 0.15) -> List[Document]:
    """Create overlapping chunks to prevent context loss at boundaries."""
    overlapped_chunks = []

    for i, chunk in enumerate(chunks):
        if i > 0:
            # Calculate overlap text from previous chunk
            prev_content = chunks[i-1].page_content
            overlap_size = int(len(prev_content) * overlap_percentage)
            overlap_text = prev_content[-overlap_size:] if overlap_size > 0 else ""

            # Add overlap to current chunk
            new_content = overlap_text + "\n" + chunk.page_content
            chunk.page_content = new_content
            chunk.metadata["has_overlap"] = True

        overlapped_chunks.append(chunk)

    return overlapped_chunks
```

#### 4. Enhanced Metadata Structure

```python
def enhance_chunk_metadata(chunk: Document, file_path: str, content: str) -> None:
    """Add rich metadata to each chunk."""
    # Basic metadata
    chunk.metadata["source"] = file_path
    chunk.metadata["document_url"] = convert_path_to_url(file_path)
    chunk.metadata["last_updated"] = get_document_timestamp(file_path)

    # Content analysis
    chunk.metadata["contains_code"] = has_code_block(content)
    chunk.metadata["code_languages"] = detect_code_languages(content)

    # Create summary for better context
    chunk.metadata["summary"] = generate_chunk_summary(content)

    # If this is a code block, add code-specific summary
    if chunk.metadata["contains_code"]:
        chunk.metadata["code_summary"] = generate_code_summary(
            code=content,
            context=chunk.metadata.get("surrounding_text", ""),
            parent_title=chunk.metadata.get("parent_title", "")
        )
```

#### 5. Main Processing Function

```python
def process_documentation(docs_dir: str, output_path: str) -> None:
    """Enhanced documentation processing with improved chunking."""
    # Load and process documents
    raw_chunks = load_and_split_documents(docs_dir)

    # Enhance chunks with code block preservation
    code_preserved_chunks = preserve_code_blocks(raw_chunks)

    # Create hierarchical structure
    hierarchical_chunks = create_hierarchical_chunks(code_preserved_chunks)

    # Add chunk overlap
    overlapped_chunks = create_overlapping_chunks(hierarchical_chunks)

    # Enhance metadata
    enhanced_chunks = []
    for chunk in overlapped_chunks:
        enhance_chunk_metadata(chunk, chunk.metadata["source"], chunk.page_content)
        enhanced_chunks.append(chunk)

    # Analyze topic relationships (improved)
    relationships = analyze_topic_relationships_enhanced(enhanced_chunks)

    # Store enhanced data
    store_enhanced_metadata(enhanced_chunks, relationships, output_path)
```

By implementing these improvements, we can transform the current header-based chunking into a sophisticated semantic chunking system that preserves code block integrity, maintains document hierarchy, and contains rich metadata for improved retrieval and citation.

## Hierarchical Retrieval Implementation

The hierarchical chunk structure and enhanced metadata are utilized during retrieval to significantly improve the context quality and citation accuracy. Here's how the retrieval process leverages this information:

### 1. Enhanced Retrieval Algorithm

```python
async def get_context_with_hierarchical_expansion(
    query: str,
    vector_store: FAISS,
    enhanced_chunks: List[Dict[str, Any]],
    k: int = 5
) -> List[Dict[str, Any]]:
    """
    Retrieval algorithm that leverages hierarchical chunk relationships.

    Args:
        query: The user query
        vector_store: Vector store containing embeddings
        enhanced_chunks: List of chunks with hierarchical metadata
        k: Base number of chunks to retrieve

    Returns:
        List of relevant chunks with their context
    """
    # Initial vector retrieval - get base relevant chunks
    base_results = await vector_store.similarity_search_with_score(query, k=k)

    # Create a mapping for quick chunk lookup
    chunk_map = {chunk["id"]: chunk for chunk in enhanced_chunks}

    # Track which chunks we've included to avoid duplication
    included_chunk_ids = set()

    # Final results with hierarchical expansion
    expanded_results = []

    # Process each base result
    for doc, score in base_results:
        chunk_id = doc.metadata.get("id")
        if not chunk_id or chunk_id in included_chunk_ids:
            continue

        # Get the enhanced chunk with full metadata
        chunk = chunk_map.get(chunk_id)
        if not chunk:
            continue

        # Add the primary chunk
        included_chunk_ids.add(chunk_id)

        # Create result with the base chunk
        result = {
            "content": doc.page_content,
            "section": doc.metadata.get("section", ""),
            "source": doc.metadata.get("source", ""),
            "document_url": doc.metadata.get("document_url", ""),
            "score": float(score),
            "sdk_type": doc.metadata.get("sdk_type", "unknown"),
            "document_type": doc.metadata.get("document_type", "unknown"),
            "contains_code": doc.metadata.get("contains_code", False),
            "related_chunks": []
        }

        # Add parent chunk if it exists (for context)
        parent_id = doc.metadata.get("parent_id")
        if parent_id and parent_id not in included_chunk_ids:
            parent_chunk = chunk_map.get(parent_id)
            if parent_chunk:
                included_chunk_ids.add(parent_id)
                result["parent_chunk"] = {
                    "id": parent_id,
                    "content": parent_chunk["content"],
                    "title": parent_chunk["metadata"].get("title", ""),
                    "summary": parent_chunk["metadata"].get("summary", "")
                }

        # Add important child chunks (for details)
        child_ids = doc.metadata.get("children", [])
        for child_id in child_ids[:2]:  # Limit to prevent context bloat
            if child_id not in included_chunk_ids:
                child_chunk = chunk_map.get(child_id)
                if child_chunk:
                    included_chunk_ids.add(child_id)
                    result["related_chunks"].append({
                        "id": child_id,
                        "content": child_chunk["content"],
                        "title": child_chunk["metadata"].get("title", ""),
                        "summary": child_chunk["metadata"].get("summary", ""),
                        "relationship": "child"
                    })

        # Add sibling chunks if they exist and are relevant (for related information)
        if parent_id:
            parent_chunk = chunk_map.get(parent_id)
            if parent_chunk:
                sibling_ids = parent_chunk["metadata"].get("children", [])
                for sibling_id in sibling_ids:
                    # Skip self and already included siblings
                    if sibling_id == chunk_id or sibling_id in included_chunk_ids:
                        continue

                    sibling_chunk = chunk_map.get(sibling_id)
                    if sibling_chunk:
                        # Only include siblings that are semantically related to the query
                        sibling_relevance = calculate_relevance(sibling_chunk["content"], query)
                        if sibling_relevance > 0.6:  # Threshold for relevance
                            included_chunk_ids.add(sibling_id)
                            result["related_chunks"].append({
                                "id": sibling_id,
                                "content": sibling_chunk["content"],
                                "title": sibling_chunk["metadata"].get("title", ""),
                                "summary": sibling_chunk["metadata"].get("summary", ""),
                                "relationship": "sibling",
                                "relevance": sibling_relevance
                            })

        expanded_results.append(result)

    # Add any strongly topic-related chunks that weren't included through hierarchy
    await add_topic_related_chunks(
        expanded_results,
        query,
        chunk_map,
        included_chunk_ids,
        threshold=0.75  # Higher threshold for topic-related additions
    )

    # Re-rank the final results
    expanded_results.sort(key=lambda x: (
        x.get("sdk_match", False),  # SDK-specific chunks first
        x.get("document_type_relevance", 0),  # More relevant document types higher
        x.get("score", 0)  # Then by similarity score
    ), reverse=True)

    return expanded_results
```

### 2. Improved Context Assembly for the LLM

```python
def assemble_llm_context(expanded_results: List[Dict[str, Any]], query: str) -> str:
    """
    Assemble context for the LLM from hierarchically expanded results.

    Args:
        expanded_results: List of expanded results with hierarchical information
        query: The user query for context adaptation

    Returns:
        Formatted context string for the LLM
    """
    context_parts = []

    # Determine if query is SDK-specific
    sdk_type = detect_sdk_from_query(query)

    # Filter to prioritize relevant SDK if detected
    if sdk_type != "general":
        # Boost SDK-specific chunks
        for result in expanded_results:
            if result.get("sdk_type") == sdk_type:
                result["relevance_boost"] = 1.5

    # Sort by final relevance (combining vector score and boosts)
    sorted_results = sorted(
        expanded_results,
        key=lambda x: x.get("score", 0) * x.get("relevance_boost", 1.0),
        reverse=True
    )

    # Assemble hierarchical context with proper citations
    for i, result in enumerate(sorted_results):
        # Start with document path for citation
        doc_url = result.get("document_url", result.get("source", "unknown"))

        # Add parent context if available (provides broader context)
        parent_info = result.get("parent_chunk")
        if parent_info:
            context_parts.append(f"## Document: {doc_url}")
            context_parts.append(f"### Section: {parent_info.get('title', 'Main Section')}")
            context_parts.append(f"Overview: {parent_info.get('summary', '')}")
            context_parts.append("")

        # Add the main chunk content with proper citation
        context_parts.append(f"## Document: {doc_url}")
        context_parts.append(f"### Section: {result.get('section', 'Section')}")

        # Add SDK type if relevant
        if result.get("sdk_type") != "general":
            context_parts.append(f"SDK: {result.get('sdk_type').upper()}")

        # Add the main content
        context_parts.append(result.get("content", ""))
        context_parts.append("")

        # Add important related chunks for additional context
        related_chunks = result.get("related_chunks", [])
        if related_chunks:
            context_parts.append("### Related Information:")
            for related in related_chunks[:2]:  # Limit to prevent context overflow
                context_parts.append(f"From {related.get('title', 'Related Section')}:")
                # Include a summary or excerpt rather than full content
                if related.get("summary"):
                    context_parts.append(related.get("summary"))
                else:
                    # Extract first paragraph if no summary
                    content = related.get("content", "")
                    excerpt = content.split("\n\n")[0] if "\n\n" in content else content[:200]
                    context_parts.append(f"{excerpt}...")
                context_parts.append("")

    # Assemble the final context string
    return "\n".join(context_parts)
```

### 3. Using the Hierarchical Structure to Improve Citation

```python
def create_citation_map(expanded_results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Create a citation mapping to track which content comes from which sources.

    Args:
        expanded_results: The expanded results with hierarchical information

    Returns:
        Dictionary mapping content fingerprints to citation information
    """
    citation_map = {}

    for result in expanded_results:
        # Create a fingerprint for this chunk's content
        content = result.get("content", "")
        fingerprint = compute_content_fingerprint(content)

        # Store citation information
        citation_map[fingerprint] = {
            "url": result.get("document_url", result.get("source", "")),
            "section": result.get("section", ""),
            "title": result.get("title", ""),
            "sdk_type": result.get("sdk_type", "general"),
            "confidence": calculate_citation_confidence(result)
        }

        # Add related chunks to the citation map
        for related in result.get("related_chunks", []):
            related_content = related.get("content", "")
            related_fingerprint = compute_content_fingerprint(related_content)

            citation_map[related_fingerprint] = {
                "url": result.get("document_url", result.get("source", "")),  # Same document
                "section": related.get("title", "Related Section"),
                "relationship": related.get("relationship", "related"),
                "sdk_type": result.get("sdk_type", "general"),
                "confidence": calculate_citation_confidence(related)
            }

    return citation_map
```

This hierarchical retrieval implementation provides the LLM with rich, contextually organized information that preserves document structure and relationships. The system not only retrieves individual chunks but understands and maintains the connections between parent sections, sibling topics, and child details. This allows for more accurate citations and more comprehensive answers that reflect the true structure of the documentation.

## Enhanced Code Retrieval and Query Understanding

Code retrieval presents unique challenges in technical documentation systems, particularly for programming languages like Move that have their own specific syntax and semantics. The semantic gap between natural language queries and code snippets often leads to suboptimal retrieval results. Strategy to improve code and content retrieval without requiring separate embedding models:

### 1. Enhanced Chunk Preparation with Summaries

```python
def process_documentation_chunk(chunk, is_code_block=False):
    """Process a documentation chunk with enhanced metadata."""
    # For code blocks, add a natural language summary
    if is_code_block:
        # Use OpenAI to generate a summary of what the code does
        code_summary = generate_code_summary(
            code=chunk.content,
            context=chunk.surrounding_text,
            parent_title=chunk.metadata.get("parent_title", "")
        )

        # Add the summary to the chunk's metadata
        chunk.metadata["code_summary"] = code_summary

        # Create the content that will be embedded
        chunk.embedding_text = f"""
        {chunk.metadata.get("parent_title", "")}

        Code Summary: {code_summary}

        Code:
        {chunk.content}
        """
    else:
        # For regular text chunks, keep the original content
        chunk.embedding_text = chunk.content

    return chunk
```

### 2. Query Intent Understanding with LLM

Instead of hardcoded keyword detection, use an LLM to understand query intent and reformulate it:

```python
async def analyze_query_intent(query):
    """Use LLM to analyze query intent and generate enhanced queries."""

    prompt = f"""
    Given this user query about Aptos development:
    "{query}"

    1. What is the primary intent of this query? (e.g., "seeking code example", "conceptual understanding", "troubleshooting")
    2. What specific Aptos concepts, SDKs, or components is this related to?
    3. Generate 2-3 reformulated versions of this query that might improve retrieval, especially if the query is about code implementation.

    Return as JSON with fields: primary_intent, aptos_components, reformulated_queries
    """

    response = await openai.ChatCompletion.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0
    )

    result = json.loads(response.choices[0].message.content)
    return result
```

### 3. Multi-query Retrieval Strategy

Execute multiple queries based on the intent analysis to capture different aspects:

```python
async def enhanced_retrieval(original_query, vector_store):
    """Retrieve relevant chunks using multi-query strategy based on intent."""

    # Step 1: Analyze query intent and generate variations
    intent_analysis = await analyze_query_intent(original_query)

    # Step 2: Prepare multiple queries
    queries = [original_query] + intent_analysis["reformulated_queries"]

    # Track all results with scores
    all_results = {}

    # Step 3: Execute each query
    for query in queries:
        results = await vector_store.similarity_search_with_score(query, k=5)

        # Add results to the collection with boosting based on intent
        for doc, score in results:
            chunk_id = doc.metadata.get("id")

            # If we already have this chunk, use the highest score
            if chunk_id in all_results:
                all_results[chunk_id]["score"] = max(all_results[chunk_id]["score"], score)
                # Track which queries matched this chunk
                all_results[chunk_id]["matched_queries"].append(query)
            else:
                # Apply intent-based boosting
                boost = 1.0

                # Boost code chunks if intent suggests code is needed
                if intent_analysis["primary_intent"] == "seeking code example" and doc.metadata.get("is_code_block"):
                    boost = 1.5

                # Boost chunks that match specific components mentioned in the query
                for component in intent_analysis["aptos_components"]:
                    if component.lower() in doc.page_content.lower():
                        boost += 0.2

                all_results[chunk_id] = {
                    "doc": doc,
                    "score": score * boost,
                    "matched_queries": [query]
                }

    # Step 4: Sort and return top results
    sorted_results = sorted(
        all_results.values(),
        key=lambda x: x["score"],
        reverse=True
    )

    # Return only the documents and scores
    return [(item["doc"], item["score"]) for item in sorted_results[:7]]
```

### 4. Hierarchical Assembly with Prioritized Content

Leverage the hierarchical structure when assembling the context for the LLM:

````python
def assemble_context_by_priority(results, original_query, intent_analysis):
    """Assemble context for the LLM with prioritized content based on query intent."""

    # Categorize results
    code_examples = []
    concept_explanations = []
    related_content = []

    for doc, score in results:
        if doc.metadata.get("is_code_block"):
            code_examples.append((doc, score))
        elif doc.metadata.get("content_type") == "concept":
            concept_explanations.append((doc, score))
        else:
            related_content.append((doc, score))

    # Prioritize based on intent
    primary_intent = intent_analysis["primary_intent"]

    context_parts = ["# Relevant Information"]

    # If seeking code, prioritize code examples
    if primary_intent == "seeking code example":
        priority_order = [code_examples, concept_explanations, related_content]
    # If seeking conceptual understanding, prioritize explanations
    elif primary_intent == "conceptual understanding":
        priority_order = [concept_explanations, related_content, code_examples]
    # Default ordering
    else:
        priority_order = [related_content, concept_explanations, code_examples]

    # Assemble context in priority order
    for category in priority_order:
        for doc, score in category[:3]:  # Limit to top 3 from each category
            # Add source information for citation
            context_parts.append(f"## Source: {doc.metadata.get('document_url', 'Unknown')}")

            # Add parent context if available
            if doc.metadata.get("parent_title"):
                context_parts.append(f"### Section: {doc.metadata.get('parent_title')}")

            # For code blocks, include both summary and code
            if doc.metadata.get("is_code_block"):
                context_parts.append(f"Code Summary: {doc.metadata.get('code_summary', '')}")
                context_parts.append(f"```{doc.metadata.get('code_language', '')}")
                context_parts.append(doc.page_content)
                context_parts.append("```")
            else:
                # For regular content, just include the content
                context_parts.append(doc.page_content)

            context_parts.append("")  # Empty line between entries

    return "\n".join(context_parts)
````

### Key Benefits of This Approach:

1. **Bridging the Code-Text Semantic Gap**: By generating natural language summaries of code blocks during preprocessing, we create embeddings that capture both the code structure and its functional meaning.

2. **Intelligent Query Understanding**: Using an LLM to analyze query intent moves beyond simple keyword matching, detecting when a user is seeking code examples even when not explicitly stated.

3. **Multi-Query Retrieval Strategy**: By generating multiple query variations, we increase the chance of finding relevant code examples that might use different terminology than the user's original query.

4. **Intent-Guided Context Assembly**: The system prioritizes different content types based on detected intent, ensuring code examples appear prominently when the user is seeking implementation details.

5. **Hierarchical Presentation**: By maintaining relationships between code examples and their explanatory context, users receive both the code they need and the information required to understand it.

This approach effectively addresses the challenge of retrieving relevant code examples without requiring separate embedding models, making it a pragmatic solution that can be implemented within the existing system architecture while significantly improving retrieval performance for code-related queries.

## Future Consideration: Large Code Block Handling

For future optimization, we may need to address how large code blocks are chunked compared to smaller ones. Currently, code blocks are treated as atomic units, which could be problematic for very large functions.

### Potential Issues with Current Approach:

1. Large functions create oversized chunks that:

   - Consume too much of the context window
   - Include excessive detail when only a specific part is relevant
   - Result in imbalanced chunk sizes

2. Small functions may be:
   - Unnecessarily grouped together
   - Missing important surrounding context
   - Too granular for effective retrieval

### Proposed Solution:

```python
def process_code_block(code_block: str, context: str, max_size: int = 1500) -> List[Dict]:
    """Process code blocks based on their size and structure."""

    # For small functions (under max_size), keep as single chunk
    if len(code_block) < max_size:
        return [{
            "content": code_block,
            "context": context,
            "is_code_block": True,
            "is_partial": False
        }]

    # For large functions, split intelligently
    chunks = []

    # Try to split at logical boundaries
    logical_parts = split_at_logical_boundaries(code_block)

    for i, part in enumerate(logical_parts):
        chunks.append({
            "content": part,
            "context": context,
            "is_code_block": True,
            "is_partial": True,
            "part_number": i + 1,
            "total_parts": len(logical_parts),
            "parent_function": extract_function_name(code_block)
        })

    return chunks
```

Key improvements to consider:

1. **Size-Based Processing**:

   - Small functions (< 1500 chars) remain as single chunks
   - Large functions are split while preserving logical structure

2. **Metadata for Split Chunks**:

   - Track partial chunks of larger functions
   - Maintain part numbers and total parts
   - Keep reference to parent function name

3. **Context Preservation**:

   - Each split chunk maintains reference to its parent context
   - Important comments and documentation stay with relevant code
   - Function signatures and important imports are preserved

4. **Logical Splitting**:
   - Split at major comment blocks
   - Respect function/method boundaries
   - Keep related code together
   - Preserve code block integrity

This enhancement should be considered after implementing and evaluating the current planned improvements. If retrieval quality for large code blocks remains an issue, this solution can be implemented to better handle varying code block sizes.
