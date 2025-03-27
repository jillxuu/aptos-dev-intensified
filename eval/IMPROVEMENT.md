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
- Primary documentation citations are missing or incorrect - while the generic discussions link is intentionally added as a supplementary resource, the responses fail to cite the specific documentation pages
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

## Root Cause Analysis and Comprehensive Solutions

After researching best practices for RAG systems focused on technical documentation, the issues in our system can be categorized into three main areas:

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
  - Score factors should include: semantic similarity, code presence, documentation recency, and source reliability
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

1. **First Phase (Foundation):**
   - Redesign the chunking strategy to preserve semantic units and document structure
   - Implement metadata tagging for SDK types and document categories
   - Add comprehensive logging to the retrieval process for better diagnostics

2. **Second Phase (Retrieval Enhancement):**
   - Implement hybrid retrieval (dense + sparse)
   - Add query classification and expansion
   - Develop a re-ranking system optimized for technical documentation

3. **Third Phase (Response and Citation Improvement):**
   - Implement citation-focused prompting and inline citation format
   - Add post-generation verification
   - Develop confidence scoring for citations

4. **Fourth Phase (Continuous Improvement):**
   - Set up an automated evaluation pipeline to continuously test retrieval quality
   - Implement feedback mechanisms to capture when citations are incorrect
   - Develop a system to automatically update the knowledge base when documentation changes

By implementing these changes, the RAG system should be able to provide not just semantically relevant responses, but also accurate citations to the specific documentation sources, making it more useful for Aptos developers.
