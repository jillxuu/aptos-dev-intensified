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
- Responses about wallets and SDKs use outdated package names and APIs (e.g., "aptos-wallet-adapter" vs "@aptos-labs/wallet-adapter-react")?
- For complex topics (validator nodes, coin minting), AI tends to provide step-by-step instructions but misses citing specific guides
- Multiple responses reference wrong URL paths (e.g., references to /build/indexer/indexer-api instead of /build/en/api/reference)
- Responses for some technical topics like fungible assets have good semantic similarity (~0.90) but still cite incorrect sources
- Responses to SDK-specific questions don't differentiate between SDKs (Python, TypeScript, etc.) in their citations
- Simplified questions about APIs (e.g., Indexer filtering) have better performance, suggesting retrieval works better for focused topics
