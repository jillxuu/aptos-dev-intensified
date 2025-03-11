# Topic-Based Chunking for Aptos Developer Assistant

This feature enhances the RAG system by identifying and leveraging topic relationships between document chunks. It provides more comprehensive and coherent context for answering user queries.

## How It Works

1. **Preprocessing**: The system analyzes the Aptos documentation to identify topic relationships between document chunks.
2. **Topic Similarity**: Chunks are related based on shared keywords and topics, not just document structure.
3. **Enhanced Retrieval**: When retrieving context for a query, the system includes topically related content.
4. **Priority Topics**: Special attention is given to high-priority topics that are most relevant to Aptos developers.

## Usage

### Running the Preprocessing

Before using the topic-based RAG provider, you need to run the preprocessing script:

```bash
./scripts/run_preprocessing.sh
```

This will:
1. Process all documentation files
2. Identify topic relationships
3. Store the enhanced data in `data/enhanced_chunks.json`

### Using the Topic-Based RAG Provider

Once preprocessing is complete, the topic-based RAG provider will be available in the API. You can select it when making chat requests by setting the `rag_provider` parameter:

```json
{
  "message": "How do I create a Move module?",
  "rag_provider": "topic"
}
```

### Benefits

- **More Comprehensive Answers**: Provides broader context that spans multiple related documents
- **Topic Coherence**: Groups related content based on semantic similarity
- **Priority Focus**: Emphasizes content related to key Aptos development topics
- **Improved Multi-Topic Queries**: Better handles questions that span multiple topics

## Implementation Details

The implementation consists of:

1. **Preprocessing Script** (`scripts/preprocess_topic_chunks.py`): Analyzes documentation and identifies topic relationships
2. **Utility Module** (`app/utils/topic_chunks.py`): Provides functions to load and use enhanced chunks
3. **RAG Provider** (`app/rag_providers/topic_provider.py`): Implements the topic-based retrieval interface

## Configuration

You can adjust the preprocessing parameters in `scripts/preprocess_topic_chunks.py`:

- `SIMILARITY_THRESHOLD`: Minimum similarity score to consider chunks related (default: 0.2)
- `MAX_RELATED_CHUNKS`: Maximum number of related chunks to store (default: 10)
- `MIN_KEYWORD_LENGTH`: Minimum length for a keyword to be considered significant (default: 4)
- `MAX_KEYWORDS_PER_CHUNK`: Maximum number of keywords to extract per chunk (default: 20)
- `PRIORITY_TOPICS`: List of high-priority topics

## Maintenance

The preprocessing should be run whenever the documentation is updated. You can automate this by adding the preprocessing script to your CI/CD pipeline. 