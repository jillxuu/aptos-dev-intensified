# Aptos devdoc chatbot

A fine-tuned AI assistant powered by RAG (Retrieval-Augmented Generation) on Aptos data. This project combines modern LLMs (ChatGPT) with Aptos-specific knowledge to provide accurate and contextual responses.

## Features

- RAG implementation using LangChain
- Support for both Claude and ChatGPT models
- FastAPI backend for efficient API handling
- React frontend with modern UI
- Vector storage using FAISS
- Real-time chat interface
- Enhanced semantic search with Cohere Rerank for improved retrieval quality
- Topic-based chunking for improved context retrieval

## RAG Implementation Flow

The Retrieval-Augmented Generation (RAG) system follows this process:

1. **Query Processing**: User query is received and processed
2. **Provider Selection**: The appropriate RAG provider is selected (default: topic-based)
3. **Vector Search**: Query is converted to an embedding and used to search the vector store
4. **Topic-Based Enhancement**: Retrieved chunks are enhanced with related documents based on topic similarity
5. **Context Formatting**: Retrieved chunks are formatted into a context string
6. **Prompt Creation**: Context is inserted into a system prompt template
7. **LLM Generation**: The prompt is sent to the LLM (GPT-4o) for response generation
8. **Response Streaming**: The generated response is streamed back to the user

### Why Vector Search Before Topic Enhancement?

Vector search happens before topic-based enhancement for efficiency and relevance reasons:

1. **Initial Relevance Filter**: Vector search efficiently narrows down the most semantically relevant documents from the entire corpus
2. **Computational Efficiency**: Topic analysis is more computationally intensive, so it's better to perform it on a smaller set of already relevant documents
3. **Two-Stage Retrieval**: This creates a two-stage retrieval process where:
   - Vector search finds the most relevant documents to the query
   - Topic enhancement expands this with topically related documents
4. **Precision & Recall Balance**: Vector search provides precision (exact matches), while topic enhancement improves recall (related concepts)

The topic-based approach enhances traditional vector search by considering document relationships and topic coherence, resulting in more comprehensive and contextually relevant responses.

## Setup

### Backend
1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your API keys:
```
OPENAI_API_KEY=your_openai_api_key
COHERE_API_KEY=your_cohere_api_key  # Optional, for enhanced reranking
```

### Frontend
1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
pnpm install
```

3. Start the development server:
```bash
pnpm run dev
```

## Usage

1. Start the backend server:
```bash
uvicorn app.main:app --reload
```

2. Start the frontend development server:
```bash
cd frontend
pnpm run dev
```

3. Access the application at `http://localhost:5173`

## Using Topic-Based RAG

The application now uses topic-based RAG by default, which provides improved context retrieval by considering document relationships and topic coherence.

1. **Preprocessing**: Before first use, ensure the enhanced chunks are generated:
```bash
./scripts/run_preprocessing.sh
```

2. **Running with Topic RAG**: The application uses topic-based RAG by default, but you can explicitly start it with:
```bash
./scripts/start_with_topic_rag.sh
```

3. **Testing**: You can test the topic-based RAG provider with:
```bash
python scripts/test_topic_provider.py
```

## Project Structure

```
.
├── app/                    # Backend application
│   ├── main.py            # FastAPI application
│   ├── models.py          # Pydantic models
│   ├── rag_providers/     # RAG providers implementation
│   ├── utils/             # Utility functions
│   └── routes/            # API routes
├── scripts/               # Utility scripts
│   ├── preprocess_topic_chunks.py  # Topic preprocessing
│   ├── run_preprocessing.sh        # Preprocessing script
│   └── test_topic_provider.py      # Test script
├── frontend/              # React frontend
├── data/                  # RAG data storage
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
``` 