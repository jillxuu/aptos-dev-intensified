# Aptos Dev Intensified

A monorepo containing the Aptos documentation chatbot and related packages. This project combines modern LLMs with Aptos-specific knowledge to provide accurate and contextual responses through RAG (Retrieval-Augmented Generation).

## Features

- RAG implementation using LangChain and OpenAI embeddings
- FastAPI backend with streaming support
- Modular React components with Tailwind CSS
- Vector storage using FAISS
- Real-time chat interface with message history
- Enhanced semantic search with topic-based chunking
- Monorepo structure with shared TypeScript configurations

## Project Structure

```
.
├── packages/                      # Shared packages
│   ├── chatbot-core/             # Core chatbot functionality
│   ├── chatbot-react/            # React hooks and context
│   ├── chatbot-ui-base/          # Base UI components
│   ├── chatbot-ui-tailwind/      # Tailwind styled components
│   └── tsconfig/                 # Shared TypeScript configs
├── apps/                         # Applications
│   └── demo/                     # Demo application
├── app/                          # Backend application
│   ├── main.py                   # FastAPI application
│   ├── models.py                 # Pydantic models
│   ├── rag_providers/            # RAG providers
│   ├── routes/                   # API routes
│   └── utils/                    # Utility functions
├── scripts/                      # Utility scripts
│   ├── preprocess_topic_chunks.py  # Topic preprocessing
│   └── run_preprocessing.sh        # Preprocessing script
├── data/                         # RAG data storage
├── requirements.txt              # Python dependencies
└── pnpm-workspace.yaml          # PNPM workspace config
```

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
CHAT_TEST_MODE=false
DEFAULT_RAG_PROVIDER=topic
```

### Frontend

1. Install dependencies:

```bash
pnpm install
```

2. Build packages:

```bash
pnpm build
```

3. Start the development server:

```bash
pnpm dev
```

## Development

### Available Scripts

- `pnpm build` - Build all packages
- `pnpm dev` - Start development servers
- `pnpm lint` - Lint all packages
- `pnpm format` - Format code using Prettier
- `pnpm format:check` - Check code formatting

### Code Style

The project uses Prettier for code formatting with the following configuration:

```json
{
  "semi": true,
  "trailingComma": "all",
  "singleQuote": true,
  "printWidth": 100,
  "tabWidth": 2,
  "useTabs": false,
  "bracketSpacing": true,
  "arrowParens": "avoid"
}
```

## RAG Implementation

The Retrieval-Augmented Generation (RAG) system follows this process:

1. **Query Processing**: User query is received and processed
2. **Provider Selection**: The appropriate RAG provider is selected (default: topic-based)
3. **Vector Search**: Query is converted to an embedding and used to search the vector store
4. **Topic-Based Enhancement**: Retrieved chunks are enhanced with related documents based on topic similarity
5. **Context Formatting**: Retrieved chunks are formatted into a context string
6. **Prompt Creation**: Context is inserted into a system prompt template
7. **LLM Generation**: The prompt is sent to the LLM for response generation
8. **Response Streaming**: The generated response is streamed back to the user

### Topic-Based RAG

The application uses topic-based RAG by default for improved context retrieval. Before first use, ensure the enhanced chunks are generated:

```bash
./scripts/run_preprocessing.sh
```

## License

This project is licensed under the terms of the license found in the LICENSE file.
