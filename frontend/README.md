# Aptos Chatbot Plugin

A React component that provides an embeddable Aptos AI chatbot for any React application.

## Installation

```bash
npm install aptos-chatbot-plugin
# or
yarn add aptos-chatbot-plugin
# or
pnpm add aptos-chatbot-plugin
```

## Usage

```jsx
import { AptosChatbotPlugin } from "aptos-chatbot-plugin";

function App() {
  return (
    <div className="my-app">
      <h1>My Application</h1>

      {/* Basic usage with default hosted backend */}
      <AptosChatbotPlugin />

      {/* With customization */}
      <AptosChatbotPlugin
        buttonText="Ask Aptos AI"
        modalTitle="Aptos AI Assistant"
        buttonClassName="my-custom-button-class"
        className="my-custom-container-class"
      />

      {/* With custom backend URL (advanced usage) */}
      <AptosChatbotPlugin apiUrl="https://your-custom-backend-url.com/api" />
    </div>
  );
}
```

## Props

| Prop              | Type                | Default        | Description                                                 |
| ----------------- | ------------------- | -------------- | ----------------------------------------------------------- |
| `buttonText`      | string              | "Ask Aptos AI" | Text displayed on the trigger button                        |
| `modalTitle`      | string              | "Ask Aptos AI" | Title displayed in the modal header                         |
| `buttonClassName` | string              | ""             | Additional CSS classes for the button                       |
| `className`       | string              | ""             | Additional CSS classes for the container                    |
| `apiUrl`          | string              | undefined      | Optional custom API URL to override the default backend URL |
| `ragProvider`     | string              | undefined      | Optional custom RAG provider name to use                    |
| `ragConfig`       | Record<string, any> | undefined      | Optional configuration for the RAG provider                 |

## Backend Configuration

The Aptos Chatbot Plugin comes with a pre-configured backend URL that connects to our hosted service. This means you don't need to set up or host your own backend to use the plugin.

### Default Hosted Backend

By default, the plugin connects to our hosted backend service, which provides:

- RAG implementation using LangChain
- Support for both Claude and ChatGPT models
- Up-to-date Aptos documentation and resources
- Reliable and scalable infrastructure

### Custom Backend (Advanced)

If you need to use your own backend:

1. Set up your own backend server using the code from our [GitHub repository](https://github.com/yourusername/aptos-dev-intensified)
2. Pass your custom backend URL using the `apiUrl` prop:

```jsx
<AptosChatbotPlugin apiUrl="https://your-custom-backend-url.com/api" />
```

## Customizing the RAG System

The plugin supports customizing the Retrieval-Augmented Generation (RAG) system, allowing you to use your own knowledge base instead of the default Aptos documentation.

### Using a GitHub Repository as Knowledge Base

The easiest way to use your own knowledge base is to use a public GitHub repository:

```jsx
<AptosChatbotPlugin ragProvider="github" buttonText="Ask about my repository" />
```

Before using the GitHub RAG provider, you need to initialize it with a repository URL:

```javascript
// Initialize the GitHub RAG provider
await axios.post(
  "https://your-backend-url.com/api/rag/provider/github/initialize",
  {
    repo_url: "https://github.com/username/repository",
    branch: "main", // Optional, defaults to main
    file_types: ["md", "mdx", "txt", "py", "js", "jsx", "ts", "tsx"], // Optional
    exclude_dirs: [".git", "node_modules"], // Optional
  },
);

// Then use it in your component
<AptosChatbotPlugin ragProvider="github" />;
```

This will:

1. Clone the GitHub repository
2. Process its contents
3. Build a vector store
4. Use it to answer questions

No need to run any Python code or set up a custom knowledge base manually!

### Using a Custom RAG Provider

For more advanced use cases, you can create a custom RAG provider:

```jsx
<AptosChatbotPlugin
  ragProvider="custom"
  ragConfig={{
    vector_store_path: "/path/to/your/vector_store",
    openai_api_key: "your-openai-api-key", // Optional, defaults to environment variable
  }}
/>
```

### Creating a Custom RAG Provider

To create a custom RAG provider for your own knowledge base:

1. Create a new RAG provider class that implements the `RAGProvider` interface:

```python
from app.rag_providers import RAGProvider, RAGProviderRegistry
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import logging
import os

logger = logging.getLogger(__name__)

class MyCustomRAGProvider(RAGProvider):
    """Custom RAG provider for my knowledge base."""

    def __init__(self):
        self._initialized = False
        self._vector_store = None

    @property
    def name(self) -> str:
        return "my_custom_provider"

    @property
    def description(self) -> str:
        return "Custom RAG provider for my knowledge base"

    async def initialize(self, config):
        # Initialize your vector store and embeddings
        vector_store_path = config.get("vector_store_path")
        openai_api_key = config.get("openai_api_key", os.getenv("OPENAI_API_KEY"))

        self._embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self._vector_store = FAISS.load_local(vector_store_path, self._embeddings)
        self._initialized = True

    async def get_relevant_context(self, query, k=5, include_series=True):
        # Implement your retrieval logic
        docs_with_scores = self._vector_store.similarity_search_with_score(query, k=k)

        # Format results
        results = []
        for doc, score in docs_with_scores:
            results.append({
                "content": doc.page_content,
                "score": float(score),
                "source": doc.metadata.get("source", "Unknown"),
                "section": doc.metadata.get("section", ""),
                "summary": doc.metadata.get("summary", ""),
            })

        return results

# Register your provider
my_provider = MyCustomRAGProvider()
RAGProviderRegistry.register(my_provider)
```

2. Initialize your RAG provider before using it:

```
POST /api/rag/provider/my_custom_provider/initialize
{
  "vector_store_path": "/path/to/your/vector_store",
  "openai_api_key": "your-openai-api-key"
}
```

3. Use your custom provider in the plugin:

```jsx
<AptosChatbotPlugin ragProvider="my_custom_provider" />
```

### Available RAG Providers

To get a list of available RAG providers:

```
GET /api/rag/providers
```

Response:

```json
{
  "providers": [
    {
      "name": "aptos",
      "description": "Default RAG provider using Aptos documentation"
    },
    {
      "name": "custom",
      "description": "Custom RAG provider using your own knowledge base"
    }
  ]
}
```

## Environment Variables (Optional)

You can override the default backend URL by setting an environment variable in your application:

```
VITE_API_URL=https://your-backend-api.com
```

This is useful for development or if you want to configure the URL at build time.

## Dependencies

This package has the following peer dependencies:

- React 18+
- React DOM 18+
- Tailwind CSS (for styling)

## License

MIT
