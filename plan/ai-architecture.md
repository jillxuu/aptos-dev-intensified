# AI Architecture Plan for Extensible RAG-Based AI System

## Executive Summary

This document outlines the architecture for a unified, extensible RAG-based AI system that serves multiple downstream applications (Aptos dev docs, Aptos Learn, GitHub discussions, Telegram bot, MCP server, etc.) through a single deployable service with application-specific, pre-configured pipelines.

## High-Level System Design

### Core Architecture Principles

1. **Single Deployment, Multiple Applications**: One RAG service serves all downstream applications
2. **Application-Level Configuration**: Each app has a fixed, optimized pipeline defined at deployment time
3. **Simple Runtime API**: Applications only pass app name and query parameters at runtime
4. **Reusable Components**: Modular components that can be composed into different application pipelines
5. **No Runtime Complexity**: No dynamic workflow orchestration - just well-designed, tested pipelines per app

### System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                    Downstream Applications                       │
├─────────────┬─────────────┬─────────────┬─────────────┬─────────┤
│ Aptos Docs  │ Aptos Learn │ GitHub Bot  │ Telegram    │ MCP     │
│ Website     │ Website     │             │ Bot         │ Server  │
└─────────────┴─────────────┴─────────────┴─────────────┴─────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      API Gateway Layer                          │
├─────────────────────────────────────────────────────────────────┤
│ • Authentication & Authorization                                │
│ • Rate Limiting (per user/app)                                 │
│ • Request Routing by App Name                                  │
│ • Simple Parameter Validation                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Application Pipeline Router                  │
├─────────────────────────────────────────────────────────────────┤
│ • Route to Specific App Pipeline                               │
│ • Load App Configurations at Startup                          │
│ • No Runtime Decision Making                                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Application Pipelines                        │
├─────────────┬─────────────┬─────────────┬─────────────┬─────────┤
│ Aptos Docs  │ GitHub Bot  │ Telegram    │ MCP Server  │ Custom  │
│ Pipeline    │ Pipeline    │ Pipeline    │ Pipeline    │ App     │
├─────────────┼─────────────┼─────────────┼─────────────┼─────────┤
│ Fixed:      │ Fixed:      │ Fixed:      │ Fixed:      │ Fixed:  │
│ • Data Src  │ • Data Src  │ • Data Src  │ • Data Src  │ • ...   │
│ • Retrieval │ • Retrieval │ • Retrieval │ • Retrieval │         │
│ • LLM       │ • LLM       │ • LLM       │ • LLM       │         │
│ • Reasoning │ • Reasoning │ • Reasoning │ • Reasoning │         │
└─────────────┴─────────────┴─────────────┴─────────────┴─────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Reusable Components                          │
├─────────────┬─────────────┬─────────────┬─────────────┬─────────┤
│ Data        │ Retrieval   │ LLM         │ Reasoning   │ Output  │
│ Sources     │ Strategies  │ Providers   │ Modes       │ Formats │
├─────────────┼─────────────┼─────────────┼─────────────┼─────────┤
│ • Docs      │ • Vector    │ • OpenAI    │ • Simple    │ • Text  │
│ • GitHub    │ • BM25      │ • Anthropic │ • Code Mode │ • JSON  │
│ • Forums    │ • Hybrid    │ • Local     │ • Fast Mode │ • Stream│
│ • Custom    │ • Rerank    │ • Custom    │ • Citations │ • URLs  │
└─────────────┴─────────────┴─────────────┴─────────────┴─────────┘
```

### Application Configuration Schema

Each application has a JSON configuration file that defines its complete pipeline:

**aptos-docs-chatbot.json**
```json
{
  "app_name": "aptos-docs-chatbot",
  "version": "1.0",
  "data_sources": [
    {
      "name": "developer-docs",
      "type": "markdown_docs",
      "path": "data/developer-docs"
    },
    {
      "name": "code-examples", 
      "type": "code_repository",
      "path": "data/code-examples"
    }
  ],
  "retrieval": {
    "strategy": "multi_step",
    "embedding_model": "openai/text-embedding-3-small",
    "query_expansion_llm": {
      "provider": "openai",
      "model": "gpt-4o-mini",
      "temperature": 0.3
    }
  },
  "llm": {
    "provider": "openai",
    "model": "gpt-4o-mini", 
    "temperature": 0.1,
    "max_tokens": 2000,
    "system_prompt_template": "aptos_docs_assistant",
    "include_sources": true,
    "citation_format": "url",
    "tools": ["context_search"]
  }
}
```

**github-discussion-bot.json**
```json
{
  "app_name": "github-discussion-bot",
  "version": "1.0",
  "data_sources": [
    {
      "name": "github-discussions",
      "type": "github_discussions",
      "repo": "aptos-labs/aptos-core",
      "include_issues": true
    }
  ],
  "retrieval": {
    "strategy": "adaptive",
    "embedding_model": "openai/text-embedding-3-small",
    "analyzer_llm": {
      "provider": "openai",
      "model": "gpt-4o-mini",
      "temperature": 0.1
    }
  },
  "llm": {
    "provider": "openai",
    "model": "gpt-4o",
    "temperature": 0.2,
    "max_tokens": 1500,
    "system_prompt_template": "github_helper",
    "include_sources": true,
    "citation_format": "github_link",
    "tools": []
  }
}
```

## Object-Oriented Programming Design

### Core Architecture Classes

#### 1. Application Pipeline

```python
class ApplicationPipeline:
    """Fixed pipeline for a specific application."""
    
    def __init__(self, app_config: AppConfig):
        self.app_name = app_config.name
        # Combine all data sources into one vector DB per app
        self.vector_db = self._build_combined_vector_db(app_config.data_sources)
        self.retrieval_strategy = self._create_retrieval_strategy(app_config.retrieval)
        self.main_llm = self._create_llm(app_config.llm)
        
    async def process_query(self, query: str, params: Dict[str, Any]) -> Response:
        """Fixed pipeline execution - no runtime decisions."""
        
        if self.main_llm.has_tools:
            # LLM can call context_search when it wants
            return await self.main_llm.generate_with_tools(
                query, 
                tools={"context_search": self.context_search}
            )
        else:
            # Traditional RAG - retrieve context first, then generate
            context = await self.context_search(query, params.get('k', 5))
            prompt = self._build_prompt(query, context)
            return await self.main_llm.generate(prompt, params)
    
    async def context_search(self, query: str, k: int = 5) -> List[Chunk]:
        """Tool function for retrieving relevant context."""
        return await self.retrieval_strategy.retrieve_context(query, k, self.vector_db)
```

#### 2. Main AI Service

```python
class AIService:
    """Main service that routes to application pipelines."""
    
    def __init__(self):
        self.pipelines: Dict[str, ApplicationPipeline] = {}
        self._load_applications()
    
    def _load_applications(self):
        """Load all application configurations at startup."""
        for config_file in glob.glob("configs/*.json"):
            app_config = AppConfig.from_file(config_file)
            self.pipelines[app_config.name] = ApplicationPipeline(app_config)
    
    async def chat(self, app_name: str, query: str, **params) -> Response:
        """Simple API - just app name and query."""
        if app_name not in self.pipelines:
            raise ValueError(f"Unknown application: {app_name}")
        
        pipeline = self.pipelines[app_name]
        return await pipeline.process_query(query, params)
```

#### 3. Reusable Component Interfaces

```python
class DataSource(ABC):
    """Abstract base class for all data sources."""
    
    @abstractmethod
    async def load_documents(self) -> List[Document]
    
    @abstractmethod
    async def preprocess_documents(self, docs: List[Document]) -> List[Chunk]

class RetrievalStrategy(ABC):
    """Handles the complete retrieval process - can be simple or multi-step."""
    
    @abstractmethod
    async def retrieve_context(self, query: str, k: int, vector_db: VectorDB) -> List[Chunk]:
        """Main method that handles the entire retrieval process."""
        pass

class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    async def generate(self, prompt: str, config: LLMConfig) -> LLMResponse
    
    @abstractmethod
    async def generate_with_tools(self, query: str, tools: Dict[str, Callable]) -> LLMResponse
    
    @abstractmethod
    async def stream_generate(self, prompt: str, config: LLMConfig) -> AsyncIterator[str]
```

#### 4. Retrieval Strategy Implementations

```python
class SimpleRetrievalStrategy(RetrievalStrategy):
    """Basic similarity search - just search and return."""
    
    async def retrieve_context(self, query: str, k: int, vector_db: VectorDB) -> List[Chunk]:
        return await vector_db.similarity_search(query, k)

class MultiStepRetrievalStrategy(RetrievalStrategy):
    """Multi-step retrieval with query expansion."""
    
    def __init__(self, query_expansion_llm: LLMProvider):
        self.query_expansion_llm = query_expansion_llm
    
    async def retrieve_context(self, query: str, k: int, vector_db: VectorDB) -> List[Chunk]:
        # Step 1: Generate additional relevant questions
        additional_queries = await self._generate_related_queries(query)
        
        # Step 2: Search with all queries
        all_results = []
        for q in [query] + additional_queries:
            results = await vector_db.similarity_search(q, k)
            all_results.extend(results)
        
        # Step 3: Deduplicate and rerank combined results
        return self._deduplicate_and_rerank(all_results, query, k)
    
    async def _generate_related_queries(self, original_query: str) -> List[str]:
        """Use LLM to generate related questions."""
        prompt = f"""
        Generate 2-3 related questions that would help find comprehensive information about: {original_query}
        
        Return only the questions, one per line.
        """
        response = await self.query_expansion_llm.generate(prompt)
        return response.content.strip().split('\n')
    
    def _deduplicate_and_rerank(self, chunks: List[Chunk], original_query: str, k: int) -> List[Chunk]:
        """Remove duplicates and rerank based on relevance to original query."""
        # Deduplicate by content hash
        unique_chunks = self._remove_duplicates(chunks)
        # Rerank based on original query
        return self._rerank_by_relevance(unique_chunks, original_query, k)

class AdaptiveRetrievalStrategy(RetrievalStrategy):
    """LLM decides if it needs more context iteratively."""
    
    def __init__(self, analyzer_llm: LLMProvider):
        self.analyzer_llm = analyzer_llm
    
    async def retrieve_context(self, query: str, k: int, vector_db: VectorDB) -> List[Chunk]:
        # Step 1: Initial search
        initial_results = await vector_db.similarity_search(query, k)
        
        # Step 2: LLM analyzes if more context needed
        needs_more = await self._analyze_context_sufficiency(query, initial_results)
        
        if not needs_more:
            return initial_results
        
        # Step 3: Generate refined search queries
        refined_queries = await self._generate_refined_queries(query, initial_results)
        
        # Step 4: Additional search
        additional_results = []
        for refined_query in refined_queries:
            results = await vector_db.similarity_search(refined_query, k)
            additional_results.extend(results)
        
        # Step 5: Combine and rerank
        all_results = initial_results + additional_results
        return self._deduplicate_and_rerank(all_results, query, k)
    
    async def _analyze_context_sufficiency(self, query: str, chunks: List[Chunk]) -> bool:
        """LLM determines if current context is sufficient."""
        context_text = "\n".join([chunk.content[:200] + "..." for chunk in chunks])
        
        prompt = f"""
        Query: {query}
        Retrieved Context: {context_text}
        
        Is this context sufficient to answer the query comprehensively? 
        Answer only: YES or NO
        """
        
        response = await self.analyzer_llm.generate(prompt, max_tokens=10)
        return "NO" in response.content.upper()

class AppSpecificRetrievalStrategy(RetrievalStrategy):
    """Custom retrieval logic for specific applications."""
    
    async def retrieve_context(self, query: str, k: int, vector_db: VectorDB) -> List[Chunk]:
        # App-specific retrieval logic
        results = await vector_db.similarity_search(query, k=k*2)
        # Custom reranking based on app needs
        return self._app_specific_rerank(results, query, k)
```

#### 5. Component Implementations

```python
# Data Sources
class MarkdownDocsSource(DataSource):
    """Implementation for markdown documentation sources."""

class GitHubDiscussionsSource(DataSource):
    """Implementation for GitHub discussions and issues."""

class CodeRepositorySource(DataSource):
    """Implementation for code repository sources."""

# LLM Providers
class OpenAIProvider(LLMProvider):
    """OpenAI API implementation."""

class AnthropicProvider(LLMProvider):
    """Anthropic API implementation."""
```

### API Design

#### Simple Chat Endpoint

```python
@app.post("/api/v1/chat")
async def chat_endpoint(request: ChatRequest):
    """Simple chat endpoint."""
    return await ai_service.chat(
        app_name=request.app_name,
        query=request.query,
        user_id=request.user_id,
        **request.params
    )

# Request format
{
  "app_name": "aptos-docs-chatbot",
  "query": "How do I deploy a Move contract?",
  "user_id": "user123",
  "params": {
    "k": 7,
    "mode": "detailed"
  }
}
```

#### Tool Definition for LLM

```json
{
  "name": "context_search",
  "description": "Retrieve relevant context from knowledge base. This may involve multi-step search if configured.",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "The search query"
      },
      "max_results": {
        "type": "integer",
        "description": "Maximum number of results to return (default: 5)"
      }
    },
    "required": ["query"]
  }
}
```

#### Management Endpoints

```python
@app.get("/api/v1/applications")
async def list_applications():
    """List all available applications."""
    return {"applications": list(ai_service.pipelines.keys())}

@app.get("/api/v1/applications/{app_name}/info")
async def get_application_info(app_name: str):
    """Get information about a specific application."""
    return ai_service.get_application_info(app_name)
```

## Code Organization

```
app/
├── pipelines/
│   ├── __init__.py
│   ├── base.py              # ApplicationPipeline base class
│   └── app_specific.py      # App-specific pipeline logic (if needed)
├── components/
│   ├── data_sources/        # Reusable data source implementations
│   │   ├── __init__.py
│   │   ├── markdown_docs.py
│   │   ├── github_discussions.py
│   │   └── code_repository.py
│   ├── retrieval/           # Reusable retrieval strategies
│   │   ├── __init__.py
│   │   ├── simple_retrieval.py
│   │   ├── multi_step_retrieval.py
│   │   ├── adaptive_retrieval.py
│   │   └── app_specific_retrieval.py
│   └── llm_providers/       # Reusable LLM providers
│       ├── __init__.py
│       ├── openai_provider.py
│       └── anthropic_provider.py
├── configs/
│   ├── aptos-docs-chatbot.json
│   ├── github-discussion-bot.json
│   ├── telegram-bot.json
│   └── mcp-server.json
├── models.py                # Pydantic models
├── config.py               # Configuration loading
└── main.py                 # AIService and API endpoints
```

## Configuration Examples

**Simple Retrieval:**
```json
{
  "app_name": "simple-bot",
  "retrieval": {
    "strategy": "simple",
    "embedding_model": "openai/text-embedding-3-small"
  },
  "llm": {
    "tools": []
  }
}
```

**Multi-Step Retrieval:**
```json
{
  "app_name": "advanced-docs-bot",
  "retrieval": {
    "strategy": "multi_step",
    "embedding_model": "openai/text-embedding-3-small",
    "query_expansion_llm": {
      "provider": "openai",
      "model": "gpt-4o-mini",
      "temperature": 0.3
    }
  },
  "llm": {
    "tools": ["context_search"]
  }
}
```

**Adaptive Retrieval:**
```json
{
  "app_name": "smart-assistant",
  "retrieval": {
    "strategy": "adaptive",
    "embedding_model": "openai/text-embedding-3-small",
    "analyzer_llm": {
      "provider": "openai",
      "model": "gpt-4o-mini",
      "temperature": 0.1
    }
  },
  "llm": {
    "tools": []
  }
}
```

## Benefits of This Architecture

1. **Simplicity**: No complex workflow orchestration or runtime decisions
2. **Flexibility**: Different retrieval strategies from simple to multi-step
3. **Performance**: Fixed pipelines are optimized and fast
4. **Reliability**: Each pipeline is thoroughly tested and battle-proven
5. **Maintainability**: Clear separation between applications and reusable components
6. **Scalability**: Easy to add new applications by creating new config files
7. **Debugging**: Easy to trace issues within specific, well-defined pipelines
8. **Extensibility**: New components and retrieval strategies can be added and used across applications


## Adding New Applications

1. **Create Configuration**: Add new JSON config file in `configs/` directory
2. **Restart Service**: Configurations are loaded at startup
3. **Start Using**: Make API calls with the new `app_name`

Example for adding a new Telegram bot:

```json
// configs/telegram-bot.json
{
  "app_name": "telegram-bot",
  "data_sources": [
    {
      "name": "telegram-docs",
      "type": "markdown_docs",
      "path": "data/telegram-docs"
    }
  ],
  "retrieval": {
    "strategy": "simple",
    "embedding_model": "openai/text-embedding-3-small"
  },
  "llm": {
    "provider": "openai",
    "model": "gpt-4o-mini",
    "temperature": 0.3,
    "system_prompt_template": "telegram_assistant",
    "tools": ["context_search"]
  }
}
```

## Migration Path

1. **Preserve Current Functionality**: Extract existing RAG logic into the new ApplicationPipeline structure
2. **Create Aptos Docs Pipeline**: Convert current implementation to the new architecture
3. **Add Component Abstractions**: Gradually extract reusable components
4. **Add New Applications**: Create additional pipelines using the established patterns
5. **Optimize and Test**: Fine-tune each pipeline for optimal performance

This architecture provides the extensibility needed to serve multiple applications while maintaining simplicity and reliability through fixed, well-tested pipelines per application. The retrieval strategies can handle everything from simple similarity search to complex multi-step retrieval patterns.

## Aptos Build Integration Opportunities (To Be Discussed)

### Overview of Aptos Build

Aptos Build is Aptos Labs' comprehensive developer platform that provides API access, no-code indexing, gas station services, NFT studio, and identity management. It uses a sophisticated usage-based billing system with Compute Units (CUs) for fair and transparent pricing.

### Key Features of Aptos Build:

1. **Compute Units (CUs) Billing Model**:
   - Different API calls consume different CUs based on complexity
   - Resource-intensive operations consume more CUs than simple calls
   - Real-time usage tracking with Stripe-powered billing
   - Pay-as-you-go with no bundles or hidden multipliers

2. **Developer-Friendly Infrastructure**:
   - API key management and authentication
   - Rate limiting and usage controls
   - Organizational workspaces and budgets
   - Live billing dashboard with usage analytics

### Integration Opportunities:

#### 1. Direct Integration with Aptos Build
Our AI chatbot could be offered as a premium service within Aptos Build:

```json
{
  "ai_services": {
    "aptos_ai_assistant": {
      "compute_units_per_query": 50,
      "pricing_tier": "premium",
      "features": ["code_mode", "fast_mode", "doc_retrieval"]
    }
  }
}
```

#### 2. Leverage Existing Infrastructure
- **Authentication**: Use Aptos Build API keys for user authentication
- **Rate Limiting**: Integrate with their existing rate limiting system
- **Billing**: Piggyback on their Stripe-powered billing system
- **Usage Analytics**: Leverage their real-time usage tracking

#### 3. AI-Specific Compute Units
Define compute units for different AI operations:

```json
{
  "ai_compute_units": {
    "simple_query": 10,
    "code_generation": 25,
    "complex_reasoning": 50,
    "document_search": 15,
    "multi_step_agent": 100
  }
}
```

#### 4. Configuration Integration
Our JSON-based pipeline configuration could integrate with Aptos Build's project system:

```json
{
  "aptos_build_integration": {
    "project_id": "aptos-docs-ai",
    "api_key": "ab_...",
    "billing_tier": "premium",
    "usage_limits": {
      "monthly_cu_limit": 100000,
      "rate_limit_per_minute": 60
    }
  }
}
```

### Strategic Benefits:

1. **Leverage Existing Infrastructure**: Use proven billing, authentication, and rate limiting systems
2. **Developer Ecosystem Integration**: AI becomes part of the official Aptos developer toolkit
3. **Unified Billing**: Developers get one bill for all Aptos services including AI assistance
4. **Enterprise Features**: Access to organizational workspaces and enterprise features
5. **Market Validation**: Being part of Aptos Build validates the AI service as an official Aptos tool

### Recommendation:

Consider **partnering with Aptos Build** rather than building custom billing infrastructure. This would:

1. **Reduce Development Overhead**: Focus on AI capabilities rather than billing infrastructure
2. **Faster Time to Market**: Leverage existing, proven systems
3. **Better Developer Experience**: Developers familiar with Aptos Build can easily adopt the AI
4. **Official Endorsement**: Being part of Aptos Build gives the AI service official status

The configuration-driven architecture already supports this through an "Aptos Build integration mode" that handles authentication, billing, and rate limiting through their existing APIs.

## Class Design Structure & Method Signatures

### **Core Class Hierarchy**

```
┌─────────────────────────────────────────────────────────────────┐
│                        AIService                                │
├─────────────────────────────────────────────────────────────────┤
│ - pipelines: Dict[str, ApplicationPipeline]                    │
│ - config_loader: ConfigLoader                                  │
├─────────────────────────────────────────────────────────────────┤
│ + __init__()                                                    │
│ + _load_applications()                                          │
│ + chat(app_name: str, query: str, **params) -> Response        │
│ + get_application_info(app_name: str) -> AppInfo               │
│ + list_applications() -> List[str]                             │
│ + reload_application(app_name: str) -> bool                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ contains
┌─────────────────────────────────────────────────────────────────┐
│                   ApplicationPipeline                           │
├─────────────────────────────────────────────────────────────────┤
│ - app_name: str                                                 │
│ - vector_db: VectorDB                                          │
│ - retrieval_strategy: RetrievalStrategy                        │
│ - main_llm: LLMProvider                                        │
│ - config: AppConfig                                            │
├─────────────────────────────────────────────────────────────────┤
│ + __init__(app_config: AppConfig)                              │
│ + process_query(query: str, params: Dict) -> Response          │
│ + context_search(query: str, k: int) -> List[Chunk]            │
│ + _build_combined_vector_db(sources: List[DataSource]) -> VectorDB │
│ + _create_retrieval_strategy(config: RetrievalConfig) -> RetrievalStrategy │
│ + _create_llm(config: LLMConfig) -> LLMProvider                │
│ + _build_prompt(query: str, context: List[Chunk]) -> str       │
└─────────────────────────────────────────────────────────────────┘
```

### **Data Source Hierarchy**

```
┌─────────────────────────────────────────────────────────────────┐
│                     DataSource (ABC)                           │
├─────────────────────────────────────────────────────────────────┤
│ + load_documents() -> List[Document]                           │
│ + preprocess_documents(docs: List[Document]) -> List[Chunk]     │
│ + get_metadata() -> Dict[str, Any]                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────┼─────────┐
                    ▼         ▼         ▼
┌─────────────────────┐ ┌─────────────────┐ ┌─────────────────────┐
│ MarkdownDocsSource  │ │GitHubDiscussions│ │ CodeRepositorySource│
├─────────────────────┤ │     Source      │ ├─────────────────────┤
│ - docs_path: str    │ ├─────────────────┤ │ - repo_path: str    │
│ - file_pattern: str │ │ - repo: str     │ │ - include_patterns  │
├─────────────────────┤ │ - github_token  │ ├─────────────────────┤
│ + load_documents()  │ │ - include_issues│ │ + load_documents()  │
│ + _parse_markdown() │ ├─────────────────┤ │ + _parse_code_files │
│ + _extract_headers()│ │ + load_documents│ │ + _extract_functions│
└─────────────────────┘ │ + _fetch_issues │ │ + _get_dependencies │
                        │ + _parse_thread │ └─────────────────────┘
                        └─────────────────┘
```

### **Retrieval Strategy Hierarchy**

```
┌─────────────────────────────────────────────────────────────────┐
│                  RetrievalStrategy (ABC)                       │
├─────────────────────────────────────────────────────────────────┤
│ + retrieve_context(query: str, k: int, vector_db: VectorDB) -> List[Chunk] │
│ + _rerank_results(chunks: List[Chunk], query: str) -> List[Chunk] │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────────┐
│SimpleRetrieval  │ │MultiStepRetrieval│ │AdaptiveRetrieval    │
│   Strategy      │ │    Strategy      │ │    Strategy         │
├─────────────────┤ ├─────────────────┤ ├─────────────────────┤
│ + retrieve_     │ │ - query_expansion│ │ - analyzer_llm      │
│   context()     │ │   _llm: LLMProvider │ │ - max_iterations │
└─────────────────┘ ├─────────────────┤ ├─────────────────────┤
                    │ + retrieve_     │ │ + retrieve_context()│
                    │   context()     │ │ + _analyze_sufficiency│
                    │ + _generate_    │ │ + _generate_refined_│
                    │   related_queries│ │   queries()         │
                    │ + _deduplicate_ │ │ + _iterative_search │
                    │   and_rerank()  │ └─────────────────────┘
                    └─────────────────┘
```

### **LLM Provider Hierarchy**

```
┌─────────────────────────────────────────────────────────────────┐
│                    LLMProvider (ABC)                           │
├─────────────────────────────────────────────────────────────────┤
│ - config: LLMConfig                                            │
│ - has_tools: bool                                              │
├─────────────────────────────────────────────────────────────────┤
│ + generate(prompt: str, config: LLMConfig) -> LLMResponse      │
│ + generate_with_tools(query: str, tools: Dict[str, Callable]) -> LLMResponse │
│ + stream_generate(prompt: str, config: LLMConfig) -> AsyncIterator[str] │
│ + _format_messages(prompt: str) -> List[Message]               │
│ + _handle_tool_calls(response: Any) -> LLMResponse             │
└─────────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────┼─────────┐
                    ▼         ▼         ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│  OpenAIProvider │ │AnthropicProvider│ │  LocalLLMProvider│
├─────────────────┤ ├─────────────────┤ ├─────────────────┤
│ - client: OpenAI│ │ - client: Client│ │ - model_path    │
│ - api_key: str  │ │ - api_key: str  │ │ - device: str   │
├─────────────────┤ ├─────────────────┤ ├─────────────────┤
│ + generate()    │ │ + generate()    │ │ + generate()    │
│ + _format_tools │ │ + _format_tools │ │ + _load_model() │
│ + _handle_stream│ │ + _handle_stream│ │ + _tokenize()   │
└─────────────────┘ └─────────────────┘ └─────────────────┘
```

### **Configuration and Models**

```
┌─────────────────────────────────────────────────────────────────┐
│                       ConfigLoader                             │
├─────────────────────────────────────────────────────────────────┤
│ - config_dir: str                                              │
├─────────────────────────────────────────────────────────────────┤
│ + load_all_configs() -> Dict[str, AppConfig]                   │
│ + load_config(filename: str) -> AppConfig                      │
│ + validate_config(config: AppConfig) -> bool                   │
│ + _resolve_data_source_paths(config: AppConfig) -> AppConfig   │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                        AppConfig                               │
├─────────────────────────────────────────────────────────────────┤
│ + app_name: str                                                │
│ + version: str                                                 │
│ + data_sources: List[DataSourceConfig]                        │
│ + retrieval: RetrievalConfig                                   │
│ + llm: LLMConfig                                               │
├─────────────────────────────────────────────────────────────────┤
│ + from_file(filepath: str) -> AppConfig                        │
│ + to_dict() -> Dict[str, Any]                                  │
│ + validate() -> bool                                           │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    DataSourceConfig                            │
├─────────────────────────────────────────────────────────────────┤
│ + name: str                                                    │
│ + type: str                                                    │
│ + path: str                                                    │
│ + options: Dict[str, Any]                                      │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    RetrievalConfig                             │
├─────────────────────────────────────────────────────────────────┤
│ + strategy: str                                                │
│ + embedding_model: str                                         │
│ + k: int                                                       │
│ + query_expansion_llm: Optional[LLMConfig]                     │
│ + analyzer_llm: Optional[LLMConfig]                            │
│ + rerank_model: Optional[str]                                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                       LLMConfig                               │
├─────────────────────────────────────────────────────────────────┤
│ + provider: str                                                │
│ + model: str                                                   │
│ + temperature: float                                           │
│ + max_tokens: int                                              │
│ + system_prompt_template: str                                  │
│ + tools: List[str]                                             │
│ + include_sources: bool                                        │
│ + citation_format: str                                         │
└─────────────────────────────────────────────────────────────────┘
```

### **Vector Database and Document Models**

```
┌─────────────────────────────────────────────────────────────────┐
│                       VectorDB                                │
├─────────────────────────────────────────────────────────────────┤
│ - collection_name: str                                         │
│ - embedding_model: str                                         │
│ - chunks: List[Chunk]                                          │
├─────────────────────────────────────────────────────────────────┤
│ + add_documents(docs: List[Document])                          │
│ + similarity_search(query: str, k: int) -> List[Chunk]         │
│ + hybrid_search(query: str, k: int) -> List[Chunk]             │
│ + delete_collection()                                          │
│ + get_stats() -> DBStats                                       │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                       Document                                │
├─────────────────────────────────────────────────────────────────┤
│ + content: str                                                 │
│ + metadata: Dict[str, Any]                                     │
│ + source: str                                                  │
│ + doc_id: str                                                  │
├─────────────────────────────────────────────────────────────────┤
│ + chunk(chunk_size: int, overlap: int) -> List[Chunk]          │
│ + get_embedding() -> List[float]                               │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                        Chunk                                  │
├─────────────────────────────────────────────────────────────────┤
│ + content: str                                                 │
│ + metadata: Dict[str, Any]                                     │
│ + doc_id: str                                                  │
│ + chunk_id: str                                                │
│ + embedding: Optional[List[float]]                             │
│ + score: Optional[float]                                       │
├─────────────────────────────────────────────────────────────────┤
│ + get_citation() -> str                                        │
│ + to_context_string() -> str                                   │
└─────────────────────────────────────────────────────────────────┘
```

### **Response Models**

```
┌─────────────────────────────────────────────────────────────────┐
│                       Response                                │
├─────────────────────────────────────────────────────────────────┤
│ + content: str                                                 │
│ + sources: List[Source]                                        │
│ + metadata: Dict[str, Any]                                     │
│ + processing_time: float                                       │
│ + tokens_used: int                                             │
├─────────────────────────────────────────────────────────────────┤
│ + add_source(chunk: Chunk)                                     │
│ + to_dict() -> Dict[str, Any]                                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                        Source                                 │
├─────────────────────────────────────────────────────────────────┤
│ + title: str                                                   │
│ + url: Optional[str]                                           │
│ + content_preview: str                                         │
│ + relevance_score: float                                       │
│ + metadata: Dict[str, Any]                                     │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                     LLMResponse                               │
├─────────────────────────────────────────────────────────────────┤
│ + content: str                                                 │
│ + tool_calls: Optional[List[ToolCall]]                         │
│ + usage: TokenUsage                                            │
│ + model: str                                                   │
│ + finish_reason: str                                           │
└─────────────────────────────────────────────────────────────────┘
```

### **Key Method Implementations**

#### **AIService.chat() - Main Entry Point**
```python
async def chat(self, app_name: str, query: str, **params) -> Response:
    """Main API endpoint for all applications."""
    start_time = time.time()
    
    # Validate application exists
    if app_name not in self.pipelines:
        raise ValueError(f"Unknown application: {app_name}")
    
    # Route to specific pipeline
    pipeline = self.pipelines[app_name]
    response = await pipeline.process_query(query, params)
    
    # Add metadata
    response.metadata['app_name'] = app_name
    response.processing_time = time.time() - start_time
    
    return response
```

#### **ApplicationPipeline.process_query() - Pipeline Execution**
```python
async def process_query(self, query: str, params: Dict[str, Any]) -> Response:
    """Execute the fixed pipeline for this application."""
    
    # Check if LLM has tool access
    if self.main_llm.has_tools:
        # Tool-enabled LLM can call context_search when needed
        return await self.main_llm.generate_with_tools(
            query=query,
            tools={"context_search": self.context_search},
            config=self.config.llm
        )
    else:
        # Traditional RAG: retrieve first, then generate
        context_chunks = await self.context_search(query, params.get('k', 5))
        prompt = self._build_prompt(query, context_chunks)
        
        llm_response = await self.main_llm.generate(prompt, self.config.llm)
        
        # Build final response with sources
        response = Response(
            content=llm_response.content,
            sources=[Source.from_chunk(chunk) for chunk in context_chunks],
            tokens_used=llm_response.usage.total_tokens
        )
        
        return response
```

#### **RetrievalStrategy.retrieve_context() - Smart Retrieval**
```python
# MultiStepRetrievalStrategy implementation
async def retrieve_context(self, query: str, k: int, vector_db: VectorDB) -> List[Chunk]:
    """Multi-step retrieval with query expansion."""
    
    # Step 1: Generate related queries using LLM
    additional_queries = await self._generate_related_queries(query)
    
    # Step 2: Search with all queries
    all_chunks = []
    for search_query in [query] + additional_queries:
        chunks = await vector_db.similarity_search(search_query, k)
        all_chunks.extend(chunks)
    
    # Step 3: Deduplicate and rerank
    unique_chunks = self._remove_duplicates(all_chunks)
    final_chunks = self._rerank_by_relevance(unique_chunks, query, k)
    
    return final_chunks

async def _generate_related_queries(self, original_query: str) -> List[str]:
    """Use query expansion LLM to generate related questions."""
    prompt = f"""
    Generate 2-3 related questions that would help find comprehensive 
    information about: {original_query}
    
    Return only the questions, one per line.
    """
    
    response = await self.query_expansion_llm.generate(prompt)
    return [q.strip() for q in response.content.split('\n') if q.strip()]
```

This structure provides a clear, extensible foundation where new applications can be added by simply creating new configuration files, and new components can be developed by implementing the abstract base classes.
