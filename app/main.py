import logging
import os
import time
import asyncio
import sys
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, Request, Response, Depends, HTTPException, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.openapi.utils import get_openapi
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.routes import chat
from app.models import initialize_models
from app.rag_providers import RAGProviderRegistry
from app.rag_providers.docs_provider import docs_provider
from app.config import DEFAULT_PROVIDER

# Configure logging for Docker (stdout only)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)
logger.info("Application starting up")

# Set default RAG provider environment variable
default_provider = os.environ.get("DEFAULT_RAG_PROVIDER", "docs")
os.environ["DEFAULT_RAG_PROVIDER"] = default_provider
logger.info(f"Default RAG provider set to: {default_provider}")

app = FastAPI(
    title="Aptos Dev Assistant API",
    description="""
    # Aptos Developer Assistant API
    
    This API provides access to the Aptos Developer Assistant chatbot, which uses Retrieval-Augmented Generation (RAG) 
    to provide accurate and contextual responses about Aptos development.
    
    ## Features
    
    - **RAG Implementation**: Uses vector search and topic-based enhancement for improved context retrieval
    - **Multiple RAG Providers**: Support for different knowledge sources (Aptos docs, GitHub repos, etc.)
    - **Streaming Responses**: Real-time streaming of AI-generated responses
    - **Chat History Management**: Create, retrieve, and manage chat conversations
    
    ## Authentication
    
    Currently, the API does not require authentication. This may change in future versions.
    
    ## Rate Limiting
    
    Please be respectful of API usage. Excessive requests may be rate-limited in the future.
    """,
    version="1.0.0",
    docs_url=None,  # Disable the default docs
    redoc_url=None,  # Disable the default redoc
)

# Configure CORS to allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,  # Must be True for wildcard origin
    allow_methods=["*"],
    allow_headers=["*"],
)


# Custom OpenAPI schema
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    # Add custom examples and descriptions
    for path in openapi_schema["paths"]:
        for method in openapi_schema["paths"][path]:
            if path == "/api/message/stream" and method == "post":
                openapi_schema["paths"][path][method][
                    "description"
                ] = """
                Unified endpoint for creating a new chat or adding a message to an existing chat.
                
                If chat_id is provided, the message will be added to that chat.
                If chat_id is not provided, a new chat will be created.
                """
                # Add example request body
                if "requestBody" in openapi_schema["paths"][path][method]:
                    openapi_schema["paths"][path][method]["requestBody"]["content"][
                        "application/json"
                    ]["example"] = {
                        "content": "What is Move language?",
                        "client_id": "user123",
                        "role": "user",
                        "id": "msg123",
                        "temperature": 0.7,
                        "rag_provider": "topic",
                    }

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


# Custom documentation endpoints
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} - Swagger UI",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css",
    )


@app.get("/redoc", include_in_schema=False)
async def redoc_html():
    return get_redoc_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} - ReDoc",
        redoc_js_url="https://cdn.jsdelivr.net/npm/redoc@next/bundles/redoc.standalone.js",
    )


# Create a router for system endpoints
system_router = APIRouter()


# Health check endpoint
@system_router.get(
    "/health",
    summary="Health Check",
    description="Check if the API is running properly",
    response_description="Returns status and timestamp",
    tags=["System"],
)
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "timestamp": time.time()}


# Create a router for RAG endpoints
rag_router = APIRouter()


# API endpoint for getting relevant context
@rag_router.post(
    "/rag/context",
    summary="Get Relevant Context",
    description="Retrieve relevant context for a given query using the specified RAG provider",
    response_description="Returns context information relevant to the query",
    tags=["RAG"],
)
async def get_context(request: Request):
    """Get relevant context for a query."""
    try:
        data = await request.json()
        query = data.get("query", "")
        provider_name = data.get("provider", "topic")  # Default to topic provider
        k = data.get("k", 5)

        if not query:
            raise HTTPException(status_code=400, detail="Query is required")

        # Get the provider from the registry
        try:
            provider = RAGProviderRegistry.get_provider(provider_name)
        except ValueError as e:
            # If provider not found, use the topic provider
            logger.warning(
                f"Provider '{provider_name}' not found, using topic provider instead"
            )
            provider = RAGProviderRegistry.get_provider("topic")

        # Get relevant context
        context = await provider.get_relevant_context(query, k=k)

        return {"context": context}
    except Exception as e:
        logger.error(f"Error getting context: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# API endpoint for listing available providers
@rag_router.get(
    "/rag/providers",
    summary="List RAG Providers",
    description="List all available RAG providers with their descriptions",
    response_description="Returns a list of available RAG providers",
    tags=["RAG"],
)
async def list_providers():
    """List all available RAG providers."""
    providers = RAGProviderRegistry.list_providers()
    # Mark the topic provider as the default
    for provider in providers:
        if provider["name"] == "topic":
            provider["is_default"] = True
        else:
            provider["is_default"] = False
    return {"providers": providers}


# Include routers
app.include_router(system_router, tags=["System"])
app.include_router(rag_router, prefix="/api", tags=["RAG"])
app.include_router(chat.router, prefix="/api", tags=["Chat"])


# Async initialization of models and providers
async def async_init_models():
    """Initialize models and providers on startup."""
    try:
        # Initialize models
        await initialize_models()
        logger.info("Models initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing models: {e}")

    try:
        # Initialize docs provider with default path
        logger.info(f"Initializing docs provider with default path: {DEFAULT_PROVIDER}")
        await docs_provider.initialize({"docs_path": DEFAULT_PROVIDER})
        logger.info("Successfully initialized docs provider")

        # Initialize aptos-learn provider
        logger.info("Initializing aptos-learn provider")
        await docs_provider.initialize({"docs_path": "aptos-learn"})
        logger.info("Successfully initialized aptos-learn provider")
    except Exception as e:
        logger.error(f"Failed to initialize providers: {e}")


@app.on_event("startup")
async def startup_event():
    # Start the initialization in a background task
    # This allows the server to start up quickly while initialization happens in the background
    asyncio.create_task(async_init_models())
    logger.info("Started model initialization in background task")


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8080))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)
