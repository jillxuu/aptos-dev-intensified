import logging
import os
import time
import asyncio
import sys
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, Request, Response, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.routes import chat
from app.models import initialize_models
from app.rag_providers import RAGProviderRegistry
from app.rag_providers import aptos_provider, github_provider, custom_provider

# Import topic_provider to ensure it's registered as the default
from app.rag_providers.topic_provider import topic_provider

# Configure logging for Docker (stdout only)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)
logger.info("Application starting up")

# Set default RAG provider environment variable
default_provider = os.environ.get("DEFAULT_RAG_PROVIDER", "topic")
os.environ["DEFAULT_RAG_PROVIDER"] = default_provider
logger.info(f"Default RAG provider set to: {default_provider}")

app = FastAPI(
    title="Aptos Dev Assistant API",
    description="API for the Aptos Developer Assistant chatbot",
    version="1.0.0",
)

# Configure CORS to allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,  # Must be True for wildcard origin
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "timestamp": time.time()}


# API endpoint for getting relevant context
@app.post("/get_context")
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
@app.get("/providers")
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
app.include_router(chat.router, prefix="/api")


# Async initialization of models and providers
async def async_init_models():
    """Initialize models and providers on startup."""
    try:
        # Initialize models
        initialize_models()
        logger.info("Models initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing models: {e}")

    # Initialize topic provider first to ensure it's ready
    try:
        # Initialize topic provider
        await topic_provider.initialize({})
        logger.info("Initialized topic provider")
    except Exception as e:
        logger.error(f"Error initializing topic provider: {e}")

    try:
        # Initialize Aptos provider
        await aptos_provider.initialize({})
        logger.info("Initialized Aptos provider")
    except Exception as e:
        logger.error(f"Error initializing Aptos provider: {e}")

    try:
        # Initialize GitHub provider
        await github_provider.initialize({})
        logger.info("Initialized GitHub provider")
    except Exception as e:
        logger.error(f"Error initializing GitHub provider: {e}")

    try:
        # Initialize custom provider
        await custom_provider.initialize({})
        logger.info("Initialized custom provider")
    except Exception as e:
        logger.error(f"Error initializing custom provider: {e}")


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
