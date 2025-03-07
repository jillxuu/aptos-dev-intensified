from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import chat
from app.models import initialize_models

# Import RAG providers to ensure they're registered
from app.rag_providers import aptos_provider, github_provider, custom_provider
import os
import asyncio
import logging

logger = logging.getLogger(__name__)

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
    return {
        "status": "healthy",
        "service": "Aptos Dev Assistant API",
        "version": "1.0.0",
    }


# Include routers
app.include_router(chat.router, prefix="/api")


async def async_init_models():
    """Asynchronous initialization of models and RAG providers"""
    try:
        # Initialize models - this will also load documents if needed
        initialize_models()
        logger.info("Models initialized successfully")

        # Initialize the default RAG provider (Aptos)
        from app.rag_providers import RAGProviderRegistry

        default_provider = RAGProviderRegistry.get_provider("aptos")
        await default_provider.initialize({})
        logger.info(
            f"Default RAG provider '{default_provider.name}' initialized successfully"
        )
    except Exception as e:
        logger.error(f"Error initializing models or RAG providers: {e}")


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
