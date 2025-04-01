from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.responses import StreamingResponse
from app.models import (
    Feedback,
    ChatHistory,
    ChatMessage,
    ChatMessageRequest,
    ChatHistoriesResponse,
    StatusResponse,
)
from app.rag_providers import RAGProviderRegistry
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import os
import logging
import numpy as np
from typing import List, AsyncGenerator, Dict, Any, Optional
from datetime import datetime
import uuid
from app.db_models import FirestoreChat, firestore_chat
import asyncio
import re
from fastapi import BackgroundTasks
import json
from app.config import get_docs_url, DOCS_BASE_URLS, DEFAULT_PROVIDER
from app.path_registry import path_registry

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test mode configuration
TEST_MODE = os.getenv("CHAT_TEST_MODE", "false").lower() == "true"
logger.info(f"Chat test mode: {'enabled' if TEST_MODE else 'disabled'}")

# Set default RAG provider
DEFAULT_RAG_PROVIDER = os.getenv("DEFAULT_RAG_PROVIDER", "topic")
logger.info(f"Chat routes using default RAG provider: {DEFAULT_RAG_PROVIDER}")

# Initialize embeddings model
embeddings_model = OpenAIEmbeddings(
    model="text-embedding-3-large",
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Initialize chat model
chat_model = ChatOpenAI(
    model_name="gpt-4o",
    temperature=0.5,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    streaming=True,
)

router = APIRouter()

# In-memory storage for chat histories (replace with database in production)
chat_histories = []

# Base template that's common across all providers
BASE_TEMPLATE = """You are an AI assistant specialized in Aptos blockchain technology. Your task is to provide accurate, technical explanations based on the following documentation context:

Context:
{context}

When answering:
1. Be concise and to the point
2. Use exact technical terminology from the documentation
3. Include specific examples when relevant
4. Format your responses using proper markdown:
   - Use proper line breaks between paragraphs (double newline)
   - For numbered lists, ensure each item is on a new line with a blank line before the list starts
   - For code blocks, ALWAYS use triple backticks with language specification (e.g. ```typescript, ```python, ```move, ```json)
   - Ensure code blocks have proper indentation and formatting
   - For inline code, use single backticks
5. When linking to documentation, ALWAYS use the full URL from the base URL {base_url}. For example, if linking to 'en/build/sdks', use '{base_url}/en/build/sdks'
6. ONLY use the following validated URLs:
{valid_urls}

DO NOT construct URLs manually. Only use the exact URLs listed above."""

# Provider-specific templates
PROVIDER_TEMPLATES: Dict[str, str] = {
    "developer-docs": BASE_TEMPLATE
    + """

7. Always end your response with: "For further discussions or questions about {main_topic}, you can explore the [Aptos Dev Discussions](https://github.com/aptos-labs/aptos-developer-discussions/discussions)."
""",
    "aptos-learn": BASE_TEMPLATE
    + """

7. Focus on providing learning-oriented explanations suitable for developers at different levels.
8. When relevant, suggest appropriate workshops or tutorials from the Aptos Learn platform.
9. Always end your response with: "To continue learning about {main_topic}, check out our interactive workshops and tutorials at [Aptos Learn]({base_url}/en)."
""",
}


# Default to developer-docs template if provider not found
def get_system_template(provider: str) -> str:
    return PROVIDER_TEMPLATES.get(provider, PROVIDER_TEMPLATES["developer-docs"])


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def extract_main_topic(question: str) -> str:
    """
    Extract the main topic from the user's question.
    This is a simple implementation that extracts key nouns or phrases.
    """
    # Remove question words and common words
    question = question.lower()

    # Try to identify Aptos-specific topics
    aptos_topics = [
        "move",
        "smart contract",
        "transaction",
        "account",
        "token",
        "blockchain",
        "validator",
        "staking",
        "consensus",
        "module",
        "resource",
        "struct",
        "function",
        "aptos",
        "deployment",
        "testing",
        "framework",
        "coin",
        "nft",
        "development",
        "sdk",
        "api",
        "cli",
        "wallet",
        "node",
        "network",
    ]

    # Check for Aptos-specific topics in the question
    for topic in aptos_topics:
        if topic in question:
            return topic

    # If no specific topic is found, extract the main subject
    # Remove question words and common words
    question = re.sub(
        r"^(what|how|why|when|where|who|which|can|do|does|is|are|will|should)\s+",
        "",
        question,
    )
    question = re.sub(r"\?+$", "", question)  # Remove trailing question marks

    # If the question is about "how to" something, extract that something
    how_to_match = re.search(r"how\s+to\s+(.+?)(?:\?|$)", question)
    if how_to_match:
        return how_to_match.group(1).strip()

    # If still no topic found, return a generic topic based on the first few words
    words = question.split()
    if len(words) > 3:
        return " ".join(words[:3]) + "..."
    else:
        return question.strip()


@router.get(
    "/chat/histories",
    response_model=ChatHistoriesResponse,
    summary="Get Chat Histories",
    description="Retrieve all chat histories for a specific client",
    response_description="Returns a list of chat histories",
)
async def get_chat_histories(client_id: str):
    """Get all chat histories for a client."""
    try:
        histories = await firestore_chat.get_client_chat_histories(client_id)
        return {
            "histories": histories,
            "total_count": len(
                histories
            ),  # Add the total_count field to match the response model
        }
    except Exception as e:
        logger.error(f"Error getting chat histories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/chat/history/{chat_id}",
    response_model=ChatHistory,
    summary="Get Chat History",
    description="Retrieve a specific chat history by ID",
    response_description="Returns the chat history with the specified ID",
)
async def get_chat_history(chat_id: str):
    """Get a specific chat history."""
    try:
        history = await firestore_chat.get_chat_history(chat_id)
        if not history:
            raise HTTPException(status_code=404, detail="Chat history not found")
        return history
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting chat history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/chat/latest",
    response_model=ChatHistory,
    summary="Get Latest Chat",
    description="Retrieve the latest chat for a specific client",
    response_description="Returns the most recent chat history for the client",
)
async def get_latest_chat_history(client_id: str):
    """Get the latest chat history for a client."""
    try:
        # Validate client_id
        if not client_id or client_id == "undefined" or client_id == "null":
            raise HTTPException(status_code=400, detail="Invalid client ID")

        logger.info(f"Getting latest chat history for client {client_id}")

        # Get all chat histories for the client
        histories = await firestore_chat.get_client_chat_histories(client_id)

        if not histories:
            logger.warning(f"No chat histories found for client {client_id}")
            raise HTTPException(
                status_code=404, detail="No chat histories found for this client"
            )

        # Sort by timestamp (newest first) and return the first one
        histories.sort(key=lambda x: x.timestamp, reverse=True)
        latest_history = histories[0]

        logger.info(
            f"Found latest chat history {latest_history.id} for client {client_id}"
        )
        return latest_history
    except Exception as e:
        logger.error(f"Error getting latest chat history: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete(
    "/chat/history/{chat_id}",
    summary="Delete Chat History",
    description="Delete a specific chat history",
    response_description="Returns success status",
)
async def delete_chat_history(chat_id: str):
    """Delete a chat history."""
    try:
        # Delete from Firestore
        await firestore_chat.delete_chat_history(chat_id)
        return {"success": True, "message": "Chat history deleted"}
    except Exception as e:
        logger.error(f"Error deleting chat history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/feedback",
    response_model=StatusResponse,
    summary="Submit Feedback",
    description="Submit feedback for a specific message",
    response_description="Returns success status",
)
async def submit_feedback(feedback: Feedback):
    """Submit feedback for a message."""
    try:
        await firestore_chat.save_feedback(feedback)
        return {"success": True, "message": "Feedback submitted"}
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/rag/provider/{provider_name}/initialize",
    summary="Initialize RAG Provider",
    description="Initialize a specific RAG provider with configuration",
    response_description="Returns success status",
)
async def initialize_rag_provider(provider_name: str, config: Dict[str, Any]):
    """Initialize a RAG provider."""
    try:
        provider = RAGProviderRegistry.get_provider(provider_name)
        await provider.initialize(config)
        return {"success": True, "message": f"Provider {provider_name} initialized"}
    except Exception as e:
        logger.error(f"Error initializing RAG provider: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def stream_response(response_text: str):
    """Stream the response text word by word."""
    words = response_text.split()
    for word in words:
        yield f"{word} "
        await asyncio.sleep(0.05)  # Add a small delay between words


async def get_or_create_chat_history(
    chat_id: str, chat_request: ChatMessageRequest, firestore_chat: FirestoreChat
) -> ChatHistory:
    """
    Get an existing chat history or create a new one if it doesn't exist.

    Args:
        chat_id: The ID of the chat to get or create
        chat_request: The chat message request
        firestore_chat: The Firestore chat instance

    Returns:
        The chat history
    """
    try:
        # Check if the chat_id exists in the format "chat-uuid"
        if chat_id.startswith("chat-"):
            # Try to get the existing chat history
            chat_history = await firestore_chat.get_chat_history(chat_id)

            # If the chat history exists, add the new message
            if chat_history:
                logger.info(f"Adding message to existing chat {chat_id}")

                # Use the rag provider from the request or from the chat history metadata
                rag_provider_name = chat_request.rag_provider
                if (
                    not rag_provider_name
                    and chat_history.metadata
                    and "rag_provider" in chat_history.metadata
                ):
                    rag_provider_name = chat_history.metadata["rag_provider"]
                if not rag_provider_name:
                    rag_provider_name = DEFAULT_RAG_PROVIDER

                # Create a new message
                new_message = ChatMessage(
                    id=chat_request.message_id or str(uuid.uuid4()),
                    role=chat_request.role,
                    content=chat_request.content,
                    timestamp=datetime.now().isoformat(),
                    metadata={
                        "client_id": chat_request.client_id,
                        "rag_provider": rag_provider_name,
                        "temperature": getattr(chat_request, "temperature", 0.7),
                        "request_time": datetime.now().isoformat(),
                    },
                )

                # Add the message to the chat history
                chat_history.messages.append(new_message)

                # Add a placeholder for the assistant's response
                assistant_message = ChatMessage(
                    id=str(uuid.uuid4()),
                    role="assistant",
                    content="",  # Empty content as placeholder
                    timestamp=datetime.now().isoformat(),
                    metadata={
                        "response_to": new_message.id,
                        "rag_provider": rag_provider_name,
                    },
                )
                chat_history.messages.append(assistant_message)

                # Update the chat history in the database
                await firestore_chat.update_chat_history(chat_history)
                return chat_history

        # If the chat doesn't exist or the chat_id is not in the correct format, create a new chat
        logger.info("Creating a new chat")

        # Generate a title for the chat
        chat_title = (
            chat_request.content[:50] + "..."
            if len(chat_request.content) > 50
            else chat_request.content
        )

        # Use the rag provider from the request or the default
        rag_provider_name = chat_request.rag_provider or DEFAULT_RAG_PROVIDER

        # Create a user message
        user_message = ChatMessage(
            role=chat_request.role,
            content=chat_request.content,
            id=chat_request.message_id or str(uuid.uuid4()),
            timestamp=datetime.now().isoformat(),
            metadata={
                "client_id": chat_request.client_id,
                "rag_provider": rag_provider_name,
                "temperature": getattr(chat_request, "temperature", 0.7),
                "request_time": datetime.now().isoformat(),
                "is_first_message": True,
            },
        )

        # Create a new chat history
        chat_history = ChatHistory(
            id=chat_id,
            title=chat_title,
            timestamp=datetime.now().isoformat(),
            messages=[user_message],
            client_id=chat_request.client_id,
            metadata={
                "rag_provider": rag_provider_name,
                "creation_time": datetime.now().isoformat(),
                "topics": [],  # Will be populated as messages are added
            },
        )

        # Add a placeholder for the assistant's response
        assistant_message = ChatMessage(
            id=str(uuid.uuid4()),
            role="assistant",
            content="",  # Empty content as placeholder
            timestamp=datetime.now().isoformat(),
            metadata={
                "response_to": user_message.id,
                "rag_provider": rag_provider_name,
            },
        )
        chat_history.messages.append(assistant_message)

        # Save the chat history
        await firestore_chat.create_chat_history(chat_history)
        return chat_history
    except Exception as e:
        logger.error(f"Error in get_or_create_chat_history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def generate_ai_response(
    chat_history: ChatHistory,
    chat_id: str,
    firestore_chat: FirestoreChat,
    background_tasks: BackgroundTasks,
    rag_provider: str = None,
) -> AsyncGenerator[str, None]:
    """
    Generate AI response using RAG.

    Args:
        chat_history: The chat history
        chat_id: The ID of the chat
        firestore_chat: The Firestore chat instance
        background_tasks: FastAPI background tasks
        rag_provider: The RAG provider to use

    Returns:
        An async generator that yields chunks of the AI response
    """
    try:
        # Track generation start time
        generation_start_time = datetime.now()

        # Get the last user message
        user_message = None
        for message in reversed(chat_history.messages):
            if message.role == "user":
                user_message = message
                break

        if not user_message:
            logger.error("No user message found in chat history")
            yield "I apologize, but I couldn't find your message. Please try again."
            return

        message = user_message.content

        # Log the provider being used
        provider_type = rag_provider or chat_history.metadata.get(
            "rag_provider", DEFAULT_PROVIDER
        )
        logger.info(f"[RAG] Using provider: {provider_type} for chat {chat_id}")
        logger.info(f"[RAG] Base URL will be: {DOCS_BASE_URLS[provider_type]}")

        logger.info(
            f"[RAG] Starting response generation for message: {message[:100]}... in chat {chat_id}"
        )

        # Extract the main topic from the user's question
        main_topic = extract_main_topic(message)
        logger.info(f"[RAG] Extracted main topic: {main_topic}")

        # Check if the query is about a process or multi-step procedure
        process_keywords = [
            "how",
            "steps",
            "guide",
            "tutorial",
            "process",
            "install",
            "setup",
            "configure",
        ]
        is_process_query = any(kw in message.lower() for kw in process_keywords)

        # Get the RAG provider
        try:
            # Use the specified provider or the default
            provider_name = rag_provider or DEFAULT_RAG_PROVIDER
            rag_provider_obj = RAGProviderRegistry.get_provider(
                "docs"
            )  # Get the docs provider
            # Switch to the correct provider path
            await rag_provider_obj.switch_provider(provider_name)
            logger.info(f"[RAG] Using provider: {provider_name}")
        except ValueError as e:
            logger.error(f"Error getting RAG provider: {str(e)}")
            yield "I apologize, but I'm currently operating without access to the documentation. My responses may be limited. Please try again later or contact support."
            return

        # Get relevant context from documentation
        context_retrieval_start = datetime.now()
        logger.info("[RAG] Retrieving relevant context...")
        context_chunks = await rag_provider_obj.get_relevant_context(
            message, k=3, include_series=is_process_query, provider_type=provider_name
        )
        context_retrieval_time = (
            datetime.now() - context_retrieval_start
        ).total_seconds()

        if not context_chunks:
            logger.warning("[RAG] No context chunks retrieved")
            yield "I apologize, but I'm currently operating without access to the documentation. My responses may be limited."
            return

        # Separate chunks into series and non-series
        series_chunks = {}
        non_series_chunks = []
        for chunk in context_chunks:
            if chunk.get("series_title") and chunk.get("series_position"):
                if chunk["series_title"] not in series_chunks:
                    series_chunks[chunk["series_title"]] = []
                series_chunks[chunk["series_title"]].append(chunk)
            else:
                non_series_chunks.append(chunk)

        # Sort series chunks by position
        for series in series_chunks.values():
            series.sort(key=lambda x: x.get("series_position", 0))

        # Get valid URLs from the path registry
        valid_urls = path_registry.get_all_urls()
        if not valid_urls:
            logger.warning("[RAG] No valid URLs available")

        # Format valid URLs for the prompt with base URL
        base_url = DOCS_BASE_URLS[provider_type]
        formatted_urls = "\n".join(
            [f"- {base_url}/{url.lstrip('/')}" for url in valid_urls]
        )

        # Format context for the prompt
        formatted_context = ""

        # First add series chunks with series context
        for series_title, chunks in series_chunks.items():
            formatted_context += f"\n\nSeries: {series_title} ({len(chunks)} parts)\n"

            # Add overview of the series
            if len(chunks) > 1:
                formatted_context += f"This is a {len(chunks)}-part series covering: "
                formatted_context += ", ".join(
                    [f"Part {c.get('series_position', 0)}" for c in chunks]
                )
                formatted_context += "\n\n"

            # Add each chunk with its position in the series
            for chunk in chunks:
                formatted_context += f"Part {chunk.get('series_position', 0)}/{chunk.get('total_steps', 0)}: {chunk.get('section', '')}\n"
                if chunk.get("summary"):
                    formatted_context += f"Summary: {chunk.get('summary')}\n"
                formatted_context += f"Content: {chunk.get('content', '')}\n\n"

        # Then add non-series chunks
        for chunk in non_series_chunks:
            formatted_context += f"\n\nSection: {chunk.get('section', '')}\n"
            if chunk.get("summary"):
                formatted_context += f"Summary: {chunk.get('summary')}\n"
            formatted_context += f"Content: {chunk.get('content', '')}\n"

        context_length = len(formatted_context)
        logger.info(
            f"[RAG] Total formatted context length: {context_length} characters"
        )

        # Get the provider type for URL formatting
        provider_type = rag_provider or chat_history.metadata.get(
            "rag_provider", DEFAULT_PROVIDER
        )

        # Get the provider-specific template
        template = get_system_template(provider_type)
        logger.info(f"[RAG] Using template for provider: {provider_type}")

        # Log the first few lines of the formatted context to verify source
        context_preview = formatted_context.split("\n")[:5]
        logger.info(
            f"[RAG] Context preview (first 5 lines): {json.dumps(context_preview, indent=2)}"
        )

        # Prepare the prompt with the context, valid URLs, and correct base URL
        prompt = template.format(
            context=formatted_context,
            question=message,
            main_topic=main_topic,
            base_url=base_url,
            valid_urls=formatted_urls,
        )

        # Generate the response
        llm_start_time = datetime.now()
        logger.info("[RAG] Generating response...")

        # Create messages for the chat
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": message},
        ]

        # Use LangChain's ChatOpenAI with streaming
        full_response = ""
        async for chunk in chat_model.astream(messages):
            if chunk.content:
                full_response += chunk.content
                yield chunk.content

        llm_generation_time = (datetime.now() - llm_start_time).total_seconds()
        total_generation_time = (datetime.now() - generation_start_time).total_seconds()

        logger.info(
            f"[RAG] Completed response generation. Total response length: {len(full_response)} characters"
        )

        # Prepare metadata for storage
        message_metadata = {
            "generation_metrics": {
                "total_time_seconds": total_generation_time,
                "llm_time_seconds": llm_generation_time,
                "context_retrieval_time_seconds": context_retrieval_time,
                "response_length": len(full_response),
                "context_length": context_length,
            },
            "retrieval_analytics": {
                "num_context_chunks": len(context_chunks),
                "num_series_chunks": sum(
                    len(chunks) for chunks in series_chunks.values()
                ),
                "num_non_series_chunks": len(non_series_chunks),
                "series_titles": list(series_chunks.keys()),
            },
            "query_analysis": {
                "main_topic": main_topic,
                "is_process_query": is_process_query,
            },
            "rag_provider": provider_name,
        }

        # Update chat history
        try:
            # Extract sources from context chunks
            sources = []
            for chunk in context_chunks:
                if "source" in chunk and chunk["source"] not in sources:
                    sources.append(chunk["source"])

            # Find the last assistant message
            for i in range(len(chat_history.messages) - 1, -1, -1):
                if (
                    chat_history.messages[i].role == "assistant"
                    and chat_history.messages[i].content == ""
                ):
                    # Update the content
                    chat_history.messages[i].content = full_response

                    # Initialize metadata if needed
                    if not chat_history.messages[i].metadata:
                        chat_history.messages[i].metadata = {}

                    # Add sources and used chunks to metadata
                    chat_history.messages[i].metadata["sources"] = sources
                    chat_history.messages[i].metadata["used_chunks"] = context_chunks

                    # Add other metadata
                    chat_history.messages[i].metadata.update(message_metadata)

                    # Update the chat history
                    await firestore_chat.update_chat_history(chat_history)

                    # Also update chat metadata with some aggregated information
                    if not chat_history.metadata:
                        chat_history.metadata = {}

                    # Update or initialize chat metadata
                    if "rag_provider" not in chat_history.metadata:
                        chat_history.metadata["rag_provider"] = provider_name

                    # Add or update topic information in chat metadata
                    if "topics" not in chat_history.metadata:
                        chat_history.metadata["topics"] = [main_topic]
                    elif main_topic not in chat_history.metadata["topics"]:
                        chat_history.metadata["topics"].append(main_topic)

                    # Update chat metadata
                    await firestore_chat.update_chat_history(chat_history)
                    break
        except Exception as e:
            logger.error(
                f"Error updating chat history with response: {str(e)}",
                exc_info=True,
            )

    except Exception as e:
        logger.error(f"Error generating AI response: {str(e)}", exc_info=True)
        yield "I apologize, but I encountered an error while generating a response. Please try again later."


def format_source_to_url(
    source_path: str, provider_type: str = "developer-docs"
) -> str:
    """
    Convert a documentation source path to a proper URL.

    Args:
        source_path: The relative path from the documentation directory
        provider_type: The provider type to use for the base URL

    Returns:
        A properly formatted URL to the documentation
    """
    return get_docs_url(source_path, provider_type) if source_path else ""


def get_firestore_chat() -> FirestoreChat:
    """
    Dependency to get the Firestore chat instance.
    """
    return firestore_chat


@router.post(
    "/message/stream",
    response_model=None,
    description="Unified endpoint for creating a new chat or adding a message to an existing chat. Returns a streaming response with the assistant's message. The chat_id is included in the X-Chat-ID response header.",
    responses={
        200: {
            "description": "Streaming response with the assistant's message. The chat_id is included in the X-Chat-ID response header.",
            "content": {"text/plain": {}},
        }
    },
    summary="Stream Chat Message",
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "example": json.loads(
                        """
                    {
                        "content": "What is Aptos Move?",
                        "client_id": "client-123",
                        "message_id": "msg-123",
                        "chat_id": null,
                        "role": "user",
                        "rag_provider": null,
                        "temperature": 0.7
                    }
                    """
                    )
                }
            }
        }
    },
)
async def unified_chat_message_stream(
    request: Request,
    chat_request: ChatMessageRequest,
    background_tasks: BackgroundTasks,
    firestore_chat: FirestoreChat = Depends(get_firestore_chat),
    rag_provider: Optional[str] = None,
):
    """
    Unified endpoint for creating a new chat or adding a message to an existing chat.
    Returns a streaming response with the assistant's message.
    The chat_id is included in the X-Chat-ID response header.

    Parameters:
    - **chat_request**: The chat message request containing:
      - **content** (required): The message content
      - **client_id** (required): Unique identifier for the client
      - **message_id** (optional): Unique identifier for the message (generated if not provided)
      - **chat_id** (optional): Chat ID to add message to (creates new chat if not provided)
      - **role** (optional): Message role (defaults to "user")
      - **rag_provider** (optional): RAG provider to use (uses server default if not provided)
      - **temperature** (optional): Temperature for LLM generation (defaults to 0.7)
    - **rag_provider**: Optional override for the RAG provider specified in the request

    Returns:
    - Streaming response with the assistant's message
    - X-Chat-ID header containing the chat ID
    """
    # Get the chat_id from the request or generate a new one
    chat_id = chat_request.chat_id or f"chat-{uuid.uuid4()}"

    # Get or create the chat history
    chat_history = await get_or_create_chat_history(
        chat_id, chat_request, firestore_chat
    )

    # Generate the AI response
    response_generator = generate_ai_response(
        chat_history=chat_history,
        chat_id=chat_id,
        firestore_chat=firestore_chat,
        background_tasks=background_tasks,
        rag_provider=rag_provider or chat_request.rag_provider,
    )

    # Create a custom generator that streams the AI response directly
    async def direct_stream():
        # Buffer for accumulating content
        content_buffer = ""
        buffer_size_threshold = 50  # Characters to accumulate before sending

        # Stream the content chunks in larger pieces
        async for content_chunk in response_generator:
            content_buffer += content_chunk

            # Send the buffer when it reaches the threshold or contains complete sentences
            if len(content_buffer) >= buffer_size_threshold or any(
                c in content_buffer for c in [".", "!", "?", "\n"]
            ):
                yield content_buffer
                content_buffer = ""  # Reset the buffer

        # Send any remaining content in the buffer
        if content_buffer:
            yield content_buffer

    # Create the streaming response with the chat_id in the headers
    response = StreamingResponse(
        direct_stream(),
        media_type="text/plain",
    )

    # Add the chat_id to the response headers
    response.headers["X-Chat-ID"] = chat_id

    return response
