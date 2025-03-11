from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.responses import StreamingResponse
from app.models import (
    ChatRequest,
    ChatResponse,
    Feedback,
    ChatHistory,
    ChatMessage,
    ChatMessageRequest,
    ChatMessageResponse,
    ChatHistoryResponse,
    ChatHistoriesResponse,
    StatusResponse,
    get_relevant_context,
)
from app.rag_providers import RAGProviderRegistry
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
import os
import logging
import numpy as np
from typing import List, AsyncGenerator, Dict, Any
from datetime import datetime
import uuid
from app.db_models import firestore_chat
import asyncio
import json
import re

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
embeddings_model = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

# Initialize chat model
chat_model = ChatOpenAI(
    model_name="gpt-4o-2024-11-20",
    temperature=0.5,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    streaming=True,
)

router = APIRouter()

# In-memory storage for chat histories (replace with database in production)
chat_histories = []


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


SYSTEM_TEMPLATE = """You are an AI assistant specialized in Aptos blockchain technology. You have access to the official Aptos documentation from aptos.dev. Your task is to provide accurate, technical explanations based on the following documentation context:

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
5. When linking to Aptos documentation, ALWAYS use the full URL with the correct structure: https://aptos.dev/en/[section]/[page]. For example:
   - Use https://aptos.dev/en/build/cli instead of https://aptos.dev/cli
   - Use https://aptos.dev/en/sdks/ts-sdk instead of https://aptos.dev/sdks/ts-sdk
6. Always end your response with: "For further discussions or questions about {main_topic}, you can explore the [Aptos Dev Discussions](https://github.com/aptos-labs/aptos-developer-discussions/discussions)." where {main_topic} is the main topic of the user's question.

User's question: {question}"""


@router.get("/chat/histories", response_model=ChatHistoriesResponse)
async def get_chat_histories(client_id: str):
    """Get all chat histories for a specific client."""
    try:
        histories = await firestore_chat.get_client_chat_histories(client_id)
        return {"histories": histories, "total_count": len(histories)}
    except Exception as e:
        logger.error(f"Error getting chat histories: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/chat/history/{chat_id}", response_model=ChatHistory)
async def get_chat_history(chat_id: str):
    history = await firestore_chat.get_chat_history(chat_id)
    if not history:
        raise HTTPException(status_code=404, detail="Chat history not found")
    return history


@router.post("/chat/history", response_model=StatusResponse)
async def create_chat_history(history: ChatHistory):
    await firestore_chat.create_chat_history(history)
    return {"status": "success", "message": "Chat history created"}


@router.put("/chat/history/{chat_id}", response_model=StatusResponse)
async def update_chat_history(chat_id: str, history: ChatHistory):
    if chat_id != history.id:
        raise HTTPException(status_code=400, detail="Chat ID mismatch")
    await firestore_chat.update_chat_history(history)
    return {"status": "success", "message": "Chat history updated"}


async def stream_response(response_text: str):
    """Stream the response text word by word."""
    words = response_text.split()
    for word in words:
        yield f"{word} "
        await asyncio.sleep(0.05)  # Add a small delay between words


async def generate_ai_response(
    message: str, chat_id: str = None, rag_provider_name: str = None
) -> AsyncGenerator[str, None]:
    """Generate AI response using RAG."""
    try:
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

        # Get the RAG provider - always default to topic provider if not specified
        try:
            # Use the specified provider or the default topic provider
            provider_name = rag_provider_name or DEFAULT_RAG_PROVIDER
            rag_provider = RAGProviderRegistry.get_provider(provider_name)
            logger.info(f"[RAG] Using provider: {rag_provider.name}")
        except ValueError as e:
            logger.warning(
                f"[RAG] Provider error: {str(e)}, falling back to topic provider"
            )
            try:
                # Try to get the topic provider explicitly
                rag_provider = RAGProviderRegistry.get_provider("topic")
                logger.info(f"[RAG] Using topic provider as fallback")
            except ValueError as e:
                # No topic provider registered, try default provider
                try:
                    rag_provider = RAGProviderRegistry.get_provider()
                    logger.info(f"[RAG] Using default provider: {rag_provider.name}")
                except ValueError as e:
                    # No default provider registered, handle gracefully
                    logger.error(f"Error generating AI response: {str(e)}")
                    yield "I apologize, but I'm currently operating without access to the documentation. My responses may be limited. Please try again later or contact support."
                    return

        # Get relevant context from documentation
        logger.info("[RAG] Retrieving relevant context...")
        context_chunks = await rag_provider.get_relevant_context(
            message, k=3, include_series=is_process_query
        )

        if not context_chunks:
            logger.warning(
                "[RAG] No context chunks retrieved - RAG may not be properly initialized"
            )
            yield "I apologize, but I'm currently operating without access to the documentation. My responses may be limited."
            return

        logger.info(f"[RAG] Retrieved {len(context_chunks)} context chunks")

        # Group chunks by series
        series_chunks = {}
        non_series_chunks = []

        for chunk in context_chunks:
            if chunk.get("is_part_of_series"):
                series_title = chunk.get("series_title", "Unknown Series")
                if series_title not in series_chunks:
                    series_chunks[series_title] = []
                series_chunks[series_title].append(chunk)
            else:
                non_series_chunks.append(chunk)

        # Sort series chunks by position
        for series_title, chunks in series_chunks.items():
            chunks.sort(key=lambda x: x.get("series_position", 0))
            logger.info(f"[RAG] Series '{series_title}' has {len(chunks)} chunks")

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

        logger.info(
            f"[RAG] Total formatted context length: {len(formatted_context)} characters"
        )

        # Prepare the prompt with the context
        prompt = SYSTEM_TEMPLATE.format(
            context=formatted_context, question=message, main_topic=main_topic
        )

        # Generate the response
        logger.info("[RAG] Generating response...")

        # Create messages for the chat
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": message},
        ]

        # Use astream instead of invoke for streaming
        full_response = ""
        async for chunk in chat_model.astream(messages):
            if chunk.content:
                full_response += chunk.content
                yield chunk.content

        logger.info(
            f"[RAG] Completed response generation. Total response length: {len(full_response)} characters"
        )

        # Update chat history if chat_id is provided
        if chat_id:
            try:
                # Get the chat history
                history = await firestore_chat.get_chat_history(chat_id)
                if history and history.messages:
                    # Find the last assistant message
                    for i in range(len(history.messages) - 1, -1, -1):
                        if (
                            history.messages[i].role == "assistant"
                            and history.messages[i].content == ""
                        ):
                            # Update the content
                            history.messages[i].content = full_response
                            # Update the chat history
                            await firestore_chat.update_chat_history(history)
                            break
            except Exception as e:
                logger.error(
                    f"Error updating chat history with response: {str(e)}",
                    exc_info=True,
                )

    except Exception as e:
        logger.error(f"Error generating AI response: {str(e)}", exc_info=True)
        yield "I apologize, but I encountered an error while generating a response. Please try again later."


def format_source_to_url(source_path: str) -> str:
    """
    Convert a documentation source path to a proper aptos.dev URL.

    Args:
        source_path: The relative path from the documentation directory

    Returns:
        A properly formatted URL to the documentation on aptos.dev
    """
    if not source_path:
        return ""

    # Remove file extension (.md or .mdx)
    if source_path.endswith(".md") or source_path.endswith(".mdx"):
        source_path = os.path.splitext(source_path)[0]

    # Handle index files
    if source_path.endswith("/index") or source_path == "index":
        source_path = source_path[:-6] if source_path.endswith("/index") else ""

    # Format the URL
    url_path = source_path.replace(os.path.sep, "/")

    # Ensure the URL starts with /en/ for the English documentation
    if not url_path.startswith("en/"):
        url_path = f"en/{url_path}"

    return f"https://aptos.dev/{url_path}"


@router.post("/chat/new/stream")
async def new_chat_stream(request: ChatRequest, request_obj: Request):
    """Create a new chat and stream the response."""
    try:
        # Create a new chat history
        chat_id = str(uuid.uuid4())
        client_id = request.client_id

        # Get the user's message
        user_message = next(
            (msg for msg in request.messages if msg.role == "user"), None
        )

        if not user_message:
            raise HTTPException(status_code=400, detail="No user message found")

        # Generate a title for the chat
        chat_title = (
            user_message.content[:50] + "..."
            if len(user_message.content) > 50
            else user_message.content
        )

        # Use the topic provider by default unless specified otherwise
        rag_provider_name = (
            getattr(request, "rag_provider", None) or DEFAULT_RAG_PROVIDER
        )

        # Create a new chat history
        chat_history = ChatHistory(
            id=chat_id,
            title=chat_title,
            timestamp=datetime.now().isoformat(),
            messages=request.messages,
            client_id=client_id,
            metadata={"rag_provider": rag_provider_name},
        )

        # Add a placeholder for the assistant's response
        assistant_message = ChatMessage(
            id=str(uuid.uuid4()),
            role="assistant",
            content="",  # Empty content as placeholder
            timestamp=datetime.now().isoformat(),
        )
        chat_history.messages.append(assistant_message)

        # Save the chat history
        await firestore_chat.create_chat_history(chat_history)

        # Generate the response
        return StreamingResponse(
            generate_ai_response(user_message.content, chat_id, rag_provider_name),
            media_type="text/event-stream",
        )
    except Exception as e:
        logger.error(f"Error creating new chat: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/chat/{chat_id}/messages", response_model=ChatHistoryResponse)
async def get_chat_messages(chat_id: str):
    try:
        history = await firestore_chat.get_chat_history(chat_id)
        if not history:
            raise HTTPException(status_code=404, detail="Chat history not found")

        return {
            "chat_id": chat_id,
            "title": history.title,
            "messages": history.messages,
        }
    except Exception as e:
        logger.error(f"Error getting chat messages: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/chat/latest", response_model=ChatHistory)
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


@router.post("/chat/{chat_id}/message/stream")
async def add_chat_message_stream(
    chat_id: str, message: ChatMessageRequest, request: Request
):
    """Add a message to an existing chat and stream the response."""
    try:
        # Get the chat history
        chat_history = await firestore_chat.get_chat_history(chat_id)

        if not chat_history:
            raise HTTPException(status_code=404, detail=f"Chat {chat_id} not found")

        # Use the topic provider by default unless specified in metadata or headers
        rag_provider_name = None

        # First check if the chat has a provider in its metadata
        if chat_history.metadata and "rag_provider" in chat_history.metadata:
            rag_provider_name = chat_history.metadata["rag_provider"]

        # If not, check the request headers
        if not rag_provider_name:
            rag_provider_name = request.headers.get("X-RAG-Provider")

        # If still not set, use the default
        if not rag_provider_name:
            rag_provider_name = DEFAULT_RAG_PROVIDER

        # Create a new message
        new_message = ChatMessage(
            id=str(uuid.uuid4()),
            role=message.role,
            content=message.content,
            timestamp=datetime.now().isoformat(),
        )

        # Add the message to the chat history
        chat_history.messages.append(new_message)

        # Add a placeholder for the assistant's response
        assistant_message = ChatMessage(
            id=str(uuid.uuid4()),
            role="assistant",
            content="",  # Empty content as placeholder
            timestamp=datetime.now().isoformat(),
        )
        chat_history.messages.append(assistant_message)

        # Save the updated chat history
        await firestore_chat.update_chat_history(chat_history)

        # Generate the response
        return StreamingResponse(
            generate_ai_response(message.content, chat_id, rag_provider_name),
            media_type="text/event-stream",
        )
    except Exception as e:
        logger.error(f"Error adding message to chat: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/feedback", response_model=StatusResponse)
async def submit_feedback(feedback: Feedback):
    try:
        await firestore_chat.save_feedback(feedback)
        return {"status": "success", "message": "Feedback received"}
    except Exception as e:
        logger.error(f"Error processing feedback: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/chat/history/{chat_id}")
async def delete_chat_history(chat_id: str):
    try:
        # Delete from Firestore
        await firestore_chat.delete_chat_history(chat_id)
        return {"status": "success", "message": "Chat history deleted"}
    except Exception as e:
        logger.error(f"Error deleting chat history: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rag/providers")
async def list_rag_providers():
    """List all available Knowledge providers."""
    try:
        providers = RAGProviderRegistry.list_providers()
        return {"providers": providers}
    except Exception as e:
        logger.error(f"Error listing Knowledge providers: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rag/provider/{provider_name}/initialize")
async def initialize_rag_provider(provider_name: str, config: Dict[str, Any]):
    """Initialize a Knowledge provider with the given configuration."""
    try:
        provider = RAGProviderRegistry.get_provider(provider_name)
        await provider.initialize(config)
        return {
            "status": "success",
            "message": f"Initialized Knowledge provider {provider_name}",
        }
    except ValueError as e:
        logger.error(f"Error initializing Knowledge provider: {str(e)}", exc_info=True)
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error initializing Knowledge provider: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
