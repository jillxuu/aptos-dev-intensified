from fastapi import APIRouter, HTTPException, Request
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
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
import os
import logging
import numpy as np
from typing import List, AsyncGenerator
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
5. Always end your response with: "For further discussions or questions about {main_topic}, you can explore the [Aptos Dev Discussions](https://github.com/aptos-labs/aptos-developer-discussions/discussions)." where {main_topic} is the main topic of the user's question.

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
    message: str, chat_id: str = None
) -> AsyncGenerator[str, None]:
    """Generate AI response using RAG."""
    try:
        logger.info(
            f"[RAG] Starting response generation for message: {message[:100]}... in chat {chat_id}"
        )

        # Extract the main topic from the user's question
        main_topic = extract_main_topic(message)
        logger.info(f"[RAG] Extracted main topic: {main_topic}")

        # Get relevant context from documentation
        logger.info("[RAG] Retrieving relevant context...")
        context_chunks = get_relevant_context(message, k=3)

        if not context_chunks:
            logger.warning(
                "[RAG] No context chunks retrieved - RAG may not be properly initialized"
            )
            yield "I apologize, but I'm currently operating without access to the documentation. My responses may be limited."
            return

        logger.info(f"[RAG] Retrieved {len(context_chunks)} context chunks")
        for i, chunk in enumerate(context_chunks):
            logger.info(
                f"[RAG] Context chunk {i+1}: Section={chunk['section']}, Score={chunk.get('score', 'N/A')}"
            )

        # Format context for the prompt
        formatted_context = "\n\n".join(
            [
                f"Section: {chunk['section']}\nContent: {chunk['content']}"
                for chunk in context_chunks
            ]
        )
        logger.info(
            f"[RAG] Total formatted context length: {len(formatted_context)} characters"
        )

        # Create messages for the chat
        messages = [
            {
                "role": "system",
                "content": SYSTEM_TEMPLATE.format(
                    context=formatted_context, question=message, main_topic=main_topic
                ),
            },
            {"role": "user", "content": message},
        ]

        logger.info("[RAG] Starting streaming response generation...")
        full_response = ""

        # Use astream instead of invoke for streaming
        async for chunk in chat_model.astream(messages):
            if chunk.content:
                full_response += chunk.content
                yield chunk.content

        logger.info(
            f"[RAG] Completed response generation. Total response length: {len(full_response)} characters"
        )

        # Update the assistant message in the chat history with the full response
        if chat_id:
            try:
                # Get the chat history
                history = await firestore_chat.get_chat_history(chat_id)
                if history:
                    # Find the last assistant message (which should be empty)
                    assistant_message_found = False
                    for i in range(len(history.messages) - 1, -1, -1):
                        if (
                            history.messages[i].role == "assistant"
                            and not history.messages[i].content
                        ):
                            # Update the content
                            history.messages[i].content = full_response
                            # Update the chat history
                            await firestore_chat.update_chat_history(history)
                            assistant_message_found = True
                            logger.info(
                                f"[RAG] Updated assistant message in chat {chat_id}"
                            )
                            break

                    if not assistant_message_found:
                        logger.warning(
                            f"[RAG] No empty assistant message found in chat {chat_id}"
                        )
            except Exception as e:
                logger.error(
                    f"[RAG] Error updating assistant message: {str(e)}", exc_info=True
                )

    except Exception as e:
        logger.error(f"[RAG] Error generating AI response: {str(e)}", exc_info=True)
        yield "I apologize, but I encountered an error while processing your request. Please try again."


@router.post("/chat/new/stream")
async def new_chat_stream(request: ChatRequest):
    try:
        if not request.messages or len(request.messages) != 1:
            raise HTTPException(
                status_code=400, detail="New chat must start with exactly one message"
            )

        user_message = request.messages[0]
        if user_message.role != "user":
            raise HTTPException(
                status_code=400, detail="First message must be from user"
            )

        # Validate client_id
        if (
            not request.client_id
            or request.client_id == "undefined"
            or request.client_id == "null"
        ):
            raise HTTPException(status_code=400, detail="Invalid client ID")

        # Create new chat with UUID and initialize with both messages
        chat_id = str(uuid.uuid4())
        assistant_message_id = str(uuid.uuid4())

        logger.info(f"Creating new chat {chat_id} for client {request.client_id}")

        # Get context and prepare metadata before streaming
        context_chunks = get_relevant_context(user_message.content, k=3)
        if not context_chunks:
            logger.warning("[RAG] No context chunks retrieved for new chat")
            metadata = {"sources": [], "used_chunks": []}
        else:
            metadata = {
                "sources": [
                    chunk["source"] for chunk in context_chunks if "source" in chunk
                ],
                "used_chunks": [
                    {
                        "content": chunk["content"],
                        "section": chunk["section"],
                        "source": chunk.get("source", ""),
                    }
                    for chunk in context_chunks
                ],
            }

        # Create initial chat history with both messages
        new_chat = ChatHistory(
            id=chat_id,
            title=user_message.content[:50] + "...",
            timestamp=datetime.now().isoformat(),
            client_id=request.client_id,
            messages=[
                ChatMessage(
                    id=str(uuid.uuid4()),
                    role="user",
                    content=user_message.content,
                    timestamp=datetime.now().isoformat(),
                ),
                ChatMessage(
                    id=assistant_message_id,
                    role="assistant",
                    content="",  # Will be filled during streaming
                    timestamp=datetime.now().isoformat(),
                    sources=metadata["sources"],
                    used_chunks=metadata["used_chunks"],
                ),
            ],
        )

        # Save initial chat history with metadata
        await firestore_chat.create_chat_history(new_chat)
        logger.info(f"Created new chat history {chat_id}")

        return StreamingResponse(
            generate_ai_response(user_message.content, chat_id),
            media_type="text/event-stream",
        )

    except Exception as e:
        logger.error(f"Error creating new chat stream: {str(e)}", exc_info=True)
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
async def add_chat_message_stream(chat_id: str, message: ChatMessageRequest):
    try:
        if message.role != "user":
            raise HTTPException(
                status_code=400, detail="Only user messages can be added directly"
            )

        # Ensure we're using the path parameter chat_id, not any chat_id in the message body
        # This ensures we're updating the correct chat
        logger.info(f"Adding message to existing chat: {chat_id}")

        # Validate chat_id
        if not chat_id or chat_id == "undefined" or chat_id == "null":
            raise HTTPException(status_code=400, detail="Invalid chat ID")

        # Get existing chat history
        history = await firestore_chat.get_chat_history(chat_id)
        if not history:
            logger.error(f"Chat history not found: {chat_id}")
            raise HTTPException(status_code=404, detail="Chat history not found")

        # Create assistant message with metadata before streaming
        context_chunks = get_relevant_context(message.content, k=3)
        if not context_chunks:
            logger.warning("[RAG] No context chunks retrieved for message")
            metadata = {"sources": [], "used_chunks": []}
        else:
            metadata = {
                "sources": [
                    chunk["source"] for chunk in context_chunks if "source" in chunk
                ],
                "used_chunks": [
                    {
                        "content": chunk["content"],
                        "section": chunk["section"],
                        "source": chunk.get("source", ""),
                    }
                    for chunk in context_chunks
                ],
            }

        # Add both user and assistant messages to history
        user_message_id = str(uuid.uuid4())
        assistant_message_id = str(uuid.uuid4())

        # Add user message
        history.messages.append(
            ChatMessage(
                id=user_message_id,
                role="user",
                content=message.content,
                timestamp=datetime.now().isoformat(),
            )
        )

        # Add empty assistant message (will be filled during streaming)
        assistant_message = ChatMessage(
            id=assistant_message_id,
            role="assistant",
            content="",  # Will be filled during streaming
            timestamp=datetime.now().isoformat(),
            sources=metadata["sources"],
            used_chunks=metadata["used_chunks"],
        )
        history.messages.append(assistant_message)

        # Update timestamp to ensure it's the most recent
        history.timestamp = datetime.now().isoformat()

        # Update history with new messages including metadata
        await firestore_chat.update_chat_history(history)

        logger.info(f"Updated chat history {chat_id} with new messages")

        return StreamingResponse(
            generate_ai_response(message.content, chat_id),
            media_type="text/event-stream",
        )

    except Exception as e:
        logger.error(f"Error adding chat message stream: {str(e)}", exc_info=True)
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
