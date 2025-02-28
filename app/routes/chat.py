from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from app.models import (
    ChatRequest, ChatResponse, Feedback, ChatHistory, 
    ChatMessageRequest, ChatMessageResponse, ChatHistoryResponse,
    ChatHistoriesResponse, StatusResponse, get_relevant_context
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test mode configuration
TEST_MODE = os.getenv('CHAT_TEST_MODE', 'false').lower() == 'true'
logger.info(f"Chat test mode: {'enabled' if TEST_MODE else 'disabled'}")

# Initialize embeddings model
embeddings_model = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

# Initialize chat model
chat_model = ChatOpenAI(
    model_name="gpt-3.5-turbo-16k",
    temperature=0.7,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    streaming=True
)

router = APIRouter()

# In-memory storage for chat histories (replace with database in production)
chat_histories = []

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

SYSTEM_TEMPLATE = """You are an AI assistant specialized in Aptos blockchain technology. You have access to the official Aptos documentation from aptos.dev. Your task is to provide accurate, technical explanations based on the following documentation context:

Context:
{context}

When answering:
1. Be concise and to the point
2. Use exact technical terminology from the documentation
3. Include specific examples when relevant

User's question: {question}"""

@router.get("/chat/histories", response_model=ChatHistoriesResponse)
async def get_chat_histories(client_id: str):
    """Get all chat histories for a specific client."""
    try:
        histories = await firestore_chat.get_client_chat_histories(client_id)
        return {
            "histories": histories,
            "total_count": len(histories)
        }
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

async def generate_ai_response(message: str, chat_id: str = None) -> AsyncGenerator[str, None]:
    """Generate AI response using RAG."""
    try:
        # Get relevant context from documentation
        context_chunks = get_relevant_context(message, k=3)
        
        # Format context for the prompt
        formatted_context = "\n\n".join([
            f"Section: {chunk['section']}\nContent: {chunk['content']}"
            for chunk in context_chunks
        ])
        
        # Create messages for the chat
        messages = [
            {
                "role": "system",
                "content": SYSTEM_TEMPLATE.format(
                    context=formatted_context,
                    question=message
                )
            },
            {
                "role": "user",
                "content": message
            }
        ]
        
        # Use astream instead of invoke for streaming
        async for chunk in chat_model.astream(messages):
            if chunk.content:
                yield chunk.content

    except Exception as e:
        logger.error(f"Error generating AI response: {str(e)}", exc_info=True)
        yield "I apologize, but I encountered an error while processing your request. Please try again."

@router.post("/chat/new/stream")
async def new_chat_stream(request: ChatRequest):
    try:
        if not request.messages or len(request.messages) != 1:
            raise HTTPException(status_code=400, detail="New chat must start with exactly one message")
        
        user_message = request.messages[0]
        if user_message.role != "user":
            raise HTTPException(status_code=400, detail="First message must be from user")

        # Create new chat with UUID
        chat_id = str(uuid.uuid4())
        
        return StreamingResponse(
            generate_ai_response(user_message.content, chat_id),
            media_type='text/event-stream'
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
            "messages": history.messages
        }
    except Exception as e:
        logger.error(f"Error getting chat messages: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat/{chat_id}/message/stream")
async def add_chat_message_stream(chat_id: str, message: ChatMessageRequest):
    try:
        if message.role != "user":
            raise HTTPException(status_code=400, detail="Only user messages can be added directly")
        
        # Get existing chat history
        history = await firestore_chat.get_chat_history(chat_id)
        if not history:
            raise HTTPException(status_code=404, detail="Chat history not found")
        
        return StreamingResponse(
            generate_ai_response(message.content, chat_id),
            media_type='text/event-stream'
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
        global chat_histories
        chat_histories = [h for h in chat_histories if h.id != chat_id]
        return {"status": "success", "message": "Chat history deleted"}
    except Exception as e:
        logger.error(f"Error deleting chat history: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) 