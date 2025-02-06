from fastapi import APIRouter, HTTPException
from app.models import ChatRequest, ChatResponse, Feedback, ChatHistory, get_relevant_context
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
import os
import logging
import numpy as np
from typing import List
from datetime import datetime
import uuid

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize embeddings model
embeddings_model = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

# In-memory storage for chat histories (replace with database in production)
chat_histories = []

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def is_technical_question(message: str, threshold: float = 0.75) -> bool:
    """
    Check if a message is related to Aptos/blockchain technology using semantic similarity.
    
    Args:
        message: The user's message
        threshold: Similarity threshold (0-1) for technical question detection
        
    Returns:
        bool: True if the message is semantically similar to technical/Aptos-related topics
    """
    # Representative technical topics and concepts from Aptos
    technical_topics = [
        "How does the Move programming language work?",
        "What is blockchain consensus?",
        "How do I deploy a smart contract?",
        "What are transactions in Aptos?",
        "How does Aptos handle account management?",
        "What is the token standard in Aptos?",
        "How do I create an NFT on Aptos?",
        "What are modules in Move?",
        "How does Aptos achieve scalability?",
        "What is the Aptos framework?"
    ]
    
    try:
        # Get embeddings for the message and technical topics
        message_embedding = embeddings_model.embed_query(message.lower())
        topic_embeddings = [embeddings_model.embed_query(topic) for topic in technical_topics]
        
        # Calculate maximum similarity with any technical topic
        max_similarity = max(
            cosine_similarity(message_embedding, topic_emb)
            for topic_emb in topic_embeddings
        )
        
        logger.debug(f"Technical topic similarity score: {max_similarity}")
        return max_similarity > threshold
        
    except Exception as e:
        logger.error(f"Error in technical question detection: {e}")
        # Fall back to keyword-based heuristic if embedding fails
        keywords = {'aptos', 'blockchain', 'move', 'smart contract', 'token', 'nft', 
                   'transaction', 'wallet', 'crypto', 'module', 'consensus', 'node'}
        return any(keyword in message.lower() for keyword in keywords)

SYSTEM_TEMPLATE = """You are an AI assistant specialized in Aptos blockchain technology. You have access to the official Aptos documentation from aptos.dev. Your task is to provide accurate, technical explanations based on the following documentation context:

Context:
{context}

When answering:
1. Be concise and to the point
2. Use exact technical terminology from the documentation
3. Include specific examples when relevant
4. If you're not completely certain or documentation is incomplete, refer users to:
   - aptos.dev
   - learn.aptoslabs.com
   - developers.aptoslabs.com/docs/introduction
5. ALWAYS reference documentation pages using the format: aptos.dev/en/[path]

User's question: {question}"""

@router.get("/chat/histories")
async def get_chat_histories():
    return chat_histories

@router.get("/chat/history/{chat_id}")
async def get_chat_history(chat_id: str):
    history = next((h for h in chat_histories if h.id == chat_id), None)
    if not history:
        raise HTTPException(status_code=404, detail="Chat history not found")
    return history

@router.post("/chat/history")
async def create_chat_history(history: ChatHistory):
    chat_histories.append(history)
    return {"status": "success", "message": "Chat history created"}

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        logger.debug("Received chat request")
        # Get the last user message
        last_message = next((msg for msg in reversed(request.messages) if msg.role == "user"), None)
        if not last_message:
            raise HTTPException(status_code=400, detail="No user message found")
        
        logger.debug(f"Processing user message: {last_message.content}")
        
        # Check if the message is technical/Aptos-related
        is_technical = is_technical_question(last_message.content)
        
        context_docs = []
        if is_technical:
            # For technical questions, get more context
            logger.debug("Getting relevant context for technical question")
            # Get more documents for technical questions
            context_docs = get_relevant_context(last_message.content, k=8)
            logger.debug(f"Found {len(context_docs) if context_docs else 0} relevant documents")
            
            # If the question contains specific technical terms, try to get additional context
            technical_terms = ['keyless', 'authentication', 'account', 'transaction', 'smart contract']
            if any(term in last_message.content.lower() for term in technical_terms):
                logger.debug("Getting additional context for specific technical terms")
                # Get additional context specifically about these terms
                additional_docs = get_relevant_context(
                    f"technical details and implementation of {' '.join(term for term in technical_terms if term in last_message.content.lower())}", 
                    k=3
                )
                # Add new docs that aren't duplicates
                existing_sources = {doc['source'] for doc in context_docs}
                for doc in additional_docs:
                    if doc['source'] not in existing_sources:
                        context_docs.append(doc)
                        existing_sources.add(doc['source'])
        
        # Format context with clear section markers and source references
        context_text = "\n\n".join([
            f"[Section: {doc['section']}]\n{doc['content']}\nSource: aptos.dev/en/{doc['source'].replace('data/developer-docs/apps/nextra/pages/en/', '').replace('.mdx', '').replace('.md', '')}"
            for doc in context_docs
        ]) if context_docs else "No specific context available."
        
        logger.debug("Creating prompt")
        # Create prompt
        prompt = ChatPromptTemplate.from_template(SYSTEM_TEMPLATE)
        formatted_prompt = prompt.format(
            context=context_text,
            question=last_message.content
        )
        
        logger.debug("Initializing GPT-4 model")
        # Initialize GPT-4 model
        model = ChatOpenAI(
            model="gpt-4-turbo-preview",
            temperature=request.temperature,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        logger.debug("Generating response")
        # Generate response
        response = model.invoke(formatted_prompt)
        logger.debug("Response generated successfully")

        # Format sources
        sources = []
        if context_docs:
            seen_sources = set()
            for doc in context_docs:
                source = f"aptos.dev/en/{doc['source'].replace('data/developer-docs/apps/nextra/pages/en/', '').replace('.mdx', '').replace('.md', '')}"
                if source not in seen_sources:
                    sources.append(source)
                    seen_sources.add(source)

        # Handle chat history
        chat_id = request.messages[0].id if hasattr(request.messages[0], 'id') else None
        existing_history = next((h for h in chat_histories if h.id == chat_id), None) if chat_id else None

        if existing_history:
            # Update existing history
            existing_history.messages = request.messages
        else:
            # Create new history
            history = ChatHistory(
                id=str(uuid.uuid4()),
                title=request.messages[0].content[:50] + "...",
                timestamp=datetime.now().isoformat(),
                messages=request.messages
            )
            chat_histories.append(history)
            chat_id = history.id

        logger.debug("Returning response")
        return {
            "response": response.content,
            "sources": sources if sources else None,
            "used_chunks": context_docs if context_docs else None,
            "chat_id": chat_id
        }

    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/feedback")
async def submit_feedback(feedback: Feedback):
    try:
        logger.debug(f"Received feedback for message {feedback.message_id}")
        
        # Here we'll store the feedback - for now we'll just log it
        # In a production system, this would be stored in a database
        logger.info(f"""
        Feedback received:
        Message ID: {feedback.message_id}
        Rating: {"👍" if feedback.rating else "👎"}
        Query: {feedback.query}
        Feedback text: {feedback.feedback_text if feedback.feedback_text else "None"}
        Number of chunks used: {len(feedback.used_chunks) if feedback.used_chunks else 0}
        """)
        
        # TODO(jill): Implement feedback processing logic:
        # 1. Store feedback in database
        # 2. Update chunk relevance scores
        # 3. Flag responses for review if needed
        # 4. Collect training data for fine-tuning
        
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