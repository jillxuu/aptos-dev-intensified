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
from app.config import (
    get_docs_url,
    DOCS_BASE_URLS,
    DEFAULT_PROVIDER,
    USE_MULTI_STEP_RAG,
)
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
    model="text-embedding-3-large", openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Initialize chat model
chat_model = ChatOpenAI(
    model_name="gpt-4.1",
    temperature=0.05,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    streaming=True,
)

router = APIRouter()

# In-memory storage for chat histories (replace with database in production)
chat_histories = []

# Base template that's common across all providers
BASE_TEMPLATE = """You are an AI assistant specialized in Aptos blockchain technology and Aptos Move code. Your task is to provide accurate, technical explanations based on the following documentation context:

Context:
{context}

When answering developer questions:

1. ACCURACY FIRST: 
   - Only answer based on information provided in the context
   - Clearly identify when you're uncertain or when information is missing
   - Never hallucinate features, functions, or capabilities not explicitly mentioned in the documentation
   - If multiple documents contain relevant information, synthesize them into a coherent answer
   - If the retrieved context doesn't contain relevant information to the question:
     * Clearly state that the you don't have the information
     * Don't attempt to answer with information not present in the context
     * Suggest possible alternative search terms the user might try
     * Recommend relevant sections of documentation that might contain the answer

2. QUESTION UNDERSTANDING:
   - Begin by identifying the core intent behind the developer's question
   - Recognize when questions have multiple parts or implied sub-questions
   - Identify the developer's likely use case or problem they're trying to solve
   - Determine the appropriate level of detail based on question complexity
   - For ambiguous questions, address the most likely interpretation first, then mention alternatives

3. CITING SOURCES:
   - Always identify the specific document source for your information
   - Use exact section titles and page numbers when available
   - Format citations like: [Document Title](full_url_to_document)
   - For multiple sources, list them at the end of your answer
   - Only reference external resources that are explicitly mentioned in the Aptos documentation

4. TECHNICAL PRECISION:
   - Use exact technical terminology from the Aptos documentation
   - Maintain precise technical meanings - don't simplify at the expense of accuracy
   - For Move code, follow exact Aptos Move syntax conventions
   - Prefer modern and latest features or code examples over deprecated or legacy code if there are multiple similar options.
   - Differentiate between Aptos-specific implementations and general blockchain concepts
   - Differentiate between similar looking concepts and code. For example, when asked about creating Fungile assets in Move, do not respond with coinV1 initialization code which looks very similar.

5. CODE EXAMPLES:
   - Provide complete, working code examples when relevant
   - All code must be runnable and follow Aptos best practices
   - Always specify the language with code blocks: ```move, ```typescript, ```python, etc.
   - Include comments explaining key parts of the code
   - For complex examples, break down the explanation step-by-step

6. FORMATTING:
   - Use proper markdown formatting for readability
   - Structure responses with clear headings and subheadings
   - Use bulleted or numbered lists for multi-step processes
   - Ensure proper spacing between paragraphs (double newline)
   - Highlight important warnings or notes in **bold**

7. LINKS AND REFERENCES:
   - ONLY use the following validated URLs:
{valid_urls}
   - When linking to documentation, ALWAYS use the full URL from base URL {base_url}. For example, if linking to 'en/build/sdks', use '{base_url}/en/build/sdks'
   - DO NOT construct URLs manually - only use exact URLs from the list above
   - If referring to API endpoints, include the complete endpoint path

8. ANSWER STRUCTURE AND RESOURCE DEDUPLICATION:
   - Begin with a direct answer to the question
   - Follow with deeper technical explanation
   - Include relevant code examples with proper citations
   - For highly technical questions, include a brief explanation of underlying concepts
   - When citing resources in your answer, keep track of which documents you've referenced
   - End with an "Additional Resources" section that ONLY includes:
     * Resources relevant to the topic but NOT already cited in your main answer
     * Documentation that provides supplementary information beyond what was covered
     * Resources that would help the developer explore related concepts or implementation details
   - Never repeat in the "Additional Resources" section any documents that were already cited in the main answer
   
9. HANDLING INFORMATION GAPS:
   - If you can identify that specific information is missing:
     * State exactly what additional information would be needed
     * Suggest specific documentation sections that might contain that information
     * Propose a reformulated question that might retrieve better context
     * When appropriate, suggest API references or GitHub repositories that might contain the answer
   - For multi-part questions where only some parts can be answered:
     * Answer the parts you can based on the context
     * Clearly identify which parts cannot be answered with the current context

Remember: You are supporting Aptos developers who need accurate technical information. Prioritize precision and correctness over simplification.
"""

# Provider-specific templates
PROVIDER_TEMPLATES: Dict[str, str] = {
    "developer-docs": BASE_TEMPLATE
    + """

10. DEVELOPER FOCUS:
    - Frame answers in terms of practical implementation
    - Highlight best practices and common pitfalls
    - Include performance considerations when relevant
    - Reference specific SDK functions and methods when applicable
   
11. COMMUNITY RESOURCES:
    Always end your response with: "For further discussions or questions about {main_topic}, you can explore the [Aptos Dev Discussions](https://github.com/aptos-labs/aptos-developer-discussions/discussions)."
""",
    "aptos-learn": BASE_TEMPLATE
    + """

10. LEARNING PROGRESSION:
    - Focus on providing learning-oriented explanations suitable for developers at different levels.
    - Begin with foundational concepts before advanced details
    - Include "Why" explanations along with "How" instructions
    - Link concepts to broader blockchain principles when helpful

11. EDUCATIONAL RESOURCES:
    - Suggest specific workshops, tutorials or learning paths when relevant from the Aptos Learn platform
    - For complex topics, break learning into manageable steps
    - End your response with: "To continue learning about {main_topic}, check out our interactive workshops and tutorials at [Aptos Learn]({base_url}/en)."
,

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
    """
    try:
        # Try to get the existing chat history first
        chat_history = await firestore_chat.get_chat_history(chat_id)

        # If the chat history exists, add the new message
        if chat_history:

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
        logger.error(
            f"[ChatAPI] Error in get_or_create_chat_history: {e}", exc_info=True
        )
        raise HTTPException(status_code=500, detail=str(e))


async def generate_ai_response(
    chat_history: ChatHistory,
    chat_id: str,
    firestore_chat: FirestoreChat,
    background_tasks: BackgroundTasks,
    rag_provider: str = None,
    use_multi_step: Optional[bool] = None,  # Now optional to handle frontend override
) -> AsyncGenerator[str, None]:
    """
    Generate AI response using RAG.

    Args:
        chat_history: The chat history
        chat_id: The ID of the chat
        firestore_chat: The Firestore chat instance
        background_tasks: FastAPI background tasks
        rag_provider: The RAG provider to use
        use_multi_step: Whether to use multi-step retrieval, overridden by frontend's fast mode selection
    """
    try:
        # If use_multi_step is None (not set by frontend), use the config default
        if use_multi_step is None:
            use_multi_step = USE_MULTI_STEP_RAG

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
            message,
            k=7,
            include_series=is_process_query,
            provider_type=provider_name,
            use_multi_step=use_multi_step,  # Enable adaptive multi-step retrieval
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
                        f"""
                    {{
                        "content": "What is Aptos Move?",
                        "client_id": "client-123",
                        "message_id": "msg-123",
                        "chat_id": null,
                        "role": "user",
                        "rag_provider": null,
                        "temperature": 0.1,
                        "use_multi_step": {str(USE_MULTI_STEP_RAG).lower()}
                    }}
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
    """
    logger.info(
        f"[ChatAPI] Received message request: chat_id={chat_request.chat_id}, client_id={chat_request.client_id}"
    )

    # First try to get existing chat if chat_id is provided
    chat_id = chat_request.chat_id
    if chat_id:
        # Try to get existing chat
        existing_chat = await firestore_chat.get_chat_history(chat_id)
        if existing_chat:
            logger.info(f"[ChatAPI] Found existing chat: {chat_id}")
        else:
            logger.warning(f"[ChatAPI] Chat ID provided but not found: {chat_id}")
            # If chat_id was provided but not found, we'll create a new one
            chat_id = None

    # Generate new chat ID only if we don't have a valid existing one
    if not chat_id:
        chat_id = f"chat-{uuid.uuid4()}"

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
        use_multi_step=chat_request.use_multi_step,
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
