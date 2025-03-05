"""Models for the chat application."""

from pydantic import BaseModel, validator
from typing import List, Optional, Dict, Any
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)
import os
from dotenv import load_dotenv
import yaml
import re
import logging
import json
from datetime import datetime
import cohere
from openai import OpenAI
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import hashlib
import pickle

load_dotenv()

logger = logging.getLogger(__name__)

# Initialize Cohere client
cohere_api_key = os.getenv("COHERE_API_KEY")
cohere_client = cohere.Client(api_key=cohere_api_key) if cohere_api_key else None

# Initialize OpenAI client for summarization
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize NLTK resources
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)
# Add punkt_tab download
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)

# Track vector store state
VECTOR_STORE_STATE_FILE = "data/vector_store_state.json"
# Summary cache file
SUMMARY_CACHE_FILE = "data/summary_cache.pkl"

# In-memory summary cache
summary_cache = {}


def load_summary_cache():
    """Load the summary cache from disk."""
    global summary_cache
    try:
        if os.path.exists(SUMMARY_CACHE_FILE):
            with open(SUMMARY_CACHE_FILE, "rb") as f:
                summary_cache = pickle.load(f)
                logger.info(
                    f"[SUMMARY] Loaded {len(summary_cache)} summaries from cache"
                )
        else:
            summary_cache = {}
            logger.info("[SUMMARY] No summary cache found, starting with empty cache")
    except Exception as e:
        logger.error(f"[SUMMARY] Error loading summary cache: {e}")
        summary_cache = {}


def save_summary_cache():
    """Save the summary cache to disk."""
    try:
        os.makedirs(os.path.dirname(SUMMARY_CACHE_FILE), exist_ok=True)
        with open(SUMMARY_CACHE_FILE, "wb") as f:
            pickle.dump(summary_cache, f)
        logger.info(f"[SUMMARY] Saved {len(summary_cache)} summaries to cache")
    except Exception as e:
        logger.error(f"[SUMMARY] Error saving summary cache: {e}")


def get_content_hash(content):
    """Generate a hash for the content to use as a cache key."""
    return hashlib.md5(content.encode("utf-8")).hexdigest()


def get_cached_summary(content):
    """Get a summary from the cache if it exists."""
    if not content:
        return None

    content_hash = get_content_hash(content)
    return summary_cache.get(content_hash)


def cache_summary(content, summary):
    """Add a summary to the cache."""
    if not content or not summary:
        return

    content_hash = get_content_hash(content)
    summary_cache[content_hash] = summary

    # Periodically save the cache to disk (every 100 new entries)
    if len(summary_cache) % 100 == 0:
        save_summary_cache()


def get_vector_store_state():
    """Get the state of the vector store from the state file."""
    if os.path.exists(VECTOR_STORE_STATE_FILE):
        try:
            with open(VECTOR_STORE_STATE_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"[RAG-INIT] Error reading vector store state: {e}")
    return {}


def save_vector_store_state(state):
    """Save the state of the vector store to the state file."""
    os.makedirs(os.path.dirname(VECTOR_STORE_STATE_FILE), exist_ok=True)
    try:
        with open(VECTOR_STORE_STATE_FILE, "w") as f:
            json.dump(state, f)
    except Exception as e:
        logger.error(f"[RAG-INIT] Error saving vector store state: {e}")


def rerank_documents(
    query: str, documents: List[Dict[str, Any]], top_n: int = 5
) -> List[Dict[str, Any]]:
    """
    Rerank documents using Cohere's reranking model.

    Args:
        query: The user query
        documents: List of document dictionaries with 'content' field
        top_n: Number of top documents to return

    Returns:
        List of reranked documents with updated scores
    """
    if not cohere_client:
        logger.warning("Cohere client not initialized. Skipping reranking.")
        return documents[:top_n]

    try:
        # Extract document texts for reranking
        docs_for_rerank = [doc["content"] for doc in documents]

        # Call Cohere's rerank API
        rerank_results = cohere_client.rerank(
            query=query,
            documents=docs_for_rerank,
            top_n=top_n,
            model="rerank-english-v3.0",
        )

        # Create a new list of reranked documents
        reranked_docs = []
        for result in rerank_results.results:
            # Get the original document
            original_doc = documents[result.index]

            # Create a new document with the reranked score
            reranked_doc = original_doc.copy()
            reranked_doc["score"] = result.relevance_score
            reranked_docs.append(reranked_doc)

        logger.info(f"Reranked {len(reranked_docs)} documents using Cohere")
        return reranked_docs

    except Exception as e:
        logger.error(f"Error during reranking: {str(e)}", exc_info=True)
        # Fallback to original ranking
        return documents[:top_n]


class ChatMessage(BaseModel):
    """A message in a chat conversation."""

    id: Optional[str] = None
    role: str
    content: str
    timestamp: Optional[str] = None
    sources: Optional[List[str]] = None
    used_chunks: Optional[List[Dict[str, Any]]] = None

    @validator("content")
    def content_must_be_valid(cls, v):
        """Validate that content is not None and is a string."""
        if v is None:
            raise ValueError("content cannot be None")
        if not isinstance(v, str):
            raise ValueError("content must be a string")
        return v


class ChatHistory(BaseModel):
    """A chat history containing messages and metadata."""

    id: str
    title: str
    timestamp: str
    messages: List[ChatMessage]
    client_id: str


class ChatRequest(BaseModel):
    """Request model for creating a new chat or adding messages."""

    messages: List[ChatMessage]
    temperature: float = 0.7
    chat_id: Optional[str] = None
    client_id: str


class ChatResponse(BaseModel):
    """Response model for chat operations."""

    response: str
    sources: Optional[List[str]] = None
    used_chunks: Optional[List[Dict[str, Any]]] = None
    chat_id: str
    message_id: str


class ChatMessageRequest(BaseModel):
    """Request model for adding a new message to an existing chat."""

    role: str
    content: str
    id: Optional[str] = None
    client_id: Optional[str] = None
    chat_id: Optional[str] = None  # This should match the path parameter


class ChatMessageResponse(BaseModel):
    """Response model for message operations."""

    response: str
    sources: Optional[List[str]] = None
    used_chunks: Optional[List[Dict[str, Any]]] = None
    message_id: str
    user_message_id: str


class ChatHistoryResponse(BaseModel):
    """Response model for retrieving a chat history."""

    chat_id: str
    title: str
    messages: List[ChatMessage]


class ChatHistoriesResponse(BaseModel):
    """Response model for retrieving multiple chat histories."""

    histories: List[ChatHistory]
    total_count: int


class StatusResponse(BaseModel):
    """Generic response model for operation status."""

    status: str
    message: str


class Feedback(BaseModel):
    """Model for user feedback on chat responses."""

    message_id: str
    query: str
    response: str
    rating: bool
    feedback_text: Optional[str] = None
    category: Optional[str] = None
    used_chunks: Optional[List[Dict[str, str]]] = None
    timestamp: Optional[str] = None


# Global variables for RAG components
embeddings: Optional[OpenAIEmbeddings] = None
vector_store: Optional[FAISS] = None
bm25_index = None
bm25_corpus = []
bm25_doc_mapping = []  # Maps BM25 index positions to actual documents


def preprocess_text(text):
    """
    Preprocess text for BM25 indexing.

    Args:
        text: The text to preprocess

    Returns:
        Preprocessed text as a string (not tokens)
    """
    if not text or not isinstance(text, str):
        return ""

    try:
        # Basic cleaning
        text = text.lower()

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text)

        # Remove URLs
        text = re.sub(r"http\S+|www\S+|https\S+", "", text)

        # Remove special characters but keep spaces
        text = re.sub(r"[^\w\s]", "", text)

        return text.strip()
    except Exception as e:
        logger.warning(f"Error preprocessing text: {e}")
        # Return a basic cleaned version as fallback
        return text.lower().strip() if text else ""


def build_bm25_index(documents):
    """
    Build a BM25 index from the documents for keyword-based search.

    Args:
        documents: List of Document objects to index
    """
    global bm25_index, bm25_corpus, bm25_doc_mapping

    try:
        logger.info("[RAG-BM25] Building BM25 index...")

        # Extract text and IDs from documents
        corpus = []
        doc_ids = []

        for i, doc in enumerate(documents):
            if hasattr(doc, "page_content") and doc.page_content:
                # Preprocess the text
                processed_text = preprocess_text(doc.page_content)
                if processed_text:
                    corpus.append(processed_text)
                    doc_ids.append(i)

        if not corpus:
            logger.warning("[RAG-BM25] No valid documents for BM25 indexing")
            return

        try:
            # Tokenize the documents
            tokenized_corpus = [word_tokenize(doc.lower()) for doc in corpus]

            # Remove stopwords
            stop_words = set(stopwords.words("english"))
            tokenized_corpus = [
                [
                    word
                    for word in doc
                    if word not in stop_words and word not in string.punctuation
                ]
                for doc in tokenized_corpus
            ]

            # Create the BM25 index
            bm25_index = BM25Okapi(tokenized_corpus)
            bm25_corpus = corpus
            bm25_doc_mapping = doc_ids

            logger.info(
                f"[RAG-BM25] Successfully built BM25 index with {len(corpus)} documents"
            )
        except LookupError as e:
            # Handle NLTK resource errors gracefully
            logger.warning(f"[RAG-BM25] NLTK resource error: {e}")
            logger.warning(
                "[RAG-BM25] Attempting to download required NLTK resources..."
            )

            # Try to download the required resources
            nltk.download("punkt", quiet=True)
            nltk.download("stopwords", quiet=True)
            nltk.download("punkt_tab", quiet=True)

            # Retry with simpler tokenization if word_tokenize fails
            tokenized_corpus = []
            for doc in corpus:
                try:
                    tokens = word_tokenize(doc.lower())
                except:
                    # Fallback to simple splitting if word_tokenize fails
                    tokens = doc.lower().split()

                # Remove stopwords
                tokens = [
                    word
                    for word in tokens
                    if word not in stop_words and word not in string.punctuation
                ]
                tokenized_corpus.append(tokens)

            # Create the BM25 index with the fallback tokenization
            bm25_index = BM25Okapi(tokenized_corpus)
            bm25_corpus = corpus
            bm25_doc_mapping = doc_ids

            logger.info(
                f"[RAG-BM25] Successfully built BM25 index with fallback tokenization for {len(corpus)} documents"
            )

    except Exception as e:
        logger.error(f"[RAG-BM25] Error building BM25 index: {e}")
        bm25_index = None
        bm25_corpus = []
        bm25_doc_mapping = []


def bm25_search(query, k=10):
    """Search the BM25 index for relevant documents."""
    global bm25_index, bm25_doc_mapping

    if not bm25_index:
        logger.warning("[RAG-BM25] BM25 index not initialized")
        return []

    # Preprocess the query
    query_tokens = preprocess_text(query)

    if not query_tokens:
        logger.warning("[RAG-BM25] Empty query after preprocessing")
        return []

    # Get BM25 scores
    scores = bm25_index.get_scores(query_tokens)

    # Get top k documents
    top_n = min(k, len(scores))
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[
        :top_n
    ]

    results = []
    for idx in top_indices:
        if scores[idx] > 0:  # Only include documents with positive scores
            doc_idx = bm25_doc_mapping[idx]
            results.append((doc_idx, scores[idx]))

    return results


def batch_generate_summaries(
    contents: List[str], max_length: int = 100, batch_size: int = 10
) -> List[str]:
    """
    Generate summaries for a batch of document chunks.

    Args:
        contents: List of document chunk contents to summarize
        max_length: Maximum length of each summary in characters
        batch_size: Number of summaries to generate in parallel

    Returns:
        List of summaries corresponding to the input contents
    """
    if not contents:
        return []

    logger.info(f"[SUMMARY] Batch processing {len(contents)} documents for summaries")

    # Initialize results list
    summaries = []

    # Track which items need to be processed
    to_process = []
    to_process_indices = []

    # Check cache first for all contents
    for i, content in enumerate(contents):
        if not content or len(content) < 100:
            # For very short content, just use it as is
            summaries.append(content)
        else:
            cached_summary = get_cached_summary(content)
            if cached_summary:
                summaries.append(cached_summary)
            else:
                # Mark for processing
                to_process.append(content)
                to_process_indices.append(i)
                # Add a placeholder
                summaries.append(None)

    # If nothing to process, return early
    if not to_process:
        logger.info("[SUMMARY] All summaries found in cache")
        return summaries

    logger.info(
        f"[SUMMARY] Generating {len(to_process)} summaries (cache hit rate: {(len(contents) - len(to_process)) / len(contents):.1%})"
    )

    # Process in batches
    for i in range(0, len(to_process), batch_size):
        batch = to_process[i : i + batch_size]
        batch_indices = to_process_indices[i : i + batch_size]

        logger.info(
            f"[SUMMARY] Processing batch {i//batch_size + 1}/{(len(to_process) + batch_size - 1) // batch_size}"
        )

        # Process each item in the batch
        for j, content in enumerate(batch):
            try:
                # Generate summary
                summary = generate_chunk_summary(content, max_length)

                # Update the results list
                summaries[batch_indices[j]] = summary

            except Exception as e:
                logger.error(f"[SUMMARY] Error generating summary in batch: {e}")
                # Use a fallback summary
                fallback = (
                    content[: max_length - 3] + "..."
                    if len(content) > max_length
                    else content
                )
                summaries[batch_indices[j]] = fallback

    # Save the cache after processing all batches
    save_summary_cache()

    return summaries


def initialize_models():
    """Initialize the RAG components."""
    global embeddings, vector_store, bm25_index

    try:
        logger.info("[RAG-INIT] Starting RAG components initialization...")

        # Load the summary cache
        load_summary_cache()

        # Initialize embeddings with OpenAI API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("[RAG-INIT] OpenAI API key not found in environment variables")
            raise ValueError("OpenAI API key not found")

        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        logger.info("[RAG-INIT] OpenAI embeddings initialized successfully")

        # Initialize vector store directory
        vector_store_path = "data/faiss"
        rebuild_vectorstore = False

        # Get current state
        state = get_vector_store_state()

        # Check if vector store exists and if rebuild is needed
        if not os.path.exists(vector_store_path) or not os.listdir(vector_store_path):
            logger.info(
                f"[RAG-INIT] Vector store not found or empty, will create new one at {vector_store_path}"
            )
            os.makedirs(vector_store_path, exist_ok=True)
            rebuild_vectorstore = True
        else:
            # Check if the vector store is empty or corrupted
            try:
                logger.info(
                    f"[RAG-INIT] Attempting to load existing vector store from {vector_store_path}"
                )
                vector_store = FAISS.load_local(
                    folder_path=vector_store_path,
                    embeddings=embeddings,
                    allow_dangerous_deserialization=True,
                )
                # Try to do a simple search to verify the vector store is working
                vector_store.similarity_search("test", k=1)
                logger.info("[RAG-INIT] Existing vector store loaded successfully")

                # Build BM25 index from the loaded vector store
                if hasattr(vector_store, "docstore") and hasattr(
                    vector_store.docstore, "_dict"
                ):
                    all_docs = list(vector_store.docstore._dict.values())
                    build_bm25_index(all_docs)
                else:
                    logger.warning(
                        "[RAG-INIT] Could not access documents for BM25 indexing"
                    )

            except Exception as e:
                logger.warning(
                    f"[RAG-INIT] Existing vector store appears corrupted or empty, will rebuild: {str(e)}"
                )
                rebuild_vectorstore = True
                import shutil

                shutil.rmtree(vector_store_path)
                os.makedirs(vector_store_path)

        if rebuild_vectorstore:
            logger.info("[RAG-INIT] Creating new vector store")
            # Initialize with a dummy document
            from langchain_core.documents import Document

            dummy_doc = Document(
                page_content="Initialization document for FAISS vector store"
            )
            vector_store = FAISS.from_documents([dummy_doc], embeddings)

            # Save the initial vector store
            vector_store.save_local(vector_store_path)
            logger.info(f"[RAG-INIT] Initial vector store saved to {vector_store_path}")

            # Check documentation directory
            docs_parent_dir = "data/developer-docs"
            docs_dir = f"{docs_parent_dir}/apps/nextra/pages/en"

            if not os.path.exists(docs_dir):
                logger.error(
                    f"[RAG-INIT] Documentation directory not found at {docs_dir}. Please ensure it is mounted correctly."
                )
                raise ValueError("Documentation directory not found")

            logger.info(
                "[RAG-INIT] Documentation directory found, processing documents..."
            )

            # Load documentation and save state
            load_aptos_docs(docs_dir)
            save_vector_store_state(
                {"last_processed": datetime.now().isoformat(), "docs_processed": True}
            )
            logger.info("[RAG-INIT] Vector store built and persisted successfully")
            logger.info("[RAG-INIT] RAG initialization completed successfully")

    except Exception as e:
        logger.error(
            f"[RAG-INIT] Error initializing RAG components: {str(e)}", exc_info=True
        )
        raise


def get_relevant_context(
    query: str, k: int = 5, include_series: bool = True
) -> List[Dict[str, str]]:
    """
    Get relevant context from the documentation using a hybrid retrieval approach.
    Returns a list of dictionaries containing content, section, source, and summary information.

    Args:
        query: The user query
        k: Number of top documents to return
        include_series: Whether to include related documents from the same series
    """
    if not query:
        logger.warning("Empty query received in get_relevant_context")
        return []

    if not vector_store:
        logger.error("Vector store not initialized. RAG functionality is disabled.")
        return []

    try:
        logger.info(f"Processing query for RAG: {query}")

        # First, get a larger initial set of candidates using semantic search
        initial_k = k * 3  # Get more candidates initially
        docs_with_scores = vector_store.similarity_search_with_score(query, k=initial_k)

        # Also get BM25 search results
        bm25_results = bm25_search(query, k=initial_k)
        logger.info(
            f"[RAG-BM25] Retrieved {len(bm25_results)} results from BM25 search"
        )

        # Create a set to track documents we've seen
        seen_doc_ids = set()

        # Process vector search results
        scored_docs = []
        seen_contents = set()

        # Process vector search results
        for doc, semantic_score in docs_with_scores:
            if not doc or not hasattr(doc, "page_content") or not doc.page_content:
                continue

            metadata = doc.metadata or {}
            source = metadata.get("source", "")
            section = metadata.get("section", "")
            content = doc.page_content.strip()
            summary = metadata.get("summary", "")  # Get the summary from metadata

            # Get series information
            is_part_of_series = metadata.get("is_part_of_series", False)
            series_title = metadata.get("series_title", "")
            series_position = metadata.get("series_position", 0)
            total_steps = metadata.get("total_steps", 0)

            if not content or not isinstance(content, str):
                continue

            # Skip if we've seen this content before
            content_hash = hash(content)
            if content_hash in seen_contents:
                continue
            seen_contents.add(content_hash)

            # Track this document ID if available
            if hasattr(doc, "id"):
                seen_doc_ids.add(doc.id)

            # Calculate additional relevance signals
            keywords = set(query.lower().split())
            doc_words = set(content.lower().split())
            keyword_overlap = (
                len(keywords.intersection(doc_words)) / len(keywords) if keywords else 0
            )

            # Check for keyword overlap in summary if available
            summary_score = 0.0
            if summary:
                summary_words = set(summary.lower().split())
                summary_overlap = (
                    len(keywords.intersection(summary_words)) / len(keywords)
                    if keywords
                    else 0
                )
                # Summary overlap is weighted higher since summaries are more concentrated
                summary_score = summary_overlap * 1.5
                logger.debug(f"Summary score for document: {summary_score:.3f}")

            section_score = 1.0
            section_keywords = set(section.lower().split("/"))
            if any(kw in section_keywords for kw in keywords):
                section_score = 1.2

            content_length = len(content.split())
            length_score = 1.0
            if 50 <= content_length <= 200:
                length_score = 1.1

            context_score = 1.1 if content.startswith("Context:") else 1.0

            # Add series boost for documents that are part of a series
            series_score = 1.0
            if is_part_of_series:
                # Check if query is asking about a process or steps
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
                if any(kw in query.lower() for kw in process_keywords):
                    series_score = 1.3
                    logger.debug(
                        f"Applied series boost for document in '{series_title}' series"
                    )

                # Give higher score to first documents in a series for overview questions
                if series_position == 1 and any(
                    kw in query.lower()
                    for kw in ["overview", "introduction", "what is", "how to"]
                ):
                    series_score *= 1.2
                    logger.debug(
                        f"Applied first-in-series boost for document in '{series_title}' series"
                    )

            # Normalize FAISS score (lower is better in FAISS)
            # Convert to a 0-1 scale where 1 is best
            normalized_score = 1.0 / (1.0 + semantic_score)

            # BM25 score is initially 0
            bm25_score = 0.0

            # Include summary score in the final calculation
            final_score = (
                normalized_score * 0.25  # Reduced weight for semantic score
                + keyword_overlap * 0.15  # Reduced weight for content overlap
                + summary_score * 0.15  # Weight for summary relevance
                + section_score * 0.10
                + length_score * 0.10
                + context_score * 0.05
                + series_score * 0.10  # Weight for series relevance
                + bm25_score
                * 0.10  # Initial BM25 score (will be updated for BM25 results)
            )

            scored_docs.append(
                {
                    "content": content,
                    "section": section,
                    "source": source,
                    "summary": summary,  # Include summary in the result
                    "score": final_score,
                    "doc_id": getattr(
                        doc, "id", None
                    ),  # Store document ID if available
                    "vector_score": normalized_score,
                    "bm25_score": 0.0,  # Initial BM25 score
                    # Add series information
                    "is_part_of_series": is_part_of_series,
                    "series_title": series_title,
                    "series_position": series_position,
                    "total_steps": total_steps,
                    "parent_document": metadata.get("parent_document", ""),
                    "next_document": metadata.get("next_document", ""),
                    "previous_document": metadata.get("previous_document", ""),
                }
            )

        # Process BM25 search results
        if bm25_results:
            # Get all documents from the vector store
            if hasattr(vector_store, "docstore") and hasattr(
                vector_store.docstore, "_dict"
            ):
                all_docs = list(vector_store.docstore._dict.values())

                # Add BM25 results to scored_docs
                for doc_idx, bm25_score in bm25_results:
                    if doc_idx < len(all_docs):
                        doc = all_docs[doc_idx]

                        # Skip if we've already processed this document from vector search
                        if hasattr(doc, "id") and doc.id in seen_doc_ids:
                            # Find the existing entry and update its BM25 score
                            for existing_doc in scored_docs:
                                if existing_doc.get("doc_id") == doc.id:
                                    # Normalize BM25 score (typically 0-15 range)
                                    normalized_bm25 = min(1.0, bm25_score / 15.0)
                                    existing_doc["bm25_score"] = normalized_bm25

                                    # Recalculate final score with BM25 component
                                    existing_doc["score"] = (
                                        existing_doc["vector_score"] * 0.30
                                        + (
                                            existing_doc["score"]
                                            - existing_doc["vector_score"] * 0.30
                                            - 0.0 * 0.10
                                        )  # Remove old components
                                        + normalized_bm25 * 0.10  # Add BM25 component
                                    )
                                    break
                            continue

                        if (
                            not doc
                            or not hasattr(doc, "page_content")
                            or not doc.page_content
                        ):
                            continue

                        metadata = doc.metadata or {}
                        source = metadata.get("source", "")
                        section = metadata.get("section", "")
                        content = doc.page_content.strip()
                        summary = metadata.get("summary", "")

                        if not content or not isinstance(content, str):
                            continue

                        # Skip if we've seen this content before
                        content_hash = hash(content)
                        if content_hash in seen_contents:
                            continue
                        seen_contents.add(content_hash)

                        # Track this document ID
                        if hasattr(doc, "id"):
                            seen_doc_ids.add(doc.id)

                        # Calculate other relevance signals (similar to vector search)
                        keywords = set(query.lower().split())
                        doc_words = set(content.lower().split())
                        keyword_overlap = (
                            len(keywords.intersection(doc_words)) / len(keywords)
                            if keywords
                            else 0
                        )

                        summary_score = 0.0
                        if summary:
                            summary_words = set(summary.lower().split())
                            summary_overlap = (
                                len(keywords.intersection(summary_words))
                                / len(keywords)
                                if keywords
                                else 0
                            )
                            summary_score = summary_overlap * 1.5

                        section_score = 1.0
                        section_keywords = set(section.lower().split("/"))
                        if any(kw in section_keywords for kw in keywords):
                            section_score = 1.2

                        content_length = len(content.split())
                        length_score = 1.0
                        if 50 <= content_length <= 200:
                            length_score = 1.1

                        context_score = 1.1 if content.startswith("Context:") else 1.0

                        # Normalize BM25 score (typically 0-15 range)
                        normalized_bm25 = min(1.0, bm25_score / 15.0)

                        # For BM25-only results, set a lower vector score
                        vector_score = 0.3  # Lower baseline for non-vector results

                        # Calculate final score with emphasis on BM25
                        final_score = (
                            vector_score * 0.30
                            + keyword_overlap * 0.20
                            + summary_score * 0.15
                            + section_score * 0.10
                            + length_score * 0.10
                            + context_score * 0.05
                            + normalized_bm25 * 0.10
                        )

                        scored_docs.append(
                            {
                                "content": content,
                                "section": section,
                                "source": source,
                                "summary": summary,
                                "score": final_score,
                                "doc_id": getattr(doc, "id", None),
                                "vector_score": vector_score,
                                "bm25_score": normalized_bm25,
                            }
                        )

        # Sort by final score and take top candidates
        scored_docs.sort(key=lambda x: x["score"], reverse=True)

        # Check if we need to include related documents from the same series
        if include_series:
            # Find series documents among top results
            series_docs = [
                doc for doc in scored_docs[:k] if doc.get("is_part_of_series")
            ]

            # If we have series documents, try to include related documents from the same series
            if series_docs:
                logger.info(
                    f"[RAG-SERIES] Found {len(series_docs)} documents that are part of a series"
                )

                # Track series we've seen to avoid duplicates
                processed_series = set()
                additional_docs = []

                for doc in series_docs:
                    series_title = doc.get("series_title", "")

                    # Skip if we've already processed this series
                    if not series_title or series_title in processed_series:
                        continue

                    processed_series.add(series_title)

                    # Find all documents from this series in our candidate pool
                    series_candidates = [
                        d
                        for d in scored_docs
                        if d.get("series_title") == series_title
                        and d.get("doc_id")
                        not in [doc.get("doc_id") for doc in scored_docs[:k]]
                    ]

                    # Sort by position in the series
                    series_candidates.sort(key=lambda x: x.get("series_position", 0))

                    # Add up to 2 additional documents from this series
                    # Prioritize documents that come before or after the matched document
                    current_position = doc.get("series_position", 0)

                    # First, try to add the document that comes before (if not already in top results)
                    prev_docs = [
                        d
                        for d in series_candidates
                        if d.get("series_position", 0) == current_position - 1
                    ]
                    if prev_docs:
                        additional_docs.append(prev_docs[0])

                    # Then, try to add the document that comes after (if not already in top results)
                    next_docs = [
                        d
                        for d in series_candidates
                        if d.get("series_position", 0) == current_position + 1
                    ]
                    if next_docs:
                        additional_docs.append(next_docs[0])

                    # If we still have room, add the first document in the series for context
                    if current_position > 1:
                        first_docs = [
                            d
                            for d in series_candidates
                            if d.get("series_position", 0) == 1
                        ]
                        if first_docs and len(additional_docs) < 2:
                            additional_docs.append(first_docs[0])

                # Add the additional series documents to our results
                if additional_docs:
                    logger.info(
                        f"[RAG-SERIES] Adding {len(additional_docs)} related documents from the same series"
                    )

                    # Combine original top docs with additional series docs
                    combined_docs = scored_docs[:k] + additional_docs

                    # Sort by series position for documents in the same series
                    def sort_key(doc):
                        # First sort by score for non-series docs
                        if not doc.get("is_part_of_series"):
                            return (0, -doc.get("score", 0), 0)
                        # Then sort by series title and position for series docs
                        return (
                            1,
                            doc.get("series_title", ""),
                            doc.get("series_position", 0),
                        )

                    combined_docs.sort(key=sort_key)

                    # Limit to k+2 documents maximum
                    top_docs = combined_docs[: k + 2]
                else:
                    # Fallback to original top k
                    top_docs = scored_docs[:k]
            else:
                # No series documents found, use original top k
                top_docs = scored_docs[:k]
        else:
            # Series inclusion disabled, use original top k
            top_docs = scored_docs[:k]

        # Apply Cohere reranking if available
        if cohere_client:
            logger.info(
                f"Applying Cohere reranking to {len(top_docs)} candidate documents"
            )
            # Only rerank the top candidates from our hybrid search
            top_docs = rerank_documents(
                query, top_docs, top_n=min(k + 2, len(top_docs))
            )

        # Log retrieval metrics
        if top_docs:
            avg_score = sum(doc.get("score", 0) for doc in top_docs) / len(top_docs)
            logger.info(
                f"Retrieved {len(top_docs)} documents with average score: {avg_score:.3f}"
            )
            logger.debug("Top document sections retrieved:")
            for i, doc in enumerate(top_docs[:3]):
                logger.debug(
                    f"  {i+1}. {doc['section']} (score: {doc.get('score', 0):.3f})"
                )
                if doc.get("is_part_of_series"):
                    logger.debug(
                        f"     Series: {doc.get('series_title')} (Part {doc.get('series_position')}/{doc.get('total_steps')})"
                    )
                if doc.get("bm25_score", 0) > 0:
                    logger.debug(f"     BM25 score: {doc.get('bm25_score', 0):.3f}")
                if doc.get("summary"):
                    logger.debug(f"     Summary: {doc.get('summary')}")

        # Clean up the results before returning (remove internal scoring details)
        for doc in top_docs:
            doc.pop("doc_id", None)
            doc.pop("vector_score", None)
            doc.pop("bm25_score", None)

        return top_docs

    except Exception as e:
        logger.error(f"Error retrieving context: {str(e)}", exc_info=True)
        return []


def extract_frontmatter(content: str) -> tuple[Dict[str, Any], str]:
    """Extract frontmatter from MDX/MD content and return both metadata and content."""
    frontmatter_match = re.match(r"^---\n(.*?)\n---\n(.*)", content, re.DOTALL)
    if frontmatter_match:
        try:
            metadata = yaml.safe_load(frontmatter_match.group(1))
            content = frontmatter_match.group(2)
            return metadata, content
        except yaml.YAMLError:
            return {}, content
    return {}, content


def process_markdown_document(content: str) -> List[str]:
    """Process markdown/MDX content and split it into sections based on headers."""
    # Extract frontmatter if present
    metadata, clean_content = extract_frontmatter(content)

    if not clean_content or not isinstance(clean_content, str):
        logger.warning(f"Invalid content type or empty content: {type(clean_content)}")
        return []

    clean_content = clean_content.strip()
    if not clean_content:
        return []

    # Remove JSX/TSX imports and exports but preserve important content
    clean_content = re.sub(r"import.*?;\n", "", clean_content, flags=re.MULTILINE)
    clean_content = re.sub(r"export.*?}\n", "", clean_content, flags=re.MULTILINE)

    # More carefully handle JSX/TSX components to preserve content
    clean_content = re.sub(
        r"<Callout.*?>(.*?)</Callout>", r"Important: \1", clean_content, flags=re.DOTALL
    )
    clean_content = re.sub(
        r"<Steps.*?>(.*?)</Steps>", r"Steps: \1", clean_content, flags=re.DOTALL
    )
    clean_content = re.sub(
        r"<Card.*?>(.*?)</Card>", r"\1", clean_content, flags=re.DOTALL
    )

    # Remove remaining JSX tags but preserve their content
    clean_content = re.sub(
        r"<[^>]+/>", "", clean_content
    )  # Remove self-closing components
    clean_content = re.sub(
        r"</?[^>]+>", "", clean_content
    )  # Remove tags but keep content

    # Clean up any double newlines or spaces created by removals
    clean_content = re.sub(r"\n{3,}", "\n\n", clean_content)
    clean_content = re.sub(r" {2,}", " ", clean_content)
    clean_content = clean_content.strip()

    if not clean_content:
        return []

    # Split on headers with more granular control
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on
    )

    try:
        splits = markdown_splitter.split_text(clean_content)
        # If no headers found, use a size-based splitter with smaller chunks and more overlap
        if not splits:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,  # Smaller chunks
                chunk_overlap=100,  # More overlap to maintain context
                separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", ". ", " ", ""],
            )
            chunks = text_splitter.split_text(clean_content)

            # Prepare content for batch summary generation
            chunk_contents = [
                chunk.strip()
                for chunk in chunks
                if chunk and isinstance(chunk, str) and chunk.strip()
            ]

            # Generate summaries in batch
            if chunk_contents:
                summaries = batch_generate_summaries(chunk_contents)

                # Create Document objects with summaries
                valid_splits = []
                for i, chunk_content in enumerate(chunk_contents):
                    from langchain_core.documents import Document

                    doc = Document(
                        page_content=chunk_content,
                        metadata={
                            "summary": summaries[i] if i < len(summaries) else ""
                        },
                    )
                    valid_splits.append(doc)
                return valid_splits
            return []

        # Process header-based splits
        valid_splits = []
        contents_to_summarize = []
        content_metadata = []

        # First pass: collect content and prepare metadata
        for split in splits:
            if split.page_content and isinstance(split.page_content, str):
                content = split.page_content.strip()
                if content:
                    # Build a rich context header
                    header_context = []
                    for level in ["Header 1", "Header 2", "Header 3", "Header 4"]:
                        if split.metadata.get(level):
                            header_context.append(split.metadata[level])

                    # Add metadata from frontmatter if relevant
                    if metadata.get("title"):
                        header_context.insert(0, metadata["title"])
                    if metadata.get("description"):
                        content = f"{metadata['description']}\n\n{content}"

                    # Combine headers into a context string
                    if header_context:
                        content = f"Context: {' > '.join(header_context)}\n\n{content}"

                    # Store content for batch summarization
                    contents_to_summarize.append(content)
                    # Store metadata for later use
                    content_metadata.append(
                        {
                            **{
                                k: v
                                for k, v in split.metadata.items()
                                if k != "page_content"
                            }
                        }
                    )

        # Generate summaries in batch
        if contents_to_summarize:
            summaries = batch_generate_summaries(contents_to_summarize)

            # Second pass: create Document objects with summaries
            for i, content in enumerate(contents_to_summarize):
                from langchain_core.documents import Document

                doc = Document(
                    page_content=content,
                    metadata={
                        "summary": summaries[i] if i < len(summaries) else "",
                        **content_metadata[i],
                    },
                )
                valid_splits.append(doc)

        return valid_splits
    except Exception as e:
        logger.error(f"Error splitting markdown content: {e}")
        # Fall back to size-based splitting if header splitting fails
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", ". ", " ", ""],
        )
        chunks = text_splitter.split_text(clean_content)

        # Prepare content for batch summary generation
        chunk_contents = [
            chunk.strip()
            for chunk in chunks
            if chunk and isinstance(chunk, str) and chunk.strip()
        ]

        # Generate summaries in batch
        if chunk_contents:
            summaries = batch_generate_summaries(chunk_contents)

            # Create Document objects with summaries
            valid_splits = []
            for i, chunk_content in enumerate(chunk_contents):
                from langchain_core.documents import Document

                doc = Document(
                    page_content=chunk_content,
                    metadata={"summary": summaries[i] if i < len(summaries) else ""},
                )
                valid_splits.append(doc)
            return valid_splits
        return []


def analyze_document_relationships(docs_dir: str) -> Dict[str, Dict]:
    """
    Analyze document relationships to identify document series and hierarchies.

    Args:
        docs_dir: The root directory of the documentation

    Returns:
        A dictionary mapping document paths to their relationship information
    """
    logger.info("[RAG-DOCS] Analyzing document relationships")
    document_map = {}
    series_map = {}

    # First pass: collect all documents and their basic info
    for root, _, files in os.walk(docs_dir):
        for file in files:
            if file.endswith((".md", ".mdx")) and not file.endswith("_meta.ts"):
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, docs_dir)
                section_path = os.path.dirname(relative_path)

                # Check for _meta.ts files in the same directory for series information
                meta_file = os.path.join(root, "_meta.ts")
                is_part_of_series = False
                series_title = ""
                series_position = 0
                total_steps = 0

                if os.path.exists(meta_file):
                    try:
                        with open(meta_file, "r", encoding="utf-8") as f:
                            meta_content = f.read()

                        # Extract series information from meta file
                        # This is a simple heuristic - adjust based on your meta file format
                        if (
                            "installation" in meta_content.lower()
                            or "guide" in meta_content.lower()
                            or "tutorial" in meta_content.lower()
                        ):
                            is_part_of_series = True

                            # Try to extract series title
                            title_match = re.search(
                                r'title: ["\'](.+?)["\']', meta_content
                            )
                            if title_match:
                                series_title = title_match.group(1)
                            else:
                                # Fallback to directory name
                                series_title = (
                                    os.path.basename(section_path)
                                    .replace("-", " ")
                                    .title()
                                )
                    except Exception as e:
                        logger.error(
                            f"[RAG-DOCS] Error processing meta file {meta_file}: {e}"
                        )

                # Store document info
                document_map[relative_path] = {
                    "section": (
                        section_path.replace(os.path.sep, "/")
                        if section_path != "."
                        else "root"
                    ),
                    "is_part_of_series": is_part_of_series,
                    "series_title": series_title,
                    "series_position": 0,  # Will be updated in second pass
                    "total_steps": 0,  # Will be updated in second pass
                    "parent_document": None,
                    "related_documents": [],
                }

                # Group by section for series detection
                if section_path not in series_map:
                    series_map[section_path] = []
                series_map[section_path].append(relative_path)

    # Second pass: identify series and positions
    for section, documents in series_map.items():
        if len(documents) > 1:
            # Sort documents to determine sequence
            sorted_docs = sorted(documents)

            # Check if this section is likely a series
            is_series = False

            # Check if any document in this section is marked as part of a series
            if any(document_map[doc]["is_part_of_series"] for doc in sorted_docs):
                is_series = True

            # Check for numeric prefixes in filenames (e.g., 01-intro.md, 02-setup.md)
            numeric_prefixes = [
                re.match(r"^\d+", os.path.basename(doc)) for doc in sorted_docs
            ]
            if any(numeric_prefixes):
                is_series = True

            # Check for sequential terms like "step", "part", etc.
            sequential_terms = ["step", "part", "chapter", "section", "phase", "stage"]
            if any(
                term in doc.lower() for term in sequential_terms for doc in sorted_docs
            ):
                is_series = True

            if is_series:
                # Get series title from the first document if not already set
                if not document_map[sorted_docs[0]]["series_title"]:
                    series_title = os.path.basename(section).replace("-", " ").title()
                    if series_title == ".":
                        series_title = "Root Documentation Series"
                else:
                    series_title = document_map[sorted_docs[0]]["series_title"]

                # Update all documents in this series
                for i, doc in enumerate(sorted_docs):
                    document_map[doc].update(
                        {
                            "is_part_of_series": True,
                            "series_title": series_title,
                            "series_position": i + 1,
                            "total_steps": len(sorted_docs),
                            "related_documents": sorted_docs,
                        }
                    )

                    # Set parent-child relationships
                    if i > 0:
                        document_map[doc]["parent_document"] = sorted_docs[i - 1]

                    # Add next document reference
                    if i < len(sorted_docs) - 1:
                        document_map[doc]["next_document"] = sorted_docs[i + 1]

                    # Add previous document reference
                    if i > 0:
                        document_map[doc]["previous_document"] = sorted_docs[i - 1]

    logger.info(
        f"[RAG-DOCS] Analyzed {len(document_map)} documents, found {sum(1 for doc in document_map.values() if doc['is_part_of_series'])} in series"
    )
    return document_map


def load_aptos_docs(docs_dir: str = "data/developer-docs/apps/nextra/pages/en") -> None:
    """Load and process Aptos documentation from the English documentation directory."""
    global vector_store

    if not os.path.exists(docs_dir):
        logger.error(f"[RAG-DOCS] Documentation directory not found: {docs_dir}")
        return

    if vector_store is None:
        logger.warning("[RAG-DOCS] Vector store not initialized, cannot load documents")
        return

    try:
        logger.info("[RAG-DOCS] Starting documentation processing")

        # Analyze document relationships first
        document_map = analyze_document_relationships(docs_dir)

        documents = []
        file_count = 0
        processed_count = 0

        for root, _, files in os.walk(docs_dir):
            for file in files:
                if file.endswith((".md", ".mdx")) and not file.endswith("_meta.ts"):
                    file_count += 1
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()

                        # Extract section info from file path
                        relative_path = os.path.relpath(file_path, docs_dir)
                        section_path = os.path.dirname(relative_path)
                        section = (
                            section_path.replace(os.path.sep, "/")
                            if section_path != "."
                            else "root"
                        )

                        # Get document relationship info
                        doc_info = document_map.get(relative_path, {})

                        # Process the document
                        doc_sections = process_markdown_document(content)
                        processed_count += 1

                        # Add each section to documents with metadata
                        for doc_section in doc_sections:
                            if doc_section:
                                try:
                                    # Check if doc_section is already a Document object
                                    from langchain_core.documents import Document

                                    if isinstance(doc_section, Document):
                                        # Update metadata with file information and relationship info
                                        doc_section.metadata.update(
                                            {
                                                "source": relative_path,
                                                "section": section,
                                                "file_type": (
                                                    "mdx"
                                                    if file.endswith(".mdx")
                                                    else "md"
                                                ),
                                                # Add document relationship information
                                                "is_part_of_series": doc_info.get(
                                                    "is_part_of_series", False
                                                ),
                                                "series_title": doc_info.get(
                                                    "series_title", ""
                                                ),
                                                "series_position": doc_info.get(
                                                    "series_position", 0
                                                ),
                                                "total_steps": doc_info.get(
                                                    "total_steps", 0
                                                ),
                                                "parent_document": doc_info.get(
                                                    "parent_document", ""
                                                ),
                                                "next_document": doc_info.get(
                                                    "next_document", ""
                                                ),
                                                "previous_document": doc_info.get(
                                                    "previous_document", ""
                                                ),
                                            }
                                        )

                                        # If this is part of a series, enhance the content with series context
                                        if doc_info.get("is_part_of_series", False):
                                            series_context = f"This document is part {doc_info.get('series_position', 0)} of {doc_info.get('total_steps', 0)} in the '{doc_info.get('series_title', '')}' series."

                                            if doc_info.get("previous_document"):
                                                series_context += f" Previous: {doc_info.get('previous_document', '')}"

                                            if doc_info.get("next_document"):
                                                series_context += f" Next: {doc_info.get('next_document', '')}"

                                            # Add series context to the beginning of the content
                                            doc_section.page_content = f"Series Context: {series_context}\n\n{doc_section.page_content}"

                                        documents.append(doc_section)
                                    else:
                                        # Handle string content (for backward compatibility)
                                        if (
                                            isinstance(doc_section, str)
                                            and doc_section.strip()
                                        ):
                                            # Create metadata including relationship info
                                            metadata = {
                                                "source": relative_path,
                                                "section": section,
                                                "file_type": (
                                                    "mdx"
                                                    if file.endswith(".mdx")
                                                    else "md"
                                                ),
                                                # Add a basic summary if not already processed
                                                "summary": generate_chunk_summary(
                                                    doc_section.strip()
                                                ),
                                                # Add document relationship information
                                                "is_part_of_series": doc_info.get(
                                                    "is_part_of_series", False
                                                ),
                                                "series_title": doc_info.get(
                                                    "series_title", ""
                                                ),
                                                "series_position": doc_info.get(
                                                    "series_position", 0
                                                ),
                                                "total_steps": doc_info.get(
                                                    "total_steps", 0
                                                ),
                                                "parent_document": doc_info.get(
                                                    "parent_document", ""
                                                ),
                                                "next_document": doc_info.get(
                                                    "next_document", ""
                                                ),
                                                "previous_document": doc_info.get(
                                                    "previous_document", ""
                                                ),
                                            }

                                            content = doc_section.strip()

                                            # If this is part of a series, enhance the content with series context
                                            if doc_info.get("is_part_of_series", False):
                                                series_context = f"This document is part {doc_info.get('series_position', 0)} of {doc_info.get('total_steps', 0)} in the '{doc_info.get('series_title', '')}' series."

                                                if doc_info.get("previous_document"):
                                                    series_context += f" Previous: {doc_info.get('previous_document', '')}"

                                                if doc_info.get("next_document"):
                                                    series_context += f" Next: {doc_info.get('next_document', '')}"

                                                # Add series context to the beginning of the content
                                                content = f"Series Context: {series_context}\n\n{content}"

                                            doc = Document(
                                                page_content=content, metadata=metadata
                                            )
                                            documents.append(doc)
                                except Exception as e:
                                    logger.error(
                                        f"[RAG-DOCS] Error creating document for section in {file_path}: {e}"
                                    )
                                    continue
                    except Exception as e:
                        logger.error(
                            f"[RAG-DOCS] Error processing file {file_path}: {e}"
                        )

        logger.info(
            f"[RAG-DOCS] Processed {processed_count}/{file_count} files, created {len(documents)} document sections"
        )

        if documents:
            try:
                logger.info(
                    f"[RAG-DOCS] Adding {len(documents)} document sections to vector store"
                )
                vector_store_path = "data/faiss"

                # Process documents in batches to avoid memory issues
                batch_size = 500
                total_batches = (len(documents) + batch_size - 1) // batch_size

                for i in range(0, len(documents), batch_size):
                    batch_end = min(i + batch_size, len(documents))
                    batch = documents[i:batch_end]
                    batch_num = (i // batch_size) + 1

                    logger.info(
                        f"[RAG-DOCS] Processing batch {batch_num}/{total_batches} with {len(batch)} documents"
                    )

                    # Create a new FAISS instance with the batch of documents
                    try:
                        batch_vector_store = FAISS.from_documents(batch, embeddings)

                        # Merge with existing vector store
                        if vector_store is not None:
                            logger.info(
                                f"[RAG-DOCS] Merging batch {batch_num} with existing vector store"
                            )
                            vector_store.merge_from(batch_vector_store)
                        else:
                            vector_store = batch_vector_store

                        # Save after each batch
                        vector_store.save_local(vector_store_path)
                        logger.info(
                            f"[RAG-DOCS] Batch {batch_num} processed and saved successfully"
                        )
                    except Exception as e:
                        logger.error(
                            f"[RAG-DOCS] Error processing batch {batch_num}: {e}"
                        )

                logger.info(
                    f"[RAG-DOCS] Successfully processed {processed_count}/{file_count} files and added {len(documents)} document sections"
                )

                # Build BM25 index after all documents are loaded
                if (
                    vector_store
                    and hasattr(vector_store, "docstore")
                    and hasattr(vector_store.docstore, "_dict")
                ):
                    all_docs = list(vector_store.docstore._dict.values())
                    build_bm25_index(all_docs)
                else:
                    logger.warning(
                        "[RAG-DOCS] Could not access documents for BM25 indexing"
                    )

            except Exception as e:
                logger.error(f"[RAG-DOCS] Error adding documents to vector store: {e}")
        else:
            logger.error(
                "[RAG-DOCS] No documents were processed from the Aptos documentation"
            )
    except Exception as e:
        logger.error(f"[RAG-DOCS] Error processing documentation: {e}")


def generate_chunk_summary(content: str, max_length: int = 100) -> str:
    """
    Generate a concise summary of a document chunk using OpenAI.
    Uses a cache to avoid regenerating summaries for unchanged content.

    Args:
        content: The document chunk content to summarize
        max_length: Maximum length of the summary in characters

    Returns:
        A concise summary of the document chunk
    """
    if not content or len(content) < 100:
        # For very short content, just return it as is
        return content

    # Check if we have a cached summary
    cached_summary = get_cached_summary(content)
    if cached_summary:
        logger.debug(f"[SUMMARY] Using cached summary: {cached_summary[:30]}...")
        return cached_summary

    try:
        # Use OpenAI to generate a summary
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",  # Using a smaller model for efficiency
            messages=[
                {
                    "role": "system",
                    "content": f"Create a concise summary (maximum {max_length} characters) that captures the key information in this text. Focus on the main concepts, definitions, or procedures.",
                },
                {"role": "user", "content": content},
            ],
            max_tokens=100,  # Limit token usage
            temperature=0.3,  # Lower temperature for more focused summaries
        )

        summary = response.choices[0].message.content.strip()

        # Ensure the summary isn't too long
        if len(summary) > max_length:
            summary = summary[: max_length - 3] + "..."

        logger.debug(f"Generated summary: {summary}")

        # Cache the summary
        cache_summary(content, summary)

        return summary
    except Exception as e:
        logger.error(f"Error generating summary: {e}")
        # If summarization fails, return a truncated version of the original content
        return (
            content[: max_length - 3] + "..." if len(content) > max_length else content
        )


def rebuild_summary_cache():
    """
    Rebuild the summary cache for all documents in the vector store.
    This is useful when you want to regenerate all summaries or when
    the summary generation logic has changed.
    """
    global vector_store

    if not vector_store:
        logger.error(
            "[SUMMARY] Vector store not initialized, cannot rebuild summary cache"
        )
        return False

    try:
        logger.info("[SUMMARY] Starting summary cache rebuild")

        # Get all documents from the vector store
        if hasattr(vector_store, "docstore") and hasattr(
            vector_store.docstore, "_dict"
        ):
            all_docs = list(vector_store.docstore._dict.values())
            logger.info(f"[SUMMARY] Found {len(all_docs)} documents in vector store")

            # Extract content from documents
            contents = []
            doc_ids = []

            for i, doc in enumerate(all_docs):
                if hasattr(doc, "page_content") and doc.page_content:
                    contents.append(doc.page_content)
                    doc_ids.append(i)

            if not contents:
                logger.warning("[SUMMARY] No valid document contents found")
                return False

            # Clear existing cache
            global summary_cache
            summary_cache = {}

            # Generate summaries in batch
            logger.info(f"[SUMMARY] Generating summaries for {len(contents)} documents")
            batch_size = 20  # Larger batch size for bulk processing
            summaries = batch_generate_summaries(contents, batch_size=batch_size)

            # Update document metadata with new summaries
            updated_docs = []
            for i, doc_idx in enumerate(doc_ids):
                if i < len(summaries) and summaries[i]:
                    doc = all_docs[doc_idx]
                    if hasattr(doc, "metadata"):
                        doc.metadata["summary"] = summaries[i]
                        updated_docs.append(doc)

            # Rebuild vector store with updated documents if needed
            if updated_docs:
                logger.info(
                    f"[SUMMARY] Updated summaries for {len(updated_docs)} documents"
                )

                # Save the summary cache
                save_summary_cache()

                return True
            else:
                logger.warning("[SUMMARY] No documents were updated with summaries")
                return False
        else:
            logger.error("[SUMMARY] Could not access documents in vector store")
            return False
    except Exception as e:
        logger.error(f"[SUMMARY] Error rebuilding summary cache: {e}")
        return False


# Add a command-line interface for rebuilding the cache
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "rebuild_summaries":
        logger.info("Starting summary cache rebuild from command line")
        initialize_models()
        success = rebuild_summary_cache()
        if success:
            logger.info("Summary cache rebuild completed successfully")
        else:
            logger.error("Summary cache rebuild failed")
