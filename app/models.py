"""Models for the chat application."""

from pydantic import BaseModel, validator
from typing import List, Optional, Dict, Any
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter
)
import os
from dotenv import load_dotenv
import yaml
import re
import logging
import json
from datetime import datetime

load_dotenv()

logger = logging.getLogger(__name__)

# Track vector store state
VECTOR_STORE_STATE_FILE = "data/vector_store_state.json"

def get_vector_store_state():
    """Get the state of the vector store from the state file."""
    if os.path.exists(VECTOR_STORE_STATE_FILE):
        try:
            with open(VECTOR_STORE_STATE_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"[RAG-INIT] Error reading vector store state: {e}")
    return {}

def save_vector_store_state(state):
    """Save the state of the vector store to the state file."""
    os.makedirs(os.path.dirname(VECTOR_STORE_STATE_FILE), exist_ok=True)
    try:
        with open(VECTOR_STORE_STATE_FILE, 'w') as f:
            json.dump(state, f)
    except Exception as e:
        logger.error(f"[RAG-INIT] Error saving vector store state: {e}")

class ChatMessage(BaseModel):
    """A message in a chat conversation."""
    id: Optional[str] = None
    role: str
    content: str
    timestamp: Optional[str] = None
    sources: Optional[List[str]] = None
    used_chunks: Optional[List[Dict[str, Any]]] = None
    
    @validator('content')
    def content_must_be_valid(cls, v):
        """Validate that content is not None and is a string."""
        if v is None:
            raise ValueError('content cannot be None')
        if not isinstance(v, str):
            raise ValueError('content must be a string')
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
vector_store: Optional[Chroma] = None

def initialize_models():
    """Initialize the RAG components."""
    global embeddings, vector_store
    
    try:
        logger.info("[RAG-INIT] Starting RAG components initialization...")
        
        # Initialize embeddings with OpenAI API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("[RAG-INIT] OpenAI API key not found in environment variables")
            raise ValueError("OpenAI API key not found")
            
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        logger.info("[RAG-INIT] OpenAI embeddings initialized successfully")
        
        # Initialize vector store directory
        vector_store_path = "data/chroma"
        rebuild_vectorstore = False
        
        # Get current state
        state = get_vector_store_state()
        
        # Check if vector store exists and if rebuild is needed
        if not os.path.exists(vector_store_path):
            logger.info(f"[RAG-INIT] Vector store not found, will create new one at {vector_store_path}")
            os.makedirs(vector_store_path, exist_ok=True)
            rebuild_vectorstore = True
        else:
            # Check if the vector store is empty or corrupted
            try:
                vector_store = Chroma(
                    persist_directory=vector_store_path,
                    embedding_function=embeddings
                )
                # Try to do a simple search to verify the vector store is working
                vector_store.similarity_search("test", k=1)
                logger.info("[RAG-INIT] Existing vector store loaded successfully")
            except Exception as e:
                logger.warning(f"[RAG-INIT] Existing vector store appears corrupted or empty, will rebuild: {str(e)}")
                rebuild_vectorstore = True
                import shutil
                shutil.rmtree(vector_store_path)
                os.makedirs(vector_store_path)
        
        if rebuild_vectorstore:
            vector_store = Chroma(
                persist_directory=vector_store_path,
                embedding_function=embeddings
            )
            logger.info("[RAG-INIT] New vector store initialized")
            
            # Check documentation directory
            docs_parent_dir = "data/developer-docs"
            docs_dir = f"{docs_parent_dir}/apps/nextra/pages/en"
            
            if not os.path.exists(docs_dir):
                logger.error(f"[RAG-INIT] Documentation directory not found at {docs_dir}. Please ensure it is mounted correctly.")
                raise ValueError("Documentation directory not found")
            
            logger.info("[RAG-INIT] Documentation directory found, processing documents...")
            
            # Load documentation and save state
            load_aptos_docs(docs_dir)
            save_vector_store_state({
                'last_processed': datetime.now().isoformat(),
                'docs_processed': True
            })
            logger.info("[RAG-INIT] Vector store built and persisted successfully")
            logger.info("[RAG-INIT] RAG initialization completed successfully")
            
    except Exception as e:
        logger.error(f"[RAG-INIT] Error initializing RAG components: {str(e)}", exc_info=True)
        raise

def get_relevant_context(query: str, k: int = 5) -> List[Dict[str, str]]:
    """
    Get relevant context from the documentation using a hybrid retrieval approach.
    Returns a list of dictionaries containing content, section, and source information.
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
        docs_and_scores = vector_store.similarity_search_with_score(query, k=initial_k)
        
        # Process and score documents
        scored_docs = []
        seen_contents = set()  # Track unique contents
        
        for doc, semantic_score in docs_and_scores:
            if not doc or not hasattr(doc, 'page_content') or not doc.page_content:
                continue
                
            metadata = doc.metadata or {}
            source = metadata.get('source', '')
            section = metadata.get('section', '')
            content = doc.page_content.strip()
            
            if not content or not isinstance(content, str):
                continue
                
            # Skip if we've seen this content before
            content_hash = hash(content)
            if content_hash in seen_contents:
                continue
            seen_contents.add(content_hash)
            
            # Calculate additional relevance signals
            keywords = set(query.lower().split())
            doc_words = set(content.lower().split())
            keyword_overlap = len(keywords.intersection(doc_words)) / len(keywords) if keywords else 0
            
            section_score = 1.0
            section_keywords = set(section.lower().split('/'))
            if any(kw in section_keywords for kw in keywords):
                section_score = 1.2
            
            content_length = len(content.split())
            length_score = 1.0
            if 50 <= content_length <= 200:
                length_score = 1.1
            
            context_score = 1.1 if content.startswith('Context:') else 1.0
            
            final_score = (
                (1 - semantic_score) * 0.4 +
                keyword_overlap * 0.3 +
                section_score * 0.15 +
                length_score * 0.1 +
                context_score * 0.05
            )
            
            scored_docs.append({
                'content': content,
                'section': section,
                'source': source,
                'score': final_score
            })
        
        # Sort by final score and take top k
        scored_docs.sort(key=lambda x: x['score'], reverse=True)
        top_docs = scored_docs[:k]
        
        # Log retrieval metrics
        if top_docs:
            avg_score = sum(doc['score'] for doc in top_docs) / len(top_docs)
            logger.info(f"Retrieved {len(top_docs)} documents with average score: {avg_score:.3f}")
            logger.debug("Top document sections retrieved:")
            for i, doc in enumerate(top_docs[:3]):  # Log top 3 docs
                logger.debug(f"Doc {i+1}: {doc['section']} (score: {doc['score']:.3f})")
        else:
            logger.warning("No relevant documents found for query")
        
        return top_docs
        
    except Exception as e:
        logger.error(f"Error in get_relevant_context: {e}")
        return []

def extract_frontmatter(content: str) -> tuple[Dict[str, Any], str]:
    """Extract frontmatter from MDX/MD content and return both metadata and content."""
    frontmatter_match = re.match(r'^---\n(.*?)\n---\n(.*)', content, re.DOTALL)
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
    clean_content = re.sub(r'import.*?;\n', '', clean_content, flags=re.MULTILINE)
    clean_content = re.sub(r'export.*?}\n', '', clean_content, flags=re.MULTILINE)
    
    # More carefully handle JSX/TSX components to preserve content
    clean_content = re.sub(r'<Callout.*?>(.*?)</Callout>', r'Important: \1', clean_content, flags=re.DOTALL)
    clean_content = re.sub(r'<Steps.*?>(.*?)</Steps>', r'Steps: \1', clean_content, flags=re.DOTALL)
    clean_content = re.sub(r'<Card.*?>(.*?)</Card>', r'\1', clean_content, flags=re.DOTALL)
    
    # Remove remaining JSX tags but preserve their content
    clean_content = re.sub(r'<[^>]+/>', '', clean_content)  # Remove self-closing components
    clean_content = re.sub(r'</?[^>]+>', '', clean_content)  # Remove tags but keep content
    
    # Clean up any double newlines or spaces created by removals
    clean_content = re.sub(r'\n{3,}', '\n\n', clean_content)
    clean_content = re.sub(r' {2,}', ' ', clean_content)
    clean_content = clean_content.strip()
    
    if not clean_content:
        return []
    
    # Split on headers with more granular control
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4")
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
                separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", ". ", " ", ""]
            )
            return text_splitter.split_text(clean_content)
        
        # Filter out any empty splits and ensure all content is string
        valid_splits = []
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
                    if metadata.get('title'):
                        header_context.insert(0, metadata['title'])
                    if metadata.get('description'):
                        content = f"{metadata['description']}\n\n{content}"
                    
                    # Combine headers into a context string
                    if header_context:
                        content = f"Context: {' > '.join(header_context)}\n\n{content}"
                    
                    valid_splits.append(content)
        
        return valid_splits
    except Exception as e:
        logger.error(f"Error splitting markdown content: {e}")
        # Fall back to size-based splitting if header splitting fails
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", ". ", " ", ""]
        )
        return text_splitter.split_text(clean_content)

def load_aptos_docs(docs_dir: str = "data/developer-docs/apps/nextra/pages/en") -> None:
    """Load and process Aptos documentation from the English documentation directory."""
    global vector_store
    
    if not os.path.exists(docs_dir):
        logger.error(f"[RAG-DOCS] Documentation directory not found: {docs_dir}")
        return
    
    if vector_store is None:
        logger.warning("[RAG-DOCS] Vector store not initialized, attempting initialization")
        initialize_models()
        return  # Return here as initialize_models will call load_aptos_docs if needed
    
    try:
        logger.info("[RAG-DOCS] Starting documentation processing")
        documents = []
        file_count = 0
        processed_count = 0
        
        for root, _, files in os.walk(docs_dir):
            for file in files:
                if file.endswith(('.md', '.mdx')) and not file.endswith('_meta.ts'):
                    file_count += 1
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        # Extract section info from file path
                        relative_path = os.path.relpath(file_path, docs_dir)
                        section_path = os.path.dirname(relative_path)
                        section = section_path.replace(os.path.sep, '/') if section_path != '.' else 'root'
                        
                        # Process the document
                        doc_sections = process_markdown_document(content)
                        processed_count += 1
                        
                        # Add each section to documents with metadata
                        for doc_section in doc_sections:
                            if doc_section and isinstance(doc_section, str) and doc_section.strip():
                                try:
                                    from langchain_core.documents import Document
                                    doc = Document(
                                        page_content=doc_section.strip(),
                                        metadata={
                                            "source": relative_path,
                                            "section": section,
                                            "file_type": "mdx" if file.endswith('.mdx') else "md"
                                        }
                                    )
                                    documents.append(doc)
                                except Exception as e:
                                    logger.error(f"[RAG-DOCS] Error creating document for section in {file_path}: {e}")
                                    continue
                    except Exception as e:
                        logger.error(f"[RAG-DOCS] Error processing file {file_path}: {e}")
        
        if documents:
            try:
                logger.info(f"[RAG-DOCS] Adding {len(documents)} document sections to vector store")
                vector_store.add_documents(documents)
                logger.info(f"[RAG-DOCS] Successfully processed {processed_count}/{file_count} files and added {len(documents)} document sections")
            except Exception as e:
                logger.error(f"[RAG-DOCS] Error adding documents to vector store: {e}")
        else:
            logger.error("[RAG-DOCS] No documents were processed from the Aptos documentation")
    except Exception as e:
        logger.error(f"[RAG-DOCS] Error processing documentation: {e}") 