"""Models for the chat application."""

from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter
)
from langchain_community.document_loaders import (
    TextLoader,
    DirectoryLoader,
    UnstructuredMarkdownLoader
)
import os
from dotenv import load_dotenv
import yaml
import re
import logging

load_dotenv()

logger = logging.getLogger(__name__)

class ChatMessage(BaseModel):
    """A chat message with metadata."""
    role: str
    content: str
    id: Optional[str] = None
    timestamp: Optional[str] = None
    sources: Optional[List[str]] = None
    used_chunks: Optional[List[Dict[str, Any]]] = None

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
        logger.info("Initializing RAG components...")
        
        # Initialize embeddings with OpenAI API key
        embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
        logger.info("OpenAI embeddings initialized successfully")
        
        # Initialize vector store
        vector_store_path = "data/chroma"
        if not os.path.exists(vector_store_path):
            logger.info(f"Creating new vector store directory at {vector_store_path}")
            os.makedirs(vector_store_path, exist_ok=True)
        else:
            logger.info("Clearing existing vector store")
            import shutil
            shutil.rmtree(vector_store_path)
            os.makedirs(vector_store_path)
        
        vector_store = Chroma(
            persist_directory=vector_store_path,
            embedding_function=embeddings
        )
        logger.info("Vector store initialized successfully")
        
        # Load documentation if the directory exists
        docs_dir = "data/developer-docs/apps/nextra/pages/en"
        if os.path.exists(docs_dir):
            logger.info(f"Loading documentation from {docs_dir}")
            load_aptos_docs(docs_dir)
        else:
            logger.warning(f"Documentation directory not found at {docs_dir}")
            
    except Exception as e:
        logger.error(f"Error initializing RAG components: {str(e)}")
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
        for doc, semantic_score in docs_and_scores:
            if not doc or not hasattr(doc, 'page_content') or not doc.page_content:
                continue
                
            metadata = doc.metadata or {}
            source = metadata.get('source', '')
            section = metadata.get('section', '')
            content = doc.page_content.strip()
            
            if not content or not isinstance(content, str):
                continue
            
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
        print(f"Documentation directory not found: {docs_dir}")
        return
    
    if vector_store is None:
        initialize_models()
    
    documents = []
    
    for root, _, files in os.walk(docs_dir):
        for file in files:
            if file.endswith(('.md', '.mdx')) and not file.endswith('_meta.ts'):
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
                                print(f"Error creating document for section in {file_path}: {e}")
                                continue
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")
    
    if documents:
        try:
            vector_store.add_documents(documents)
            vector_store.persist()
            print(f"Successfully processed {len(documents)} document sections from Aptos documentation.")
        except Exception as e:
            print(f"Error adding documents to vector store: {e}")
    else:
        print("No documents were processed from the Aptos documentation.") 