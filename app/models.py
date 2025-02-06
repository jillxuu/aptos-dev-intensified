from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader, UnstructuredMarkdownLoader
import os
from dotenv import load_dotenv
import yaml
import re

load_dotenv()

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    temperature: float = 0.7
    chat_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    sources: Optional[List[str]] = None
    used_chunks: Optional[List[Dict[str, str]]] = None  # Store the chunks used for this response

class Feedback(BaseModel):
    message_id: str
    query: str
    response: str
    rating: bool  # True for thumbs up, False for thumbs down
    feedback_text: Optional[str] = None
    used_chunks: Optional[List[Dict[str, str]]] = None  # Store the chunks used for this response
    timestamp: Optional[str] = None

class ChatHistory(BaseModel):
    id: str
    title: str
    timestamp: str
    messages: List[ChatMessage]

# Global variables for RAG components
embeddings = None
vector_store = None

def initialize_models():
    """Initialize the RAG components."""
    global embeddings, vector_store
    
    # Initialize embeddings with OpenAI API key
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    
    # Initialize vector store
    if not os.path.exists("data/chroma"):
        os.makedirs("data/chroma", exist_ok=True)
    else:
        # Clear existing vector store
        import shutil
        shutil.rmtree("data/chroma")
        os.makedirs("data/chroma")
    
    vector_store = Chroma(
        persist_directory="data/chroma",
        embedding_function=embeddings
    )

def get_relevant_context(query: str, k: int = 5) -> List[Dict[str, str]]:
    """
    Get relevant context from the documentation based on the query.
    Returns a list of dictionaries containing content, section, and source information.
    """
    if not query or not vector_store:
        return []

    try:
        # Get relevant documents with scores
        docs_and_scores = vector_store.similarity_search_with_score(query, k=k*2)  # Get more docs initially
        
        # Filter and sort by relevance
        filtered_docs = []
        for doc, score in docs_and_scores:
            if not doc or not hasattr(doc, 'page_content') or not doc.page_content:
                continue
                
            metadata = doc.metadata or {}
            source = metadata.get('source', '')
            section = metadata.get('section', '')
            content = doc.page_content.strip()
            
            if not content or not isinstance(content, str):
                continue
            
            # Create document entry with score
            filtered_docs.append({
                'content': content,
                'section': section,
                'source': source,
                'score': float(score)
            })
        
        # Sort by score (lower is better in this case)
        filtered_docs.sort(key=lambda x: x['score'])
        
        # Group by source to avoid too many duplicates from the same source
        source_groups = {}
        for doc in filtered_docs:
            source = doc['source']
            if source not in source_groups:
                source_groups[source] = []
            source_groups[source].append(doc)
        
        # Take the best document from each source first, then fill with remaining best docs
        final_docs = []
        # First, take the best doc from each source
        for source_docs in source_groups.values():
            if source_docs and len(final_docs) < k:
                final_docs.append(source_docs[0])
        
        # If we still need more docs, take the next best ones regardless of source
        remaining_slots = k - len(final_docs)
        if remaining_slots > 0:
            # Flatten remaining docs (excluding ones we've already taken)
            remaining_docs = [
                doc for source_docs in source_groups.values()
                for doc in source_docs[1:]  # Skip the first one we already took
            ]
            # Sort by score and take the best remaining ones
            remaining_docs.sort(key=lambda x: x['score'])
            final_docs.extend(remaining_docs[:remaining_slots])
        
        # Remove scores from final output
        for doc in final_docs:
            doc.pop('score', None)
        
        return final_docs
        
    except Exception as e:
        print(f"Error in get_relevant_context: {e}")
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
        print(f"Invalid content type or empty content: {type(clean_content)}")
        return []
    
    clean_content = clean_content.strip()
    if not clean_content:
        return []
    
    # Remove JSX/TSX imports and exports but preserve important content
    clean_content = re.sub(r'import.*?;\n', '', clean_content, flags=re.MULTILINE)
    clean_content = re.sub(r'export.*?}\n', '', clean_content, flags=re.MULTILINE)
    
    # More carefully handle JSX/TSX components to preserve content
    clean_content = re.sub(r'<Callout.*?>(.*?)</Callout>', r'\1', clean_content, flags=re.DOTALL)
    clean_content = re.sub(r'<Steps.*?>(.*?)</Steps>', r'\1', clean_content, flags=re.DOTALL)
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
        # If no headers found, use a size-based splitter
        if not splits:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", " ", ""]
            )
            return text_splitter.split_text(clean_content)
        
        # Filter out any empty splits and ensure all content is string
        valid_splits = []
        for split in splits:
            if split.page_content and isinstance(split.page_content, str):
                content = split.page_content.strip()
                if content:
                    # Add header context to the content
                    header_prefix = ""
                    if split.metadata.get("Header 1"):
                        header_prefix += f"{split.metadata['Header 1']}\n"
                    if split.metadata.get("Header 2"):
                        header_prefix += f"{split.metadata['Header 2']}\n"
                    if split.metadata.get("Header 3"):
                        header_prefix += f"{split.metadata['Header 3']}\n"
                    if header_prefix:
                        content = f"{header_prefix}\n{content}"
                    valid_splits.append(content)
        
        return valid_splits
    except Exception as e:
        print(f"Error splitting markdown content: {e}")
        # Fall back to size-based splitting if header splitting fails
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", " ", ""]
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

def load_documents(directory: str = "data/documents"):
    """Load additional documents into the vector store."""
    global vector_store
    
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        print(f"Created documents directory at {directory}")
        return
    
    # Initialize the loader for txt files
    loader = DirectoryLoader(
        directory,
        glob="**/*.txt",
        loader_cls=TextLoader
    )
    
    try:
        # Load documents
        documents = loader.load()
        if not documents:
            print("No documents found to load.")
            return
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(documents)
        
        # Add to vector store
        if vector_store is None:
            initialize_models()
        
        vector_store.add_documents(splits)
        vector_store.persist()
        print(f"Loaded {len(splits)} document chunks into the vector store.")
    except Exception as e:
        print(f"Error loading documents: {e}") 