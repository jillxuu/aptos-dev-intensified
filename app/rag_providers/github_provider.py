"""GitHub RAG provider that uses a GitHub repository as a knowledge base."""

from typing import List, Dict, Any, Optional
import logging
import os
import tempfile
import subprocess
import shutil
import glob
from pathlib import Path
import asyncio
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from app.rag_providers import RAGProvider, RAGProviderRegistry

logger = logging.getLogger(__name__)


class GitHubRAGProvider(RAGProvider):
    """
    RAG provider that uses a GitHub repository as a knowledge base.

    This provider:
    1. Clones a GitHub repository
    2. Processes its contents
    3. Builds a vector store
    4. Uses it for retrievals
    """

    def __init__(self):
        """Initialize the GitHub RAG provider."""
        self._initialized = False
        self._vector_store = None

        # Initialize embeddings in constructor
        try:
            self._embeddings = OpenAIEmbeddings()
        except Exception as e:
            logger.warning(
                f"Could not initialize OpenAI embeddings in constructor: {str(e)}"
            )
            self._embeddings = None

        self._config = {}
        self._repo_dir = None
        self._vector_store_dir = None

    @property
    def name(self) -> str:
        """Get the name of the RAG provider."""
        return "github"

    @property
    def description(self) -> str:
        """Get the description of the RAG provider."""
        return "RAG provider using a GitHub repository as a knowledge base"

    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the GitHub RAG provider with the given configuration."""
        repo_url = config.get("repo_url")
        branch = config.get("branch", "main")
        file_types = config.get(
            "file_types",
            [
                "md",
                "txt",
                "py",
                "js",
                "ts",
                "jsx",
                "tsx",
                "html",
                "css",
                "json",
                "yaml",
                "yml",
            ],
        )
        exclude_dirs = config.get(
            "exclude_dirs", [".git", "node_modules", "dist", "build", "__pycache__"]
        )

        if not repo_url:
            raise ValueError("Repository URL is required")

        # Clean up any existing repository directory
        if (
            hasattr(self, "_repo_dir")
            and self._repo_dir
            and os.path.exists(self._repo_dir)
        ):
            try:
                shutil.rmtree(self._repo_dir)
            except Exception as e:
                logger.warning(
                    f"Failed to clean up existing repository directory: {str(e)}"
                )

        # Create a new temporary directory for the repository
        self._repo_dir = tempfile.mkdtemp()
        logger.info(f"Created temporary directory for repository: {self._repo_dir}")

        try:
            # Try a direct clone with limited depth
            logger.info(f"Cloning repository {repo_url} (branch: {branch})...")

            # Use a simpler direct clone approach
            clone_process = await asyncio.create_subprocess_exec(
                "git",
                "clone",
                "--depth=1",
                "--single-branch",
                "--branch",
                branch,
                repo_url,
                self._repo_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    clone_process.communicate(), timeout=60  # 60 second timeout
                )

                if clone_process.returncode != 0:
                    error_message = stderr.decode().strip()
                    raise ValueError(f"Failed to clone repository: {error_message}")

            except asyncio.TimeoutError:
                logger.warning("Git clone operation timed out.")
                raise ValueError(
                    "Repository cloning timed out. Please try a smaller repository."
                )

        except Exception as e:
            # If all attempts fail, raise a more helpful error
            logger.error(f"Failed to initialize GitHub RAG provider: {str(e)}")
            raise ValueError(
                f"Failed to clone repository: {str(e)}. Please try a smaller repository or check the URL."
            )

        logger.info(f"Repository cloned successfully to {self._repo_dir}")

        # Create a directory for the vector store
        self._vector_store_dir = os.path.join(
            tempfile.gettempdir(), f"vector_store_{os.path.basename(self._repo_dir)}"
        )
        os.makedirs(self._vector_store_dir, exist_ok=True)

        # Process the repository files
        await self._process_repository(file_types, exclude_dirs)

        self._config = config
        self._initialized = True
        logger.info(f"Initialized GitHub RAG provider for {repo_url}")

    async def _process_repository(
        self, file_types: List[str], exclude_dirs: List[str]
    ) -> None:
        """Process the repository files and create a vector store."""
        logger.info(f"Processing repository files with types: {file_types}")

        # Ensure embeddings are available
        if self._embeddings is None:
            try:
                logger.info("Embeddings not initialized, attempting to initialize now")
                self._embeddings = OpenAIEmbeddings()
            except Exception as e:
                logger.error(f"Failed to initialize embeddings: {str(e)}")
                raise ValueError(
                    "Failed to initialize embeddings. Please check your OpenAI API key."
                )

        # Get all files in the repository
        all_files = []
        for root, dirs, files in os.walk(self._repo_dir):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in exclude_dirs]

            for file in files:
                # Only include files with the specified extensions
                if any(file.endswith(f".{ext}") for ext in file_types):
                    all_files.append(os.path.join(root, file))

        logger.info(f"Found {len(all_files)} files to process")

        # Limit the number of files to process to avoid timeouts
        max_files = 100
        if len(all_files) > max_files:
            logger.warning(
                f"Repository has too many files ({len(all_files)}). Limiting to {max_files} files."
            )
            all_files = all_files[:max_files]

        # Process files
        chunks = []
        for file_path in all_files:
            try:
                # Get relative path for source tracking
                rel_path = os.path.relpath(file_path, self._repo_dir)

                # Skip files that are too large
                file_size = os.path.getsize(file_path)
                if file_size > 1_000_000:  # Skip files larger than 1MB
                    logger.warning(
                        f"Skipping large file: {rel_path} ({file_size} bytes)"
                    )
                    continue

                # Read file content
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()

                # Skip empty files
                if not content.strip():
                    continue

                # Create text splitter based on file type
                if file_path.endswith((".md", ".txt")):
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200,
                    )
                else:
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200,
                        separators=["\n\n", "\n", " ", ""],
                    )

                # Split text into chunks
                file_chunks = text_splitter.create_documents(
                    [content], metadatas=[{"source": rel_path}]
                )

                chunks.extend(file_chunks)

            except Exception as e:
                logger.warning(f"Error processing file {file_path}: {str(e)}")

        if not chunks:
            raise ValueError("No processable text found in the repository")

        logger.info(f"Created {len(chunks)} chunks from repository files")

        # Double-check embeddings before creating vector store
        if self._embeddings is None:
            logger.error("Embeddings still not available before creating vector store")
            raise ValueError(
                "Embeddings not available. Please check your OpenAI API key."
            )

        # Create vector store
        self._vector_store = FAISS.from_documents(chunks, self._embeddings)

        # Save vector store
        self._vector_store.save_local(self._vector_store_dir)
        logger.info(f"Vector store saved to {self._vector_store_dir}")

    async def get_relevant_context(
        self, query: str, k: int = 5, include_series: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get relevant context from the GitHub repository.

        Args:
            query: The user query
            k: Number of top documents to return
            include_series: Whether to include related documents from the same series

        Returns:
            List of dictionaries containing content, section, source, and summary information
        """
        if not self._initialized or not self._vector_store:
            logger.warning("GitHub RAG provider not initialized")
            return []

        try:
            # Perform similarity search
            docs_with_scores = self._vector_store.similarity_search_with_score(
                query, k=k
            )

            # Format results
            results = []
            for doc, score in docs_with_scores:
                # Extract metadata
                metadata = doc.metadata

                # Create result dictionary
                result = {
                    "content": doc.page_content,
                    "score": float(score),
                    "source": metadata.get("source", "Unknown"),
                    "section": (
                        os.path.relpath(metadata.get("source", ""), self._repo_dir)
                        if metadata.get("source")
                        else ""
                    ),
                    "summary": f"From file: {os.path.basename(metadata.get('source', 'Unknown'))}",
                }

                results.append(result)

            return results
        except Exception as e:
            logger.error(f"Error in GitHub RAG provider: {str(e)}")
            return []

    def __del__(self):
        """Clean up temporary directories when the provider is deleted."""
        try:
            if self._repo_dir and os.path.exists(self._repo_dir):
                shutil.rmtree(self._repo_dir)
            if self._vector_store_dir and os.path.exists(self._vector_store_dir):
                shutil.rmtree(self._vector_store_dir)
        except Exception as e:
            logger.error(f"Error cleaning up GitHub RAG provider: {str(e)}")


# Register the GitHub RAG provider
github_provider = GitHubRAGProvider()
RAGProviderRegistry.register(github_provider)
