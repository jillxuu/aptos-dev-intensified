# GitHub Repository Integration for RAG System

## Overview

This document outlines a simplified plan to integrate GitHub repositories as data sources for our Retrieval-Augmented Generation (RAG) system, with particular focus on Move language code. The goal is to enhance developer support by providing relevant code examples, implementations, and context from repositories, while managing the priority of these repositories relative to official documentation.

## Current System and Extension Points

Our current RAG system is primarily focused on documentation retrieval, with the following components:

1. **RAG Providers**: Abstract interface (`RAGProvider`) with implementations for different data sources
2. **Vector Store**: Storage for embeddings of chunked content
3. **Retrieval Logic**: Methods for retrieving and ranking content based on queries
4. **Response Generation**: Processing retrieved context into coherent LLM responses

To support GitHub repositories as a data source, we need to extend each of these components while maintaining backward compatibility.

## Repository Priority Management

To ensure that GitHub repositories don't overshadow official documentation, we'll implement a priority-based integration:

### 1. Priority Scaling System

```python
class GitHubRepoConfig:
    """Configuration for GitHub repository."""
    
    def __init__(
        self, 
        repo_url: str, 
        branch: str = "main",
        priority: float = 0.7,  # Default priority (0-1)
        include_patterns: List[str] = None,
        exclude_patterns: List[str] = None,
        metadata: Dict[str, Any] = None
    ):
        self.repo_url = repo_url
        self.branch = branch
        self.priority = priority  # How results are weighted relative to docs
        self.include_patterns = include_patterns or ["**/*.move", "**/*.md"]
        self.exclude_patterns = exclude_patterns or ["**/node_modules/**", "**/target/**"]
        self.metadata = metadata or {}
```

### 2. Result Merging with Priority

```python
async def merge_retrieval_results(
    doc_results: List[Dict[str, Any]],
    github_results: List[Dict[str, Any]],
    repo_configs: Dict[str, GitHubRepoConfig],
    top_k: int = 10
) -> List[Dict[str, Any]]:
    """
    Merge results from documentation and GitHub repositories with priority weighting.
    
    Args:
        doc_results: Results from documentation
        github_results: Results from GitHub repositories
        repo_configs: Repository configurations with priority settings
        top_k: Maximum number of results to return
        
    Returns:
        Merged and prioritized results
    """
    # Apply priority weighting to GitHub results
    for result in github_results:
        repo_url = result.get("metadata", {}).get("repo_url", "")
        repo_config = repo_configs.get(repo_url)
        
        if repo_config:
            # Apply the repository's priority as a multiplier to the score
            result["score"] = result["score"] * repo_config.priority
            
            # Add priority to metadata for future reference
            if "metadata" not in result:
                result["metadata"] = {}
            result["metadata"]["priority"] = repo_config.priority
    
    # Combine results from both sources
    merged_results = doc_results + github_results
    
    # Sort by score (descending)
    merged_results.sort(key=lambda x: x.get("score", 0), reverse=True)
    
    # Return top-k results
    return merged_results[:top_k]
```

## Simplified Code Chunking Strategy

We'll implement a pattern-based chunking approach that focuses on identifying useful code files and documentation without complex language-specific parsing:

### 1. File Pattern Selection

```python
def is_relevant_file(file_path: str, config: GitHubRepoConfig) -> bool:
    """
    Determine if a file should be processed based on patterns.
    
    Args:
        file_path: Path to the file
        config: Repository configuration with include/exclude patterns
        
    Returns:
        Boolean indicating if the file should be processed
    """
    # Check exclude patterns first
    for pattern in config.exclude_patterns:
        if fnmatch.fnmatch(file_path, pattern):
            return False
    
    # Then check include patterns
    for pattern in config.include_patterns:
        if fnmatch.fnmatch(file_path, pattern):
            return True
    
    # If no include pattern matches, exclude by default
    return False
```

### 2. Simple Code Chunking

We'll use a straightforward chunking approach based on file size and natural boundaries:

```python
def chunk_code_file(
    file_path: str, 
    file_content: str, 
    repo_metadata: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Chunk a code file using a simplified approach.
    
    Args:
        file_path: Path to the file
        file_content: Content of the file
        repo_metadata: Repository metadata
        
    Returns:
        List of chunks with metadata
    """
    chunks = []
    file_extension = os.path.splitext(file_path)[1]
    
    # Extract basic file metadata
    repo_name = repo_metadata.get("repo_name", "")
    repo_url = repo_metadata.get("repo_url", "")
    
    # Add the entire file as a chunk for smaller files
    if len(file_content.split('\n')) <= 100:  # Reasonable size for full context
        chunks.append({
            "content": file_content,
            "metadata": {
                "chunk_type": "file",
                "file_path": file_path,
                "file_extension": file_extension,
                "repo_name": repo_name,
                "repo_url": repo_url,
                "is_full_file": True
            }
        })
        return chunks
    
    # For larger files, chunk by natural boundaries
    sections = split_by_natural_boundaries(file_content, file_extension)
    
    for i, section in enumerate(sections):
        chunks.append({
            "content": section.content,
            "metadata": {
                "chunk_type": "code_section",
                "file_path": file_path,
                "file_extension": file_extension,
                "section_index": i,
                "section_type": section.section_type,  # e.g., "imports", "function", "class"
                "repo_name": repo_name,
                "repo_url": repo_url,
                "is_full_file": False
            }
        })
    
    return chunks
```

### 3. Natural Boundary Detection

```python
def split_by_natural_boundaries(
    content: str, 
    file_extension: str
) -> List[Section]:
    """
    Split content by natural boundaries based on file type.
    
    Args:
        content: File content
        file_extension: File extension to determine language
        
    Returns:
        List of sections
    """
    sections = []
    lines = content.split('\n')
    
    # Default chunk size if no natural boundaries found
    max_chunk_size = 50  # lines
    
    # For Move files
    if file_extension == '.move':
        # Split on module, struct and function definitions
        current_section = []
        current_type = "unknown"
        
        for line in lines:
            # Check for new section boundary
            if line.strip().startswith(('module ', 'struct ', 'public fun ', 'fun ')):
                # Save previous section if not empty
                if current_section:
                    sections.append(Section(
                        content='\n'.join(current_section),
                        section_type=current_type
                    ))
                
                # Start new section
                current_section = [line]
                if line.strip().startswith('module '):
                    current_type = "module_definition"
                elif line.strip().startswith('struct '):
                    current_type = "struct_definition"
                else:
                    current_type = "function_definition"
            else:
                current_section.append(line)
        
        # Add the last section
        if current_section:
            sections.append(Section(
                content='\n'.join(current_section),
                section_type=current_type
            ))
    else:
        # For other file types, use simpler chunking by groups of lines
        for i in range(0, len(lines), max_chunk_size):
            chunk_lines = lines[i:min(i + max_chunk_size, len(lines))]
            sections.append(Section(
                content='\n'.join(chunk_lines),
                section_type="code_block"
            ))
    
    return sections
```

### 4. Documentation File Handling

For README files and other documentation that explain concepts and usage:

```python
def chunk_documentation_file(
    file_path: str, 
    file_content: str, 
    repo_metadata: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Chunk documentation files like README.md.
    
    Args:
        file_path: Path to the file
        file_content: Content of the file
        repo_metadata: Repository metadata
        
    Returns:
        List of chunks with metadata
    """
    chunks = []
    
    # For markdown files, split by headers
    if file_path.endswith(('.md', '.markdown')):
        # Simple splitting by markdown headers
        header_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
        
        # Find all headers with their positions
        headers = [(m.group(1), m.group(2), m.start()) for m in header_pattern.finditer(file_content)]
        
        # Split content by headers
        if not headers:
            # No headers, just use the whole file
            chunks.append({
                "content": file_content,
                "metadata": {
                    "chunk_type": "documentation",
                    "section_title": os.path.basename(file_path),
                    "file_path": file_path,
                    "repo_name": repo_metadata.get("repo_name", ""),
                    "repo_url": repo_metadata.get("repo_url", ""),
                }
            })
        else:
            # Process each section
            for i, (level, title, start) in enumerate(headers):
                # Get section content
                if i < len(headers) - 1:
                    end = headers[i + 1][2]
                    section_content = file_content[start:end]
                else:
                    section_content = file_content[start:]
                
                # Create chunk
                chunks.append({
                    "content": section_content,
                    "metadata": {
                        "chunk_type": "documentation",
                        "section_title": title,
                        "section_level": len(level),  # e.g., # = 1, ## = 2
                        "file_path": file_path,
                        "repo_name": repo_metadata.get("repo_name", ""),
                        "repo_url": repo_metadata.get("repo_url", ""),
                        "is_example": "example" in title.lower() or "usage" in title.lower(),
                    }
                })
    else:
        # For non-markdown documentation, use the whole file
        chunks.append({
            "content": file_content,
            "metadata": {
                "chunk_type": "documentation",
                "file_path": file_path,
                "repo_name": repo_metadata.get("repo_name", ""),
                "repo_url": repo_metadata.get("repo_url", ""),
            }
        })
    
    return chunks
```

## LLM-Enhanced Code Summaries

We'll use LLM to generate summaries for code chunks to improve retrieval without complex language-specific parsing:

```python
async def add_code_summaries(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Add natural language summaries to code chunks using LLM.
    
    Args:
        chunks: List of code chunks
        
    Returns:
        Enhanced chunks with summaries
    """
    enhanced_chunks = []
    
    for chunk in chunks:
        if chunk["metadata"]["chunk_type"] in ["file", "code_section"]:
            # Only summarize code files and sections
            file_path = chunk["metadata"]["file_path"]
            
            # Generate summary using LLM
            prompt = f"""
            Summarize the following code from {file_path}. 
            Focus on what it does, its purpose, and any key functionality:
            
            ```
            {chunk["content"]}
            ```
            
            Provide a concise summary (2-3 sentences).
            """
            
            summary = await get_llm_summary(prompt)
            
            # Add summary to chunk metadata
            chunk["metadata"]["summary"] = summary
            
            # Create embedding text that combines code and summary
            chunk["embedding_text"] = f"""
            {summary}
            
            {chunk["content"]}
            """
        else:
            # For documentation, use the content as is
            chunk["embedding_text"] = chunk["content"]
        
        enhanced_chunks.append(chunk)
    
    return enhanced_chunks
```

## Implementation Plan

### Phase 1: Core Infrastructure (Week 1)

1. Create the `GitHubRepositoryProvider` class implementing the `RAGProvider` interface
2. Implement repository cloning and content extraction logic
3. Develop the configuration system for repositories with priority settings

### Phase 2: Basic Chunking Implementation (Week 1-2)

1. Implement pattern-based file selection
2. Build the simplified chunking system for code and documentation files
3. Set up the LLM summarization pipeline for code chunks

### Phase 3: Integration and Testing (Week 2-3)

1. Implement the result merging logic with priority weighting
2. Create embeddings for code and documentation chunks
3. Test with various types of developer queries and repositories

### Phase 4: Refinement and Deployment (Week 3-4)

1. Optimize chunking based on test results
2. Add caching for repository content and embeddings
3. Deploy and monitor the enhanced RAG system

## Success Metrics

1. **Relevance**: % of code examples that correctly demonstrate the queried concept
2. **Retrieval Effectiveness**: Improvement in correct code retrieval vs. documentation-only
3. **User Satisfaction**: Feedback on code example quality and usefulness
4. **Performance Impact**: Minimal latency increase when adding GitHub as a source

## Next Steps

After successfully implementing this simplified approach, we can consider future enhancements:

1. Adding language-specific parsing for more intelligent chunking
2. Implementing code relationship mapping to track dependencies
3. Developing more sophisticated code understanding features
4. Adding support for automatic code updates and versioning 