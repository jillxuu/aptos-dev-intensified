# Documentation URL Reference Improvement Plan

## Current System Architecture

### Adaptive Multi-Step Retrieval Implementation

Our current RAG system utilizes an adaptive multi-step retrieval approach implemented in `app/utils/adaptive_retrieval.py`. The system works as follows:

1. **Query Analysis**: When a user asks a question, the system analyzes the query to determine its complexity and generates multiple retrieval queries using `generate_retrieval_queries()`.

2. **Primary Retrieval**: The system executes these queries against the vector store, fetching semantically similar chunks of documentation.

3. **Follow-up Analysis**: The system analyzes if the retrieved chunks fully address the user's question using `analyze_follow_up_needs()`.

4. **Follow-up Retrieval**: If needed, the system generates and executes additional targeted queries to fill knowledge gaps.

5. **Result Combination**: The system combines all retrieved chunks, sorts them by relevance score, and returns the top results.

### Document Reference System

The reference system resolves document paths to URLs through:

1. **Path Registry**: A centralized `PathRegistry` class in `app/path_registry.py` that normalizes file paths and maps them to URLs.

2. **Source Path to URL Conversion**: When formatting retrieval results, `source_path` values are converted to URLs using `path_registry.get_url()`.

3. **URL Generation**: The `get_url()` method in the `PathRegistry` class looks up the URL corresponding to the normalized path.

### LLM Context Processing

The LLM receives chunk information through the following process:

1. **Chunk Format**: Each retrieved chunk is formatted into a dictionary containing:
   ```python
   formatted_result = {
       "content": result["content"],      # The actual text content
       "section": result["section"],      # Section title or header
       "source": source_url or "",        # URL to the documentation (potentially truncated)
       "source_path": source_path,        # Raw source path from the chunk metadata
       "summary": result["summary"],      # Brief summary of the content
       "score": result["score"],          # Relevance score
       "metadata": {
           "related_documents": result.get("related_documents", []),
           "is_priority": result.get("is_priority", False),
           "docs_path": self._current_path,
           "retrieval_type": result.get("retrieval_type", "primary"),
           "retrieval_query": result.get("retrieval_query", query),
       },
   }
   ```

2. **Path Information Flow**:
   - During document chunking, each chunk is assigned a `source` attribute that contains the file path
   - When formatting chunks for the LLM, the `source_path` is extracted and converted to a URL using `path_registry.get_url()`
   - The LLM receives the URL in the `source` field of each chunk, which it uses to construct references

3. **Prompt Construction**: The chunks are inserted into the prompt for the LLM, with instructions to use the provided sources when answering questions:
   ```
   Here are the sources I found that might help answer your question:
   
   SOURCE 1: {source_url}
   {content}
   
   SOURCE 2: {source_url}
   {content}
   ...
   
   When providing references, use the exact URLs provided in the sources.
   ```

4. **Reference Generation**: The LLM constructs references based on the URLs it receives in the chunks. If a URL is truncated or points to a parent directory, the LLM has no way to know and will use the incorrect URL in its response.

## Identified Issues

The current system has a critical limitation with URL references:

### 1. Section Referencing Issues

When the chatbot tries to reference specific sections with anchor tags:
- It cannot validate if the anchor actually exists on the target page
- It sometimes applies anchors to parent pages where they don't exist

For example:
- **Correct**: `https://aptos.dev/en/build/guides/system-integrators-guide#current-balance-for-a-coin`
- **Incorrect**: `https://aptos.dev/en/build/guides#current-balance-for-a-coin` (anchor on wrong page)

- **Correct**: `https://aptos.dev/en/build/guides/exchanges#fungible-asset-balances`
- **Incorrect**: `https://aptos.dev/en/build/guides#fungible-asset-balances` (anchor on wrong page)

### 2. Path Truncation Problem

When providing references to specific sections in documentation, the system often returns URLs that point to parent pages rather than specific pages. For example:

**Correct URL (specific page)**: `https://aptos.dev/en/concepts/staking`  
**Incorrect URL (parent page)**: `https://aptos.dev/en/concepts`

This occurs because:
- Source paths in chunks are not always preserving the full path to specific pages
- The URL resolution system sometimes truncates paths when mapping to URLs


### 3. Root Causes

1. **Path Truncation**: During preprocessing or retrieval, specific page paths are sometimes truncated to parent directory paths.
2. **URL Mapping**: The path-to-URL mapping system doesn't consistently preserve full paths.
3. **Source Attribution**: Source information in chunks doesn't always contain the complete path.
4. **Section-URL Mismatch**: The system doesn't validate whether section anchors actually exist on referenced pages.

### 4. How Anchor URLs Are Formed

The anchor tags (like `#current-balance-for-a-coin` in URLs such as `https://aptos.dev/en/build/guides#current-balance-for-a-coin`) can be generated at multiple points in the pipeline:

1. **During Document Processing**: 
   - The documentation processing pipeline extracts headings from Markdown files
   - These headings are converted to anchor IDs (lowercase, spaces replaced with hyphens)
   - However, this information is not always preserved when creating chunks

2. **In the LLM's Response Generation**:
   - When the LLM receives documentation chunks, it might identify section headers within the content
   - It attempts to construct anchor links based on the header text it sees
   - The LLM follows standard markdown/HTML conventions for creating anchors (lowercase, replace spaces with hyphens)
   - Example: A heading "Current Balance for a Coin" becomes `#current-balance-for-a-coin`

3. **URL Formation Issues**:
   - The primary issue occurs when the LLM attaches valid anchor tags to incorrect base URLs
   - For example, it might see content from `coin-and-token/coin.md` but get a URL for just `coin-and-token/`
   - The resulting URL `https://aptos.dev/en/concepts/coin-and-token/#current-balance-for-a-coin` is invalid because the anchor exists in a child page, not the parent directory

4. **Debugging Example**:
   ```
   Original file: /en/concepts/coin-and-token/coin.md 
   Section: "Current Balance for a Coin"
   
   During chunking: 
   - Chunk is created with content including this section
   - Source is truncated to "/en/concepts/coin-and-token/"
   
   URL generation:
   - path_registry.get_url("/en/concepts/coin-and-token/") 
     returns "https://aptos.dev/en/concepts/coin-and-token/"
   
   LLM response:
   - LLM sees header "Current Balance for a Coin" in the content
   - Creates anchor "#current-balance-for-a-coin"
   - Attaches to base URL: "https://aptos.dev/en/concepts/coin-and-token/#current-balance-for-a-coin"
   - This URL fails because the anchor exists on "coin.md" page, not the directory page
   ```

## Proposed Solutions

### 1. Source Path Preservation

Enhance the chunking process to ensure each chunk's `source` attribute contains the full path to the specific page:

```python
def enhance_chunk_source_path(chunk, doc_path):
    """Ensure chunk source contains complete path to specific page."""
    if "source" not in chunk or not chunk["source"]:
        chunk["source"] = doc_path
    elif not chunk["source"].endswith((".md", ".mdx")) and doc_path.endswith((".md", ".mdx")):
        # If source is a directory but we know the specific file, use the file
        chunk["source"] = doc_path
    return chunk
```

### 2. Path-to-URL Enhancements

Update the path mapping mechanism to preserve specific page paths:

```python
def get_complete_url(path: str, provider_type: str = "developer-docs") -> str:
    """
    Get a complete URL preserving full path to specific pages.
    
    Args:
        path: Document path
        provider_type: Documentation provider
        
    Returns:
        Complete URL with preserved path
    """
    # First check if path is already in registry
    url = path_registry.get_url(path)
    if url:
        return url
        
    # If not in registry, ensure we're using full path
    if path.endswith(("/", "index")):
        # This is a directory path, which is fine
        pass
    elif not path.endswith((".md", ".mdx")) and "/" in path:
        # This might be a truncated path, check if we have a more specific path
        possible_full_paths = [
            f"{path}.md",
            f"{path}.mdx",
            f"{path}/index.md",
            f"{path}/index.mdx"
        ]
        
        for full_path in possible_full_paths:
            full_url = path_registry.get_url(full_path)
            if full_url:
                return full_url
    
    # Fall back to standard URL generation
    return get_docs_url(path, provider_type)
```

### 3. Reference Construction Logic

Implement a post-processing step to correct truncated URLs in responses:

```python
def correct_reference_urls(response_text: str, anchor_page_map: Dict[str, str]) -> str:
    """
    Correct URLs in a response to ensure they point to specific pages.
    
    Args:
        response_text: The generated response
        anchor_page_map: Mapping of anchors to their specific pages
        
    Returns:
        Corrected response text
    """
    # Use regex to find URLs with fragments
    url_pattern = r'(https?://[^\s]+?/en/[^\s#]+)(#[^\s]+)?'
    
    def replace_url(match):
        base_url = match.group(1)
        fragment = match.group(2) or ""
        
        if fragment:
            # Strip the # from the fragment for lookup
            anchor = fragment[1:]
            if anchor in anchor_page_map:
                # Replace with the correct page URL
                correct_page = anchor_page_map[anchor]
                if not base_url.endswith(correct_page):
                    # URL is pointing to wrong page, fix it
                    correct_base = re.sub(r'/en/[^\s#]+$', f'/en/{correct_page}', base_url)
                    return f"{correct_base}{fragment}"
        
        return f"{base_url}{fragment}"
    
    return re.sub(url_pattern, replace_url, response_text)
```

### 4. URL-Anchor Mapping

Build a mapping of anchors to their specific pages during preprocessing:

```python
async def build_anchor_page_map() -> Dict[str, str]:
    """
    Build a mapping of anchor IDs to their specific pages.
    
    Returns:
        Dictionary mapping anchor IDs to their page paths
    """
    anchor_map = {}
    
    # Process all documentation pages
    for page_url in path_registry.get_all_urls():
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(page_url) as response:
                    if response.status == 200:
                        html = await response.text()
                        
                        # Extract all heading IDs (common anchor targets)
                        heading_pattern = r'<h[1-6][^>]*id=["\']([^"\']+)["\'][^>]*>'
                        matches = re.findall(heading_pattern, html)
                        
                        # Get relative path from URL
                        path = page_url.split('/en/')[1] if '/en/' in page_url else ""
                        
                        # Add to anchor map
                        for anchor_id in matches:
                            anchor_map[anchor_id] = path
        except Exception as e:
            logger.error(f"Error processing anchors for {page_url}: {e}")
    
    logger.info(f"Built anchor map with {len(anchor_map)} entries")
    return anchor_map
```

### 5. Enhanced Prompt Instructions 

Update system prompts to guide the construction of references:

```
When linking to documentation in your responses:

1. URL FORMAT:
   - Always use complete URLs that include the full path to the specific page
   - Never link to parent directories when referring to specific content
   - Example: Use "https://aptos.dev/en/concepts/staking" NOT "https://aptos.dev/en/concepts"

2. ANCHOR USAGE:
   - When referencing specific sections, ensure the anchor exists on that page
   - Format section references as: [Section Title](https://aptos.dev/en/path/to/page#section-id)
   - If uncertain about an anchor, link to the page without the anchor

3. REFERENCE STYLE:
   - Use descriptive link text that indicates what information is found at the link
   - Format: [descriptive text](complete URL with anchor)
   - For code references, include the module/function name in the link text

4. SOURCE ATTRIBUTION:
   - Clearly attribute information to its source
   - When combining information from multiple sources, reference each source
   - Use the source URLs exactly as provided in the context
```


