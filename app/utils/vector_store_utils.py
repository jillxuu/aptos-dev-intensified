"""
Vector store utilities and extensions.

This module provides utility functions and extensions for vector stores,
including batch operations for improved performance.
"""

import logging
import time
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)

async def batch_similarity_search_with_score(
    vector_store: FAISS,
    queries: List[str],
    k: int = 4,
) -> List[List[Tuple[Document, float]]]:
    """
    Perform batch similarity search for multiple queries with improved performance.
    
    This function optimizes FAISS search by batching multiple queries together
    into a single operation, which is significantly faster than running
    individual queries in parallel.
    
    Args:
        vector_store: Initialized FAISS vector store
        queries: List of query strings to search for
        k: Number of results to return per query
        
    Returns:
        List of document-score pairs for each query
    """
    start_time = time.time()
    logger.info(f"[VECTOR-STORE] Starting batch similarity search for {len(queries)} queries")
    
    if not queries:
        logger.warning("[VECTOR-STORE] No queries provided for batch search")
        return []
    
    try:
        # Access the internal FAISS index and embeddings
        index = vector_store.index
        embedding_function = vector_store.embedding_function
        docstore = vector_store.docstore
        
        # Get embeddings for all queries in a single batch operation
        embed_start = time.time()
        
        # Generate embeddings for all queries at once (OpenAI can handle batching internally)
        query_embeddings = embedding_function.embed_documents(queries)
        
        embed_time = time.time() - embed_start
        logger.info(f"[VECTOR-STORE] Batch embedding completed in {embed_time:.4f}s")
        
        # Perform batch search in FAISS
        search_start = time.time()
        
        # Convert to numpy array for FAISS
        query_embeds_np = np.array(query_embeddings, dtype=np.float32)
        
        # Batch search using FAISS's native search_batch capability
        scores_batch, indices_batch = index.search(query_embeds_np, k)
        
        search_time = time.time() - search_start
        logger.info(f"[VECTOR-STORE] FAISS batch search completed in {search_time:.4f}s")
        
        # Process results
        process_start = time.time()
        all_results = []
        
        # Check if we have access to the docstore._dict which contains all documents
        docstore_dict = getattr(docstore, "_dict", None)
        if not docstore_dict:
            logger.warning("[VECTOR-STORE] Could not access docstore._dict, falling back to individual lookups")
        
        # Create a mapping of document IDs for faster lookups
        doc_mapping = {}
        if hasattr(vector_store, "index_to_docstore_id"):
            logger.info("[VECTOR-STORE] Using docstore_id mapping for document retrieval")
            
        # For each query, process its results
        for i, (scores, indices) in enumerate(zip(scores_batch, indices_batch)):
            query_results = []
            
            # For each result of this query
            for j, (score, idx) in enumerate(zip(scores, indices)):
                # Skip if idx is -1 (no result) or score is too low
                if idx == -1:
                    continue
                
                try:
                    # Convert FAISS index to docstore ID if needed
                    if hasattr(vector_store, "index_to_docstore_id"):
                        docstore_id = vector_store.index_to_docstore_id.get(idx)
                        if not docstore_id:
                            logger.warning(f"[VECTOR-STORE] No docstore ID found for index {idx}")
                            continue
                    else:
                        docstore_id = str(idx)
                    
                    # Try to get the document from the docstore
                    doc = None
                    
                    # First try using search method
                    doc = docstore.search(docstore_id)
                    
                    # If that failed, try direct dictionary access
                    if not doc and docstore_dict:
                        doc = docstore_dict.get(docstore_id)
                    
                    # If still no document but we have the docstore_id, create a fallback Document
                    if not doc:
                        logger.warning(f"[VECTOR-STORE] Document with ID {docstore_id} not found in docstore")
                        
                        # Try to get document from the vector store's document store
                        if hasattr(vector_store, "_documents") and isinstance(vector_store._documents, list):
                            # Try to find document in the cached list
                            for cached_doc in vector_store._documents:
                                if str(idx) == getattr(cached_doc, "id", None) or docstore_id == getattr(cached_doc, "id", None):
                                    doc = cached_doc
                                    break
                        
                        # If we still don't have a document, log and skip
                        if not doc:
                            logger.error(f"[VECTOR-STORE] Could not retrieve document for index {idx} / docstore_id {docstore_id}")
                            continue
                    
                    # Verify that we have a proper Document object
                    if not isinstance(doc, Document):
                        # If we got a string or other primitive, try to convert it to a Document
                        if isinstance(doc, str):
                            logger.warning(f"[VECTOR-STORE] Got string instead of Document, creating fallback Document")
                            doc = Document(page_content=doc, metadata={})
                        else:
                            logger.warning(f"[VECTOR-STORE] Expected Document object but got {type(doc)}, skipping")
                            continue
                    
                    # Add to results
                    query_results.append((doc, float(score)))
                except Exception as e:
                    logger.error(f"[VECTOR-STORE] Error processing result {j} for query {i}: {str(e)}")
                    continue
            
            all_results.append(query_results)
        
        process_time = time.time() - process_start
        logger.info(f"[VECTOR-STORE] Results processing completed in {process_time:.4f}s")
        
        # Log the number of results found
        result_count = sum(len(results) for results in all_results)
        logger.info(f"[VECTOR-STORE] Found {result_count} documents across {len(queries)} queries")
        
        total_time = time.time() - start_time
        logger.info(f"[VECTOR-STORE] Batch search completed in {total_time:.4f}s")
        logger.info(f"[VECTOR-STORE] PERFORMANCE SUMMARY:")
        logger.info(f"[VECTOR-STORE] - Embedding: {embed_time:.4f}s ({(embed_time/total_time)*100:.1f}%)")
        logger.info(f"[VECTOR-STORE] - FAISS search: {search_time:.4f}s ({(search_time/total_time)*100:.1f}%)")
        logger.info(f"[VECTOR-STORE] - Results processing: {process_time:.4f}s ({(process_time/total_time)*100:.1f}%)")
        
        return all_results
    
    except Exception as e:
        logger.error(f"[VECTOR-STORE] Error during batch similarity search: {str(e)}")
        # Return empty results for each query
        return [[] for _ in queries]


async def similarity_search_with_batch_support(
    vector_store: FAISS,
    query: str,
    queries: Optional[List[str]] = None,
    k: int = 4,
) -> List[Tuple[Document, float]]:
    """
    Smart similarity search that dispatches to either single or batch search.
    
    This function acts as a router that decides whether to use the standard
    similarity_search_with_score or the optimized batch_similarity_search_with_score
    based on the input. Use this as a drop-in replacement for the standard search.
    
    Args:
        vector_store: Initialized FAISS vector store
        query: Single query string (for backward compatibility)
        queries: Optional list of queries (if provided, uses batch search)
        k: Number of results to return per query
        
    Returns:
        List of document-score pairs for the query
    """
    # If a list of queries is provided, use batch search and return the first result
    if queries and len(queries) > 1:
        logger.info(f"[VECTOR-STORE] Dispatching to batch search for {len(queries)} queries")
        all_results = await batch_similarity_search_with_score(vector_store, queries, k)
        
        # Merge all results (useful for primary search where we want all results)
        # This combines results from all queries, uniquely by document ID
        merged_results = {}
        
        for query_results in all_results:
            for doc, score in query_results:
                # Ensure we're working with a Document object
                if not isinstance(doc, Document):
                    logger.warning(f"[VECTOR-STORE] Unexpected result type: {type(doc)}, skipping")
                    continue
                
                # Create a unique ID for this document
                # Try to use existing ID from metadata, fall back to content hash
                doc_id = None
                
                # Try to get ID from metadata
                if hasattr(doc, 'metadata') and isinstance(doc.metadata, dict):
                    doc_id = doc.metadata.get("id") or doc.metadata.get("chunk_id")
                
                # Fall back to content hash if no ID in metadata
                if not doc_id and hasattr(doc, 'page_content'):
                    # Use first 50 chars of content as a simple hash if no ID
                    content_sample = doc.page_content[:50] if len(doc.page_content) > 0 else "empty"
                    doc_id = hash(content_sample)
                
                # If we still don't have an ID, use object hash
                if not doc_id:
                    doc_id = id(doc)
                
                # Keep the highest scoring version of each document
                if doc_id not in merged_results or score > merged_results[doc_id][1]:
                    merged_results[doc_id] = (doc, score)
        
        # Convert back to list and sort by score
        results = list(merged_results.values())
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Log the total number of unique documents found
        logger.info(f"[VECTOR-STORE] Found {len(results)} unique documents after merging results")
        
        # Limit to top k results
        return results[:k]
    
    # For single query, use the standard search
    elif query:
        logger.info(f"[VECTOR-STORE] Using standard search for single query")
        try:
            return vector_store.similarity_search_with_score(query, k=k)
        except Exception as e:
            logger.error(f"[VECTOR-STORE] Error during standard similarity search: {str(e)}")
            return []
    
    # If neither query nor queries are provided
    else:
        logger.warning("[VECTOR-STORE] No queries provided for search")
        return [] 