#!/usr/bin/env python3
"""Test script to check current response for the fungible assets question."""

import asyncio
import os
import json
from openai import OpenAI
from app.rag_providers import RAGProviderRegistry
from app.rag_providers.docs_provider import docs_provider
from app.config import DOCS_BASE_URLS

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def test_current_response():
    """Test the current response for the fungible assets question."""
    
    print("=== Testing Current Response ===")
    
    question = "How can I query fungible assets and coin balance?"
    print(f"Question: {question}")
    
    # Initialize RAG provider
    try:
        rag_provider = RAGProviderRegistry.get_provider("docs")
        await rag_provider.switch_provider("developer-docs")
        print("✅ RAG provider initialized")
    except Exception as e:
        print(f"❌ Failed to initialize RAG provider: {e}")
        return
    
    # Get relevant context
    try:
        context_chunks = await rag_provider.get_relevant_context(
            question,
            k=7,  # Same as in the actual code
            include_series=False,  # Not a process query
            provider_type="developer-docs",
            use_multi_step=False
        )
        print(f"✅ Retrieved {len(context_chunks)} context chunks")
    except Exception as e:
        print(f"❌ Failed to retrieve context: {e}")
        return
    
    # Analyze the chunks
    print(f"\n=== Chunk Analysis ===")
    unique_sources = set()
    for i, chunk in enumerate(context_chunks):
        source = chunk.get('source', 'Unknown')
        content_preview = chunk.get('content', '')[:100] + "..."
        unique_sources.add(source)
        print(f"Chunk {i+1}: {source}")
        print(f"  Content: {content_preview}")
        print()
    
    print(f"Unique sources: {len(unique_sources)}")
    for source in sorted(unique_sources):
        print(f"  - {source}")
    
    # Check if we have the expected sources for the 3 methods
    expected_sources = [
        "en/build/guides/exchanges",  # Method 1 & 2: coin balance and fungible asset balance
        "en/build/guides/system-integrators-guide",  # Method 1: coin balance  
        "en/build/indexer/indexer-api/fungible-asset-balances"  # Method 3: GetFungibleAssetBalances
    ]
    
    print(f"\n=== Expected Sources Check ===")
    for expected in expected_sources:
        found = any(expected in chunk.get('source', '') for chunk in context_chunks)
        print(f"{'✅' if found else '❌'} {expected}: {'Found' if found else 'Missing'}")
    
    # Look for specific content patterns
    print(f"\n=== Content Pattern Analysis ===")
    patterns = {
        "coin::balance": "0x1::coin::balance",
        "primary_fungible_store::balance": "0x1::primary_fungible_store::balance", 
        "GetFungibleAssetBalances": "GetFungibleAssetBalances"
    }
    
    for pattern_name, pattern in patterns.items():
        found_chunks = []
        for i, chunk in enumerate(context_chunks):
            if pattern in chunk.get('content', ''):
                found_chunks.append(i+1)
        
        if found_chunks:
            print(f"✅ {pattern_name}: Found in chunks {found_chunks}")
        else:
            print(f"❌ {pattern_name}: Not found in any chunk")

if __name__ == "__main__":
    asyncio.run(test_current_response()) 