#!/usr/bin/env python3
"""Test script to get the actual LLM response for the fungible assets question."""

import asyncio
import os
import json
import re
from openai import OpenAI
from app.rag_providers import RAGProviderRegistry
from app.rag_providers.docs_provider import docs_provider
from app.config import DOCS_BASE_URLS
from app.path_registry import path_registry

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Simplified template for testing
TEST_TEMPLATE = """You are an AI assistant specialized in Aptos blockchain technology. Based on the following documentation context, provide a comprehensive answer to the user's question.

Context:
{context}

Instructions:
- Provide a complete answer covering all methods mentioned in the context
- Use exact URLs provided in the context for citations
- Include code examples when available
- Be comprehensive and don't omit any methods that are documented

Question: {question}"""

async def test_full_response():
    """Test the full LLM response to see what's being generated."""
    
    print("=== Testing Full LLM Response ===")
    
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
    
    # Initialize path registry
    await path_registry.initialize_from_config('data/generated/developer-docs/url_mappings.yaml')
    
    # Get relevant context
    try:
        context_chunks = await rag_provider.get_relevant_context(
            question,
            k=7,
            include_series=False,
            provider_type="developer-docs",
            use_multi_step=False
        )
        print(f"✅ Retrieved {len(context_chunks)} context chunks")
    except Exception as e:
        print(f"❌ Failed to retrieve context: {e}")
        return
    
    # Format context (simplified version)
    base_url = DOCS_BASE_URLS['developer-docs']
    formatted_context = ""
    
    for chunk in context_chunks:
        source_path = chunk.get('source', '')
        if source_path:
            chunk_url = path_registry.get_url(source_path)
            if chunk_url:
                full_url = f"{base_url}/{chunk_url.lstrip('/')}"
                formatted_context += f"\n\nSection: {chunk.get('section', '')} - {full_url}\n"
            else:
                formatted_context += f"\n\nSection: {chunk.get('section', '')}\n"
        else:
            formatted_context += f"\n\nSection: {chunk.get('section', '')}\n"
        
        if chunk.get("summary"):
            formatted_context += f"Summary: {chunk.get('summary')}\n"
        formatted_context += f"Content: {chunk.get('content', '')}\n"
    
    # Prepare prompt
    prompt = TEST_TEMPLATE.format(context=formatted_context, question=question)
    
    # Call OpenAI API
    print("\n=== Calling OpenAI API ===")
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": question}
            ],
            temperature=0.05
        )
        
        ai_response = response.choices[0].message.content
        print("✅ OpenAI API call successful")
        
    except Exception as e:
        print(f"❌ OpenAI API call failed: {e}")
        return
    
    print(f"\n=== AI Response ===")
    print("=" * 80)
    print(ai_response)
    print("=" * 80)
    
    # Analyze the response for the 3 methods
    print(f"\n=== Response Analysis ===")
    methods = {
        "Method 1 (coin::balance)": ["coin::balance", "0x1::coin::balance"],
        "Method 2 (primary_fungible_store)": ["primary_fungible_store", "0x1::primary_fungible_store::balance"],
        "Method 3 (GetFungibleAssetBalances)": ["GetFungibleAssetBalances", "fungible asset balances query"]
    }
    
    for method_name, patterns in methods.items():
        found = any(pattern.lower() in ai_response.lower() for pattern in patterns)
        print(f"{'✅' if found else '❌'} {method_name}: {'Present' if found else 'Missing'}")
        
        if found:
            # Find which patterns matched
            matched_patterns = [p for p in patterns if p.lower() in ai_response.lower()]
            print(f"    Matched patterns: {matched_patterns}")

if __name__ == "__main__":
    asyncio.run(test_full_response()) 