#!/usr/bin/env python3
"""Test script using the exact production template and logic."""

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

# Import the actual production template
from app.routes.chat import BASE_TEMPLATE, PROVIDER_TEMPLATES

async def test_production_template():
    """Test using the exact production template and logic."""
    
    print("=== Testing Production Template ===")
    
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
    
    # Get relevant context (same as production)
    try:
        context_chunks = await rag_provider.get_relevant_context(
            question,
            k=7,
            include_series=False,  # Not a process query
            provider_type="developer-docs",
            use_multi_step=False
        )
        print(f"✅ Retrieved {len(context_chunks)} context chunks")
    except Exception as e:
        print(f"❌ Failed to retrieve context: {e}")
        return
    
    # Format context using the EXACT same logic as production
    base_url = DOCS_BASE_URLS['developer-docs']
    formatted_context = ""
    
    for chunk in context_chunks:
        # Fix malformed source paths on-the-fly for URL lookup (same as production)
        source_path = chunk.get('source', '')
        if source_path:
            # Handle malformed paths from old chunk processing
            if "data/developer-docs/apps/nextra/pages/en/" in source_path:
                # Extract the correct relative path from the last /en/ occurrence
                en_index = source_path.rfind("/en/")
                if en_index != -1:
                    corrected_path = "en/" + source_path[en_index + 4:]
                    # Try to get URL with corrected path
                    chunk_url = path_registry.get_url(corrected_path)
                    if chunk_url:
                        # Add the base URL to create full URL
                        full_url = f"{base_url}/{chunk_url.lstrip('/')}"
                        # Extract section title from content if title field is empty
                        section_title = chunk.get('title', '')
                        if not section_title:
                            # Try to extract from content pattern like "Context: ... > Section Title"
                            content = chunk.get('content', '')
                            if content.startswith('Context:'):
                                # Extract the last part after the last '>'
                                lines = content.split('\n')
                                if lines:
                                    first_line = lines[0]
                                    if '>' in first_line:
                                        section_title = first_line.split('>')[-1].strip()
                                    else:
                                        # If no '>', take everything after "Context: "
                                        section_title = first_line.replace('Context:', '').strip()
                        
                        if section_title:
                            # Convert section title to proper URL anchor format
                            anchor = section_title.lower().replace(' ', '-').replace('(', '').replace(')', '').replace(',', '').replace(':', '').replace('–', '-').replace('—', '-')
                            # Remove any remaining special characters and multiple dashes
                            anchor = re.sub(r'[^a-z0-9\-]', '', anchor)
                            anchor = re.sub(r'-+', '-', anchor).strip('-')
                            if anchor:
                                full_url += f"#{anchor}"
                        formatted_context += f"\n\nSection: {chunk.get('section', '')} - {full_url}\n"
                    else:
                        formatted_context += f"\n\nSection: {chunk.get('section', '')}\n"
                else:
                    formatted_context += f"\n\nSection: {chunk.get('section', '')}\n"
            else:
                # Try normal URL lookup
                chunk_url = path_registry.get_url(source_path)
                if chunk_url:
                    # Add the base URL to create full URL
                    full_url = f"{base_url}/{chunk_url.lstrip('/')}"
                    # Extract section title from content if title field is empty
                    section_title = chunk.get('title', '')
                    if not section_title:
                        # Try to extract from content pattern like "Context: ... > Section Title"
                        content = chunk.get('content', '')
                        if content.startswith('Context:'):
                            # Extract the last part after the last '>'
                            lines = content.split('\n')
                            if lines:
                                first_line = lines[0]
                                if '>' in first_line:
                                    section_title = first_line.split('>')[-1].strip()
                                else:
                                    # If no '>', take everything after "Context: "
                                    section_title = first_line.replace('Context:', '').strip()
                    
                    if section_title:
                        # Convert section title to proper URL anchor format
                        anchor = section_title.lower().replace(' ', '-').replace('(', '').replace(')', '').replace(',', '').replace(':', '').replace('–', '-').replace('—', '-')
                        # Remove any remaining special characters and multiple dashes
                        anchor = re.sub(r'[^a-z0-9\-]', '', anchor)
                        anchor = re.sub(r'-+', '-', anchor).strip('-')
                        if anchor:
                            full_url += f"#{anchor}"
                    formatted_context += f"\n\nSection: {chunk.get('section', '')} - {full_url}\n"
                else:
                    formatted_context += f"\n\nSection: {chunk.get('section', '')}\n"
        else:
            formatted_context += f"\n\nSection: {chunk.get('section', '')}\n"
        
        if chunk.get("summary"):
            formatted_context += f"Summary: {chunk.get('summary')}\n"
        formatted_context += f"Content: {chunk.get('content', '')}\n"
    
    # Use the exact production template
    template = PROVIDER_TEMPLATES["developer-docs"]
    main_topic = "fungible assets"
    
    # Prepare the prompt with the context (same as production)
    prompt = template.format(
        context=formatted_context,
        question=question,
        main_topic=main_topic,
    )
    
    # Call OpenAI API with same settings as production
    print("\n=== Calling OpenAI API (Production Settings) ===")
    try:
        response = client.chat.completions.create(
            model="gpt-4",  # Same as production
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": question}
            ],
            temperature=0.05  # Same as production
        )
        
        ai_response = response.choices[0].message.content
        print("✅ OpenAI API call successful")
        
    except Exception as e:
        print(f"❌ OpenAI API call failed: {e}")
        return
    
    print(f"\n=== Production AI Response ===")
    print("=" * 80)
    print(ai_response)
    print("=" * 80)
    
    # Analyze the response for the 3 methods
    print(f"\n=== Production Response Analysis ===")
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
    asyncio.run(test_production_template()) 