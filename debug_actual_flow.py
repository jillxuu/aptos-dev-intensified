#!/usr/bin/env python3
"""Debug script to trace the actual URL generation flow."""

import asyncio
import re
import json
from app.path_registry import path_registry
from app.config import DOCS_BASE_URLS

async def debug_actual_flow():
    """Debug the actual flow to see what's happening."""
    
    print("=== Debugging Actual URL Generation Flow ===")
    
    # Initialize path registry
    await path_registry.initialize_from_config('data/generated/developer-docs/url_mappings.yaml')
    
    # Simulate the actual chunks that would be retrieved
    # Based on the head output, let's use real chunk data
    mock_chunks = [
        {
            'source': 'en/data/developer-docs/apps/nextra/pages/en/build/indexer/indexer-api/indexer-reference',
            'section': 'Fungible Asset Balances',
            'content': 'Content about fungible asset balances...'
        },
        {
            'source': 'en/data/developer-docs/apps/nextra/pages/en/build/guides/exchanges',
            'section': 'Exchange Integration',
            'content': 'Content about exchange integration...'
        },
        {
            'source': 'en/build/guides/exchanges',  # Normal path
            'section': 'Normal Exchange Guide',
            'content': 'Content about normal exchange guide...'
        }
    ]
    
    base_url = DOCS_BASE_URLS['developer-docs']
    formatted_context = ""
    
    print(f"Base URL: {base_url}")
    print(f"Path registry initialized with {len(path_registry.get_all_urls())} URLs")
    
    # Show some sample URLs from registry
    sample_urls = list(path_registry.get_all_urls())[:10]
    print(f"Sample URLs in registry: {sample_urls}")
    
    print("\n=== Processing Chunks ===")
    
    for i, chunk in enumerate(mock_chunks, 1):
        print(f"\n--- Chunk {i} ---")
        source_path = chunk.get('source', '')
        section_title = chunk.get('section', '')
        
        print(f"Source: '{source_path}'")
        print(f"Section: '{section_title}'")
        
        if source_path:
            # Apply the current logic from chat.py
            malformed_pattern = "data/developer-docs/apps/nextra/pages/en/"
            print(f"Contains malformed pattern: {malformed_pattern in source_path}")
            
            if malformed_pattern in source_path:
                print("  -> Entering malformed path correction")
                # Extract the correct relative path from the last /en/ occurrence
                en_index = source_path.rfind("/en/")
                print(f"  -> Last /en/ index: {en_index}")
                
                if en_index != -1:
                    corrected_path = "en/" + source_path[en_index + 4:]
                    print(f"  -> Corrected path: '{corrected_path}'")
                    
                    # Try to get URL with corrected path
                    chunk_url = path_registry.get_url(corrected_path)
                    print(f"  -> Registry lookup result: '{chunk_url}'")
                    
                    if chunk_url:
                        # Add the base URL to create full URL
                        full_url = f"{base_url}/{chunk_url.lstrip('/')}"
                        print(f"  -> Base URL + chunk URL: '{full_url}'")
                        
                        # Add anchor based on section title
                        if section_title:
                            # Convert section title to proper URL anchor format
                            anchor = section_title.lower().replace(' ', '-').replace('(', '').replace(')', '').replace(',', '').replace(':', '').replace('–', '-').replace('—', '-')
                            # Remove any remaining special characters and multiple dashes
                            anchor = re.sub(r'[^a-z0-9\-]', '', anchor)
                            anchor = re.sub(r'-+', '-', anchor).strip('-')
                            print(f"  -> Generated anchor: '{anchor}'")
                            
                            if anchor:
                                full_url += f"#{anchor}"
                        
                        print(f"  -> Final URL: '{full_url}'")
                        formatted_context += f"\n\nSection: {section_title} - {full_url}\n"
                    else:
                        print("  -> No URL found in registry")
                        formatted_context += f"\n\nSection: {section_title}\n"
                else:
                    print("  -> Could not find /en/ in path")
                    formatted_context += f"\n\nSection: {section_title}\n"
            else:
                print("  -> Using normal URL lookup")
                # Try normal URL lookup
                chunk_url = path_registry.get_url(source_path)
                print(f"  -> Registry lookup result: '{chunk_url}'")
                
                if chunk_url:
                    # Add the base URL to create full URL
                    full_url = f"{base_url}/{chunk_url.lstrip('/')}"
                    print(f"  -> Base URL + chunk URL: '{full_url}'")
                    
                    # Add anchor based on section title
                    if section_title:
                        # Convert section title to proper URL anchor format
                        anchor = section_title.lower().replace(' ', '-').replace('(', '').replace(')', '').replace(',', '').replace(':', '').replace('–', '-').replace('—', '-')
                        # Remove any remaining special characters and multiple dashes
                        anchor = re.sub(r'[^a-z0-9\-]', '', anchor)
                        anchor = re.sub(r'-+', '-', anchor).strip('-')
                        print(f"  -> Generated anchor: '{anchor}'")
                        
                        if anchor:
                            full_url += f"#{anchor}"
                    
                    print(f"  -> Final URL: '{full_url}'")
                    formatted_context += f"\n\nSection: {section_title} - {full_url}\n"
                else:
                    print("  -> No URL found in registry")
                    formatted_context += f"\n\nSection: {section_title}\n"
        else:
            formatted_context += f"\n\nSection: {section_title}\n"
        
        formatted_context += f"Content: {chunk.get('content', '')}\n"
    
    print(f"\n=== Final Context (what LLM sees) ===")
    print(formatted_context)
    
    print(f"\n=== Analysis ===")
    print("If the LLM is still generating incorrect URLs like:")
    print("  https://aptos.dev/en/build/guides/exchanges#buildguides")
    print("Then the issue is NOT in our context formatting.")
    print("The issue must be in:")
    print("  1. The LLM ignoring our provided URLs")
    print("  2. The LLM creating its own URLs despite instructions")
    print("  3. Some other part of the system overriding our URLs")

if __name__ == "__main__":
    asyncio.run(debug_actual_flow()) 