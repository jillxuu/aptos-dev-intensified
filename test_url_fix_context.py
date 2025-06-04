#!/usr/bin/env python3
"""Test script to verify URL fix in context formatting."""

import asyncio
from app.path_registry import path_registry
from app.config import DOCS_BASE_URLS

async def test_context_formatting():
    """Test the context formatting with URL fix."""
    
    print("=== Testing Context URL Fix ===")
    
    # Initialize path registry
    await path_registry.initialize_from_config('data/generated/developer-docs/url_mappings.yaml')
    
    # Mock chunk data similar to what we see in the system
    # Adding more test cases based on the actual problematic URLs
    mock_chunks = [
        {
            'source': 'en/data/developer-docs/apps/nextra/pages/en/build/guides/system-integrators-guide',
            'section': 'Current Balance for a Coin',
            'summary': 'How to check current balance for a coin',
            'content': 'To check the current balance...'
        },
        {
            'source': 'en/data/developer-docs/apps/nextra/pages/en/build/guides/exchanges',
            'section': 'Fungible Asset Balances',
            'summary': 'How to check fungible asset balances',
            'content': 'To check fungible asset balances...'
        },
        # Add test cases for the problematic URLs you mentioned
        {
            'source': 'en/data/developer-docs/apps/nextra/pages/en/build/smart-contracts/fungible-asset',
            'section': 'Build Smart Contracts',
            'summary': 'How to build smart contracts',
            'content': 'To build smart contracts...'
        },
        {
            'source': 'en/build/smart-contracts/fungible-asset',  # Already correct format
            'section': 'Fungible Asset Creation',
            'summary': 'How to create fungible assets',
            'content': 'To create fungible assets...'
        },
        {
            'source': 'en/build/guides/exchanges',  # Already correct format
            'section': 'Exchange Integration',
            'summary': 'How to integrate with exchanges',
            'content': 'To integrate with exchanges...'
        }
    ]
    
    base_url = DOCS_BASE_URLS['developer-docs']
    formatted_context = ""
    
    # Apply the same logic as in the fixed code
    for chunk in mock_chunks:
        print(f"\nProcessing chunk: {chunk['section']}")
        source_path = chunk.get('source', '')
        print(f"Original source path: '{source_path}'")
        
        if source_path:
            # Handle malformed paths from old chunk processing
            malformed_pattern = "data/developer-docs/apps/nextra/pages/en/"
            print(f"Checking if malformed pattern in source_path: {malformed_pattern in source_path}")
            
            if malformed_pattern in source_path:
                print("Entering malformed path correction logic")
                # Extract the correct relative path
                en_index = source_path.rfind("/en/")
                print(f"Found /en/ at index: {en_index}")
                if en_index != -1:
                    corrected_path = "en/" + source_path[en_index + 4:]
                    print(f"Corrected path: '{corrected_path}'")
                    # Try to get URL with corrected path
                    chunk_url = path_registry.get_url(corrected_path)
                    print(f"Chunk URL from registry: '{chunk_url}'")
                    if chunk_url:
                        # Add the base URL to create full URL
                        full_url = f"{base_url}/{chunk_url.lstrip('/')}"
                        # Add anchor if we have a title that could be an anchor
                        title = chunk.get('section', '').lower()
                        print(f"Section title: '{title}'")
                        print(f"Corrected path last part: '{corrected_path.split('/')[-1]}'")
                        if title and title != corrected_path.split('/')[-1]:
                            # Convert title to URL anchor format
                            anchor = title.replace(' ', '-').replace('(', '').replace(')', '').replace(',', '')
                            print(f"Generated anchor: '{anchor}'")
                            full_url += f"#{anchor}"
                        formatted_context += f"\n\nSection: {chunk.get('section', '')} - {full_url}\n"
                        print(f"Generated URL: {full_url}")
                    else:
                        formatted_context += f"\n\nSection: {chunk.get('section', '')}\n"
                        print("No URL found in registry")
                else:
                    formatted_context += f"\n\nSection: {chunk.get('section', '')}\n"
                    print("Could not find /en/ in path")
            else:
                print("Using normal URL lookup")
                # Try normal URL lookup
                chunk_url = path_registry.get_url(source_path)
                print(f"Normal lookup result: '{chunk_url}'")
                if chunk_url:
                    full_url = f"{base_url}/{chunk_url.lstrip('/')}"
                    title = chunk.get('section', '').lower()
                    print(f"Section title: '{title}'")
                    print(f"Source path last part: '{source_path.split('/')[-1]}'")
                    if title and title != source_path.split('/')[-1]:
                        anchor = title.replace(' ', '-').replace('(', '').replace(')', '').replace(',', '')
                        print(f"Generated anchor: '{anchor}'")
                        full_url += f"#{anchor}"
                    formatted_context += f"\n\nSection: {chunk.get('section', '')} - {full_url}\n"
                    print(f"Generated URL: {full_url}")
                else:
                    formatted_context += f"\n\nSection: {chunk.get('section', '')}\n"
                    print("No URL found in registry")
        else:
            formatted_context += f"\n\nSection: {chunk.get('section', '')}\n"
        
        if chunk.get("summary"):
            formatted_context += f"Summary: {chunk.get('summary')}\n"
        formatted_context += f"Content: {chunk.get('content', '')}\n"
        print("---")
    
    print("\nFormatted Context:")
    print(formatted_context)

if __name__ == "__main__":
    asyncio.run(test_context_formatting()) 