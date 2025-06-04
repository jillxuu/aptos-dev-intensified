#!/usr/bin/env python3
"""Test script to verify complete URL generation."""

import asyncio
import re
from app.path_registry import path_registry
from app.config import DOCS_BASE_URLS

def generate_anchor(section_title):
    """Generate URL anchor from section title."""
    if not section_title:
        return ""
    
    anchor = section_title.lower().replace(' ', '-').replace('(', '').replace(')', '').replace(',', '').replace(':', '').replace('–', '-').replace('—', '-')
    anchor = re.sub(r'[^a-z0-9\-]', '', anchor)
    anchor = re.sub(r'-+', '-', anchor).strip('-')
    return anchor

async def test_full_url_generation():
    """Test complete URL generation with real examples."""
    
    print("=== Testing Complete URL Generation ===")
    
    # Initialize path registry
    await path_registry.initialize_from_config('data/generated/developer-docs/url_mappings.yaml')
    
    # Test cases based on your examples
    test_cases = [
        {
            'source': 'en/data/developer-docs/apps/nextra/pages/en/build/guides/exchanges',
            'section': 'Fungible Asset Balances',
            'expected_base': 'https://aptos.dev/en/build/guides/exchanges',
            'expected_anchor': 'fungible-asset-balances'
        },
        {
            'source': 'en/data/developer-docs/apps/nextra/pages/en/build/guides/system-integrators-guide',
            'section': 'Current Balance for a Coin',
            'expected_base': 'https://aptos.dev/en/build/guides/system-integrators-guide',
            'expected_anchor': 'current-balance-for-a-coin'
        },
        {
            'source': 'en/data/developer-docs/apps/nextra/pages/en/build/smart-contracts/fungible-asset',
            'section': 'Creating Fungible Assets',
            'expected_base': 'https://aptos.dev/en/build/smart-contracts/fungible-asset',
            'expected_anchor': 'creating-fungible-assets'
        }
    ]
    
    base_url = DOCS_BASE_URLS['developer-docs']
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i} ---")
        source_path = test_case['source']
        section_title = test_case['section']
        
        print(f"Source: {source_path}")
        print(f"Section: {section_title}")
        
        # Apply the same logic as the fixed code
        if "data/developer-docs/apps/nextra/pages/en/" in source_path:
            # Extract the correct relative path from the last /en/ occurrence
            en_index = source_path.rfind("/en/")
            if en_index != -1:
                corrected_path = "en/" + source_path[en_index + 4:]
                print(f"Corrected path: {corrected_path}")
                
                # Try to get URL with corrected path
                chunk_url = path_registry.get_url(corrected_path)
                print(f"Registry URL: {chunk_url}")
                
                if chunk_url:
                    # Add the base URL to create full URL
                    full_url = f"{base_url}/{chunk_url.lstrip('/')}"
                    
                    # Add anchor based on section title
                    if section_title:
                        anchor = generate_anchor(section_title)
                        if anchor:
                            full_url += f"#{anchor}"
                    
                    print(f"Generated URL: {full_url}")
                    expected_full = f"{test_case['expected_base']}#{test_case['expected_anchor']}"
                    print(f"Expected URL:  {expected_full}")
                    
                    if full_url == expected_full:
                        print("✅ PASS")
                    else:
                        print("❌ FAIL")
                else:
                    print("❌ No URL found in registry")
            else:
                print("❌ Could not find /en/ in path")
        else:
            print("❌ Malformed pattern not detected")

if __name__ == "__main__":
    asyncio.run(test_full_url_generation()) 