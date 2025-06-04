#!/usr/bin/env python3
"""Debug script to test URL mapping issues."""

import asyncio
import json
import yaml
from app.path_registry import path_registry

async def debug_url_mapping():
    """Debug the URL mapping issue."""
    
    print("=== URL Mapping Debug ===")
    
    # 1. Load and check URL mappings
    print("\n1. Loading URL mappings...")
    try:
        with open('data/generated/developer-docs/url_mappings.yaml', 'r') as f:
            url_mappings = yaml.safe_load(f)
        print(f"Loaded {len(url_mappings)} URL mappings")
        
        # Show some examples
        print("\nSample URL mappings:")
        count = 0
        for key, value in url_mappings.items():
            if 'system-integrators-guide' in key or 'exchanges' in key:
                print(f"  {key} -> {value}")
                count += 1
            if count >= 5:
                break
                
    except Exception as e:
        print(f"Error loading URL mappings: {e}")
        return
    
    # 2. Initialize path registry
    print("\n2. Initializing path registry...")
    try:
        await path_registry.initialize_from_config('data/generated/developer-docs/url_mappings.yaml')
        print("Path registry initialized successfully")
    except Exception as e:
        print(f"Error initializing path registry: {e}")
        return
    
    # 3. Test problematic paths
    print("\n3. Testing problematic paths...")
    test_paths = [
        'en/build/guides/system-integrators-guide.mdx',
        'en/build/guides/system-integrators-guide',
        'en/build/guides/exchanges.mdx', 
        'en/build/guides/exchanges',
        'en/data/developer-docs/apps/nextra/pages/en/build/guides/system-integrators-guide.mdx',
        'data/developer-docs/apps/nextra/pages/en/build/guides/system-integrators-guide.mdx'
    ]
    
    for path in test_paths:
        url = path_registry.get_url(path)
        print(f"Path: '{path}'")
        print(f"URL:  '{url}'")
        print("---")
    
    # 4. Check what's actually stored in enhanced chunks
    print("\n4. Checking enhanced chunks source paths...")
    try:
        with open('data/generated/developer-docs/enhanced_chunks.json', 'r') as f:
            chunks = json.load(f)
        
        print(f"Loaded {len(chunks)} chunks")
        
        # Find chunks related to system-integrators-guide
        system_integrator_chunks = []
        exchange_chunks = []
        
        for chunk in chunks:
            source = chunk.get('metadata', {}).get('source', '')
            if 'system-integrator' in source.lower():
                system_integrator_chunks.append(chunk)
            elif 'exchange' in source.lower():
                exchange_chunks.append(chunk)
        
        print(f"\nFound {len(system_integrator_chunks)} system-integrator chunks")
        for chunk in system_integrator_chunks[:3]:
            source = chunk.get('metadata', {}).get('source', '')
            url = chunk.get('metadata', {}).get('url', '')
            title = chunk.get('metadata', {}).get('title', '')
            print(f"  Source: '{source}'")
            print(f"  URL: '{url}'")
            print(f"  Title: '{title}'")
            print("  ---")
            
        print(f"\nFound {len(exchange_chunks)} exchange chunks")
        for chunk in exchange_chunks[:3]:
            source = chunk.get('metadata', {}).get('source', '')
            url = chunk.get('metadata', {}).get('url', '')
            title = chunk.get('metadata', {}).get('title', '')
            print(f"  Source: '{source}'")
            print(f"  URL: '{url}'")
            print(f"  Title: '{title}'")
            print("  ---")
            
    except Exception as e:
        print(f"Error loading enhanced chunks: {e}")

if __name__ == "__main__":
    asyncio.run(debug_url_mapping()) 