#!/usr/bin/env python3
"""Test script to verify improved anchor generation."""

import re

def generate_anchor(section_title):
    """Generate URL anchor from section title using the same logic as the fixed code."""
    if not section_title:
        return ""
    
    # Convert section title to proper URL anchor format
    anchor = section_title.lower().replace(' ', '-').replace('(', '').replace(')', '').replace(',', '').replace(':', '').replace('–', '-').replace('—', '-')
    # Remove any remaining special characters and multiple dashes
    anchor = re.sub(r'[^a-z0-9\-]', '', anchor)
    anchor = re.sub(r'-+', '-', anchor).strip('-')
    return anchor

def test_anchor_generation():
    """Test anchor generation with various section titles."""
    
    test_cases = [
        # Original problematic cases
        ("Current Balance for a Coin", "current-balance-for-a-coin"),
        ("Fungible Asset Balances", "fungible-asset-balances"),
        
        # Cases with special characters
        ("Exchange Integration Guide – Retrieving Balances", "exchange-integration-guide-retrieving-balances"),
        ("Build Smart Contracts: Overview", "build-smart-contracts-overview"),
        ("API Reference (v1.0)", "api-reference-v10"),
        ("Getting Started — Quick Setup", "getting-started-quick-setup"),
        
        # Edge cases
        ("", ""),
        ("Simple", "simple"),
        ("Multiple   Spaces", "multiple-spaces"),
        ("Special!@#$%Characters", "specialcharacters"),
        ("Dashes-Already-Present", "dashes-already-present"),
    ]
    
    print("=== Testing Anchor Generation ===")
    
    for section_title, expected in test_cases:
        result = generate_anchor(section_title)
        status = "✅" if result == expected else "❌"
        print(f"{status} '{section_title}' -> '{result}' (expected: '{expected}')")
        
        if result != expected:
            print(f"   MISMATCH: got '{result}', expected '{expected}'")

if __name__ == "__main__":
    test_anchor_generation() 