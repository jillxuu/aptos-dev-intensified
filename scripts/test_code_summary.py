#!/usr/bin/env python3
"""Test script for code summary generation."""

import os
import sys
import asyncio
import logging
from pathlib import Path

# Add the project root to Python path
project_root = str(Path(__file__).parent.parent.absolute())
sys.path.insert(0, project_root)

# Load environment variables from .env file
from dotenv import load_dotenv
env_path = os.path.join(project_root, '.env')
load_dotenv(env_path)
print(f"Loaded environment variables from {env_path}")
print(f"OPENAI_API_KEY exists: {os.getenv('OPENAI_API_KEY') is not None}")
print(f"CHAT_TEST_MODE: {os.getenv('CHAT_TEST_MODE')}")

from scripts.preprocess_topic_chunks import generate_code_summary

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Test code block
CODE_BLOCK = """```rust
// Example of Move module
module 0x1::hello_world {
    use std::string;
    use std::signer;

    struct MessageHolder has key {
        message: string::String,
    }

    public fun set_message(account: &signer, message: string::String) acquires MessageHolder {
        let account_addr = signer::address_of(account);
        if (!exists<MessageHolder>(account_addr)) {
            move_to(account, MessageHolder { message })
        } else {
            let old_message_holder = borrow_global_mut<MessageHolder>(account_addr);
            old_message_holder.message = message;
        }
    }

    public fun get_message(addr: address): string::String acquires MessageHolder {
        assert!(exists<MessageHolder>(addr), 0);
        *&borrow_global<MessageHolder>(addr).message
    }
}
```"""

async def test_code_summary():
    """Test the code summary generation function."""
    logger.info("Testing code summary generation")
    
    # Test with valid OpenAI API key
    summary = await generate_code_summary(
        CODE_BLOCK, 
        parent_title="Move Smart Contracts", 
        block_title="Hello World Example"
    )
    logger.info(f"Generated summary: {summary}")

def main():
    """Main entry point."""
    asyncio.run(test_code_summary())

if __name__ == "__main__":
    main() 