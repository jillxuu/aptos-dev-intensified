#!/usr/bin/env python3

import os
import sys

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
from app.utils.topic_chunks import load_enhanced_chunks, initialize_vector_store
from app.config import get_vector_store_path, get_generated_data_path


async def init():
    # Load enhanced chunks from the correct path
    provider = "developer-docs"
    chunks_path = os.path.join(
        get_generated_data_path(provider), "enhanced_chunks.json"
    )
    chunks = await load_enhanced_chunks(file_path=chunks_path)
    if not chunks:
        sys.exit(1)
    await initialize_vector_store(
        chunks, vector_store_path=get_vector_store_path(provider)
    )


if __name__ == "__main__":
    asyncio.run(init())
