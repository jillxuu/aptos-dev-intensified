#!/usr/bin/env python3

import asyncio
from app.utils.topic_chunks import load_enhanced_chunks, initialize_vector_store
from app.config import get_vector_store_path


async def init():
    # Load enhanced chunks directly from the file
    chunks = await load_enhanced_chunks(file_path="data/enhanced_chunks.json")
    await initialize_vector_store(chunks, vector_store_path=get_vector_store_path('developer-docs'))

if __name__ == "__main__":
    asyncio.run(init())
