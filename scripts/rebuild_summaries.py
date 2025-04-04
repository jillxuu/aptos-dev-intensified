#!/usr/bin/env python3
"""
Script to rebuild the summary cache for all documents in the vector store.
This is useful when you want to regenerate all summaries or when
the summary generation logic has changed.

Usage:
    python rebuild_summaries.py [--force]

Options:
    --force    Force rebuild all summaries, even if they exist in the cache
"""

import os
import sys
import logging
import asyncio
from app.models import (
    initialize_models,
    rebuild_summary_cache,
    load_summary_cache,
    save_summary_cache,
    summary_cache,
    DEFAULT_PROVIDER,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("rebuild_summaries")


async def async_main():
    # Check for force flag
    force = False
    if len(sys.argv) > 1 and sys.argv[1] == "--force":
        force = True
        logger.info("Force rebuild enabled - will regenerate all summaries")

    # Initialize models and load existing cache
    logger.info("Initializing models and loading cache...")
    initialize_models()

    if force:
        # Clear the cache if force rebuild is requested
        logger.info("Clearing existing summary cache due to force flag")
        global summary_cache
        summary_cache = {}

    # Rebuild summaries asynchronously
    logger.info("Starting asynchronous summary cache rebuild...")
    success = await rebuild_summary_cache()

    if success:
        logger.info("Summary cache rebuild completed successfully")
        return 0
    else:
        logger.error("Summary cache rebuild failed")
        return 1


def main():
    """Run the async main function"""
    return asyncio.run(async_main())


async def rebuild_cache():
    """Rebuild the summary cache."""
    # Load existing cache
    load_summary_cache(DEFAULT_PROVIDER)

    # Your rebuilding logic here

    # Save updated cache
    save_summary_cache(DEFAULT_PROVIDER)


if __name__ == "__main__":
    sys.exit(main())
