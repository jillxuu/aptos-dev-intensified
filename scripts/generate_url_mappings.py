#!/usr/bin/env python3
"""Script to generate URL mappings."""

import os
import sys
import asyncio

# Add the app directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils.generate_url_mappings import main

if __name__ == "__main__":
    asyncio.run(main())
