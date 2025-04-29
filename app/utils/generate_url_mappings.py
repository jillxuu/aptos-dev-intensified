"""Script to generate comprehensive URL mappings from developer-docs repository."""

import os
import yaml
import logging
import aiohttp
import asyncio
from typing import Dict, List, Set, Tuple
from xml.etree import ElementTree
from app.config import DOCS_BASE_URLS, DEFAULT_PROVIDER
from app.path_registry import path_registry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def fetch_sitemap_urls(
    base_url: str = DOCS_BASE_URLS[DEFAULT_PROVIDER],
) -> Set[str]:
    """Fetch all URLs from the sitemap."""
    urls = set()
    try:
        async with aiohttp.ClientSession() as session:
            # First fetch English sitemap specifically
            async with session.get(f"{base_url}/sitemap-en.xml") as response:
                if response.status == 200:
                    content = await response.text()
                    root = ElementTree.fromstring(content)

                    # Extract URLs from sitemap
                    for url in root.findall(
                        ".//{http://www.sitemaps.org/schemas/sitemap/0.9}loc"
                    ):
                        full_url = url.text
                        # Convert full URL to path
                        path = full_url.replace(base_url, "").strip("/")
                        if path and path.startswith("en/"):
                            urls.add(path)

                    logger.info(f"Processed English sitemap, found {len(urls)} URLs")
                else:
                    logger.error(f"Failed to fetch English sitemap: {response.status}")
    except Exception as e:
        logger.error(f"Error fetching sitemap: {e}")

    return urls


def scan_docs_directory(docs_dir: str) -> List[str]:
    """Scan the docs directory for English markdown files."""
    markdown_files = []

    for root, _, files in os.walk(docs_dir):
        for file in files:
            if file.endswith((".md", ".mdx")) and not file.endswith("_meta.ts"):
                rel_path = os.path.relpath(os.path.join(root, file), docs_dir)
                # Only include English files
                if rel_path.startswith("en/"):
                    markdown_files.append(rel_path)

    logger.info(f"Found {len(markdown_files)} English markdown files in {docs_dir}")
    return markdown_files


async def generate_mappings(docs_dir: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Generate comprehensive URL mappings for English content."""
    # Get all English markdown files
    markdown_files = scan_docs_directory(docs_dir)

    # Get all URLs from English sitemap
    sitemap_urls = await path_registry._fetch_sitemap()

    mappings = {}
    for file_path in markdown_files:
        # Get normalized path from registry
        normalized_path = path_registry.register_file(file_path)

        # Check if URL exists in sitemap
        if normalized_path in sitemap_urls:
            mappings[file_path] = normalized_path
        else:
            logger.warning(f"No matching URL found for {file_path}")

    # Return both mappings and empty redirects dictionary
    redirects = {}
    return mappings, redirects


async def main():
    """Main function to generate and save URL mappings."""
    try:
        # Configure paths
        docs_dir = "data/developer-docs/apps/nextra/pages"  # Adjust this path
        output_file = "data/generated/developer-docs/url_mappings.yaml"

        # Generate mappings
        logger.info("Generating URL mappings for English content...")
        mappings, redirects = await generate_mappings(docs_dir)

        # Prepare YAML content
        content = {"mappings": mappings, "redirects": redirects}

        # Save to YAML file
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w") as f:
            yaml.safe_dump(content, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Generated {len(mappings)} mappings")
        logger.info(f"Saved to {output_file}")

    except Exception as e:
        logger.error(f"Error generating mappings: {e}")


if __name__ == "__main__":
    asyncio.run(main())
