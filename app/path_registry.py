"""Service for managing file paths, normalized paths, and URL mappings."""

import os
import yaml
import logging
import aiohttp
from typing import Dict, List, Optional, Set
from xml.etree import ElementTree
from app.config import DOCS_BASE_URLS, DEFAULT_PROVIDER

logger = logging.getLogger(__name__)


class PathRegistry:
    """Central registry for managing file paths and URL mappings."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PathRegistry, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the registry."""
        if self._initialized:
            return

        self._file_to_normalized: Dict[str, str] = {}
        self._normalized_to_url: Dict[str, str] = {}
        self._url_to_normalized: Dict[str, str] = {}
        self._initialized = True

    def _normalize_path(self, file_path: str) -> str:
        """Normalize a file path to a consistent format."""
        if not file_path:
            return ""

        # Remove file extension
        path = os.path.splitext(file_path)[0]

        # Handle index files
        if path.endswith("/index") or path == "index":
            path = path[:-6] if path.endswith("/index") else ""

        # Convert path separators and ensure en/ prefix
        path = path.replace(os.path.sep, "/")
        if not path.startswith("en/"):
            path = f"en/{path}"

        return path

    def register_file(self, file_path: str) -> str:
        """Register a file path and return its normalized form."""
        if not file_path:
            return ""

        normalized = self._normalize_path(file_path)
        self._file_to_normalized[file_path] = normalized
        return normalized

    def register_url(self, url: str, normalized_path: str):
        """Register a URL mapping."""
        self._normalized_to_url[normalized_path] = url
        self._url_to_normalized[url] = normalized_path

    def get_normalized_path(self, file_path: str) -> str:
        """Get normalized path from file path."""
        if not file_path:
            return ""
        return self._file_to_normalized.get(file_path) or self._normalize_path(
            file_path
        )

    def get_url(self, path: str) -> Optional[str]:
        """Get URL from either file path or normalized path."""
        if not path:
            return None

        normalized = (
            self._file_to_normalized.get(path) or path
            if path in self._normalized_to_url
            else self._normalize_path(path)
        )
        return self._normalized_to_url.get(normalized)

    def get_all_urls(self) -> List[str]:
        """Get all registered URLs."""
        return list(self._normalized_to_url.values())

    async def initialize_from_config(self, config_path: str):
        """Initialize mappings from a config file."""
        try:
            if not os.path.exists(config_path):
                logger.warning(f"URL mappings file not found at {config_path}")
                return

            # Load mappings from config
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)

            # Register all file paths and URLs
            for file_path, url in config.get("mappings", {}).items():
                normalized = self.register_file(file_path)
                self.register_url(url, normalized)

            logger.info(f"Loaded {len(self._normalized_to_url)} URL mappings")

            # Validate against sitemap
            await self._validate_urls()

        except Exception as e:
            logger.error(f"Error initializing path registry: {e}")

    async def _validate_urls(self):
        """Validate URLs against the sitemap."""
        try:
            sitemap_urls = await self._fetch_sitemap()

            # Validate each URL
            valid_urls = set()
            for normalized_path, url in self._normalized_to_url.items():
                if normalized_path in sitemap_urls:
                    valid_urls.add(url)
                else:
                    logger.warning(f"Invalid URL mapping: {normalized_path} -> {url}")

            logger.info(f"Validated {len(valid_urls)} URLs")

        except Exception as e:
            logger.error(f"Error validating URLs: {e}")

    async def _fetch_sitemap(
        self, base_url: str = DOCS_BASE_URLS[DEFAULT_PROVIDER]
    ) -> Set[str]:
        """Fetch URLs from the sitemap."""
        urls = set()
        try:
            async with aiohttp.ClientSession() as session:
                # Try both sitemap-en.xml and sitemap.xml
                for sitemap_file in ["sitemap-en.xml", "sitemap.xml"]:
                    async with session.get(f"{base_url}/{sitemap_file}") as response:
                        if response.status == 200:
                            content = await response.text()
                            root = ElementTree.fromstring(content)

                            # Extract URLs from sitemap
                            for url in root.findall(
                                ".//{http://www.sitemaps.org/schemas/sitemap/0.9}loc"
                            ):
                                full_url = url.text
                                if full_url.startswith(base_url):
                                    path = full_url[len(base_url) :].strip("/")
                                    if path and path.startswith("en/"):
                                        urls.add(path)

                            logger.info(f"Fetched {len(urls)} URLs from {sitemap_file}")
                            if urls:  # If we found URLs, no need to try the other file
                                break
                        else:
                            logger.warning(
                                f"Failed to fetch {sitemap_file}: {response.status}"
                            )
        except Exception as e:
            logger.error(f"Error fetching sitemap: {e}")

        return urls


# Create singleton instance
path_registry = PathRegistry()
