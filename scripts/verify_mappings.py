import os
import yaml
import aiohttp
import asyncio
import logging
from bs4 import BeautifulSoup
from typing import Dict, Tuple, List
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def fetch_url_content(
    session: aiohttp.ClientSession, url: str
) -> Tuple[bool, str]:
    """Fetch content from a URL."""
    try:
        async with session.get(f"https://aptos.dev/{url}") as response:
            if response.status == 200:
                html = await response.text()
                soup = BeautifulSoup(html, "html.parser")
                content_div = soup.find("article")
                if content_div:
                    return True, content_div.get_text().strip()
                return True, ""
            else:
                logger.error(f"Failed to fetch {url}: {response.status}")
                return False, ""
    except Exception as e:
        logger.error(f"Error fetching {url}: {e}")
        return False, ""


def read_mdx_content(file_path: str) -> str:
    """Read content from local MDX file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
        return ""


def content_similarity(mdx_content: str, url_content: str) -> float:
    """Calculate rough content similarity."""
    if not mdx_content or not url_content:
        return 0.0

    # Convert both contents to lowercase and remove whitespace
    mdx_clean = "".join(mdx_content.lower().split())
    url_clean = "".join(url_content.lower().split())

    # Get the shorter and longer of the two strings
    shorter = mdx_clean if len(mdx_clean) < len(url_clean) else url_clean
    longer = mdx_clean if len(mdx_clean) >= len(url_clean) else url_clean

    # Calculate how much of the shorter string appears in the longer string
    matching_chars = sum(1 for char in shorter if char in longer)
    return matching_chars / len(shorter)


async def verify_mapping(
    session: aiohttp.ClientSession, local_path: str, url_path: str
) -> Tuple[bool, float]:
    """Verify a single mapping."""
    # Only verify English content
    if not local_path.startswith("en/"):
        return False, 0.0

    # Fetch URL content
    url_accessible, url_content = await fetch_url_content(session, url_path)
    if not url_accessible:
        return False, 0.0

    # Read local MDX content
    mdx_content = read_mdx_content(
        os.path.join("data/developer-docs/apps/nextra/pages", local_path)
    )
    if not mdx_content:
        return False, 0.0

    # Calculate similarity
    similarity = content_similarity(mdx_content, url_content)
    return True, similarity


async def verify_all_mappings(
    mappings: Dict[str, str],
) -> List[Tuple[str, str, bool, float]]:
    """Verify all mappings."""
    results = []
    chunks = []
    chunk_size = 10  # Process in chunks to avoid overwhelming the server

    # Split mappings into chunks
    items = list(mappings.items())
    for i in range(0, len(items), chunk_size):
        chunks.append(items[i : i + chunk_size])

    async with aiohttp.ClientSession() as session:
        for chunk_num, chunk in enumerate(chunks, 1):
            logger.info(
                f"Processing chunk {chunk_num}/{len(chunks)} ({len(chunk)} mappings)"
            )

            # Process chunk concurrently
            tasks = []
            for local_path, url_path in chunk:
                if local_path.startswith("en/"):  # Only process English files
                    task = verify_mapping(session, local_path, url_path)
                    tasks.append((local_path, url_path, task))

            # Wait for all tasks in chunk to complete
            for local_path, url_path, task in tasks:
                accessible, similarity = await task
                results.append((local_path, url_path, accessible, similarity))

            # Small delay between chunks to be nice to the server
            if chunk_num < len(chunks):
                await asyncio.sleep(1)

    return results


async def main():
    """Main verification function."""
    try:
        # Load mappings
        with open("config/url_mappings.yaml", "r") as f:
            config = yaml.safe_load(f)
            mappings = config.get("mappings", {})

        # Only verify English mappings
        en_mappings = {k: v for k, v in mappings.items() if k.startswith("en/")}
        logger.info(f"Loaded {len(en_mappings)} English mappings to verify...")

        # Verify mappings
        results = await verify_all_mappings(en_mappings)

        # Analyze results
        total = len(results)
        successful = sum(
            1
            for _, _, accessible, similarity in results
            if accessible and similarity > 0.5
        )

        # Group results by similarity ranges
        similarity_ranges = defaultdict(list)
        for local_path, url_path, accessible, similarity in results:
            if not accessible:
                similarity_ranges["inaccessible"].append((local_path, url_path))
            elif similarity < 0.3:
                similarity_ranges["low"].append((local_path, url_path, similarity))
            elif similarity < 0.5:
                similarity_ranges["medium"].append((local_path, url_path, similarity))
            else:
                similarity_ranges["high"].append((local_path, url_path, similarity))

        # Print summary
        print("\nVerification Results Summary:")
        print("=" * 80)
        print(f"Total mappings verified: {total}")
        print(
            f"Successful mappings (accessible & similarity > 50%): {successful} ({successful/total*100:.1f}%)"
        )
        print("\nDetailed Results:")
        print("-" * 80)

        if similarity_ranges["inaccessible"]:
            print("\n❌ Inaccessible URLs:")
            for local_path, url_path in similarity_ranges["inaccessible"]:
                print(f"   {local_path} -> {url_path}")

        if similarity_ranges["low"]:
            print("\n⚠️  Low Similarity (<30%):")
            for local_path, url_path, similarity in similarity_ranges["low"]:
                print(f"   {local_path} -> {url_path} ({similarity:.1%})")

        if similarity_ranges["medium"]:
            print("\n⚠️  Medium Similarity (30-50%):")
            for local_path, url_path, similarity in similarity_ranges["medium"]:
                print(f"   {local_path} -> {url_path} ({similarity:.1%})")

        print("\n✅ High Similarity (>50%):")
        print(f"   {len(similarity_ranges['high'])} mappings")

    except Exception as e:
        logger.error(f"Error during verification: {e}")


if __name__ == "__main__":
    asyncio.run(main())
