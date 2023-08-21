import argparse
import asyncio
import json
from pathlib import Path

import aiohttp
import bs4
import chardet

PAUL_GRAHAM_URI = "http://paulgraham.com/articles.html"

"""
A script to download Paul Graham's essays as an illustrative example. This script is used by the download-essays run.

To extend this example to fit with another dataset, either replace this script with another to download that dataset, or
ignore this script and upload your dataset as an artifact directly:
https://slingshot-ai.gitbook.io/slingshot-ai-docs/using-slingshot/concepts/storage/artifacts
"""


async def get_document(url: str) -> str:  # Rename
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers={"User-Agent": "Mozilla/5.0"}) as response:
            bytes_ = await response.content.read()
            encoding = chardet.detect(bytes_)['encoding']  # About 26 articles are not valid UTF-8.
            return bytes_.decode(encoding)


async def get_article(url: str) -> str:  # Rename
    document = await get_document(url)
    soup = bs4.BeautifulSoup(document, "html.parser")

    # Replace all br tags with a double line-break.
    for br in soup.select("table td font br"):
        br.replace_with('\n')

    text = soup.select_one("table td font").text  # type: ignore
    return text


async def get_document_hrefs(url: str) -> list[str]:  # Rename
    document = await get_document(url)
    soup = bs4.BeautifulSoup(document, "html.parser")
    return [tag['href'] for tag in soup.select("table td font a")]  # type: ignore


async def parse_document(url: str, output_dir: Path) -> None:  # Should this be called main?
    links = await get_document_hrefs(url)
    links = [link for link in links if isinstance(link, str)]  # TODO: What's this?

    links = [f"http://paulgraham.com/{link}" for link in links if
             not link.startswith("http")]  # What if the link does start with http?
    print(f"Found {len(links)} links: {links}")  # Do we want to print this, like this?

    articles = await asyncio.gather(*[get_article(link) for link in links])
    articles = [article for article in articles if isinstance(article, str)]  # Why isinstance?

    # Save JSON with all articles
    with open(output_dir / "articles.json", 'w') as f:  # Use Pandas?
        json.dump(articles, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("/mnt/documents"))
    args = parser.parse_args()

    asyncio.run(parse_document(PAUL_GRAHAM_URI, args.output_dir))
