from mcp.server.fastmcp import FastMCP
from tavily import TavilyClient
import os
from dotenv import load_dotenv
import feedparser
import urllib.parse
# from langchain.tools import tool

load_dotenv()

# Create MCP server
mcp = FastMCP("internet-and-research-search")

# Init Tavily
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


@mcp.tool()
def internet_search(query: str, max_results: int = 5) -> list[dict]:
    """
    Search the internet for up-to-date general information.
    """
    response = tavily.search(
        query=query,
        max_results=max_results,
        include_answer=False,
        include_raw_content=False
    )

    results = []
    for r in response.get("results", []):
        results.append({
            "title": r.get("title"),
            "url": r.get("url"),
            "content": r.get("content")
        })

    return results

@mcp.tool()
def arxiv_paper_search(query: str, max_results: int = 5) -> list[dict]:
    encoded_query = urllib.parse.quote(query)

    url = (
        f"http://export.arxiv.org/api/query?"
        f"search_query=all:{encoded_query}"
        f"&start=0&max_results={max_results}"
    )

    feed = feedparser.parse(url)

    results = []
    for entry in feed.entries:
        results.append({
            "title": entry.title,
            "authors": [a.name for a in entry.authors],
            "published": entry.published,
            "summary": entry.summary,
            "url": entry.link
        })

    return results


if __name__ == "__main__":
    mcp.run()
    
