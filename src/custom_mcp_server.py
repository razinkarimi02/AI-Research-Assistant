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
    Search the public internet for up-to-date, general-purpose information.

    Use this tool when the user asks for:
    - Current events, news, or recent developments
    - General factual information not limited to academic papers
    - Explanations, tutorials, or overviews from blogs, documentation, or articles
    - Topics where freshness or real-world context matters

    Inputs:
    - query (str): Natural language search query.
    - max_results (int): Maximum number of search results to return.

    Returns:
    A list of dictionaries, each containing:
    - title (str): Title of the web page
    - url (str): Source URL
    - content (str): Extracted summary or snippet of the page
    
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
    """
    Search arXiv for academic papers matching a natural language query.

    Use this tool when the user asks for:
    - Research papers
    - Academic references
    - Scientific publications
    - arXiv-specific searches

    Returns a list of papers with title, authors, publication date, summary, and URL.
    """
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
    
