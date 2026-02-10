import os
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from typing import List

def create_search_tools() -> List:
    """Factory to create search tools. Includes web search (Tavily) and academic search (arXiv)."""
    tools = []

    # 1. Web Search (Tavily)
    if not os.environ.get("TAVILY_API_KEY"):
        print("WARNING: TAVILY_API_KEY not found in environment. Web search will be disabled.", flush=True)
    else:
        try:
            search_tool = TavilySearch(max_results=5)
            tools.append(search_tool)
        except Exception as e:
            print(f"ERROR: Failed to initialize Tavily search tool: {str(e)}", flush=True)

    # 2. arXiv Academic Search (no API key required)
    try:
        arxiv_tool = create_arxiv_search_tool()
        tools.append(arxiv_tool)
    except Exception as e:
        print(f"WARNING: Failed to initialize arXiv search tool: {str(e)}", flush=True)

    return tools


def create_arxiv_search_tool():
    """Creates an arXiv search tool for academic paper research."""

    @tool
    def arxiv_search(query: str, max_results: int = 5) -> str:
        """
        Search arXiv for academic papers and research.
        Returns title, authors, summary, and URL for relevant papers.

        Best for:
        - Academic research on algorithms, AI, ML, systems
        - Understanding state-of-the-art approaches
        - Finding formal definitions and theoretical foundations
        - Discovering cutting-edge techniques

        Args:
            query: Search query (e.g., "large language model reasoning", "hierarchical planning agents")
            max_results: Maximum number of papers to return (default 5, max 10)

        Examples:
            - "reinforcement learning hierarchical planning"
            - "chain of thought reasoning transformers"
            - "multi-agent coordination systems"
        """
        try:
            import arxiv
        except ImportError:
            return "ERROR: arxiv library not installed. Run: pip install arxiv"

        # Limit max_results to prevent overload
        max_results = min(max_results, 10)

        try:
            # Create arXiv client
            client = arxiv.Client()

            # Search arXiv
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )

            results = []
            for paper in client.results(search):
                # Extract key information
                result = {
                    "title": paper.title,
                    "authors": ", ".join([author.name for author in paper.authors[:3]]),  # First 3 authors
                    "published": paper.published.strftime("%Y-%m-%d"),
                    "summary": paper.summary[:500] + "..." if len(paper.summary) > 500 else paper.summary,
                    "url": paper.entry_id,
                    "categories": ", ".join(paper.categories[:3])  # First 3 categories
                }
                results.append(result)

            if not results:
                return f"No papers found for query: {query}"

            # Format results as readable text
            output = f"arXiv Search Results for '{query}' ({len(results)} papers):\n\n"
            for i, paper in enumerate(results, 1):
                output += f"{i}. {paper['title']}\n"
                output += f"   Authors: {paper['authors']}\n"
                output += f"   Published: {paper['published']} | Categories: {paper['categories']}\n"
                output += f"   Summary: {paper['summary']}\n"
                output += f"   URL: {paper['url']}\n\n"

            return output

        except Exception as e:
            return f"ERROR searching arXiv: {str(e)}"

    return arxiv_search
