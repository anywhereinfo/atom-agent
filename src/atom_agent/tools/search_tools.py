import os
from langchain_tavily import TavilySearch
from typing import List

def create_search_tools() -> List:
    """Factory to create search tools. Requires TAVILY_API_KEY in environment."""
    if not os.environ.get("TAVILY_API_KEY"):
        print("WARNING: TAVILY_API_KEY not found in environment. Search tools will be disabled.", flush=True)
        return []
        
    try:
        search_tool = TavilySearch(max_results=5)
        return [search_tool]
    except Exception as e:
        print(f"ERROR: Failed to initialize Tavily search tool: {str(e)}", flush=True)
        return []
