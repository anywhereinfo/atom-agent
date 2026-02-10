# atom_agent.tools.search_tools Documentation

## Overall Flow
`src/atom_agent/tools/search_tools.py` provides the agent with external research capabilities. It bundles two primary search engines:
1.  **Tavily**: For general web search (requires API key).
2.  **arXiv**: For academic paper search (open access).

The flow is:
1.  **Factory**: `create_search_tools()` initializes both tools.
2.  **Environment Check**: Checks for `TAVILY_API_KEY`. If missing, disables web search.
3.  **Tool Creation**: Wraps the search functionality in LangChain `@tool` decorators or uses existing wrappers.

## Use Cases
-   **Research Phase**: The Planner uses these tools to understand new libraries, algorithms, or frameworks before creating a plan.
-   **Problem Solving**: The Executor uses web search to find documentation or fix errors during implementation.

## Edge Cases
-   **Missing API Key**: Handles missing Tavily key gracefully by simply not adding the tool to the list (agent will just have arXiv).
-   **Network Errors**: Wraps searches in try/except blocks to prevent crashing the agent.
-   **Empty Results**: Returns a friendly "No papers found" message instead of empty lists.

## Method Documentation

### `create_search_tools() -> List`
Factory function to return a list of executable tools.
-   **Returns**: List containing `TavilySearch` (if configured) and `arxiv_search`.

### `arxiv_search(query: str, max_results: int = 5) -> str`
Custom tool for academic research.
-   **Arguments**:
    -   `query`: Search terms.
    -   `max_results`: Limit (capped at 10 internally).
-   **Returns**: Formatted string with Title, Authors, Published Date, Summary, and URL for each paper.
-   **Error Handling**: Returns error string if `arxiv` library is missing or network fails.
