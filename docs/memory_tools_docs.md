# atom_agent.tools.memory_tools Documentation

## Overall Flow
`src/atom_agent/tools/memory_tools.py` provides tools for the agent to access Long Term Memory (Tier 3). It specifically allows the agent to retrieve detailed logs, code artifacts, and reports from *previously committed steps*.

The flow is:
1.  **Factory**: `create_memory_tools(workspace)` binds the tool to the current workspace configuration.
2.  **Path Resolution**: Resolves the location of committed artifacts (`steps/{step_id}/committed/`) using the `Workspace` contract.
3.  **Content Retrieval**: Reads JSON logs (`history.json`) and reports (`report.json`) from disk.
4.  **Formatting**: Returns the content as a formatted string useful for LLM context.

## Use Cases
-   **Deep Dive**: Asking "How did we implement X in step Y?" requires reading the code or full history of that step.
-   **Debugging Regressions**: Checking previous test results or implementation details to understand why a change broke something later.

## Edge Cases
-   **Invalid Step ID**: Returns "No report/history found" if the directory doesn't exist.
-   **JSON Errors**: Handles potential corruption in log files gracefully.

## Method Documentation

### `create_memory_tools(workspace: Workspace) -> list`
Factory function.

### `get_committed_step_history(step_id: str) -> str`
Tool to retrieve full context of a past step.

-   **Arguments**: `step_id` (e.g., "setup_database").
-   **Returns**: String containing:
    -   Full Report (`report.json`).
    -   Full Message History (`history.json`).
-   **Logic**:
    -   Constructs paths.
    -   Reads files.
    -   Combines them into a single markdown-friendly string.
