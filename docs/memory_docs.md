# atom_agent.memory Documentation

## Overall Flow
`src/atom_agent/memory.py` implements a tiered memory retrieval system for the agent. It allows the agent to recall information from past steps and conversations, ensuring context retention across long-running tasks.

The flow involves:
1.  **Tier structured context**:
    -   **Tier 0**: System prompt and current user instruction.
    -   **Tier 1**: Local context (full message history of the current step/attempt).
    -   **Tier 2**: Global context (summaries of previous successful steps).
2.  **Retrieval**: `load_step_history` reads JSON logs from disk.
3.  **Ranking**: `load_previous_step_reports` uses a `CrossEncoder` model (`cross-encoder/ms-marco-MiniLM-L-6-v2`) to semantic rank previous step reports against the current query, filtering out irrelevant information.
4.  **Context Construction**: `reconstruct_context` assembles these pieces into a final list of `BaseMessage` objects for the LLM.

## Use Cases
-   **Context Window Optimization**: Instead of stuffing the entire history into the LLM, it selects only the most relevant past steps.
-   **Long-term Memory**: Allows the agent to reference decisions made in step 1 while working on step 10.
-   **Refinement Context**: When refining a step (attempt > 1), it loads previous attempts to avoid repeating mistakes.

## Edge Cases
-   **Missing History**: If logs don't exist, it returns an empty list or default message.
-   **Ranking Failure**: If the Cross-Encoder fails (e.g., model load error), it falls back to the most recent K reports.
-   **First Attempt**: Explicitly ignores legacy history on the first attempt of a step to ensure a fresh start.

## Class Documentation

### `MemoryManager` (Class)
Static manager for memory operations.

### `_get_encoder()`
Singleton accessor for the CrossEncoder model. lazy-loads to save startup time.

### `load_step_history(workspace: Workspace, step_id: str) -> List[BaseMessage]`
Loads the full chat history for a specific step.

-   **Returns**: List of LangChain `BaseMessage` objects (Human, AI, System, Tool).
-   **Error Handling**: Returns empty list on JSON errors or missing files.

### `load_previous_step_reports(...)`
Retrieves and ranks summaries of previous steps.

-   **Arguments**:
    -   `workspace`, `plan`, `current_step_idx`: Context.
    -   `query`: Search query (usually current step description).
    -   `top_k`: Number of reports to return.
    -   `min_score`: Relevance threshold.
-   **Logic**:
    -   Reads `report.json` from `committed/` directories of previous steps.
    -   Ranks them using Cross-Encoder.
    -   Returns formatted string of top K reports.

### `reconstruct_context(...)`
Assembles the final prompt for the LLM.

-   **Arguments**: `workspace`, `plan`, `current_step_idx`, `system_prompt`, `user_prompt`.
-   **Returns**: List of `BaseMessage`.
-   **Logic**:
    -   Adds System Prompt.
    -   Adds Global Context (Tier 2).
    -   Adds Local History (Tier 1) if attempt > 1.
    -   Adds User Instruction.
