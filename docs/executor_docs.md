# atom_agent.nodes.executor Documentation

## Overall Flow
`src/atom_agent/nodes/executor.py` implements the core execution engine of the agent. It uses a ReAct pattern (Reasoning + Acting) to decide which tools to call in order to satisfy the current step's acceptance criteria.

The flow is:
1.  **State Loading**: Retrieves the current step, attempt number, and workspace context.
2.  **Tool Creation**: Loads all available tools (`file_tools`, `code_tools`, `memory_tools`, `search_tools`) and binds them to the workspace.
3.  **Prompt Construction**:
    -   Builds a tiered context prompt using `MemoryManager.reconstruct_context`.
    -   Includes system prompt with strict **anti-hardcoded-output rules** (forbidding large string literals in `impl.py`).
    -   Injects `{dependency_context}` containing explicit paths to committed artifacts from all dependency steps.
    -   Includes global context (Tier 2), local history (Tier 1), and the specific user instruction identifying the files to create (`impl.py`, `test.py`).
4.  **Agent Execution**:
    -   Creates a `create_react_agent` with `gemini-3-pro-preview`.
    -   Invokes the agent loop, capturing tool calls and thoughts via a callback handler (`ToolLogHandler`).
5.  **Result Processing**:
    -   Cleans the message history (strips heavy metadata/signatures) to keep tokens low.
    -   Persists the full conversation history to `history.json` in the messages directory.
    -   Determines if the agent is "done" (returned final answer) or "awaiting input" (asked a question).
6.  **State Update**: Returns the updated history and phase (`reflecting` or `awaiting_user_input`).

## Use Cases
-   **Implementation**: Writing the actual code (`impl.py`) for a feature.
-   **Testing**: Writing and running tests (`test.py`) to verify the implementation.
-   **Debugging**: Reading error logs and analyzing why a previous attempt failed.

## Edge Cases
-   **Agent Loops**: The ReAct agent has internal recursion limits (default LangGraph behavior).
-   **Tool Failures**: Captured in `history.json`.
-   **Context Limit**: `_clean_message` aggressively prunes metadata to prevent blowing up the context window over many turns.
-   **First Attempt**: Explicitly overwrites any legacy `history.json` to ensure a clean slate.

## Method Documentation

### `_clean_message(msg: dict) -> dict`
Helper to sanitize LangChain message dictionaries.
-   **Logic**: Removes `id`, `usage_metadata`, `response_metadata`, and internal Gemini signatures.

### `_load_executor_tools(...)`
Factory to combine all toolsets.
-   **Returns**: List of `StructuredTool` instances.

### `_build_dependency_context(workspace, current_step) -> str`
Resolves and formats paths to committed artifacts from dependency steps.
-   **Logic**: Uses the `Workspace` contract to find the `committed/artifacts` directory for each dependency.
-   **Returns**: A formatted string list of available dependency paths.

### `_prepare_executor_messages(...)`
Constructs the input messages for the LLM.
-   **Logic**:
    -   Formats `user_msg_content` with acceptance criteria and staging paths.
    -   Calls `MemoryManager` to inject appropriate historical context.

### `_process_executor_result(...)`
Post-processes the agent's output.
-   **Logic**:
    -   Extracts final response text.
    -   Persists `history.json` to disk.
    -   Detects if agent is asking a question (`is_asking`).
-   **Returns**: Dictionary with `awaiting_input` and `full_history`.

### `executor_node(state: AgentState) -> dict`
The primary node function.

-   **Logic**:
    -   Checks if plan is complete (if so, moves to reflecting/end).
    -   Builds dependency context via `_build_dependency_context`.
    -   Builds tools and messages.
    -   Runs the ReAct agent.
    -   Processes result and updates state.
