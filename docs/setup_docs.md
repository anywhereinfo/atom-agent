# atom_agent.nodes.setup Documentation

## Overall Flow
`src/atom_agent/nodes/setup.py` initializes the entire task execution environment. It creates the unique directory structure for a new task, generates a `Workspace` contract based on the user's request, and sets up the initial state files.

The flow is:
1.  **Request normalization**: Trims whitespace and hashes the request to create a unique ID.
2.  **ID Generation**: Combines a slug (from description), timestamp (YYYYMMDD_HHMMSS_mmm), and hash to create a collision-resistant `task_id`.
3.  **Directory Scaffold**: Creates `tasks/{task_id}/` and subdirectories `inputs`, `steps`, `state`, `logs`.
4.  **Workspace Creation**: Instantiates a `Workspace` object with strict configuration for:
    -   Naming conventions (steps, attempts).
    -   Path templates (steps/{step_id}/attempts/{attempt_id}/...).
    -   Read/Write permissions (enforcing boundaries between executor and reflector).
5.  **State Persistence**: Writes `request.txt`, `task.json`, `workspace.json`, and an initial empty `plan.json` to the `state/` directory.
6.  **Return**: Returns the initial `AgentState` components (`task_context`, `workspace`).

## Use Cases
-   **Task Initialization**: The very first node that runs. Sets the stage for everything else.
-   **Reproducibility**: By saving the exact request and workspace config, any run can be inspected later.

## Edge Cases
-   **Empty Request**: Raises `ValueError` if `task_description` is empty.
-   **Permissions**: Errors if it cannot create directories (e.g., read-only filesystem).

## Method Documentation

### `slugify(text: str) -> str`
Helper to create URL-friendly path segments.
-   **Logic**: Lowercase, non-alphanumeric -> hyphen, trim to 48 chars.

### `task_setup_node(state: AgentState) -> dict`
The primary setup node function.

-   **Logic**:
    -   Calculates `task_id`.
    -   Creates directory tree.
    -   Defines `Workspace` contract (HARDCODED configuration of paths/rules).
    -   Persists state files.
-   **Returns**: Dictionary with `task_context`, `workspace`, and `task_id`.
-   **Key feature**: Defines the "Attempt/Commit" architecture paths here.
