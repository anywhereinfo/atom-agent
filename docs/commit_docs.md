# atom_agent.nodes.commit Documentation

## Overall Flow
`src/atom_agent/nodes/commit.py` is the pivotal stage where a successful implementation attempt is promoted to the "committed" state. This ensures that the agent only builds upon code that has passed verification.

The flow is:
1.  **Validation**: Ensures the reflector's decision was `proceed` (accepted). If not, aborts.
2.  **Path Resolution**: Identifies the *staged* source files (attempt directory) and the *destination* paths (committed directory) using the `Workspace`.
3.  **Promotion (Copy)**:
    -   Copies `impl.py` and `test.py` from attempt -> committed.
    -   Copies the entire `artifacts/` directory.
    -   Cleans up previous committed artifacts (overwrites).
4.  **Manifest Creation**: Writes a `manifest.json` metadata file in the committed folder, recording acceptance criteria, score, timestamp, and file list.
5.  **State Update**: Marks the step status as `completed` and advances the `current_step_index`.

## Use Cases
-   **Step Finalization**: This is the only way a step becomes officially "done".
-   **History Preservation**: Ensures that the `committed/` folder contains only working, verified code snapshots.

## Edge Cases
-   **Premature Call**: If called when the plan index is out of bounds, returns empty.
-   **Reflector Rejection**: If `reflector_review.decision` != "proceed", it logs a debug message and skips the commit (returning empty to likely satisfy the graph router).
-   **File Errors**: Wraps file operations in try/except; if promotion fails, returns an error phase.

## Method Documentation

### `commit_node(state: AgentState) -> Dict[str, Any]`
The primary node function.

-   **Logic**:
    -   Resolves current step and attempt.
    -   Checks reflector decision.
    -   Copies files.
    -   Writes manifest.
    -   Updates state (`phase`, `current_step_index`).
-   **Returns**: Updated state with incremented step index, effectively moving the agent to the next step.
