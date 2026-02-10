# atom_agent.tools.planning_tools Documentation

## Overall Flow
`src/atom_agent/tools/planning_tools.py` equips the Planner agent with the ability to research past work and formally submit the implementation plan. It acts as the bridge between the Planner's reasoning and the structured Agent State.

The flow is:
1.  **Factory**: `create_planning_tools(workspace)` bundles research tools (search + file reading) and the schema-enforced `submit_plan` tool.
2.  **Research**: The planner uses `read_committed_artifact` and `list_completed_steps` to understand what has already been built (Tier 3 memory access).
3.  **Submission**:
    -   The agent calls `submit_plan` with a JSON string.
    -   The tool validates the JSON against the `ImplementationPlanInput` Pydantic model.
    -   It enforces regex patterns for step IDs and valid complexity levels.

## Use Cases
-   **Plan Generation**: The defining tool for the Planner node. It forces the LLM to structure its loose reasoning into a strict execution graph.
-   **Historical Research**: Before planning step 5, the agent reads the report from step 4 to ensure continuity.

## Edge Cases
-   **Invalid JSON**: Returns a detailed error message if the LLM's JSON is malformed.
-   **Schema Violations**: Returns Pydantic validation errors (e.g., "step_id must be lowercase snake_case") so the agent can self-correct.
-   **No Previous Steps**: Initializes gracefully even if the `steps/` directory is empty (first run).

## Method Documentation

### `submit_plan(plan_json: str) -> str`
The core output tool for the Planner.
-   **Arguments**: JSON string matching `ImplementationPlanInput`.
-   **Validation**:
    -   `task_id` must match.
    -   `steps` must be a list.
    -   `step_id` regex `^[a-z][a-z0-9_]{0,47}$`.
    -   `estimated_complexity` enum: low, medium, high.
-   **Returns**: Success message with step count.

### `read_committed_artifact(step_id: str, artifact_name: str) -> str`
Reads a file from a *completed* step's artifacts.
-   **Example**: `read_committed_artifact("research_api", "api_docs.md")`.

### `list_completed_steps() -> str`
Returns a summary of all past work.
-   **Returns**: JSON list of step IDs and their artifact filenames.
