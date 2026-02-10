# atom_agent.graph Documentation

## Overall Flow
`src/atom_agent/graph.py` defines the main `LangGraph` workflow for the `atom_agent`. It stitches together the various nodes (setup, planner, executor, reflector, commit) into a cohesive StateMachine.

The flow is:
1.  **Entry Point**: `setup` node initializes the workspace and task context.
2.  **Sequential Execution**: `setup` -> `planner` -> `executor` -> `reflector`.
3.  **Conditional Routing**:
    -   After `reflector`, `route_reflection(state)` decides the next step based on the Reflector's review:
        -   `proceed` -> `commit` (success)
        -   `refine` -> `executor` (retry/fix)
        -   `rollback` -> `planner` (fundamental replan needed)
    -   After `commit`, `route_post_commit(state)` checks if there are more steps in the plan:
        -   Yes -> `executor` (next step)
        -   No -> `END` (task complete)
4.  **Guardrails**: The router includes logic to prevent infinite loops (e.g., checking `max_attempts`).

## Use Cases
-   **Workflow orchestration**: This file is the "brain" that decides which component runs next.
-   **Debugging flow**: If the agent gets stuck or loops unexpectedly, this is the place to check the conditional edges.

## Edge Cases
-   **Max Attempts Exceeded**: If the executor fails repeatedly and exceeds `max_attempts`, the router forces a rollback to the planner to reconsider the strategy.
-   **Plan Index Out of Bounds**: Handled gracefully by routing to `END` or `planner`.

## Method Documentation

### `route_reflection(state: AgentState) -> str`
Determines the next node after the Reflector's evaluation.

-   **Logic**:
    -   Checks `state["reflector_review"]["decision"]`.
    -   Checks `current_attempt` vs `max_attempts`.
    -   Returns node name: "commit", "executor", or "planner".

### `create_graph() -> StateGraph`
Constructs and compiles the LangGraph application.

-   **nodes**: Adds `setup`, `planner`, `executor`, `reflector`, `commit`.
-   **edges**:
    -   `setup` -> `planner`
    -   `planner` -> `executor`
    -   `executor` -> `reflector`
    -   `reflector` -> (conditional)
    -   `commit` -> (conditional)
-   **Returns**: A compiled `CompiledGraph` object ready for execution.
