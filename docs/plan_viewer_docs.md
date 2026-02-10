# atom_agent.nodes.plan_viewer Documentation

## Overall Flow
`src/atom_agent/nodes/plan_viewer.py` is a simple node responsible for visualizing the implementation plan to the console. It performs a topological sort on the plan steps to ensure they are presented in the correct execution order (respecting dependencies) and then prints a formatted summary of each step.

## Use Cases
-   **Debug Visibility**: Allows the user (and developer) to see exactly what the agent plans to do, in what order, and with what criteria.
-   **Plan Validation**: Visual confirmation that dependencies are correctly understood (e.g., Step 2 comes after Step 1).

## Edge Cases
-   **Empty Plan**: Prints a warning if `implementation_plan` is missing or empty.
-   **Cycles**: The topological sort has a basic cycle detection (skips node if cycle detected, though a robust sort would raise an error).

## Method Documentation

### `plan_viewer_node(state: AgentState) -> dict`
The main node function.

-   **Logic**:
    -   Gets plan from state.
    -   Calls `_topological_sort`.
    -   Iterates through sorted steps and prints details (Title, Status, Complexity, Description, Dependencies, Skills, Criteria, Contract Paths).
-   **Returns**: Dictionary containing the (unchanged) implementation plan to pass it forward.

### `_topological_sort(steps: List[PlanStep]) -> List[PlanStep]`
Helper to order steps by dependency.

-   **Logic**:
    -   Builds a map of step IDs.
    -   Uses depth-first search (DFS) with `visited` and `temp_visited` sets to traverse dependencies.
    -   Returns list of steps in execution order.
