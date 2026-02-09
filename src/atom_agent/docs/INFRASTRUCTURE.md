# Infrastructure Deep Dive

This document provides a technical deep dive into the core infrastructure components of the ATOM Agent.

## 1. State Management (`state.py`)

The system state is defined using `TypedDict` and serves as the backbone of the LangGraph orchestrator.

### `AgentState`
- **Purpose**: The central object passed between nodes.
- **Fields**:
  - `task_context`: Static metadata about the task.
  - `workspace`: An instance of the `Workspace` model (authoritative paths).
  - `implementation_plan`: The list of `PlanStep` objects.
  - `current_step_index`: Pointer to the active step.
  - `messages`: List of recent `BaseMessage` objects for the active LLM loop.
  - `reflector_review`: The clinical review packet from the Reflector.
  - `phase`: Current execution phase (e.g., "planning", "executing").

### `PlanStep`
Represents a discrete unit of work within the plan.
- **Accepted Statuses**: `pending`, `in_progress`, `completed`, `failed`, `refined`, `blocked`.
- **Logic**: Tracks its own `current_attempt` and defines its own `acceptance_criteria` and `dependencies`.

---

## 2. Workspace Contract (`workspace.py`)

The `Workspace` class (Pydantic) provides the authoritative contract for file path resolution.

### Key Methods
- `get_path(key, **kwargs)`: Resolves templates from `workspace.json` (e.g., `attempt_impl`).
- `get_staging_paths(step_id, attempt_id)`: Provides a dictionary of all staging targets for a specific attempt.
- `get_task_dir()`: Returns the relative `Path` to the task root.

### Edge Case Handling
- **Path Resolution Errors**: Raises `KeyError` if a template key is missing, forcing the system to rely on strict configuration rather than "guessing" paths.
- **Staging Isolation**: Path templates ensure that every attempt is physically separated, preventing cross-attempt pollution.

---

## 3. Tiered Memory (`memory.py`)

The `MemoryManager` class implements a three-tier retrieval system.

### Retrieval Tiers
1.  **Tier 0**: System and User prompts (Active Instructions).
2.  **Tier 1 (Local)**: Message history for the active step, loaded from `history.json`.
3.  **Tier 2 (Global)**: Summaries of completed steps, ranked via a **Cross-Encoder**.

### Key Methods
- `reconstruct_context()`: The main entry point. selectivly populates tiers based on the `current_attempt`.
- `load_previous_step_reports()`: Gathers `report.json` files from `committed/` directories and ranks them by relevancy to the current query.

### Edge Case Handling
- **Identity Isolation**: On `attempt == 1`, legacy `history.json` is ignored to ensure a "Fresh Start" after a rollback.
- **Ranking Failure**: if the Cross-Encoder fails, the system falls back to the last `K` chronological reports.

---

## 4. Orchestration Flow (`graph.py`)

The `create_graph()` function compiles the LangGraph.

### Conditional Routing
- `route_reflection(state)`: Implements the "Clinical Decision Tree".
  - **Proceed**: Decisions to `commit`.
  - **Refine**: Loop to `executor` (increments attempt).
  - **Rollback**: Strategic return to `planner` (clears disk).
  - **Guardrail**: If `attempt > max_attempts` and not `proceed`, forces a rollback to prevent infinite refine loops.
