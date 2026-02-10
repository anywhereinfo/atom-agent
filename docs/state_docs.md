# atom_agent.state Documentation

## Overall Flow
`src/atom_agent/state.py` defines the core data structures and types used to maintain state within the agentic workflow. It acts as the "database" or "schema" for the entire application, defining what information is passed between nodes in the `langgraph` state.

It primarily defines `TypedDict` schemas for:
-   `PlanStep`: A single step in the implementation plan.
-   `SkillMetadata`: Metadata for available skills.
-   `TaskContext`: Information about the current task (paths, ID, description).
-   `AgentState`: The central state object passed through the computation graph.

## Use Cases
-   **Node Development**: Developers modify this file when adding new fields to the global state (e.g., adding a `reflector_feedback` field).
-   **Type Hinting**: Used extensively across the codebase for type safety and clarity.

## Edge Cases
-   **None**: This file only contains type definitions and has no runtime logic.

## Class Documentation

### `PlanStep` (TypedDict)
Represents a single step in the execution plan.
-   **Fields**:
    -   `step_id`: Unique ID.
    -   `title`: Step name.
    -   `description`: Detailed description.
    -   `acceptance_criteria`: List of criteria for success.
    -   `max_attempts`: Retry limit.
    -   `estimated_complexity`: low/medium/high.
    -   `dependencies`: List of step IDs this step depends on.
    -   `status`: Current status (pending, in_progress, completed, failed, refined, blocked).
    -   `uses_skills`: Skills required.
    -   `skill_instructions`: Instructions for specific skills.
    -   `can_run_in_parallel`: Boolean flag.
    -   `current_attempt`: Attempt counter.

### `SkillMetadata` (TypedDict)
Metadata for a registered skill.
-   **Fields**: `name`, `description`, `version`, `capabilities`, `triggers`, `dependencies`, `provides`, `path`, `last_updated`.

### `TaskContext` (TypedDict)
Context metadata for the running task.
-   **Fields**: `task_id`, `task_description`, `task_dir_rel` (relative path), `task_dir_abs` (absolute path), `created_at`, `timezone` (UTC), `request_raw`, `request_normalized`, `request_hash8`, `slug`, `slug_metadata`.

### `AgentState` (TypedDict)
The main state object for the `StateGraph`.
-   **Fields**:
    -   **Context**: `task_context`, `workspace`.
    -   **Input**: `task_description`, `skill_name`.
    -   **Skills**: `available_skills`, `skills_to_use`, `loaded_skill_content`.
    -   **Planning**: `implementation_plan`, `current_step_index`, `is_skill_learning_task`, `task_id`.
    -   **Runtime**: `phase`, `messages` (chat history), `progress_reports`, `reflector_review` (structured feedback), `awaiting_user_input`, `waiting_reason`.
