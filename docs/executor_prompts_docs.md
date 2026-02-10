# atom_agent.prompts.executor_prompts Documentation

## Overall Flow
`src/atom_agent/prompts/executor_prompts.py` defines the instructions for the Lead Executor agent. These prompts enforce the "ReAct" (Reasoning + Acting) loop and strict adherence to the implementation plan.

The flow is:
1.  **System Prompt**: Establishes the Executor's role (implement one step, do not replan) and the "Workspace Contract" (where to write).
2.  **User Prompt**: Provides the specific step details (ID, description, criteria) and mandatory staging targets.

## Key Prompt Components

### `EXECUTOR_SYSTEM_PROMPT`
-   **Role**: "Lead Executor... responsible for implementing ONE step."
-   **Authoritative Inputs**: Defines `state/plan.json` as the Single Source of Truth.
-   **Workspace Rules**:
    -   Must write ONLY to `steps/{step_id}/attempts/{attempt_id}/`.
    -   Must NOT use `..` to escape.
    -   Must NOT write to `committed/`.
-   **Execution Loop**: Impl -> Test -> Run -> Log.
-   **Analytical Depth**: Conditional logic requiring deeper analysis (complexity, failure modes) if the task is an evaluation.

### `EXECUTOR_USER_PROMPT`
-   **Step Context**: `{step_id}`, `{step_title}`, `{step_description}`, `{acceptance_criteria}`.
-   **Staging Targets**: Explicit paths for `impl.py` and `test.py` (e.g., `steps/login/attempts/a01/impl.py`).
-   **Mandatory Testing**: Explicit instruction to run `run_pytest` and create `test_results.json`.

## Use Cases
-   **Context Isolation**: Prevents the executor from "hallucinating" requirements by grounding it in a rigid prompt structure.
-   **Path Enforcement**: The prompt explicitly tells the LLM where to write, reducing `PermissionError`s during tool execution.
