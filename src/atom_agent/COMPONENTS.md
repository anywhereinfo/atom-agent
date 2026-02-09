# Component Documentation

This document details every major component within the ATOM Agent system.

## 1. Nodes (`src/atom_agent/nodes/`)

### `setup.py` (Setup Node)
- **Responsibility**: Orchestrates the initial workspace and context.
- **Outputs**: `task_id`, `TaskContext`, `Workspace`.
- **Key Logic**: Generates unique task slugs, creates directory structures, and prepares the workspace contract.

### `planner.py` (Planner Node)
- **Responsibility**: Strategic delegation.
- **Logic**: Uses the `PLANNER_SYSTEM_PROMPT` to analyze the request and generate a structured JSON plan (`PlanStep[]`).
- **Rollback Handling**: Performs a "Complete Reset" of the steps directory if a rollback is triggered.

### `executor.py` (Executor Node)
- **Responsibility**: Tactical execution.
- **Logic**: Loads tools, prepares a specific staging directory, and manages the LLM loop to follow the plan. Writes implementation to `impl.py` and tests to `test.py`.

### `reflector.py` (Reflector Node)
- **Responsibility**: Clinical verification.
- **Logic**: Gathers evidence (test results, file logs), audits them against the step's `acceptance_criteria`, and produces a JSON review packet.
- **Scoring**: Applies deterministic caps (e.g., test failure = max 0.49).

### `commit.py` (Commit Node)
- **Responsibility**: Artifact promotion.
- **Logic**: Copies staged files from `attempts/` to `committed/`, creates a `manifest.json`, and updates the global step state to `completed`.

---

## 2. Tools (`src/atom_agent/tools/`)

### `file_tools.py`
- **Tools**: `write_file`, `read_file`, `list_dir`.
- **Safeguards**: Uses `_resolve_path` to jail operations within the task root. Denies absolute paths and `..` traversal.

### `code_tools.py`
- **Tools**: `execute_python_code`, `run_pytest`.
- **Sandbox**: Wraps execution in `run_in_bubblewrap`.
- **Safeguards**: Jails pytest targets to relative paths within the task root.

### `memory_tools.py`
- **Tools**: `get_committed_step_history`.
- **Responsibility**: Allows the current step to pull detailed history from previous successful (committed) steps.

---

## 3. Core Logic

### `workspace.py`
Defines the `Workspace` Pydantic model. Centralizes path templates and provides authoritative path resolution for nodes and tools.

### `memory.py`
Implements the `MemoryManager`. Handles the reconstruction of context, cross-encoder ranking of historical reports, and persistence of step history.

### `sandbox.py`
Interface to the `bubblewrap` runtime. Ensures that untrusted code execution is isolated from the host machine.

### `state.py`
Defines `AgentState`, `PlanStep`, and `TaskContext`. This is the shared backbone of the LangGraph.
