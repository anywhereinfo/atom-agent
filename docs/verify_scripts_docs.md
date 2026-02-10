# Verification Scripts Documentation

## Overview
These scripts (`verify_*.py`) are standalone test harnesses used to assert the correctness of critical agent components *without* running the full agent loop. They serve as unit tests and integration tests for the agent's infrastructure.

## Scripts

### 1. `verify_arc_setup.py`
**Goal**: Verifies that the `setup` node correctly initializes the Attempt-Reflect-Commit (ARC) file structure.
-   **Checks**:
    -   Creation of `inputs/`, `steps/`, `state/`, `logs/`.
    -   Structure of `workspace.json` (regex rules, path templates, permissions).
    -   Initialization of `task.json` and `plan.json`.
-   **Key Assertion**: Ensures `workspace.json` contains the machine-enforceable rules that downstream nodes rely on.

### 2. `verify_executor_prompt_v2.py`
**Goal**: Verifies that the `Executor` prompt is constructed correctly with all necessary context.
-   **Checks**:
    -   Dynamic injection of `step_id`, `attempt_id`, and `acceptance_criteria`.
    -   Presence of mandatory `staged_impl_path` and `staged_test_path`.
-   **Key Assertion**: The prompt must contain JSON-formatted instructions that match the current state.

### 3. `verify_path_resolution.py`
**Goal**: Verifies that the `Planner` node respects the paths defined in the `Workspace` contract.
-   **Checks**:
    -   Ability to read `config.py`.
    -   Resolution of `plan.json` path from the workspace object.
-   **Key Assertion**: The planner must not hardcode paths; it must look them up in the workspace.

### 4. `verify_search_tool.py`
**Goal**: Verifies that external tools load correctly based on environment variables.
-   **Checks**:
    -   `tavily_search` is NOT loaded if API key is missing.
    -   `tavily_search` IS loaded if API key is present.
-   **Key Assertion**: The agent gracefully degrades if external services are unavailable.

### 5. `verify_workspace_bound_tools.py`
**Goal**: Verifies that file system tools (`write_file`, `execute_python_code`) are strictly confined to the task directory.
-   **Checks**:
    -   Relative path resolution (`sub/test.txt` -> `{task_dir}/sub/test.txt`).
    -   Execution of code in the correct CWD.
    -   Cleanup of test directories.
-   **Key Assertion**: Tools operation within the sandbox boundary.
