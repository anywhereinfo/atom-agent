# atom_agent.tools.code_tools Documentation

## Overall Flow
`src/atom_agent/tools/code_tools.py` allows the agent to execute Python code safely and run tests. It is the primary way the agent performs actions and verifies them.

The flow is:
1.  **Factory**: `create_code_tools(workspace)` binds tools to the current task directory.
2.  **Path Resolution & Security**: Uses `_resolve_relative_path` to ensure all execution happens within the task boundary (prevents path traversal outside the task directory).
3.  **Sandboxing**: Wraps execution in `run_in_bubblewrap` (from `src/atom_agent/sandbox.py`) to isolate the process.
4.  **Temporary Files**: Creates temporary scripts (`_temp_exec_*.py`) to run the code, then cleans them up.

## Use Cases
-   **Execution**: Running scripts to generate artifacts, process data, or demonstrate functionality.
-   **Testing**: Running `pytest` to verify that the implementation meets the requirements.

## Edge Cases
-   **Timeouts**: Enforces strict timeouts (30s for code, 60s for tests) to prevent infinite loops.
-   **Path Escape**: Raises `PermissionError` if the agent tries to access `/etc/passwd` or parent directories.
-   **Cleanup**: Uses `finally` blocks to ensure temporary scripts are deleted even if execution crashes.

## Method Documentation

### `execute_python_code(code: str) -> str`
Executes an arbitrary Python snippet.
-   **Logic**: Writes code to a temp file, runs it in bubblewrap, captures stdout/stderr.
-   **Returns**: Combined output or error message.

### `run_pytest(test_path: str) -> str`
Runs the pytest framework on a specific file.
-   **Logic**:
    -   Creates a driver script that imports `pytest` and calls `pytest.main()`.
    -   Passes the `test_path` as an argument to the driver inside the sandbox.
-   **Returns**: Pytest output (pass/fail summary).
