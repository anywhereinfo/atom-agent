# Tools & Sandbox Specification

This document details the toolbelt and execution environment governing the agent's interaction with the system.

## 1. File Tools (`file_tools.py`)
- **Design Philosophy**: Jail by default.
- **Tools**: `write_file`, `read_file`, `list_dir`.
- **Method: `_resolve_path(path)`**:
  - Forces all paths to be relative to the task root.
  - Blocks absolute path hijacking (e.g., writing to `/etc/`).
  - Blocks directory traversal attacks (e.g., `../../../`).
- **Edge cases**: If a path escape is detected, it raises a `PermissionError`, resulting in a clinical failure in the executor loop.

## 2. Code Tools (`code_tools.py`)
- **Tools**: `execute_python_code`, `run_pytest`.
- **Method: `_resolve_relative_path(path)`**: Ensures that the target file for execution or testing is strictly within the task workspace.
- **Workflow**:
  - Generates a temporary driver script in the task directory.
  - Invokes `bubblewrap` to execute the script in an isolated PID/Mount namespace.
  - Cleans up driver scripts regardless of execution success.

## 3. Memory Tools (`memory_tools.py`)
- **Tool**: `get_committed_step_history(step_id)`.
- **Purpose**: Allows the executor to "remember" details from past successful steps that were not included in the lean Tier 2 summary.
- **Logic**: Reads the `history.json` from the target step's `committed/` directory.

## 4. Sandbox Provider (`sandbox.py`)
- **Backbone**: `run_in_bubblewrap`.
- **Capabilities**:
  - PID Isolation.
  - Mount namespace isolation (maps the task workspace to `/work` in the sandbox).
  - Network isolation (disabled by default for security).
  - Resource limits (CPU/Memory via cgroups if configured).
- **Edge Cases**:
  - **Timeouts**: Terminates scripts after a configured limit (default 30s-60s) to prevent infinite loops.
