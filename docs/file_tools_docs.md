# atom_agent.tools.file_tools Documentation

## Overall Flow
`src/atom_agent/tools/file_tools.py` provides basic file system operations (read, write, list) to the agent. Like other toolsets, it is strictly bound to the `Workspace` to prevent unauthorized access.

The flow is:
1.  **Factory**: `create_file_tools(workspace)` initializes the tools.
2.  **Path Resolution**: All operations go through `_resolve_path`, which:
    -   Converts absolute paths to relative ones (stripping root).
    -   Resolves canonical paths.
    -   Checks for path traversal (ensures path starts with task root).
3.  **Operation**: Performs the requested file I/O using standard Python `pathlib`.

## Use Cases
-   **Reading Code**: The agent reads existing files to understand the current state.
-   **Listing Directories**: The agent explores the file structure to find where files are located.
-   **Writing Configs**: Creating non-executable files (like JSON or Markdown). Note that actual *code* execution is handled by `code_tools`, but `write_file` can be used to set up data files.

## Edge Cases
-   **Missing Files**: `read_file` returns a clear error message string instead of raising an exception, allowing the agent to react (e.g., "File not found").
-   **Directory Creation**: `write_file` automatically creates parent directories if they don't exist.
-   **Path Traversal**: Explicitly blocks attempts to read `../` or absolute paths outside the sandbox.

## Method Documentation

### `write_file(path: str, content: str) -> str`
Writes text content to a file.
-   **Returns**: Success message with byte count.

### `read_file(path: str) -> str`
Reads text content.
-   **Returns**: File content or error message.

### `list_dir(path: str = ".") -> str`
Lists directory contents.
-   **Returns**: Newline-separated list of filenames (directories suffixed with `/`).
