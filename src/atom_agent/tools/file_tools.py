from langchain_core.tools import tool
from pathlib import Path
from typing import List

from ..workspace import Workspace

def create_file_tools(workspace: Workspace) -> List:
    """Factory to create workspace-bound file tools."""
    task_dir = workspace.get_task_dir()

    def _resolve_path(path: str) -> Path:
        """Helper to safely resolve a path within the task directory."""
        # 1. Deny absolute paths
        p = Path(path)
        if p.is_absolute():
            # If absolute, we force it relative by stripping root
            p = Path(*p.parts[1:])
        
        # 2. Join and Resolve
        full_path = (task_dir / p).resolve()
        
        # 3. Prevent Escape (Path traversal check)
        if not str(full_path).startswith(str(task_dir.resolve())):
            raise PermissionError(f"Path escape detected! Access to {path} is denied.")
            
        return full_path

    @tool
    def write_file(path: str, content: str) -> str:
        """Write content to a file. Overwrites if exists. Paths are relative to task root."""
        print(f"DEBUG TOOL: write_file called for: {path}", flush=True)
        try:
            full_path = _resolve_path(path)
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content, encoding="utf-8")
            return f"Successfully wrote {len(content)} bytes to {path}"
        except Exception as e:
            return f"Error writing file: {str(e)}"

    @tool
    def read_file(path: str) -> str:
        """Read content from a file. Paths are relative to task root."""
        try:
            full_path = _resolve_path(path)
            if not full_path.exists():
                return f"Error: File {path} does not exist."
            return full_path.read_text(encoding="utf-8")
        except Exception as e:
            return f"Error reading file: {str(e)}"

    @tool
    def list_dir(path: str = ".") -> str:
        """List contents of a directory. Paths are relative to task root."""
        try:
            full_path = _resolve_path(path)
            if not full_path.exists():
                return f"Directory {path} does not exist"
            items = [f"{x.name}{'/' if x.is_dir() else ''}" for x in full_path.iterdir()]
            return "\n".join(items)
        except Exception as e:
            return f"Error listing directory: {str(e)}"

    return [write_file, read_file, list_dir]
