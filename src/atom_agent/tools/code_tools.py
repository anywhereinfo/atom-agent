import sys
import uuid
import os
from pathlib import Path
from langchain_core.tools import tool
from typing import List
from ..sandbox import run_in_bubblewrap

from ..workspace import Workspace

def create_code_tools(workspace: Workspace) -> List:
    """Factory to create workspace-bound code execution tools."""
    task_dir_abs = workspace.get_task_dir().absolute()

    def _resolve_relative_path(path: str) -> str:
        """Ensures the path is relative and stays within the task directory."""
        p = Path(path)
        if p.is_absolute():
            p = Path(*p.parts[1:])
        
        # Resolve to check for escape
        full_path = (task_dir_abs / p).resolve()
        if not str(full_path).startswith(str(task_dir_abs.resolve())):
            raise PermissionError(f"Path escape detected! Access to {path} is denied.")
        
        # Return path relative to task_dir_abs for use in sandbox
        return str(p)

    @tool
    def execute_python_code(code: str) -> str:
        """
        Execute a snippet of Python code within the task workspace using a sandbox.
        Output(stdout/stderr) is captured and returned.
        """
        print(f"DEBUG TOOL: execute_python_code called. CWD: {task_dir_abs}", flush=True)
        
        # Create a unique temp script in the task dir
        script_name = f"_temp_exec_{uuid.uuid4().hex}.py"
        script_path = task_dir_abs / script_name
        
        try:
            # Ensure workspace exists
            task_dir_abs.mkdir(parents=True, exist_ok=True)
            
            # Write the code to the script
            script_path.write_text(code, encoding="utf-8")
            
            # Run in sandbox
            result = run_in_bubblewrap(
                attempt_dir=task_dir_abs,
                script_path=script_path,
                timeout_s=30
            )
            
            output = result["stdout"]
            if result["stderr"]:
                output += f"\nSTDERR:\n{result['stderr']}"
            if result["timed_out"]:
                output += "\nError: Execution timed out (30s limit)"
                
            return output if output else "No output produced"
            
        except Exception as e:
            return f"Error executing code: {str(e)}"
        finally:
            # Cleanup
            if script_path.exists():
                script_path.unlink()

    @tool
    def run_pytest(test_path: str) -> str:
        """Run pytest on a specific file or directory relative to task root."""
        print(f"DEBUG TOOL: run_pytest called for {test_path}", flush=True)
        
        try:
            # Validate path
            safe_test_path = _resolve_relative_path(test_path)
            
            # Create a driver script for pytest
            driver_name = f"_temp_pytest_driver_{uuid.uuid4().hex}.py"
            driver_path = task_dir_abs / driver_name
            
            driver_code = (
                "import sys, pytest\n"
                # We pass arguments via sys.argv to the driver
                "sys.exit(pytest.main(sys.argv[1:]))"
            )
            
            task_dir_abs.mkdir(parents=True, exist_ok=True)
            driver_path.write_text(driver_code, encoding="utf-8")
            
            # Run sandbox targeting the driver, passing the test path as an arg
            result = run_in_bubblewrap(
                attempt_dir=task_dir_abs,
                script_path=driver_path,
                args=[safe_test_path],
                timeout_s=60
            )
            
            output = result["stdout"]
            if result["stderr"]:
                output += f"\nSTDERR:\n{result['stderr']}"
            if result["timed_out"]:
                output += "\nError: Test execution timed out (60s limit)"
                
            return output
            
        except Exception as e:
            return f"Error running tests: {str(e)}"
        finally:
            if 'driver_path' in locals() and driver_path.exists():
                driver_path.unlink()

    return [execute_python_code, run_pytest]
