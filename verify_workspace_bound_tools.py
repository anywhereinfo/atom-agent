from src.atom_agent.tools.file_tools import create_file_tools
from src.atom_agent.tools.code_tools import create_code_tools
from pathlib import Path
import os
import shutil

def test_workspace_bound_tools():
    # Setup a mock workspace
    test_task_dir = Path("test_task_folder")
    if test_task_dir.exists():
        shutil.rmtree(test_task_dir)
    test_task_dir.mkdir()

    workspace = {
        "task_directory_rel": str(test_task_dir)
    }

    # 1. Test File Tools
    file_tools = create_file_tools(workspace)
    write_tool = [t for t in file_tools if t.name == "write_file"][0]
    read_tool = [t for t in file_tools if t.name == "read_file"][0]

    rel_path = "sub/test.txt"
    content = "Hello Workspace!"
    
    # Call tool with relative path
    result = write_tool.invoke({"path": rel_path, "content": content})
    print(f"Write Result: {result}")

    # Check if file exists in the correct place
    expected_abs_path = test_task_dir / rel_path
    assert expected_abs_path.exists(), f"File should be at {expected_abs_path}"
    assert expected_abs_path.read_text() == content
    print("File Tools: SUCCESS")

    # 2. Test Code Tools (CWD)
    code_tools = create_code_tools(workspace)
    exec_tool = [t for t in code_tools if t.name == "execute_python_code"][0]

    # Script results should be in the task dir
    code = "import os; open('marker.txt', 'w').write('here')"
    result = exec_tool.invoke({"code": code})
    print(f"Exec Result: {result}")

    marker_path = test_task_dir / "marker.txt"
    assert marker_path.exists(), f"Marker file should be at {marker_path}"
    print("Code Tools: SUCCESS")

    # Cleanup
    shutil.rmtree(test_task_dir)
    print("ALL VERIFICATIONS PASSED")

if __name__ == "__main__":
    test_workspace_bound_tools()
