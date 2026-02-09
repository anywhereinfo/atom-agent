import os
import json
from src.atom_agent.nodes.executor import _load_executor_tools
from src.atom_agent.workspace import Workspace

def verify_tools_loading():
    print("Verifying Tool Loading in Executor...")
    
    # Mock workspace
    ws = Workspace(
        task_id="test",
        task_directory_rel=".",
        task_directory_abs=".",
        allowed_top_level_dirs=[],
        paths={"attempt_impl": "test.py", "attempt_test": "test.py", "attempt_artifacts_dir": "test/"}
    )
    
    # 1. Test without API KEY
    os.environ["TAVILY_API_KEY"] = ""
    tools = _load_executor_tools({}, ws.model_dump())
    tool_names = [t.name for t in tools]
    print(f"Tools loaded (No Key): {tool_names}")
    assert "tavily_search_results_json" not in tool_names
    
    # 2. Test with API KEY (Mock key)
    os.environ["TAVILY_API_KEY"] = "tvly-test-key"
    tools = _load_executor_tools({}, ws.model_dump())
    tool_names = [t.name for t in tools]
    print(f"Tools loaded (With Key): {tool_names}")
    assert "tavily_search" in tool_names
    
    print("Verification SUCCESS: Tools load correctly based on environment.")

if __name__ == "__main__":
    verify_tools_loading()
