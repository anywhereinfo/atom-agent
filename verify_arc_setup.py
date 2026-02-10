import os
import json
import sys
from pathlib import Path

# Add src to path so we can import atom_agent
sys.path.append(str(Path(__file__).parent / "src"))

from atom_agent.nodes.setup import task_setup_node
from atom_agent.state import AgentState

def test_refined_arc_setup():
    print("Testing Refined ARC Setup Node...")
    
    # Mock State
    state: AgentState = {
        "request_raw": "Build a simple web scraper for news headlines.",
        "task_description": "Build a news headline scraper",
        "available_skills": [],
        "messages": [],
        "current_step_index": 0,
        "phase": "setup"
    }
    
    try:
        # Run Setup Node
        result = task_setup_node(state)
        
        ctx = result["task_context"]
        task_id = ctx["task_id"]
        task_dir_rel = ctx["task_dir_rel"]
        task_dir_abs = ctx["task_dir_abs"]
        
        print(f"Generated Task ID: {task_id}")
        print(f"Relative Dir: {task_dir_rel}")
        print(f"Absolute Dir: {task_dir_abs}")
        
        # 1. Verify Directory Scaffold
        # Should only have: inputs, steps, state, logs
        expected_top_dirs = {"inputs", "steps", "state", "logs"}
        actual_top_dirs = {d.name for d in Path(task_dir_abs).iterdir() if d.is_dir()}
        
        print(f"Top-level directories: {actual_top_dirs}")
        assert actual_top_dirs == expected_top_dirs, f"Scaffold mismatch! Expected {expected_top_dirs}, got {actual_top_dirs}"
        
        # 2. Verify workspace.json
        workspace_path = Path(task_dir_abs) / "state" / "workspace.json"
        assert workspace_path.exists(), "workspace.json missing!"
        
        with open(workspace_path, "r") as f:
            workspace = json.load(f)
            
        print("Verifying machine-enforceable rules in workspace.json...")
        assert "naming" in workspace
        assert workspace["naming"]["step_id"]["regex"] == r"^[a-z][a-z0-9_]{0,47}$"
        assert workspace["naming"]["attempt_id"]["regex"] == r"^a\d{2,3}$"
        
        assert "paths" in workspace
        assert "write_rules" in workspace
        assert "read_rules" in workspace
        assert "step_metadata" in workspace
        
        # Verify new contract fields
        assert "defaults" in workspace
        assert workspace["defaults"]["max_attempts"] == 3
        assert "step_structure" in workspace
        assert "attempt_required_subdirs" in workspace["step_structure"]
        assert "authority" in workspace
        assert "authoritative_root" in workspace["authority"]
        
        # Verify specific paths
        assert "steps/{step_id}/attempts/{attempt_id}/impl.py" in workspace["paths"]["attempt_impl"]
        
        print("Verifying task.json location and metadata...")
        task_json_path = Path(task_dir_abs) / "state" / "task.json"
        assert task_json_path.exists(), "task.json should be in state/ directory"
        
        with open(task_json_path, "r") as f:
            task_meta = json.load(f)
            assert "blurb" in task_meta["slug_metadata"]
            assert task_meta["slug_metadata"]["blurb"].startswith("Build a news")
            
        print("Verifying plan.json location and stub...")
        plan_json_path = Path(task_dir_abs) / "state" / "plan.json"
        assert plan_json_path.exists(), "plan.json should be initialized in state/"
        with open(plan_json_path, "r") as f:
            plan_meta = json.load(f)
            assert plan_meta["task_id"] == task_id
            assert plan_meta["steps"] == []
        
        print("\n✅ Refined ARC Setup Verifications PASSED!")
        
    except Exception as e:
        print(f"\n❌ Refined ARC Setup Verifications FAILED: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_refined_arc_setup()
