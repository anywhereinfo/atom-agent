import json
import shutil
from pathlib import Path
from typing import Dict, Any
from ..state import AgentState
from ..workspace import Workspace

def commit_node(state: AgentState) -> Dict[str, Any]:
    """
    Promotes the successful attempt to the authoritative 'committed/' directory.
    This fulfills the logic: Attempt -> Reflect -> Commit.
    """
    workspace = state.get("workspace")
    if isinstance(workspace, dict):
        workspace = Workspace.from_dict(workspace)
        
    plan = state.get("implementation_plan", [])
    current_step_idx = state.get("current_step_index", 0)
    
    if current_step_idx >= len(plan):
        return {}

    current_step = plan[current_step_idx]
    step_id = current_step.get("step_id", "unknown")
    attempt_num = current_step.get("current_attempt", 1)
    attempt_id = f"a{attempt_num:02d}"
    
    # Check if we should actually commit
    review = state.get("reflector_review", {})
    if review.get("decision") != "proceed":
        print(f"DEBUG COMMIT: Decision is {review.get('decision')}. Skipping commit.", flush=True)
        return {}

    task_dir = Path(workspace.task_directory_rel)
    
    # Source: Attempt staging
    staging_paths = workspace.get_staging_paths(step_id, attempt_id)
    attempt_dir = task_dir / staging_paths.get("staging_dir", f"steps/{step_id}/attempts/{attempt_id}/") # Fallback if key missing
    if not attempt_dir.exists():
        # Fallback manual resolution if staging_paths is incomplete
        attempt_dir = task_dir / f"steps/{step_id}/attempts/{attempt_id}"
    
    # Destination: Committed
    committed_dir = task_dir / f"steps/{step_id}/committed"
    
    print(f"DEBUG COMMIT: Promoting {attempt_id} to committed for step {step_id}", flush=True)
    
    try:
        committed_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Promote impl.py
        src_impl = task_dir / staging_paths.get("impl")
        dst_impl = committed_dir / "impl.py"
        if src_impl.exists():
            shutil.copy2(src_impl, dst_impl)
            
        # 2. Promote test.py
        src_test = task_dir / staging_paths.get("test")
        dst_test = committed_dir / "test.py"
        if src_test.exists():
            shutil.copy2(src_test, dst_test)
            
        # 3. Promote Artifacts
        src_artifacts = task_dir / staging_paths.get("artifacts_dir")
        dst_artifacts = committed_dir / "artifacts"
        if src_artifacts.exists():
            if dst_artifacts.exists():
                shutil.rmtree(dst_artifacts)
            shutil.copytree(src_artifacts, dst_artifacts)
            
        # 4. Create Manifest
        manifest = {
            "step_id": step_id,
            "promoted_attempt": attempt_id,
            "timestamp": review.get("timestamp"), # Use reflector's timestamp if available
            "score": review.get("confidence_score"),
            "acceptance_criteria": current_step.get("acceptance_criteria"),
            "files": [f.name for f in committed_dir.glob("**/*") if f.is_file()]
        }
        with open(committed_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
            
        print(f"DEBUG COMMIT: Successfully promoted {step_id} to committed.", flush=True)
        
        # Update step status in the plan
        current_step["status"] = "completed"
        # The graph will increment the index, or we can do it here to be safe and deterministic
        return {
            "implementation_plan": plan,
            "current_step_index": current_step_idx + 1,
            "phase": "executing" if (current_step_idx + 1) < len(plan) else "completed"
        }
        
    except Exception as e:
        print(f"DEBUG COMMIT: Failed to promote artifacts: {str(e)}", flush=True)
        # We don't increment index if commit fails? 
        return {"phase": "error", "progress_reports": [f"Commit failed: {str(e)}"]}
