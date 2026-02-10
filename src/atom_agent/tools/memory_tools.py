import json
from pathlib import Path
from langchain_core.tools import tool
from typing import Dict, Any

from ..workspace import Workspace

def create_memory_tools(workspace: Workspace) -> list:
    """
    Factory to create memory-related tools bound to the current workspace.
    """
    task_dir_rel = workspace.task_directory_rel
    paths = workspace.paths
    
    @tool
    def get_committed_step_history(step_id: str) -> str:
        """
        Retrieves the full message history and final report for a previously committed step.
        Use this to 'deep dive' into architectural details or code changes from earlier steps.
        """
        committed_dir_template = paths.get("committed_dir", "steps/{step_id}/committed/").format(step_id=step_id)
        committed_dir = Path(task_dir_rel) / committed_dir_template
        
        report_path = committed_dir / "report.json"
        history_path = committed_dir / "messages" / "history.json"
        
        output = []
        
        # 1. Load Report
        if report_path.exists():
            try:
                with open(report_path, "r", encoding="utf-8") as f:
                    report_data = json.load(f)
                output.append(f"### FULL REPORT: {step_id}")
                output.append(json.dumps(report_data, indent=2))
            except Exception as e:
                output.append(f"Error loading report for {step_id}: {str(e)}")
        else:
            output.append(f"No report.json found for step {step_id}")
            
        # 2. Load History
        if history_path.exists():
            try:
                with open(history_path, "r", encoding="utf-8") as f:
                    history_data = json.load(f)
                output.append(f"\n### FULL MESSAGE HISTORY: {step_id}")
                # We limit the output to avoid token overflow, but provide the core chain
                output.append(json.dumps(history_data, indent=2))
            except Exception as e:
                output.append(f"Error loading history for {step_id}: {str(e)}")
        else:
            output.append(f"No history.json found for step {step_id}")
            
        return "\n\n".join(output)

    return [get_committed_step_history]
