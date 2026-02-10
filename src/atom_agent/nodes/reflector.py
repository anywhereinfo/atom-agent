import json
import re
from pathlib import Path
from typing import Dict, Any, List
from ..state import AgentState
from ..workspace import Workspace
from ..memory import MemoryManager
from ..config import get_llm
from ..prompts.reflector_prompts import REFLECTOR_SYSTEM_PROMPT, REFLECTOR_USER_PROMPT
from langchain_core.messages import SystemMessage, HumanMessage

def reflector_node(state: AgentState) -> Dict[str, Any]:
    """
    Evaluates the Executor's attempt using an LLM to produce a structured Review JSON.
    """
    workspace = state.get("workspace")
    if isinstance(workspace, dict):
        workspace = Workspace.from_dict(workspace)
        
    task_dir_rel = workspace.task_directory_rel if workspace else "."
    plan = state.get("implementation_plan", [])
    current_step_idx = state.get("current_step_index", 0)
    current_step = plan[current_step_idx]
    
    step_id = current_step.get("step_id", "unknown")
    attempt_num = current_step.get("current_attempt", 1)
    attempt_id = f"a{attempt_num:02d}"
    
    staging_paths = workspace.get_staging_paths(step_id, attempt_id) if workspace else {}
    errors_dir_abs = Path(task_dir_rel) / staging_paths.get("errors_dir", "")
    artifacts_dir_abs = Path(task_dir_rel) / staging_paths.get("artifacts_dir", "")
    
    # --- 1. EVIDENCE GATHERING ---

    # A. Test Results
    test_results_summary = "NO_TEST_RESULTS"
    test_results_path = artifacts_dir_abs / "test_results.json"
    if test_results_path.exists():
        try:
            with open(test_results_path, "r", encoding="utf-8") as f:
                res = json.load(f)
                test_results_summary = json.dumps(res, indent=2)
        except Exception as e:
            test_results_summary = f"ERROR_PARSING_RESULTS: {str(e)}"

    # B. Artifacts Summary
    artifacts_found = []
    if artifacts_dir_abs.exists():
        for p in artifacts_dir_abs.glob("*"):
            if p.is_file():
                try:
                    # Read first 500 chars of artifacts for context
                    content = p.read_text(errors='replace')[:500]
                    artifacts_found.append(f"File: {p.name}\nContent Snippet: {content}")
                except:
                    artifacts_found.append(f"File: {p.name} (Binary or unreadable)")
    artifacts_summary = "\n\n".join(artifacts_found) if artifacts_found else "No artifacts found in staging directory."

    # C. Recent Messages (Tier 1 Memory) - Always use MemoryManager for consistency
    messages = MemoryManager.load_step_history(workspace, step_id)

    recent_messages_str = ""
    for m in messages[-10:]: # Last 10 messages for context
        if hasattr(m, 'content'):
            content = m.content
        elif isinstance(m, dict):
            content = m.get('content', str(m))
        else:
            content = str(m)
        type_name = type(m).__name__ if not isinstance(m, dict) else m.get('type', 'Message')
        recent_messages_str += f"[{type_name}]: {content}\n"

    # D. Dependencies
    dep_summaries = []
    for dep_id in current_step.get("dependencies", []):
        # Find dep in plan
        dep_step = next((s for s in plan if s.get("step_id") == dep_id), None)
        status = dep_step.get("status", "unknown") if dep_step else "missing"
        dep_summaries.append(f"- {dep_id}: {status}")
    dependency_summary = "\n".join(dep_summaries) if dep_summaries else "No dependencies."

    # E. Acceptance Criteria
    criteria = current_step.get("acceptance_criteria", [])
    numbered_criteria = "\n".join([f"{i+1}. {c}" for i, c in enumerate(criteria)])

    # --- 2. LLM INVOCATION ---
    
    user_prompt = REFLECTOR_USER_PROMPT.format(
        step_id=step_id,
        step_title=current_step.get("title", "Untitled"),
        step_description=current_step.get("description", ""),
        attempt_number=attempt_num,
        max_attempts=current_step.get("max_attempts", 3),
        dependency_summary=dependency_summary,
        criteria_count=len(criteria),
        numbered_acceptance_criteria=numbered_criteria,
        execution_summary="See recent messages for detailed trace.",
        recent_messages=recent_messages_str,
        artifacts_summary=artifacts_summary,
        test_results_summary=test_results_summary
    )

    llm = get_llm("reflector")
    response = llm.invoke([
        SystemMessage(content=REFLECTOR_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt)
    ])

    # --- 3. PARSING & VALIDATION ---
    content = response.content
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text_parts.append(part.get("text", ""))
            elif isinstance(part, str):
                text_parts.append(part)
        raw_content = " ".join(text_parts)
    else:
        raw_content = str(content)
    
    raw_content = raw_content.strip()
    
    # 1. Strip Markdown Fences (Robustly)
    if "```" in raw_content:
        # Regex to extract content between ```json and ``` or just ``` and ```
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw_content, re.DOTALL)
        if match:
            raw_content = match.group(1).strip()
        else:
            # Fallback: just strip the markers if regex failed
            raw_content = re.sub(r"```(json)?", "", raw_content).strip()
    
    # 2. Heuristic: Find the first occurrence of '{' and the last occurrence of '}'
    # if the LLM added conversational prefix/suffix
    start_idx = raw_content.find("{")
    end_idx = raw_content.rfind("}")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        raw_content = raw_content[start_idx : end_idx + 1]

    try:
        review = json.loads(raw_content)
    except Exception as e:
        print(f"ERROR: Reflector failed to output valid JSON: {e}\nRaw: {raw_content}", flush=True)
        # Fallback review to force refine/error
        review = {
            "decision": "refine",
            "confidence_score": 0.0,
            "observations": ["Failed to parse Reflector JSON output."],
            "issues_identified": [str(e)],
            "criteria_evaluation": [],
            "refinement_instructions": [{"priority": 1, "file": "reflector.py", "change": "Fix JSON parsing", "verification": "Retry"}]
        }

    # --- 4. PERSISTENCE & STATE UPDATE ---
    
    # Save review to report.json in the attempt directory
    attempt_dir_rel = workspace.get_path("attempt_dir", step_id=step_id, attempt_id=attempt_id) if workspace else f"steps/{step_id}/attempts/{attempt_id}/"
    report_path = Path(task_dir_rel) / attempt_dir_rel / "report.json"
    
    try:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(review, f, indent=2)
    except Exception as e:
        print(f"DEBUG: Failed to save report.json: {e}", flush=True)

    # State update
    if review.get("decision") == "proceed":
        current_step["status"] = "accepted" # Accepted by reflector, but not yet committed
    elif review.get("decision") == "rollback":
        current_step["status"] = "failed"
    else:
        current_step["status"] = "refined"
        # Increment attempt for NEXT run
        current_step["current_attempt"] = attempt_num + 1

    return {
        "implementation_plan": plan,
        "reflector_review": review,
        "phase": "routing",
        "progress_reports": [f"Step {step_id} decision: {review.get('decision')} (Score: {review.get('confidence_score')})"]
    }
