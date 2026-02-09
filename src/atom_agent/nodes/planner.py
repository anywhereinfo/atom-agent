from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Dict
import json
from pathlib import Path
from ..prompts.planner_prompts import PLANNER_SYSTEM_PROMPT, PLANNER_USER_PROMPT
from ..state import AgentState, PlanStep
from ..workspace import Workspace
from ..config import get_llm
from langchain_core.prompts import ChatPromptTemplate

class PlanStepModel(BaseModel):
    step_id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,47}$")
    title: str
    description: str
    acceptance_criteria: List[str]
    max_attempts: int = 3
    estimated_complexity: Literal["low", "medium", "high"]
    dependencies: List[str] = Field(default_factory=list)
    uses_skills: List[str] = Field(default_factory=list)
    skill_instructions: Optional[Dict[str, str]] = None
    can_run_in_parallel: bool = False

class ImplementationPlan(BaseModel):
    task_id: str
    task_directory_rel: str
    steps: List[PlanStepModel]

def format_available_skills(skills: list) -> str:
    if not skills:
        return "None"
    return "\n".join([f"- {s['name']}: {s['description']}" for s in skills])

def planner_node(state: AgentState) -> dict:
    llm = get_llm("planner")
    structured_llm = llm.with_structured_output(ImplementationPlan)

    prompt = ChatPromptTemplate.from_messages([
        ("system", PLANNER_SYSTEM_PROMPT),
        ("user", PLANNER_USER_PROMPT)
    ])

    available_skills_summary = format_available_skills(state.get("available_skills", []))
    
    # Authoritative Workspace Contract from state
    workspace = state.get("workspace")
    if not workspace:
        # Fallback to reading disk if not in state (e.g. initial dev or resumed task)
        context = state.get("task_context", {})
        task_dir_rel = context.get("task_dir_rel")
        if task_dir_rel:
            workspace_json_path = Path(task_dir_rel) / "state" / "workspace.json"
            if workspace_json_path.exists():
                try:
                    with open(workspace_json_path, "r") as f:
                        workspace_data = json.load(f)
                        workspace = Workspace.from_dict(workspace_data)
                except Exception as e:
                    print(f"DEBUG: Failed to read workspace.json from disk: {str(e)}", flush=True)

    workspace_context_str = json.dumps(workspace.model_dump(), indent=2) if workspace else "{}"

    # Rollback / Failure Context & Cleanup
    review = state.get("reflector_review")
    rollback_context = "No previous failures reported."
    if review and review.get("decision") == "rollback":
        rollback_context = (
            f"PREVIOUS FAILURE DETECTED (Rollback triggered).\n"
            f"Reason: {review.get('rollback_reason', 'Not provided')}\n"
            f"Issues: {', '.join(review.get('issues_identified', []))}\n"
            f"Hint: {review.get('alternative_approach_hint', 'No hint provided')}"
        )
        # Complete Reset: Wipe the steps directory to avoid "Ghost Context" residues
        if workspace:
            steps_dir = Path(workspace.task_directory_rel) / "steps"
            if steps_dir.exists():
                import shutil
                try:
                    # We keep the steps/ dir but clear its children
                    for item in steps_dir.iterdir():
                        if item.is_dir():
                            shutil.rmtree(item)
                        else:
                            item.unlink()
                    print(f"DEBUG: Complete Reset performed. Cleared legacy steps in {steps_dir}", flush=True)
                except Exception as e:
                    print(f"DEBUG: Failed to clear legacy steps: {str(e)}", flush=True)

    print(f"DEBUG: Planning for task: {state['task_description'][:50]}...", flush=True)
    
    chain = prompt | structured_llm
    import time
    start_time = time.time()
    print("DEBUG: Invoking planner LLM...", flush=True)
    try:
        result = chain.invoke({
            "task_description": state["task_description"],
            "workspace_context": workspace_context_str,
            "skill_name": state.get("skill_name") or "Not Specified",
            "available_skills_summary": available_skills_summary,
            "rollback_context": rollback_context
        })
        duration = time.time() - start_time
        print(f"DEBUG: Planner LLM response received in {duration:.2f}s. Steps generated: {len(result.steps)}", flush=True)
    except Exception as e:
        print(f"DEBUG: Planner LLM invocation FAILED: {str(e)}", flush=True)
        raise e

    # Convert Pydantic models to TypedDicts for state
    plan_steps = []
    for step in result.steps:
        plan_step: PlanStep = {
            "step_id": step.step_id,
            "title": step.title,
            "description": step.description,
            "acceptance_criteria": step.acceptance_criteria,
            "max_attempts": step.max_attempts,
            "estimated_complexity": step.estimated_complexity,
            "dependencies": step.dependencies,
            "status": "pending",
            "uses_skills": step.uses_skills,
            "skill_instructions": step.skill_instructions,
            "can_run_in_parallel": step.can_run_in_parallel,
            "current_attempt": 1
        }
        plan_steps.append(plan_step)
            
    # Persist the plan as a first-class artifact using contract paths
    if workspace:
        plan_rel_path = workspace.get_path("plan_path")
        task_dir_rel = workspace.task_directory_rel
        if task_dir_rel:
            plan_path = Path(task_dir_rel) / plan_rel_path
            plan_data = {
                "task_id": result.task_id,
                "task_directory_rel": result.task_directory_rel,
                "steps": plan_steps
            }
            try:
                plan_path.parent.mkdir(parents=True, exist_ok=True)
                with open(plan_path, "w") as f:
                    json.dump(plan_data, f, indent=2)
                print(f"DEBUG: Plan persisted to {plan_path}", flush=True)
            except Exception as e:
                print(f"DEBUG: Failed to persist plan: {str(e)}", flush=True)

    return {
        "implementation_plan": plan_steps,
        "current_step_index": 0,
        "phase": "executing",
        "progress_reports": [] 
    }
