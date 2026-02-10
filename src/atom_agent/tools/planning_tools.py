"""
Planning-specific tools for research and plan submission.
"""
import json
from pathlib import Path
from langchain_core.tools import tool
from typing import List, Dict, Any
from pydantic import BaseModel, Field

from ..workspace import Workspace
from .search_tools import create_search_tools


class PlanStepInput(BaseModel):
    """Schema for a single plan step."""
    step_id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,47}$")
    title: str
    description: str
    acceptance_criteria: List[str]
    max_attempts: int = 3
    estimated_complexity: str = Field(pattern="^(low|medium|high)$")
    dependencies: List[str] = Field(default_factory=list)
    uses_skills: List[str] = Field(default_factory=list)
    skill_instructions: Dict[str, str] | None = None
    can_run_in_parallel: bool = False


class ImplementationPlanInput(BaseModel):
    """Schema for the complete implementation plan."""
    task_id: str
    task_directory_rel: str
    steps: List[PlanStepInput]


def create_planning_tools(workspace: Workspace, task_context: Dict[str, Any]) -> List:
    """
    Factory to create tools for agentic planning with research capabilities.
    """
    tools = []

    # 1. Add search tools for web research
    tools.extend(create_search_tools())

    # 2. Add file reading tools for accessing previous artifacts
    task_dir = Path(workspace.task_directory_rel) if workspace else Path(".")

    @tool
    def read_committed_artifact(step_id: str, artifact_name: str) -> str:
        """
        Read a specific artifact from a previously completed step's committed directory.
        Useful for understanding what work has already been done.

        Args:
            step_id: The step identifier (e.g., 'research_planning')
            artifact_name: The artifact filename (e.g., 'research_notes.md')
        """
        try:
            if not workspace:
                return "Error: Workspace not initialized"

            committed_dir_rel = workspace.get_path("committed_dir", step_id=step_id)
            artifact_path = task_dir / committed_dir_rel / "artifacts" / artifact_name

            if not artifact_path.exists():
                return f"Artifact {artifact_name} not found in step {step_id}"

            content = artifact_path.read_text(encoding="utf-8")
            return content
        except Exception as e:
            return f"Error reading artifact: {str(e)}"

    @tool
    def list_completed_steps() -> str:
        """
        List all completed steps with their committed artifacts.
        Useful for understanding the current state of the project.
        """
        try:
            steps_dir = task_dir / "steps"
            if not steps_dir.exists():
                return "No steps directory found. This is a new task."

            completed_steps = []
            for step_dir in steps_dir.iterdir():
                if not step_dir.is_dir():
                    continue

                committed_dir = step_dir / "committed"
                if committed_dir.exists():
                    step_id = step_dir.name
                    artifacts_dir = committed_dir / "artifacts"
                    artifacts = []
                    if artifacts_dir.exists():
                        artifacts = [f.name for f in artifacts_dir.iterdir() if f.is_file()]

                    completed_steps.append({
                        "step_id": step_id,
                        "artifacts": artifacts
                    })

            if not completed_steps:
                return "No completed steps found. This is a fresh planning session."

            return json.dumps(completed_steps, indent=2)
        except Exception as e:
            return f"Error listing steps: {str(e)}"

    @tool
    def submit_plan(plan_json: str) -> str:
        """
        Submit the final implementation plan as a JSON string.
        This should be called AFTER you've completed your research and are ready to finalize the plan.

        The plan must be a valid JSON object with:
        - task_id: str
        - task_directory_rel: str
        - steps: List of step objects

        Each step must have:
        - step_id: lowercase_snake_case matching ^[a-z][a-z0-9_]{0,47}$
        - title: Short descriptive name
        - description: Detailed technical instructions
        - acceptance_criteria: List of verifiable outcomes
        - max_attempts: Number (default 3)
        - estimated_complexity: "low" | "medium" | "high"
        - dependencies: List of step_ids this depends on
        - uses_skills: List of skills to use
        - skill_instructions: Optional dict of skill-specific instructions
        - can_run_in_parallel: Boolean

        Args:
            plan_json: The complete plan as a JSON string
        """
        try:
            # Validate JSON structure
            plan_data = json.loads(plan_json)

            # Validate using Pydantic
            validated_plan = ImplementationPlanInput(**plan_data)

            return json.dumps({
                "status": "success",
                "message": "Plan validated and ready for execution",
                "steps_count": len(validated_plan.steps),
                "validated_plan": validated_plan.model_dump()
            })
        except json.JSONDecodeError as e:
            return f"ERROR: Invalid JSON format: {str(e)}"
        except Exception as e:
            return f"ERROR: Plan validation failed: {str(e)}"

    tools.extend([read_committed_artifact, list_completed_steps, submit_plan])

    return tools
