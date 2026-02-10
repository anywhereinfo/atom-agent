from .workspace import Workspace
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing import TypedDict, Literal, Optional, List, Dict, Any, Annotated

class PlanStep(TypedDict):
    """A single step in the implementation plan."""
    step_id: str
    title: str
    description: str
    acceptance_criteria: list[str]
    max_attempts: int
    estimated_complexity: Literal["low", "medium", "high"]
    dependencies: list[str]
    status: Literal["pending", "in_progress", "completed", "failed", "refined", "blocked"]
    
    # Skill Usage
    uses_skills: list[str]              # Skills to load and use for this step
    skill_instructions: dict[str, str] | None # How to use specific skills
    
    # Parallelism
    can_run_in_parallel: bool           # If true, can run concurrently with other independent steps

    # Progress Tracking
    current_attempt: int                # Current attempt number (starts at 1)

class SkillMetadata(TypedDict):
    """Metadata for a registered skill."""
    name: str
    description: str
    version: str
    capabilities: list[str]
    triggers: list[str]
    dependencies: list[str]
    provides: list[str]
    path: str
    last_updated: str

class TaskContext(TypedDict):
    """Metadata about the current task and its environment."""
    task_id: str
    task_description: str           # Explicit first-class field
    task_dir_rel: str               # Authoritative relative path
    task_dir_abs: str               # Absolute path for local convenience
    created_at: str                 # YYYYMMDD_HHMMSS_mmm (UTC)
    timezone: str                  # Always 'UTC'
    request_raw: str
    request_normalized: str
    request_hash8: str
    slug: str
    slug_metadata: dict            # Original source of slug

class AgentState(TypedDict):
    """The central state of the skill learning agent."""
    
    # Context & Foundation
    task_context: TaskContext
    workspace: Workspace         # Authoritative Workspace Contract
    
    # Input
    task_description: str
    skill_name: Optional[str]
    
    # Skill Discovery
    available_skills: list[SkillMetadata]
    skills_to_use: list[str]
    loaded_skill_content: dict[str, dict]

    # Planning
    implementation_plan: list[PlanStep]
    current_step_index: int
    is_skill_learning_task: bool       # Whether this task should result in a new skill
    task_id: str                       # Unique ID for the current task execution

    # Runtime State
    phase: str                         # current phase: planning, executing, reflecting, etc.
    messages: List[BaseMessage]        # Recent conversation history
    progress_reports: List[str]        # Collected reports from nodes
    reflector_review: Optional[Dict[str, Any]] # NEW: Structured review from Reflector
    awaiting_user_input: bool          # HITL flag
    waiting_reason: Optional[str]      # Why are we waiting?
