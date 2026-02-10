from .workspace import Workspace, StepWorkspace, AttemptWorkspace
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing import TypedDict, Literal, Optional, List, Dict, Any, Annotated


class Attempt(BaseModel):
    """A single execution attempt within a PlanStep."""
    attempt_id: str                          # e.g. "a01", "a02"
    attempt_number: int                      # 1-based
    status: Literal["pending", "running", "accepted", "refined", "failed"] = "pending"
    workspace: Optional[AttemptWorkspace] = None  # Index into workspace for this attempt

    # Populated by reflector after evaluation
    decision: Optional[str] = None           # "proceed" | "refine" | "rollback"
    confidence_score: Optional[float] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

    model_config = {"arbitrary_types_allowed": True}


class PlanStep(BaseModel):
    """A single step in the implementation plan."""
    step_id: str
    title: str
    description: str = ""
    acceptance_criteria: list[str] = Field(default_factory=list)
    max_attempts: int = 3
    estimated_complexity: Literal["low", "medium", "high"] = "medium"
    dependencies: list[str] = Field(default_factory=list)
    status: Literal["pending", "in_progress", "completed", "failed", "refined", "blocked", "accepted"] = "pending"

    # Skill Usage
    uses_skills: list[str] = Field(default_factory=list)
    skill_instructions: Optional[dict[str, str]] = None

    # Parallelism
    can_run_in_parallel: bool = False

    # Workspace binding — set by planner at plan finalization
    step_workspace: Optional[StepWorkspace] = None

    # Attempt tracking — replaces bare `current_attempt: int`
    attempts: list[Attempt] = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}

    @property
    def current_attempt_index(self) -> int:
        """0-based index of the current (latest) attempt."""
        return max(0, len(self.attempts) - 1)

    @property
    def current_attempt_number(self) -> int:
        """1-based attempt number (backward compat with old `current_attempt` field)."""
        if self.attempts:
            return self.attempts[-1].attempt_number
        return 1

    def current_attempt(self) -> Optional[Attempt]:
        """Return the current (latest) attempt, or None if none exist."""
        return self.attempts[-1] if self.attempts else None

    def new_attempt(self) -> Attempt:
        """Create and append a new Attempt with the correct index and workspace."""
        next_num = len(self.attempts) + 1
        attempt_id = f"a{next_num:02d}"
        attempt_ws = self.step_workspace.for_attempt(attempt_id) if self.step_workspace else None
        attempt = Attempt(
            attempt_id=attempt_id,
            attempt_number=next_num,
            status="pending",
            workspace=attempt_ws,
        )
        self.attempts.append(attempt)
        return attempt

    def ensure_attempt(self) -> Attempt:
        """Return current attempt, or create the first one if none exist."""
        if not self.attempts:
            return self.new_attempt()
        return self.attempts[-1]

    # ── Dict-style compatibility ──
    # Executor/reflector currently use current_step.get("field", default).
    # These helpers provide backward compat during migration.

    def get(self, key: str, default: Any = None) -> Any:
        """Dict-style .get() for backward compatibility with consumer code."""
        # Map old field names to new model
        if key == "current_attempt":
            return self.current_attempt_number
        try:
            return getattr(self, key, default)
        except AttributeError:
            return default

    def __getitem__(self, key: str) -> Any:
        """Dict-style bracket access for backward compatibility."""
        if key == "current_attempt":
            return self.current_attempt_number
        val = getattr(self, key, None)
        if val is None and not hasattr(self, key):
            raise KeyError(key)
        return val

    def __setitem__(self, key: str, value: Any) -> None:
        """Dict-style bracket assignment for backward compatibility."""
        if key == "current_attempt":
            # Ignored — attempts are managed via new_attempt()
            return
        setattr(self, key, value)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)


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
