from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional, TYPE_CHECKING
from pathlib import Path

class Workspace(BaseModel):
    """
    Authoritative Workspace Contract.
    Encapsulates path resolution and configuration for a specific task.
    """
    task_id: str
    task_directory_rel: str
    task_directory_abs: str
    allowed_top_level_dirs: List[str]
    defaults: Dict[str, Any] = Field(default_factory=lambda: {"max_attempts": 3})
    naming: Dict[str, Any] = Field(default_factory=dict)
    step_structure: Dict[str, Any] = Field(default_factory=dict)
    authority: Dict[str, Any] = Field(default_factory=dict)
    paths: Dict[str, str] = Field(default_factory=dict)
    write_rules: Dict[str, List[str]] = Field(default_factory=dict)
    read_rules: Dict[str, List[str]] = Field(default_factory=dict)
    step_metadata: Dict[str, Any] = Field(default_factory=dict)

    def get_path(self, key: str, **kwargs) -> str:
        """
        Resolves a path template by key using provided keyword arguments.
        """
        path_template = self.paths.get(key)
        if not path_template:
            # Fallback to sensible defaults or raise if critical
            raise KeyError(f"Path key '{key}' not found in workspace configuration")
        return path_template.format(**kwargs)

    def get_staging_paths(self, step_id: str, attempt_id: str) -> Dict[str, str]:
        """
        Helper to get common staging paths for an implementation attempt.
        """
        return {
            "impl": self.get_path("attempt_impl", step_id=step_id, attempt_id=attempt_id),
            "test": self.get_path("attempt_test", step_id=step_id, attempt_id=attempt_id),
            "artifacts_dir": self.get_path("attempt_artifacts_dir", step_id=step_id, attempt_id=attempt_id),
            "messages_dir": self.get_path("attempt_messages_dir", step_id=step_id, attempt_id=attempt_id),
            "errors_dir": self.get_path("attempt_errors_dir", step_id=step_id, attempt_id=attempt_id),
        }

    def get_task_dir(self) -> Path:
        """Returns the relative task directory as a Path object."""
        return Path(self.task_directory_rel)

    def for_step(self, step_id: str) -> "StepWorkspace":
        """Factory: create a StepWorkspace bound to this workspace and step."""
        return StepWorkspace(step_id=step_id, _workspace=self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Workspace":
        """Factory to create a Workspace instance from a dictionary (e.g. from JSON)."""
        return cls(**data)


class StepWorkspace(BaseModel):
    """
    A thin workspace wrapper scoped to a specific step.
    Delegates all path resolution to the parent Workspace.
    """
    step_id: str
    _workspace: Workspace

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, step_id: str, _workspace: Workspace, **kwargs):
        super().__init__(step_id=step_id, **kwargs)
        # Store as private attr (not a Pydantic field, avoids serialization)
        object.__setattr__(self, '_workspace', _workspace)

    def get_step_dir(self) -> str:
        """Resolve the step-level directory path."""
        return self._workspace.get_path("step_dir", step_id=self.step_id)

    def get_messages_dir(self) -> str:
        """Resolve the step-level messages directory."""
        return self._workspace.get_path("step_messages_dir", step_id=self.step_id)

    def get_committed_dir(self) -> str:
        """Resolve the committed output directory for this step."""
        return self._workspace.get_path("committed_dir", step_id=self.step_id)

    def for_attempt(self, attempt_id: str) -> "AttemptWorkspace":
        """Factory: create an AttemptWorkspace bound to this step."""
        return AttemptWorkspace(
            step_id=self.step_id,
            attempt_id=attempt_id,
            _workspace=self._workspace,
        )


class AttemptWorkspace(BaseModel):
    """
    A thin workspace wrapper scoped to a specific step + attempt.
    Delegates all path resolution to the parent Workspace.
    """
    step_id: str
    attempt_id: str
    _workspace: Workspace

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, step_id: str, attempt_id: str, _workspace: Workspace, **kwargs):
        super().__init__(step_id=step_id, attempt_id=attempt_id, **kwargs)
        object.__setattr__(self, '_workspace', _workspace)

    def get_staging_paths(self) -> Dict[str, str]:
        """Get all staging paths for this attempt. Delegates to Workspace."""
        return self._workspace.get_staging_paths(self.step_id, self.attempt_id)

    def get_impl_path(self) -> str:
        return self._workspace.get_path("attempt_impl", step_id=self.step_id, attempt_id=self.attempt_id)

    def get_test_path(self) -> str:
        return self._workspace.get_path("attempt_test", step_id=self.step_id, attempt_id=self.attempt_id)

    def get_artifacts_dir(self) -> str:
        return self._workspace.get_path("attempt_artifacts_dir", step_id=self.step_id, attempt_id=self.attempt_id)

    def get_attempt_dir(self) -> str:
        return self._workspace.get_path("attempt_dir", step_id=self.step_id, attempt_id=self.attempt_id)

    def get_report_path(self, task_dir_rel: str) -> Path:
        """Resolve the report.json path for this attempt."""
        attempt_dir = self._workspace.get_path("attempt_dir", step_id=self.step_id, attempt_id=self.attempt_id)
        return Path(task_dir_rel) / attempt_dir / "report.json"
