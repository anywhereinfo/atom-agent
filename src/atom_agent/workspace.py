from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
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

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Workspace":
        """Factory to create a Workspace instance from a dictionary (e.g. from JSON)."""
        return cls(**data)
