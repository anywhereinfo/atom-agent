import re
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any

from ..state import AgentState, TaskContext
from ..workspace import Workspace

def slugify(text: str) -> str:
    """
    Lowercase, replace non-alnum with hyphen, collapse hyphens, trim to 48 chars.
    """
    s = (text or "").lower()
    s = re.sub(r"[^a-z0-9]", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s[:48].strip("-")


def task_setup_node(state: AgentState) -> dict:
    """
    Initializes the task context and creates the directory scaffold.

    Architecture:
    - Task -> Steps -> Attempts (staging) -> Commit (authoritative)
    - Executor writes only to attempts/<attempt_id>/
    - Reflector writes step-scoped reflector.json
    - Commit node promotes accepted attempt into committed/
    - Downstream reads ONLY committed/
    """
    print("DEBUG: task_setup_node entered", flush=True)

    raw_request = state.get("task_description", "")
    if not raw_request or not raw_request.strip():
        raise ValueError("Task description is empty or whitespace. Cannot initialize task.")

    normalized_request = " ".join(raw_request.split())
    hash8 = hashlib.sha1(normalized_request.encode("utf-8")).hexdigest()[:8]

    # Slug derived from task description (not skill_name). Use a short blurb for readability.
    blurb = " ".join(normalized_request.split()[:8])
    slug = slugify(blurb) or "task"

    # Timestamp with millisecond precision to reduce collisions
    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y%m%d_%H%M%S_%f")[:-3]  # YYYYMMDD_HHMMSS_mmm

    task_id = f"{slug}__{timestamp}__{hash8}"

    base_dir = Path("tasks")
    task_dir_rel = base_dir / task_id
    task_dir_abs = task_dir_rel.resolve()

    # Minimal scaffold aligned to step-local attempt/commit model
    top_level_dirs = ["inputs", "steps", "state", "logs"]
    for sd in top_level_dirs:
        (task_dir_rel / sd).mkdir(parents=True, exist_ok=True)

    # Workspace contract: machine-enforceable naming & path rules
    workspace_contract = Workspace(
        task_id=task_id,
        task_directory_rel=str(task_dir_rel),
        task_directory_abs=str(task_dir_abs),
        allowed_top_level_dirs=top_level_dirs,

        defaults={"max_attempts": 3},

        naming={
            "step_id": {
                "format": "lower_snake_case",
                "regex": r"^[a-z][a-z0-9_]{0,47}$",
                "max_len": 48
            },
            "attempt_id": {
                "format": "a{n:02d}",
                "regex": r"^a\d{2,3}$",   # allow 100+ if ever needed
                "start": 1
            }
        },

        step_structure={
            "attempt_required_subdirs": ["artifacts", "messages", "errors"],
            "committed_required_subdirs": ["artifacts"]
        },

        authority={
            "authoritative_root": "steps/{step_id}/committed/",
            "non_authoritative_roots": ["steps/{step_id}/attempts/"]
        },

        paths={
            "plan_path": "state/plan.json",
            "task_meta_path": "state/task.json",
            "workspace_path": "state/workspace.json",

            "step_dir": "steps/{step_id}/",
            "step_state_path": "steps/{step_id}/step_state.json",
            "reflector_path": "steps/{step_id}/reflector.json",

            "attempt_dir": "steps/{step_id}/attempts/{attempt_id}/",
            "attempt_impl": "steps/{step_id}/attempts/{attempt_id}/impl.py",
            "attempt_test": "steps/{step_id}/attempts/{attempt_id}/test.py",
            "attempt_artifacts_dir": "steps/{step_id}/attempts/{attempt_id}/artifacts/",
            "attempt_messages_dir": "steps/{step_id}/attempts/{attempt_id}/messages/",
            "attempt_errors_dir": "steps/{step_id}/attempts/{attempt_id}/errors/",

            "committed_dir": "steps/{step_id}/committed/",
            "committed_impl": "steps/{step_id}/committed/impl.py",
            "committed_test": "steps/{step_id}/committed/test.py",
            "committed_artifacts_dir": "steps/{step_id}/committed/artifacts/",
            "commit_manifest": "steps/{step_id}/committed/manifest.json"
        },

        step_metadata={
            # Planner must include these fields per step in plan.json
            "required_fields": ["step_id", "title", "description", "acceptance_criteria", "max_attempts"]
        },

        # These are path-based allow/deny rules your IDE can enforce
        write_rules={
            "executor_allow": ["steps/{step_id}/attempts/{attempt_id}/**"],
            "executor_deny": [
                "steps/{step_id}/committed/**",
                "steps/{step_id}/reflector.json",
                "steps/{step_id}/step_state.json",
                "state/**"
            ],

            "reflector_allow": ["steps/{step_id}/reflector.json"],
            "reflector_deny": [
                "steps/{step_id}/attempts/**",
                "steps/{step_id}/committed/**",
                "state/**"
            ],

            "commit_allow": [
                "steps/{step_id}/committed/**",
                "steps/{step_id}/step_state.json"
            ],
            "commit_deny": [
                "steps/{step_id}/attempts/**"
            ]
        },

        read_rules={
            "downstream_allow": ["steps/{dependency_step_id}/committed/**"],
            "downstream_deny": ["steps/{dependency_step_id}/attempts/**"]
        }
    )

    # Task context: include task_description explicitly + relative path authority
    context: TaskContext = {
        "task_id": task_id,
        "task_description": raw_request,          # explicit
        "request_raw": raw_request,
        "request_normalized": normalized_request,
        "request_hash8": hash8,
        "slug": slug,
        "created_at": timestamp,
        "timezone": "UTC",
        "task_dir_rel": str(task_dir_rel),
        "task_dir_abs": str(task_dir_abs),
        "slug_metadata": {
            "source": "task_description_blurb",
            "blurb": blurb,                       # Tighten for traceability
            "skill_name_provided": state.get("skill_name")
        }
        # NOTE: run_id intentionally removed (not used in this architecture)
    }

    # Write initial files
    (task_dir_rel / "inputs" / "request.txt").write_text(raw_request, encoding="utf-8")
    with (task_dir_rel / "state" / "task.json").open("w", encoding="utf-8") as f:
        json.dump(context, f, indent=2)

    with (task_dir_rel / "state" / "workspace.json").open("w", encoding="utf-8") as f:
        json.dump(workspace_contract.model_dump(), f, indent=2)

    # Initialize state/plan.json placeholder
    plan_stub = {
        "task_id": task_id,
        "task_directory_rel": str(task_dir_rel),
        "steps": []
    }
    with (task_dir_rel / "state" / "plan.json").open("w", encoding="utf-8") as f:
        json.dump(plan_stub, f, indent=2)

    print(f"DEBUG: Task setup complete. ID: {task_id}", flush=True)
    print(f"DEBUG: Task directory: {task_dir_rel}", flush=True)

    return {
        "task_context": context,
        "workspace": workspace_contract,
        "task_id": task_id
    }
