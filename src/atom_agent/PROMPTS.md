# Prompt Engineering & Contracts

The ATOM Agent relies on structured communication and strict contracts between nodes.

## 1. The Planner Contract
- **System Prompt**: `PLANNER_SYSTEM_PROMPT`
- **Output Format**: A list of `PlanStep` objects.
- **Critical Rules**:
  - Every step must have clear `acceptance_criteria`.
  - Steps must be sequenced logically (dependencies).
  - Complexity must be assigned (low/medium/high).

## 2. The Executor Contract
- **System Prompt**: `EXECUTOR_SYSTEM_PROMPT`
- **Staging Requirements**: 
  - Implementation MUST be in `impl.py`.
  - Tests MUST be in `test.py`.
  - Artifacts MUST be in the `artifacts/` subdirectory.
- **Identity Constraint**: Executor is forbidden from inventing step names; it must use the provided `step_id` for all paths.

## 3. The Reflector Contract (The Clinical Audit)
- **JSON packet**:
  ```json
  {
    "observations": "Narrative of what was seen",
    "issues_identified": ["List of clinical failures"],
    "confidence_score": 0.0,
    "decision": "proceed | refine | rollback",
    "criteria_evaluation": [
      {"criterion": "string", "status": "pass | fail"}
    ]
  }
  ```
- **Deterministic Scoring**:
  - Test Failure -> Max 0.49.
  - Test Pass + Criteria Met -> 0.8 to 1.0.

## 4. Path Templates (workspace.json)
The `workspace.json` (generated during setup) serves as the persistent contract for the entire task. It defines templates like:
- `attempt_impl`: `steps/{step_id}/attempts/{attempt_id}/impl.py`
- `committed_impl`: `steps/{step_id}/committed/impl.py`

Nodes and tools resolve these templates through the `Workspace` object to ensure absolute consistency across the graph.
