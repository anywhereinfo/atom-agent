# Node Logic & Detailed Flow

This document details the internal logic, flow, and edge case handling for each LangGraph node.

## 1. Setup Node (`setup.py`)
- **Main Workflow**: Normalizes the user request, generates a unique task ID, and initializes the `workspace.json`.
- **Edge Cases**: 
  - **Collision Prevention**: Uses `hashlib` and timestamping for task IDs.
  - **Resumption**: Detects if a task folder already exists and can logically bind to it.

## 2. Planner Node (`planner.py`)
- **Main Workflow**: Generates the `ImplementationPlan` using a structured output LLM.
- **Rollback Logic**: 
  - If a rollback is detected, it receives a `ROLLBACK_CONTEXT` detailing the previous failure.
  - **Complete Reset**: It physically wipes the `steps/` directory to ensure no legacy files (Zombie Files) interfere with the new plan.
- **Validation**: Enforces strictly formatted `step_id` patterns (snake_case, max 48 chars).

## 3. Executor Node (`executor.py`)
- **Main Workflow**: Orchestrates a ReAct agent to complete a specific `PlanStep`.
- **Handover Logic**:
  - Writes code to `impl.py` and tests to `test.py`.
  - Captures test results in `test_results.json` and logs in `artifacts/`.
- **Isolation**: Strictly writes to the `staging/` directory of the current attempt.
- **Edge Case handling**:
  - Uses `SimpleMonitor` to track agent "pulse" (internal tracing).
  - Clean History: Filters internal Gemini metadata from `history.json` to keep memory tokens low.

## 4. Reflector Node (`reflector.py`)
- **Main Workflow**: Performs a "Clinical Audit" of the Executor's work.
- **Evaluation Logic**:
  - Gathers evidence (test results, implementation snippets, logs).
  - Audits criteria scores individually.
- **Deterministic Caps**:
  - If tests fail -> `confidence_score` is capped at 0.49.
  - If dependencies are missing -> Decision is forced to `refine` or `rollback`.
- **Robustness**: Implements multi-stage JSON extraction (regex fences -> brace finding) to handle conversational LLM outputs.

## 5. Commit Node (`commit.py`)
- **Main Workflow**: The "Authority Promotion" stage.
- **Promotion Logic**:
  - Physically copies files from `staging/` to `committed/`.
  - Creates a `manifest.json` as a permanent record of the successful attempt.
- **Edge Case handling**:
  - **Atomic Commit**: If any copy operation fails, it reports an error phase instead of signaling success, preventing "Half-Committed" states.
