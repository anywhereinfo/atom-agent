# ATOM Agent Architecture

This document provides a high-level overview of the ATOM (Autonomous Task Orchestration & Management) Agent architecture, its core components, and the data flow that governs its operation.

## 1. Orchestration Layer (LangGraph)

The agent uses **LangGraph** to manage a stateful, cyclic workflow. The graph ensures that tasks move through distinct phases—planning, execution, and verification—with robust error handling and rollback capabilities.

### Graph Vertices (Nodes)
- **Setup Node**: Initializes the `TaskContext` and `Workspace` contract.
- **Planner Node**: Breaks down the user request into an `ImplementationPlan`.
- **Executor Node**: Performs tool-driven execution (coding, research, tests) for a specific step.
- **Reflector Node**: Audits the Executor's output against acceptance criteria.
- **Commit Node**: Promotes successful staging artifacts to an authoritative state.

### Routing Logic
The graph uses conditional edges to determine the next state:
- **Proceed**: Move to `Commit` -> (Next Step | End).
- **Refine**: Loop back to `Executor` for a targeted fix (incrementing `current_attempt`).
- **Rollback**: Return to `Planner` for a strategic reset if attempts are exhausted or the strategy is fundamentally flawed.

---

## 2. State Management (`AgentState`)

The `AgentState` is the single source of truth for the entire run. It captures:
- **Task Context**: metadata, directories, and normalized request.
- **Workspace Contract**: Authoritative rules for path resolution.
- **Implementation Plan**: The step-by-step roadmap and its current status.
- **Memory**: Conversation history and historical step reports.
- **Reflector Review**: The clinical evaluation from the last verification run.

---

## 3. The Workspace Contract

ATOM operates under a strict **Workspace Contract** to ensure security, isolation, and predictability.

### Staging vs. Committed
- **Staging Area**: `steps/<step_id>/attempts/aXX/`
  Every executor run is isolated. Artifacts are written to staging first.
- **Committed Area**: `steps/<step_id>/committed/`
  Upon successful reflection, the `commit_node` promotes implementation files and reports to the committed folder. This becomes the "immutable truth" for future steps.

### Path Hardening
Tools like `write_file` and `run_pytest` are jailed within the task directory. They resolve all paths relative to the task root and explicitly deny absolute paths or path traversal attacks (`..`).

---

## 4. Memory Hierarchy

ATOM uses a tiered memory system implemented in `MemoryManager`:
- **Tier 0 (Semantic)**: The active user instructions and system prompts.
- **Tier 1 (Local)**: Message history (`history.json`) for the current active step.
- **Tier 2 (Global)**: Cross-Encoder ranked summaries of previous successful steps cached in the `committed/` folder.

---

## 5. Security & Isolation

- **Bubblewrap Sandbox**: Python execution and tests run in a restricted environment.
- **Strict Scoping**: Tools cannot access files outside the `task_dir_rel`.
- **Fresh Starts**: Rollbacks wipe the `steps/` directory to prevent "Ghost Context" or poisoning from failed attempts.
