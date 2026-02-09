# Prompt Engineering & System Contracts

ATOM relies on strictly defined JSON contracts to ensure robust transitions between LLM-driven nodes.

## 1. Planner Contract (`planner_prompts.py`)
- **System Role**: Architect.
- **Goal**: Decompose a request into a non-linear but sequenced graph of steps.
- **Edge Cases Handled**:
  - **Ambiguous Requests**: instructed to create a "Research" step first if the request is ill-defined.
  - **Rollback Recovery**: When a rollback occurs, the planner receives the failure report and must adjust the strategy (not just the parameters).
- **Constraints**: 
  - `step_id` must match `^[a-z][a-z0-9_]{0,47}$`.
  - Max 3 attempts per step by default.

## 2. Executor Contract (`executor_prompts.py`)
- **System Role**: Tactical Operative.
- **Staging Contract**:
  - Implementation -> `impl.py`
  - Tests -> `test.py`
  - Logs/Data -> `artifacts/`
- **Main Use Case**: Direct execution of a single atomic task.
- **Edge Cases Handled**:
  - **Tool Failures**: Instructed to retry different parameters or alternative tools if one fails.
  - **Jailbreak Prevention**: Forbidden from using absolute paths or escaping the `steps/` boundary.
  - **Identity Drift**: Strictly forbidden from inventing new step names; it MUST use the provided `step_id`.

## 3. Reflector Contract (`reflector_prompts.py`)
- **System Role**: Clinical Auditor.
- **Scoring Model**:
  - **Critical Failure (Score < 0.5)**: Resulting in `refine` or `rollback`. Forced if tests fail.
  - **Conditional Pass (Score 0.5 - 0.7)**: Resulting in `refine` (targeted fix).
  - **Success (Score > 0.8)**: Resulting in `proceed` (commit).
- **Main Use Case**: Verifying that the Executor met all acceptance criteria.
- **Edge Cases Handled**:
  - **Hallucinated Success**: Instructed to prioritize raw logs and test results over the Executor's narrative summary.
  - **Flaky Tests**: if tests are flaky, the reflector identifies the flake and instructs the executor to harden the test suite.

## 4. Operational Flow & Edge Cases

### The "Staging-Commit" Lifecycle
1. **Executor** writes to `attempts/aXX/`.
2. **Reflector** audits `attempts/aXX/`.
3. **Commit** (on success) promotes `attempts/aXX/` -> `committed/`.

**Edge Case: Interrupted Task**
If the system crashes mid-execution, the `Workspace` contract ensures that the partial work is isolated in an `attempt` folder. Upon resumption, the agent sees the latest `attempt` and can either resume or start `a(XX+1)`.

**Edge Case: Dependency Corruption**
If a previous step's `committed/` output is found to be buggy by a later step, the later step triggers a `rollback`. This is the ONLY time the system moves backward in the graph, wiping the relevant downstream `steps/` to prevent contamination.
