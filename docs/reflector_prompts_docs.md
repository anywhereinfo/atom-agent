# atom_agent.prompts.reflector_prompts Documentation

## Overall Flow
`src/atom_agent/prompts/reflector_prompts.py` contains the system and user prompts that drive the Reflector agent. These prompts are critical for ensuring the Reflector acts as a strict, objective Quality Assurance gatekeeper.

The flow is:
1.  **System Prompt**: Defines the persona (Senior QA/Reviewer) and enforces "Hard Rules" (JSON only, no execution, evidence-based).
2.  **User Prompt**: Formats the dynamic context (Step ID, Evidence, Test Results, Artifacts) into a structured inspection request.

## Key Prompt Components

### `REFLECTOR_SYSTEM_PROMPT`
-   **Persona**: "Senior QA/Reviewer agent (Reflector)".
-   **Hard Rules**:
    -   Output ONLY valid JSON.
    -   Evaluate *every* acceptance criterion.
    -   Use objective evidence (test results, artifacts).
-   **Scoring Logic**: Defines deterministic caps:
    -   Tests failed: 0.49 cap.
    -   Hardcoded implementation detected: 0.49 cap.
    -   Existence-only tests or missing dependencies: 0.59 cap.
-   **Routing Contract**: Explicitly defines when to `proceed` (>= 0.70), `refine`, or `rollback`.

### `REFLECTOR_USER_PROMPT`
-   **Dynamic Fields**:
    -   `{step_id}`, `{step_title}`, `{step_description}`: Step context.
    -   `{acceptance_criteria}`: The rubric to check against.
    -   `{programmatic_warnings}`: Machine-level red flags from `_run_programmatic_prechecks`.
    -   `{test_results_summary}`: The output from `run_pytest` or `execute_python_code`.
    -   `{artifacts_summary}`: List of created files.
    -   `{execution_summary}`: Narrative of what the executor did.

## Use Cases
-   **Quality Control**: These prompts ensure the LLM evaluates work based on *proof* (tests passing), not just *feeling*.
-   **Standardization**: Forces the output into a specific JSON schema (`confidence_score`, `decision`, `criteria_evaluation`) that the Python code can parse reliably.
