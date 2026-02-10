# atom_agent.prompts.report_prompts Documentation

## Overall Flow
`src/atom_agent/prompts/report_prompts.py` defines the instructions for the final report generation. It focuses on objective evidence synthesis and strict quality calibration based on aggregate metrics.

The flow is:
1.  **System Prompt**: Defines the persona (Lead Analyst) and enforces "Score Calibration Rules".
2.  **User Prompt**: Provides the dynamic quality context and evidence retrieved from the task lifecycle.

## Key Prompt Components

### `REPORT_SYSTEM_PROMPT`
-   **Persona**: "Expert Lead Analyst".
-   **Score Calibration Rules**:
    -   Must include a "Known Limitations" section referencing failures, weaknesses, and retries.
    -   Self-reported scores MUST NOT exceed the **Recommended Score Ceiling**.
    -   Every score must be justified with evidence citations.

### `REPORT_USER_PROMPT`
-   **Quality Summary**: `{quality_summary}` provides a machine-calculated baseline (average confidence, pass rates, judge score).
-   **Task Evidence**: Comprehensive logs of steps, artifacts, and reviews.

## Use Cases
-   **Honest Reporting**: Prevents the agent from claiming 100% success if tests failed or required many retries.
-   **evidence-Based Summarization**: Forces the report to be grounded in the actual `committed/` artifacts and `reflector_report.json` data.
