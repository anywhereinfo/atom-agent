# atom_agent.nodes.report_generator Documentation

## Overall Flow
`src/atom_agent/nodes/report_generator.py` is responsible for generating the final comprehensive task report after all steps are completed or if the task fails. It synthesizes evidence from the entire task lifecycle.

The flow is:
1.  **Evidence Collection**: `_collect_task_data` gathers:
    -   Task metadata and plan details.
    -   Per-step evidence (artifacts, reflector reports, test results).
    -   Plan judge evaluations (scores and weaknesses) from `plan_candidates.json`.
2.  **Quality Metrics**: `_compute_quality_summary` aggregates performance data:
    -   Average reflector confidence scores.
    -   Criteria pass rate.
    -   Total retry count.
    -   Recommended Score Ceiling based on judge and reflector feedback.
3.  **LLM Synthesis**: Prompts the LLM with the gathered evidence and quality metrics.
4.  **Calibration**: Enforces strict Score Calibration Rules to prevent self-reported score inflation.

## Use Cases
-   **Project Handover**: Provides a detailed account of what was built and how it was verified.
-   **Quality Audit**: Highlights known limitations and areas where the agent struggled (retries, low confidence).
-   **Performance Tracking**: Captures aggregate metrics for system-level evaluation.

## Edge Cases
-   **Missing Data**: Gracefully handles missing plan files or step directories by providing "No data available" placeholders.
-   **Score Ceiling**: If the judge score or reflector confidence is low, the report generator forces a lower maximum self-reported score.

## Method Documentation

### `_collect_task_data(task_dir_rel: str) -> Dict[str, Any]`
Parses the file system to build a complete picture of the task execution.

### `_compute_quality_summary(task_data: Dict[str, Any]) -> str`
Computes the concrete quality baseline (avg score, pass rate) that prevents score inflation in the final report.

### `_format_step_evidence(steps_data: Dict[str, Dict]) -> str`
Formats per-step artifacts and reviews into a readable markdown summary for the LLM.

### `report_generator_node(state: AgentState) -> Dict[str, Any]`
The primary node function that orchestrates data collection and report generation.
