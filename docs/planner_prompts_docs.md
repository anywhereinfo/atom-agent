# atom_agent.prompts.planner_prompts Documentation

## Overall Flow
`src/atom_agent/prompts/planner_prompts.py` governs the "Architect" agent. This prompt is responsible for the highest level of cognitive work: researching the problem, complying with academic standards, and structuring the entire execution graph.

The flow is:
1.  **System Prompt**: Defines the "Agentic Planner" workflow (Research -> Analysis -> Plan -> Submit).
2.  **User Prompt**: Provides the raw user request, available skills, and historical context.

## Key Prompt Components

### `PLANNER_SYSTEM_PROMPT`
-   **Role**: "Architect agent responsible for creating a Python-executable implementation plan."
-   **Workflow**:
    -   **Phase 0 (Research)**: Mandates using `tavily_search` and `arxiv_search` before planning.
    -   **Phase 3 (Submission)**: Requires calling `submit_plan` with a valid JSON.
-   **Systems-Level Depth**: for evaluation tasks, it *forces* the agent to include analysis of:
    -   Topology, Control Models, Complexity (Big O), Failure Modes, Determinism.
-   **Output Format**: Provides the exact JSON schema for `submit_plan`.

### `PLANNER_USER_PROMPT`
-   **Inputs**:
    -   `{workspace_context}`: The file system rules.
    -   `{task_description}`: The user's goal.
    -   `{historical_context}`: Summary of previous steps (if any).
    -   `{rollback_context}`: Critical for self-healing—if the planner is called after a rollback, this field explains *why* the previous plan failed.

## Use Cases
-   **Reflective Replanning**: The `rollback_context` field allows the planner to say "Okay, approach A failed because of X, so I will plan approach B."
-   **Academic Rigor**: The "Systems-Level Depth" section ensures the agent doesn't just write "hello world" code for complex requests.
