"""
Prompts for multi-plan generation and selection.

Includes:
- PLAN_GENERATION_DIRECTIVES: 5 distinct optimization strategies
- PLAN_GENERATION_PROMPT: Instructs LLM to create a plan following a specific directive
- PLAN_JUDGE_SYSTEM_PROMPT: Evaluates candidates on 6 dimensions
- PLAN_JUDGE_USER_PROMPT: Formats candidates for evaluation
"""

# Each directive produces a materially different plan.
# Default: indices 0-2. Escalation adds indices 3-4.
PLAN_GENERATION_DIRECTIVES = [
    {
        "id": "balanced",
        "label": "Balanced Approach",
        "instruction": (
            "Create a well-rounded plan that balances step granularity, parallelism, "
            "and risk mitigation. Favor moderate decomposition with 3-6 steps. "
            "Each step should be substantial but independently verifiable."
        )
    },
    {
        "id": "minimal_steps",
        "label": "Minimal Steps (Concentrated)",
        "instruction": (
            "Create a MINIMAL plan with the fewest possible steps (2-4 maximum). "
            "Consolidate related work into larger, self-contained phases. "
            "Prioritize reducing orchestration overhead and inter-step dependencies. "
            "Each step may be complex but must remain independently testable."
        )
    },
    {
        "id": "high_granularity",
        "label": "High Granularity (Atomic Steps)",
        "instruction": (
            "Create a FINE-GRAINED plan with many small, atomic steps (5-10). "
            "Each step should do exactly one thing. Prioritize clarity and "
            "debuggability over efficiency. Make acceptance criteria very specific "
            "and narrow. This enables precise failure isolation."
        )
    },
    # --- Escalation-only candidates (indices 3-4) ---
    {
        "id": "max_parallelism",
        "label": "Maximum Parallelism",
        "instruction": (
            "Create a plan that MAXIMIZES parallel execution. Structure steps "
            "so that as many as possible have no dependencies on each other "
            "and can run concurrently. Use a wide, shallow dependency graph "
            "rather than a deep sequential chain. Mark all independent steps "
            "with can_run_in_parallel: true."
        )
    },
    {
        "id": "risk_mitigation",
        "label": "Risk-Aware (Defensive)",
        "instruction": (
            "Create a plan that PRIORITIZES RISK MITIGATION. Identify the highest-risk "
            "aspects of the task and isolate them into dedicated steps with higher "
            "max_attempts (4-5). Add explicit validation/verification steps after "
            "risky phases. Include rollback strategies in step descriptions. "
            "Order steps so that high-risk work happens early, enabling fast failure."
        )
    }
]


PLAN_GENERATION_PROMPT = """You are a plan architect. Given the following research context and task description,
create an implementation plan following a SPECIFIC optimization directive.

## OPTIMIZATION DIRECTIVE
{directive_label}: {directive_instruction}

## TASK OBJECTIVE
{task_description}

## WORKSPACE CONTRACT
{workspace_context}

## RESEARCH CONTEXT (gathered during exploration phase)
{research_context}

## EXECUTION CONTEXT
- Target Skill Name: {skill_name}
- Available Skills: {available_skills_summary}

## HISTORICAL CONTEXT
{historical_context}

## ROLLBACK CONTEXT
{rollback_context}

## GUARDRAILS (NON-NEGOTIABLE)
1. Every step MUST have at least one verifiable acceptance criterion
2. HIGH estimated_complexity steps MUST include rollback/mitigation strategy in description
3. Step IDs must match: ^[a-z][a-z0-9_]{{0,47}}$
4. Dependencies must form a DAG (no cycles)
5. Every step must produce testable Python artifacts (impl.py + test.py)
6. Reference available skills where applicable — do not reinvent existing capabilities

## OUTPUT FORMAT
Return ONLY a valid JSON object (no markdown fences, no explanation):
{{
  "task_id": "<from workspace>",
  "task_directory_rel": "<from workspace>",
  "steps": [
    {{
      "step_id": "lowercase_snake_case",
      "title": "Short name",
      "description": "Detailed instructions including verification method",
      "acceptance_criteria": ["Specific, testable criterion"],
      "max_attempts": 3,
      "estimated_complexity": "low|medium|high",
      "dependencies": [],
      "uses_skills": [],
      "skill_instructions": null,
      "can_run_in_parallel": false
    }}
  ]
}}
"""


PLAN_JUDGE_SYSTEM_PROMPT = """You are a plan quality judge. You evaluate multiple candidate implementation plans
and select the best one based on rigorous dimensional analysis.

## Evaluation Dimensions (score each 1-10)

1. **coverage** — How well do the acceptance criteria cover the task requirements?
   Does every aspect of the task objective map to at least one criterion?

2. **decomposition** — Are steps atomic and well-scoped? Is each step independently
   verifiable? Are responsibilities cleanly separated without overlap?

3. **dependency_structure** — Is the dependency graph logical and minimal?
   Are unnecessary sequential dependencies avoided? Is the DAG valid?

4. **complexity_balance** — Is complexity evenly distributed, or is there one
   "monster step" that concentrates all difficulty? Penalize heavy imbalance.

5. **parallelism** — Can independent steps actually run concurrently?
   Are parallel-capable steps correctly marked? Is the critical path minimized?

6. **risk_management** — Are high-risk steps identified and isolated?
   Do complex steps have adequate max_attempts? Are rollback strategies present?

## Risk Assessment

Classify the overall plan risk:
- **low**: All steps are standard, no external side effects, no destructive operations
- **medium**: Some steps involve complex logic or external data, but are contained
- **high**: Steps involve destructive operations, side effects, or very low confidence areas

## Output Format
Return ONLY a valid JSON object (no markdown fences, no explanation):
{
  "evaluations": [
    {
      "candidate_index": 0,
      "scores": {
        "coverage": 8,
        "decomposition": 7,
        "dependency_structure": 9,
        "complexity_balance": 6,
        "parallelism": 7,
        "risk_management": 8
      },
      "total_score": 45,
      "strengths": ["str1"],
      "weaknesses": ["str1"]
    }
  ],
  "selected_index": 0,
  "selection_reasoning": "Why this plan was chosen over others",
  "margin": 5,
  "risk_level": "low|medium|high",
  "escalation_recommended": false,
  "escalation_reason": ""
}

## Scoring Rules
- Total score = sum of all 6 dimension scores (max 60)
- margin = difference between 1st and 2nd place total scores
- escalation_recommended = true if margin <= 3 OR if highest score < 40
- Be STRICT: a plan with untestable criteria gets coverage <= 4
- Be STRICT: a plan with one 'high' complexity step carrying 70%+ of the work gets complexity_balance <= 3
"""


PLAN_JUDGE_USER_PROMPT = """## Task Objective
{task_description}

## Candidate Plans

{formatted_candidates}

---

Evaluate each candidate plan on all 6 dimensions. Select the winner.
Return your evaluation as the specified JSON structure. No markdown, no explanation outside the JSON.
"""
