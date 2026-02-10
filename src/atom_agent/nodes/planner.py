from typing import List, Dict, Any, Optional, Tuple
import json
import re
import time
import concurrent.futures
from pathlib import Path
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.callbacks import BaseCallbackHandler
from langgraph.prebuilt import create_react_agent

from ..prompts.planner_prompts import PLANNER_SYSTEM_PROMPT, PLANNER_USER_PROMPT
from ..prompts.plan_judge_prompts import (
    PLAN_GENERATION_DIRECTIVES,
    PLAN_GENERATION_PROMPT,
    PLAN_JUDGE_SYSTEM_PROMPT,
    PLAN_JUDGE_USER_PROMPT,
)
from ..state import AgentState, PlanStep
from ..workspace import Workspace
from ..config import get_llm, on_rate_limit_hit, on_rate_limit_clear
from ..memory import MemoryManager
from ..tools.planning_tools import create_planning_tools, ImplementationPlanInput


# ────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────

DEFAULT_CANDIDATE_COUNT = 3
ESCALATED_CANDIDATE_COUNT = 5
ESCALATION_MARGIN_THRESHOLD = 3      # Escalate if top-2 margin <= this
ESCALATION_MIN_SCORE_THRESHOLD = 40  # Escalate if best score < this (out of 60)
APPROVAL_CONFIDENCE_THRESHOLD = 35   # Auto-proceed if total_score >= this

SYSTEMS_EVAL_DATA_CRITERIA = [
    "Structured evaluation data captures systems-level fields for each evaluated approach: planning topology, control model, computational complexity, failure modes, determinism class, observability/governance, enterprise suitability, and composition patterns.",
    "Each evaluated approach includes at least one concrete failure scenario and one production use case."
]

SYSTEMS_EVAL_REPORT_CRITERIA = [
    "The report includes an architectural comparison matrix covering planning topology, control model, computational complexity, failure modes, determinism spectrum, observability/governance, enterprise readiness, and composition patterns.",
    "Each evaluated approach includes explicit complexity and cost analysis (token growth, branching behavior, latency amplification, tool overhead, or memory scaling) with formal notation where applicable.",
    "The report includes enterprise deployment guidance: cost predictability, reliability constraints, security/isolation considerations, and explainability/auditability implications."
]

SYSTEMS_EVAL_DATA_HINTS = ("populate", "schema", "data", "dataset")
SYSTEMS_EVAL_REPORT_HINTS = ("report", "analysis", "evaluate", "evaluation", "comparison", "summary")


# ────────────────────────────────────────────────────────────
# Debug Callback Handler (unchanged)
# ────────────────────────────────────────────────────────────

class DebugCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self._time = time
        self._llm_start_time = None
        self._llm_call_count = 0

    def on_llm_start(self, serialized: Dict, prompts: List[str], **kwargs):
        self._llm_call_count += 1
        self._llm_start_time = self._time.time()
        print(f"\nDEBUG: [LLM Call #{self._llm_call_count}] Sending request to Gemini...", flush=True)

    def on_llm_new_token(self, token: str, **kwargs):
        print(token, end="", flush=True)

    def on_llm_end(self, response, **kwargs):
        duration = self._time.time() - self._llm_start_time if self._llm_start_time else 0
        print(f"\nDEBUG: [LLM Call #{self._llm_call_count}] Gemini responded in {duration:.1f}s", flush=True)
        on_rate_limit_clear()

    def on_llm_error(self, error, **kwargs):
        duration = self._time.time() - self._llm_start_time if self._llm_start_time else 0
        error_str = str(error)
        if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
            on_rate_limit_hit()
            print(f"\n⚠️  DEBUG: [LLM Call #{self._llm_call_count}] RATE LIMITED after {duration:.1f}s! Retrying with backoff...", flush=True)
        else:
            print(f"\n❌ DEBUG: [LLM Call #{self._llm_call_count}] LLM ERROR after {duration:.1f}s: {error_str[:200]}", flush=True)

    def on_retry(self, retry_state, **kwargs):
        attempt = getattr(retry_state, 'attempt_number', '?')
        print(f"\n🔄 DEBUG: Retry attempt #{attempt} — waiting before next try...", flush=True)

    def on_tool_start(self, serialized: Dict, input_str: str, **kwargs):
        tool_name = serialized.get("name") if serialized else "Unknown Tool"
        print(f"\nDEBUG: [Tool] Executing: {tool_name}", flush=True)
        print(f"DEBUG: [Tool] Input: {input_str}", flush=True)

    def on_tool_end(self, output: str, **kwargs):
        print(f"DEBUG: [Tool] Output: {str(output)}", flush=True)

    def on_tool_error(self, error, **kwargs):
        print(f"❌ DEBUG: [Tool] ERROR: {str(error)[:300]}", flush=True)


# ────────────────────────────────────────────────────────────
# Helper Functions
# ────────────────────────────────────────────────────────────

def format_available_skills(skills: list) -> str:
    if not skills:
        return "None"
    return "\n".join([f"- {s['name']}: {s['description']}" for s in skills])


def _append_unique_criteria(criteria: List[str], additions: List[str]) -> List[str]:
    normalized = {c.strip().lower() for c in criteria}
    for item in additions:
        key = item.strip().lower()
        if key not in normalized:
            criteria.append(item)
            normalized.add(key)
    return criteria


def _is_systems_evaluation_task(task_description: str) -> bool:
    text = (task_description or "").lower()
    has_eval_intent = bool(re.search(r"\b(evaluate|evaluation|compare|comparison|benchmark|assess|trade[- ]?off)\b", text))
    has_strategy_subject = bool(re.search(r"\b(strategy|strategies|architecture|architectures|framework|frameworks|planning|agentic|autonomous)\b", text))
    return has_eval_intent and has_strategy_subject


def _enforce_systems_depth_requirements(task_description: str, plan_steps: List[PlanStep]) -> List[PlanStep]:
    if not _is_systems_evaluation_task(task_description):
        return plan_steps
    for step in plan_steps:
        step_text = " ".join([
            step.get("step_id", ""),
            step.get("title", ""),
            step.get("description", "")
        ]).lower()
        if any(hint in step_text for hint in SYSTEMS_EVAL_DATA_HINTS):
            step["acceptance_criteria"] = _append_unique_criteria(
                step.get("acceptance_criteria", []),
                SYSTEMS_EVAL_DATA_CRITERIA
            )
        if any(hint in step_text for hint in SYSTEMS_EVAL_REPORT_HINTS):
            step["acceptance_criteria"] = _append_unique_criteria(
                step.get("acceptance_criteria", []),
                SYSTEMS_EVAL_REPORT_CRITERIA
            )
    return plan_steps


def _extract_plan_from_messages(messages: List) -> Dict:
    """Extract the submitted plan from the agent's tool call messages."""
    for msg in reversed(messages):
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            for tool_call in msg.tool_calls:
                if tool_call.get('name') == 'submit_plan':
                    tool_call_id = tool_call.get('id')
                    for response_msg in messages:
                        if (hasattr(response_msg, 'tool_call_id') and
                            response_msg.tool_call_id == tool_call_id):
                            try:
                                response_data = json.loads(response_msg.content)
                                if response_data.get("status") == "success":
                                    validated_plan = response_data.get("validated_plan")
                                    if validated_plan:
                                        return validated_plan
                            except Exception as e:
                                print(f"DEBUG: Failed to parse submit_plan response: {str(e)}", flush=True)
                                continue
    raise ValueError("No valid submit_plan tool call found in agent messages")


def _extract_research_context(messages: List) -> str:
    """
    Extract research context from the ReAct agent's tool interactions.
    Captures tool calls and their outputs to provide shared context
    for all candidate plan generators.
    """
    context_parts = []
    for msg in messages:
        # Capture tool call results (research findings)
        if hasattr(msg, 'tool_call_id') and msg.tool_call_id:
            tool_name = getattr(msg, 'name', 'unknown_tool')
            content = str(msg.content)
            # Skip submit_plan responses — that's the plan itself
            if tool_name == 'submit_plan':
                continue
            # Truncate very long tool outputs
            if len(content) > 3000:
                content = content[:3000] + "\n... [truncated]"
            context_parts.append(f"### Tool: {tool_name}\n{content}\n")

        # Capture the agent's reasoning/synthesis messages
        elif hasattr(msg, 'content') and not hasattr(msg, 'tool_calls'):
            content = str(msg.content)
            if content and len(content) > 50:  # Skip short acknowledgements
                if len(content) > 2000:
                    content = content[:2000] + "\n... [truncated]"
                context_parts.append(f"### Agent Analysis\n{content}\n")

    if not context_parts:
        return "No research data collected."

    return "\n".join(context_parts)


# ────────────────────────────────────────────────────────────
# Phase 2: Plan Candidate Generation
# ────────────────────────────────────────────────────────────

def _normalize_llm_content(content) -> str:
    """Normalize LLM response content to a plain string.
    
    Gemini can return content as a list of content blocks
    (e.g., [{"type": "text", "text": "..."}]) instead of a string.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(part.get("text", str(part)))
            else:
                parts.append(str(part))
        return "\n".join(parts)
    return str(content)


def _extract_json_from_response(raw: str) -> str:
    """Robustly extract JSON from an LLM response that may include markdown fences."""
    stripped = raw.strip()

    # Try 1: direct parse
    if stripped.startswith("{"):
        return stripped

    # Try 2: extract from markdown code fences (```json ... ```)
    fence_match = re.search(r"```(?:json)?\s*\n?(\{.*?\})\s*```", stripped, re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()

    # Try 3: find first { to last } (greedy)
    brace_match = re.search(r"(\{.*\})", stripped, re.DOTALL)
    if brace_match:
        return brace_match.group(1).strip()

    return stripped  # last resort — let json.loads raise the error


def _generate_single_candidate(
    directive: Dict,
    task_description: str,
    workspace_context: str,
    research_context: str,
    skill_name: str,
    available_skills_summary: str,
    historical_context: str,
    rollback_context: str,
    task_id: str,
    task_directory_rel: str,
) -> Optional[Dict]:
    """Generate a single plan candidate following a specific directive."""
    try:
        prompt = PLAN_GENERATION_PROMPT.format(
            directive_label=directive["label"],
            directive_instruction=directive["instruction"],
            task_description=task_description,
            workspace_context=workspace_context,
            research_context=research_context,
            skill_name=skill_name,
            available_skills_summary=available_skills_summary,
            historical_context=historical_context,
            rollback_context=rollback_context,
        )

        llm = get_llm("plan_generator")
        
        # Retry wrapper for transient network errors
        MAX_RETRIES = 3
        RETRY_DELAYS = [10, 20, 40]  # Standard backoff
        response = None

        for attempt in range(MAX_RETRIES + 1):
            try:
                response = llm.invoke([HumanMessage(content=prompt)])
                break
            except Exception as e:
                error_name = type(e).__name__
                is_transient = any(keyword in error_name for keyword in [
                    "Timeout", "ReadTimeout", "ConnectTimeout", "ConnectionError",
                    "RemoteDisconnected", "ConnectionReset"
                ]) or "timed out" in str(e).lower()

                if is_transient and attempt < MAX_RETRIES:
                    delay = RETRY_DELAYS[attempt]
                    print(f"  ⚠️  Transient error ({error_name}) in candidate gen, retrying in {delay}s...", flush=True)
                    time.sleep(delay)
                    continue
                else:
                    raise e
        raw = _normalize_llm_content(response.content).strip()

        # Robustly extract JSON from the response
        json_str = _extract_json_from_response(raw)
        plan_data = json.loads(json_str)

        # Inject server-side values — never trust the LLM to extract these
        plan_data["task_id"] = task_id
        plan_data["task_directory_rel"] = task_directory_rel

        # Sanitize steps before Pydantic validation
        for step in plan_data.get("steps", []):
            # skill_instructions must be Dict[str, str] | None — LLM often outputs a string
            si = step.get("skill_instructions")
            if si is not None and not isinstance(si, dict):
                step["skill_instructions"] = None
            # estimated_complexity must be low|medium|high
            ec = step.get("estimated_complexity", "medium")
            if ec not in ("low", "medium", "high"):
                step["estimated_complexity"] = "medium"

        # Validate with Pydantic
        validated = ImplementationPlanInput(**plan_data)
        return {
            "directive": directive["id"],
            "directive_label": directive["label"],
            "steps_count": len(validated.steps),
            "plan": validated.model_dump()
        }

    except json.JSONDecodeError as e:
        preview = raw[:300] if raw else "(empty)"
        print(f"  ⚠️  Candidate '{directive['id']}': Invalid JSON — {str(e)[:100]}", flush=True)
        print(f"  ⚠️  Raw preview: {preview}", flush=True)
        return None
    except Exception as e:
        print(f"  ⚠️  Candidate '{directive['id']}': Failed — {str(e)[:300]}", flush=True)
        return None


def _generate_plan_candidates(
    directives: List[Dict],
    task_description: str,
    workspace_context: str,
    research_context: str,
    skill_name: str,
    available_skills_summary: str,
    historical_context: str,
    rollback_context: str,
    task_id: str,
    task_directory_rel: str,
) -> List[Dict]:
    """
    Generate plan candidates, one per directive.
    Accepts the specific directives to use (allows escalation to add new ones only).
    """
    candidates = []
    count = len(directives)

    print(f"\n{'─' * 60}", flush=True)
    print(f"PLAN GENERATOR: Generating {count} candidate plans (PARALLEL)...", flush=True)
    print(f"{'─' * 60}", flush=True)

    # Use ThreadPoolExecutor for parallel generation
    # We use a max_workers cap (e.g., 5) to avoid hitting rate limits too aggressively,
    # though our rate limit handler should manage that at the config level.
    max_workers = min(count, 5)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_directive = {}
        for directive in directives:
            print(f"  [Scheduled] {directive['label']}...", flush=True)
            future = executor.submit(
                _generate_single_candidate,
                directive=directive,
                task_description=task_description,
                workspace_context=workspace_context,
                research_context=research_context,
                skill_name=skill_name,
                available_skills_summary=available_skills_summary,
                historical_context=historical_context,
                rollback_context=rollback_context,
                task_id=task_id,
                task_directory_rel=task_directory_rel,
            )
            future_to_directive[future] = directive

        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_directive):
            directive = future_to_directive[future]
            try:
                candidate = future.result()
                if candidate:
                    print(f"  ✅ '{directive['id']}': {candidate['steps_count']} steps (Complete)", flush=True)
                    candidates.append(candidate)
                else:
                    print(f"  ❌ '{directive['id']}': Failed (returned None)", flush=True)
            except Exception as e:
                print(f"  ❌ '{directive['id']}': Exception: {str(e)}", flush=True)

    # Sort candidates to ensure deterministic order (by directive ID) for the Judge
    candidates.sort(key=lambda c: c['directive'])

    print(f"\nPLAN GENERATOR: {len(candidates)}/{count} valid candidates produced.", flush=True)
    return candidates


# ────────────────────────────────────────────────────────────
# Phase 3: Plan Judging
# ────────────────────────────────────────────────────────────

def _format_candidates_for_judge(candidates: List[Dict]) -> str:
    """Format all candidates as a numbered list for the judge."""
    parts = []
    for i, c in enumerate(candidates):
        plan = c["plan"]
        steps_summary = []
        for step in plan["steps"]:
            deps = step.get("dependencies", [])
            criteria_count = len(step.get("acceptance_criteria", []))
            steps_summary.append(
                f"    - `{step['step_id']}`: {step['title']} "
                f"[{step.get('estimated_complexity', '?')}] "
                f"(deps: {deps or 'none'}, criteria: {criteria_count}, "
                f"parallel: {step.get('can_run_in_parallel', False)}, "
                f"max_attempts: {step.get('max_attempts', 3)})"
            )

        parts.append(
            f"### Candidate {i} — {c['directive_label']}\n"
            f"**Directive**: {c['directive']}\n"
            f"**Steps**: {c['steps_count']}\n"
            f"**Step Details**:\n" + "\n".join(steps_summary) + "\n"
            f"\n**Full Plan JSON**:\n```json\n{json.dumps(plan, indent=2)}\n```\n"
        )

    return "\n---\n".join(parts)


def _judge_plans(
    candidates: List[Dict],
    task_description: str,
) -> Dict:
    """
    Evaluate all candidates and select the best one.
    Returns the full judge evaluation including scores and reasoning.
    """
    if len(candidates) == 1:
        # Only one valid candidate — auto-select
        return {
            "evaluations": [{
                "candidate_index": 0,
                "scores": {"coverage": 7, "decomposition": 7, "dependency_structure": 7,
                           "complexity_balance": 7, "parallelism": 7, "risk_management": 7},
                "total_score": 42,
                "strengths": ["Only valid candidate"],
                "weaknesses": []
            }],
            "selected_index": 0,
            "selection_reasoning": "Single valid candidate — auto-selected.",
            "margin": 0,
            "risk_level": "medium",
            "escalation_recommended": True,
            "escalation_reason": "Only one valid candidate was generated."
        }

    print(f"\n{'─' * 60}", flush=True)
    print(f"PLAN JUDGE: Evaluating {len(candidates)} candidates...", flush=True)
    print(f"{'─' * 60}", flush=True)

    formatted = _format_candidates_for_judge(candidates)
    user_prompt = PLAN_JUDGE_USER_PROMPT.format(
        task_description=task_description,
        formatted_candidates=formatted,
    )

    llm = get_llm("plan_judge")
    start = time.time()

    # Retry wrapper for transient network errors
    MAX_RETRIES = 3
    RETRY_DELAYS = [5, 10, 20]  # shorter backoff for judge
    response = None

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = llm.invoke([
                SystemMessage(content=PLAN_JUDGE_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ])
            break  # success
        except Exception as e:
            error_name = type(e).__name__
            is_transient = any(keyword in error_name for keyword in [
                "Timeout", "ReadTimeout", "ConnectTimeout", "ConnectionError",
                "RemoteDisconnected", "ConnectionReset"
            ]) or "timed out" in str(e).lower()

            if is_transient and attempt < MAX_RETRIES:
                delay = RETRY_DELAYS[attempt]
                print(f"⚠️  Transient error ({error_name}) in judge, retrying in {delay}s... (attempt {attempt + 1}/{MAX_RETRIES})", flush=True)
                time.sleep(delay)
                continue
            else:
                print(f"DEBUG: FATAL ERROR in plan judge: {str(e)}", flush=True)
                raise e

    raw = _normalize_llm_content(response.content).strip()
    elapsed = time.time() - start

    # Robustly extract JSON from the response
    json_str = _extract_json_from_response(raw)

    try:
        evaluation = json.loads(json_str)
    except json.JSONDecodeError:
        print(f"  ⚠️  Judge returned invalid JSON. Falling back to candidate 0.", flush=True)
        return {
            "evaluations": [],
            "selected_index": 0,
            "selection_reasoning": "Judge returned unparseable output — defaulted to first candidate.",
            "margin": 0,
            "risk_level": "medium",
            "escalation_recommended": True,
            "escalation_reason": "Judge evaluation failed."
        }

    selected = evaluation.get("selected_index", 0)
    margin = evaluation.get("margin", 0)
    risk = evaluation.get("risk_level", "medium")

    print(f"  JUDGE: Selected candidate {selected} ({candidates[selected]['directive_label']})", flush=True)
    print(f"  JUDGE: Margin: {margin}, Risk: {risk} ({elapsed:.1f}s)", flush=True)
    print(f"  JUDGE: Reasoning: {evaluation.get('selection_reasoning', 'N/A')[:200]}", flush=True)

    return evaluation


# ────────────────────────────────────────────────────────────
# Escalation & Approval Logic
# ────────────────────────────────────────────────────────────

def _should_escalate(evaluation: Dict, current_count: int) -> bool:
    """
    Determine if we should generate more candidates.
    Only escalates from 3 → 5.
    """
    if current_count >= ESCALATED_CANDIDATE_COUNT:
        return False  # Already at max

    # Judge explicitly recommended escalation
    if evaluation.get("escalation_recommended", False):
        return True

    # Margin too thin — candidates are too similar
    margin = evaluation.get("margin", 0)
    if margin <= ESCALATION_MARGIN_THRESHOLD:
        return True

    # Best score too low — all candidates weak
    evals = evaluation.get("evaluations", [])
    if evals:
        best_score = max(e.get("total_score", 0) for e in evals)
        if best_score < ESCALATION_MIN_SCORE_THRESHOLD:
            return True

    return False


def _needs_approval(evaluation: Dict, plan_steps: List) -> Tuple[bool, str]:
    """
    Determine if the plan needs user approval before execution.

    Returns (needs_approval: bool, reason: str)

    Approval required for:
    - High risk level
    - Low winning score
    - Any step with HIGH complexity and potential side effects
    """
    risk_level = evaluation.get("risk_level", "medium")

    # High risk always needs approval
    if risk_level == "high":
        return True, "Plan classified as HIGH risk by judge."

    # Low winning score
    evals = evaluation.get("evaluations", [])
    selected_idx = evaluation.get("selected_index", 0)
    if evals and selected_idx < len(evals):
        winner_score = evals[selected_idx].get("total_score", 0)
        if winner_score < APPROVAL_CONFIDENCE_THRESHOLD:
            return True, f"Winner score ({winner_score}/60) below confidence threshold ({APPROVAL_CONFIDENCE_THRESHOLD})."

    # Any HIGH complexity step (potential destructive/side-effect)
    high_risk_steps = [s for s in plan_steps if s.get("estimated_complexity") == "high"]
    if len(high_risk_steps) > len(plan_steps) // 2:
        return True, f"{len(high_risk_steps)}/{len(plan_steps)} steps are HIGH complexity."

    # Auto-proceed for low-risk plans
    return False, ""


def _format_approval_summary(
    evaluation: Dict,
    candidates: List[Dict],
    winner_steps: List,
) -> str:
    """Format a concise summary for user approval."""
    selected = evaluation.get("selected_index", 0)
    winner = candidates[selected]

    lines = [
        f"\n{'=' * 60}",
        f"⚠️  PLAN APPROVAL REQUIRED",
        f"{'=' * 60}",
        f"",
        f"🏆 Winner: {winner['directive_label']} ({winner['steps_count']} steps)",
        f"   Reasoning: {evaluation.get('selection_reasoning', 'N/A')[:200]}",
        f"   Risk Level: {evaluation.get('risk_level', 'unknown').upper()}",
        f"",
        f"📋 Steps:",
    ]

    for step in winner_steps:
        lines.append(f"   {step['step_id']}: {step['title']} [{step.get('estimated_complexity', '?')}]")

    # Top risks
    evals = evaluation.get("evaluations", [])
    if evals and selected < len(evals):
        weaknesses = evals[selected].get("weaknesses", [])
        if weaknesses:
            lines.append(f"\n⚠️  Top Risks:")
            for w in weaknesses[:3]:
                lines.append(f"   - {w}")

    # Alternatives
    lines.append(f"\n📊 Alternatives:")
    for i, c in enumerate(candidates):
        if i == selected:
            continue
        score = "?"
        if evals and i < len(evals):
            score = evals[i].get("total_score", "?")
        lines.append(f"   [{i}] {c['directive_label']} ({c['steps_count']} steps, score: {score})")

    lines.extend([
        f"",
        f"{'=' * 60}",
    ])

    return "\n".join(lines)


# ────────────────────────────────────────────────────────────
# Persistence
# ────────────────────────────────────────────────────────────

def _persist_candidates(
    task_dir: Path,
    candidates: List[Dict],
    evaluation: Dict,
    escalation_history: List[str],
) -> None:
    """Save all candidates and judge evaluation for auditability."""
    artifact = {
        "candidates": candidates,
        "evaluation": evaluation,
        "escalation_history": escalation_history,
        "selected_index": evaluation.get("selected_index", 0),
        "selected_directive": candidates[evaluation.get("selected_index", 0)]["directive"]
            if candidates else "none",
    }

    path = task_dir / "state" / "plan_candidates.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(artifact, f, indent=2)
    print(f"DEBUG: Plan candidates persisted to {path}", flush=True)


# ────────────────────────────────────────────────────────────
# Main Planner Node
# ────────────────────────────────────────────────────────────

def planner_node(state: AgentState) -> dict:
    """
    Agentic planner with multi-plan generation and selection.

    3-Phase Process:
    1. RESEARCH  — Single ReAct agent gathers domain context
    2. GENERATE  — 3 diverse plan candidates (escalates to 5 if needed)
    3. JUDGE     — Evaluates candidates and selects the best one

    Includes approval gates for high-risk or low-confidence plans.
    """
    # ─── Setup ───
    llm = get_llm("planner")
    available_skills_summary = format_available_skills(state.get("available_skills", []))

    workspace = state.get("workspace")
    if not workspace:
        context = state.get("task_context", {})
        task_dir_rel = context.get("task_dir_rel")
        if task_dir_rel:
            workspace_json_path = Path(task_dir_rel) / "state" / "workspace.json"
            if workspace_json_path.exists():
                try:
                    with open(workspace_json_path, "r") as f:
                        workspace_data = json.load(f)
                        workspace = Workspace.from_dict(workspace_data)
                except Exception as e:
                    print(f"DEBUG: Failed to read workspace.json: {str(e)}", flush=True)

    workspace_context_str = json.dumps(workspace.model_dump(), indent=2) if workspace else "{}"

    # ─── Rollback / Failure Context & Cleanup ───
    review = state.get("reflector_review")
    rollback_context = "No previous failures reported."
    if review and review.get("decision") == "rollback":
        rollback_context = (
            f"PREVIOUS FAILURE DETECTED (Rollback triggered).\n"
            f"Reason: {review.get('rollback_reason', 'Not provided')}\n"
            f"Issues: {', '.join(review.get('issues_identified', []))}\n"
            f"Hint: {review.get('alternative_approach_hint', 'No hint provided')}"
        )
        if workspace:
            steps_dir = Path(workspace.task_directory_rel) / "steps"
            if steps_dir.exists():
                import shutil
                try:
                    for item in steps_dir.iterdir():
                        if item.is_dir():
                            shutil.rmtree(item)
                        else:
                            item.unlink()
                    print(f"DEBUG: Complete Reset performed. Cleared {steps_dir}", flush=True)
                except Exception as e:
                    print(f"DEBUG: Failed to clear legacy steps: {str(e)}", flush=True)

    # ─── Historical Context ───
    historical_context = "No previous completed steps."
    plan = state.get("implementation_plan", [])
    current_step_idx = state.get("current_step_index", 0)
    if workspace and plan and current_step_idx > 0:
        try:
            historical_context = MemoryManager.load_previous_step_reports(
                workspace=workspace,
                plan=plan,
                current_step_idx=current_step_idx,
                query=state.get("task_description", ""),
                top_k=5,
                min_score=0.0
            )
        except Exception as e:
            print(f"DEBUG: Failed to load historical context: {str(e)}", flush=True)

    task_description = state["task_description"]
    skill_name = state.get("skill_name") or "Not Specified"

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PHASE 1: RESEARCH (Single ReAct Agent)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    print(f"\n{'=' * 60}", flush=True)
    print(f"PLANNER PHASE 1: Research & Exploration", flush=True)
    print(f"{'=' * 60}", flush=True)
    print(f"Task: {task_description[:80]}...", flush=True)

    task_context = state.get("task_context", {})
    planning_tools = create_planning_tools(workspace, task_context)

    user_prompt = PLANNER_USER_PROMPT.format(
        task_description=task_description,
        workspace_context=workspace_context_str,
        skill_name=skill_name,
        available_skills_summary=available_skills_summary,
        rollback_context=rollback_context,
        historical_context=historical_context,
    )

    planning_messages = [
        SystemMessage(content=PLANNER_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    planner_agent = create_react_agent(llm, planning_tools)

    start_time = time.time()
    print("DEBUG: Invoking agentic planner for research...", flush=True)

    try:
        agent_result = planner_agent.invoke(
            {"messages": planning_messages},
            config={"callbacks": [DebugCallbackHandler()]}
        )
        phase1_duration = time.time() - start_time
        print(f"\nDEBUG: Phase 1 (research) completed in {phase1_duration:.1f}s", flush=True)
    except Exception as e:
        print(f"DEBUG: Agentic planner FAILED: {str(e)}", flush=True)
        raise e

    # Extract research context from the agent's tool interactions
    agent_messages = agent_result.get("messages", [])
    research_context = _extract_research_context(agent_messages)

    # Also try to extract the plan the ReAct agent produced — use as candidate 0 (balanced)
    react_plan = None
    try:
        react_plan = _extract_plan_from_messages(agent_messages)
        print(f"DEBUG: ReAct agent also produced a plan with {len(react_plan.get('steps', []))} steps", flush=True)
    except ValueError:
        print("DEBUG: ReAct agent did not submit a plan (will generate all candidates fresh)", flush=True)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PHASE 2: GENERATE CANDIDATES
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    print(f"\n{'=' * 60}", flush=True)
    print(f"PLANNER PHASE 2: Multi-Plan Generation", flush=True)
    print(f"{'=' * 60}", flush=True)

    # Extract task_id and task_directory_rel from workspace (server-side)
    ws_task_id = workspace.task_id if workspace else "unknown_task"
    ws_task_dir_rel = workspace.task_directory_rel if workspace else ""

    escalation_history = []

    # Initial run: 3 candidates (directives 0-2)
    initial_directives = PLAN_GENERATION_DIRECTIVES[:DEFAULT_CANDIDATE_COUNT]
    candidates = _generate_plan_candidates(
        directives=initial_directives,
        task_description=task_description,
        workspace_context=workspace_context_str,
        research_context=research_context,
        skill_name=skill_name,
        available_skills_summary=available_skills_summary,
        historical_context=historical_context,
        rollback_context=rollback_context,
        task_id=ws_task_id,
        task_directory_rel=ws_task_dir_rel,
    )

    # If the ReAct agent produced a valid plan, include it as a candidate
    if react_plan:
        try:
            validated = ImplementationPlanInput(**react_plan)
            candidates.insert(0, {
                "directive": "react_original",
                "directive_label": "ReAct Agent Original",
                "steps_count": len(validated.steps),
                "plan": validated.model_dump()
            })
            print(f"DEBUG: Added ReAct agent's original plan as candidate 0", flush=True)
        except Exception as e:
            print(f"DEBUG: ReAct agent's plan failed validation: {str(e)[:200]}", flush=True)

    if not candidates:
        raise ValueError("No valid plan candidates were generated. All generation attempts failed.")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PHASE 3: JUDGE & SELECT
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    print(f"\n{'=' * 60}", flush=True)
    print(f"PLANNER PHASE 3: Plan Evaluation & Selection", flush=True)
    print(f"{'=' * 60}", flush=True)

    evaluation = _judge_plans(candidates, task_description)

    # ─── Adaptive Escalation ───
    if _should_escalate(evaluation, len(candidates)):
        reason = evaluation.get("escalation_reason", "Margin too thin or scores too low")
        print(f"\n🔄 ESCALATION TRIGGERED: {reason}", flush=True)
        escalation_history.append(f"Round 1: {len(candidates)} candidates, escalating because: {reason}")

        # Generate ONLY the 2 new escalation directives (indices 3-4)
        escalation_directives = PLAN_GENERATION_DIRECTIVES[DEFAULT_CANDIDATE_COUNT:ESCALATED_CANDIDATE_COUNT]
        extra = _generate_plan_candidates(
            directives=escalation_directives,
            task_description=task_description,
            workspace_context=workspace_context_str,
            research_context=research_context,
            skill_name=skill_name,
            available_skills_summary=available_skills_summary,
            historical_context=historical_context,
            rollback_context=rollback_context,
            task_id=ws_task_id,
            task_directory_rel=ws_task_dir_rel,
        )

        # Append new candidates
        candidates.extend(extra)

        print(f"DEBUG: After escalation: {len(candidates)} total candidates", flush=True)
        escalation_history.append(f"Round 2: {len(candidates)} total candidates after escalation")

        # Re-judge with all candidates
        evaluation = _judge_plans(candidates, task_description)

    # ─── Extract winner ───
    selected_idx = evaluation.get("selected_index", 0)
    if selected_idx >= len(candidates):
        selected_idx = 0
    winner = candidates[selected_idx]
    result = winner["plan"]

    print(f"\n🏆 WINNER: {winner['directive_label']} ({winner['steps_count']} steps)", flush=True)

    # ─── Convert to PlanSteps ───
    plan_steps = []
    for step in result["steps"]:
        plan_step: PlanStep = {
            "step_id": step.get("step_id"),
            "title": step.get("title"),
            "description": step.get("description"),
            "acceptance_criteria": step.get("acceptance_criteria", []),
            "max_attempts": step.get("max_attempts", 3),
            "estimated_complexity": step.get("estimated_complexity", "medium"),
            "dependencies": step.get("dependencies", []),
            "status": "pending",
            "uses_skills": step.get("uses_skills", []),
            "skill_instructions": step.get("skill_instructions"),
            "can_run_in_parallel": step.get("can_run_in_parallel", False),
            "current_attempt": 1
        }
        plan_steps.append(plan_step)

    plan_steps = _enforce_systems_depth_requirements(task_description, plan_steps)

    # ─── Approval Gate ───
    needs_approval, approval_reason = _needs_approval(evaluation, plan_steps)

    if needs_approval:
        summary = _format_approval_summary(evaluation, candidates, plan_steps)
        print(summary, flush=True)
        print(f"\n⚠️  Approval required: {approval_reason}", flush=True)
        print("   Auto-proceeding with winning plan (HITL not yet wired).", flush=True)
        # TODO: When HITL is wired, set awaiting_user_input = True
        # and let the user select proceed/alt/revise

    # ─── Persist everything ───
    if workspace:
        task_dir = Path(workspace.task_directory_rel)

        # Save plan_candidates.json (audit trail)
        _persist_candidates(task_dir, candidates, evaluation, escalation_history)

        # Save winning plan as plan.json
        plan_rel_path = workspace.get_path("plan_path")
        plan_path = task_dir / plan_rel_path
        plan_data = {
            "task_id": result["task_id"],
            "task_directory_rel": result["task_directory_rel"],
            "steps": plan_steps,
            "selected_from": {
                "directive": winner["directive"],
                "candidate_count": len(candidates),
                "judge_score": evaluation.get("evaluations", [{}])[selected_idx].get("total_score", "N/A")
                    if evaluation.get("evaluations") and selected_idx < len(evaluation.get("evaluations", []))
                    else "N/A",
                "escalated": len(escalation_history) > 0,
            }
        }
        try:
            plan_path.parent.mkdir(parents=True, exist_ok=True)
            with open(plan_path, "w") as f:
                json.dump(plan_data, f, indent=2)
            print(f"DEBUG: Winning plan persisted to {plan_path}", flush=True)
        except Exception as e:
            print(f"DEBUG: Failed to persist plan: {str(e)}", flush=True)

    total_duration = time.time() - start_time
    print(f"\n{'=' * 60}", flush=True)
    print(f"PLANNER: Complete! Total time: {total_duration:.1f}s", flush=True)
    print(f"  Candidates generated: {len(candidates)}", flush=True)
    print(f"  Winner: {winner['directive_label']}", flush=True)
    print(f"  Escalated: {'Yes' if escalation_history else 'No'}", flush=True)
    print(f"{'=' * 60}", flush=True)

    return {
        "implementation_plan": plan_steps,
        "current_step_index": 0,
        "phase": "executing",
        "progress_reports": [
            f"Plan selected: {winner['directive_label']} ({winner['steps_count']} steps)",
            f"Candidates evaluated: {len(candidates)}",
            f"Escalated: {'Yes' if escalation_history else 'No'}",
        ]
    }
