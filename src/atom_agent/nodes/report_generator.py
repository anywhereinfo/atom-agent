import json
from pathlib import Path
from typing import Dict, Any, List
from langchain_core.messages import SystemMessage, HumanMessage

from ..state import AgentState
from ..workspace import Workspace
from ..config import get_llm

from ..prompts.report_prompts import REPORT_SYSTEM_PROMPT, REPORT_USER_PROMPT


def _normalize_llm_content(content) -> str:
    """Normalize LLM response content to a plain string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                if "text" in part:
                    parts.append(part["text"])
        return "".join(parts)
    return str(content)


def _read_text_safe(path: Path, max_chars: int = 0) -> str:
    """Read a text file safely, optionally truncating."""
    try:
        content = path.read_text(encoding="utf-8")
        if max_chars > 0 and len(content) > max_chars:
            return content[:max_chars] + f"\n\n... [truncated, {len(content)} total chars]"
        return content
    except Exception:
        return "[unreadable]"


def _read_json_safe(path: Path) -> Dict:
    """Read a JSON file safely."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _collect_task_data(task_dir: Path) -> Dict[str, Any]:
    """
    Scans the task directory and collects ALL available evidence,
    including reflector reports, attempt history, and test results.
    """
    data = {
        "task": {},
        "plan": {},
        "steps": {},
        "judge": {}  # Plan judge evaluation data
    }

    # 1. Task metadata
    task_json = task_dir / "state" / "task.json"
    if task_json.exists():
        data["task"] = _read_json_safe(task_json)

    # 2. Plan
    plan_json = task_dir / "state" / "plan.json"
    if plan_json.exists():
        data["plan"] = _read_json_safe(plan_json)

    # 2b. Plan Candidates (judge evaluation with scores and weaknesses)
    candidates_json = task_dir / "state" / "plan_candidates.json"
    if candidates_json.exists():
        candidates_data = _read_json_safe(candidates_json)
        judge_eval = candidates_data.get("judge_evaluation", {})
        if judge_eval:
            selected_idx = judge_eval.get("selected_index", 0)
            evaluations = judge_eval.get("evaluations", [])
            selected_eval = next(
                (e for e in evaluations if e.get("candidate_index") == selected_idx),
                evaluations[0] if evaluations else {}
            )
            data["judge"] = {
                "total_score": selected_eval.get("total_score", 0),
                "max_possible": 60,  # 6 dimensions × 10 max
                "weaknesses": selected_eval.get("weaknesses", []),
                "strengths": selected_eval.get("strengths", []),
                "risk_level": judge_eval.get("risk_level", "unknown"),
                "selection_reasoning": judge_eval.get("selection_reasoning", "")
            }

    # 3. Per-step: committed evidence + attempt history + reflector reports
    steps_dir = task_dir / "steps"
    if not steps_dir.exists():
        return data

    for step_dir in sorted(steps_dir.iterdir()):
        if not step_dir.is_dir():
            continue
        step_id = step_dir.name
        step_data = {
            "manifest": {},
            "artifacts": {},
            "reflector_report": {},
            "attempt_history": [],
            "test_results": {},
            "message_count": 0
        }

        # --- Committed Evidence ---
        committed_dir = step_dir / "committed"

        # Manifest (scored result)
        manifest_path = committed_dir / "manifest.json"
        if manifest_path.exists():
            step_data["manifest"] = _read_json_safe(manifest_path)

        # Artifacts — full content for key files, truncated for large ones
        artifacts_dir = committed_dir / "artifacts"
        if artifacts_dir.exists():
            for f in sorted(artifacts_dir.iterdir()):
                if not f.is_file():
                    continue
                name = f.name

                # Test results get special treatment — structured data
                if name == "test_results.json":
                    step_data["test_results"] = _read_json_safe(f)
                # Skip binary-ish outputs
                elif name.endswith((".txt",)):
                    step_data["artifacts"][name] = _read_text_safe(f, max_chars=500)
                # Key data artifacts get more room
                elif name.endswith((".json", ".md", ".csv")):
                    step_data["artifacts"][name] = _read_text_safe(f, max_chars=4000)
                else:
                    step_data["artifacts"][name] = _read_text_safe(f, max_chars=1000)

        # impl.py and test.py — read first 80 lines (captures docstrings + logic)
        for code_file in ["impl.py", "test.py"]:
            code_path = committed_dir / code_file
            if code_path.exists():
                try:
                    lines = code_path.read_text(encoding="utf-8").splitlines()
                    preview = "\n".join(lines[:80])
                    if len(lines) > 80:
                        preview += f"\n... [{len(lines)} lines total]"
                    step_data["artifacts"][code_file] = preview
                except Exception:
                    pass

        # --- Reflector Report (the CRITICAL missing piece) ---
        # The reflector writes report.json in each attempt directory
        # It contains per-criterion pass/fail with evidence strings
        attempts_dir = step_dir / "attempts"
        if attempts_dir.exists():
            for attempt_dir in sorted(attempts_dir.iterdir()):
                if not attempt_dir.is_dir():
                    continue
                attempt_id = attempt_dir.name

                report_path = attempt_dir / "report.json"
                if report_path.exists():
                    report = _read_json_safe(report_path)
                    step_data["attempt_history"].append({
                        "attempt_id": attempt_id,
                        "decision": report.get("decision", "unknown"),
                        "confidence_score": report.get("confidence_score"),
                        "observations": report.get("observations", []),
                        "issues_identified": report.get("issues_identified", []),
                        "improvements_suggested": report.get("improvements_suggested", []),
                        "criteria_evaluation": report.get("criteria_evaluation", []),
                        "attempt_awareness": report.get("attempt_awareness", "")
                    })

                    # The final (promoted) attempt's report is the definitive one
                    promoted = step_data["manifest"].get("promoted_attempt", "")
                    if attempt_id == promoted:
                        step_data["reflector_report"] = report

        # --- Message history size (for methodology awareness) ---
        history_path = step_dir / "messages" / "history.json"
        if history_path.exists():
            try:
                history = _read_json_safe(history_path)
                if isinstance(history, list):
                    step_data["message_count"] = len(history)
            except Exception:
                pass

        data["steps"][step_id] = step_data

    return data


def _format_plan_summary(plan_data: Dict) -> str:
    """Formats the plan into a readable methodology section."""
    steps = plan_data.get("steps", [])
    if not steps:
        return "No plan data available."

    lines = []
    for i, step in enumerate(steps, 1):
        lines.append(f"### Phase {i}: {step.get('title', step.get('step_id', 'Unknown'))}")
        lines.append(f"**Step ID**: `{step.get('step_id', 'unknown')}`")
        lines.append(f"**Description**: {step.get('description', 'N/A')}")
        lines.append(f"**Complexity**: {step.get('estimated_complexity', 'N/A')}")

        deps = step.get("dependencies", [])
        lines.append(f"**Dependencies**: {', '.join(deps) if deps else 'None (independent)'}")

        criteria = step.get("acceptance_criteria", [])
        if criteria:
            lines.append("**Acceptance Criteria**:")
            for c in criteria:
                lines.append(f"  - {c}")

        lines.append(f"**Max Attempts Allowed**: {step.get('max_attempts', 'N/A')}")
        lines.append("")

    return "\n".join(lines)


def _format_step_evidence(steps_data: Dict[str, Dict]) -> str:
    """
    Formats per-step evidence into a comprehensive section,
    including reflector evaluations and attempt history.
    """
    if not steps_data:
        return "No step evidence available."

    lines = []
    for step_id, step_data in steps_data.items():
        manifest = step_data.get("manifest", {})
        artifacts = step_data.get("artifacts", {})
        reflector = step_data.get("reflector_report", {})
        attempts = step_data.get("attempt_history", [])
        test_results = step_data.get("test_results", {})

        lines.append(f"---")
        lines.append(f"### Step: `{step_id}`")
        lines.append(f"**Promoted Attempt**: {manifest.get('promoted_attempt', 'N/A')}")
        lines.append(f"**Total Attempts**: {len(attempts)}")
        lines.append(f"**Confidence Score**: {manifest.get('score', 'N/A')}")
        lines.append(f"**LLM Interaction Messages**: {step_data.get('message_count', 'N/A')}")

        # --- Acceptance Criteria from Manifest ---
        criteria = manifest.get("acceptance_criteria", [])
        if criteria:
            lines.append(f"\n**Acceptance Criteria**:")
            for c in criteria:
                lines.append(f"  - {c}")

        # --- Reflector's Per-Criterion Evaluation (the gold) ---
        criteria_eval = reflector.get("criteria_evaluation", [])
        if criteria_eval:
            lines.append(f"\n**Reflector's Per-Criterion Evaluation**:")
            lines.append(f"| Criterion | Status | Evidence |")
            lines.append(f"|-----------|--------|----------|")
            for ce in criteria_eval:
                criterion = ce.get("criterion", "N/A")[:80]
                status = ce.get("status", "N/A")
                evidence = ce.get("evidence", "N/A")[:120]
                lines.append(f"| {criterion} | **{status}** | {evidence} |")

        # --- Reflector Observations ---
        observations = reflector.get("observations", [])
        if observations:
            lines.append(f"\n**Reflector Observations**:")
            for obs in observations:
                lines.append(f"  - {obs}")

        # --- Issues & Improvements (if any) ---
        issues = reflector.get("issues_identified", [])
        if issues:
            lines.append(f"\n**Issues Identified**:")
            for issue in issues:
                lines.append(f"  - ⚠️ {issue}")

        improvements = reflector.get("improvements_suggested", [])
        if improvements:
            lines.append(f"\n**Improvements Suggested**:")
            for imp in improvements:
                lines.append(f"  - 💡 {imp}")

        # --- Attempt History (iteration narrative) ---
        if len(attempts) > 1:
            lines.append(f"\n**Attempt History** (shows iterative refinement):")
            for att in attempts:
                decision_icon = "✅" if att["decision"] == "proceed" else "🔄" if att["decision"] == "refine" else "⛔"
                lines.append(f"  - `{att['attempt_id']}`: {decision_icon} {att['decision']} "
                             f"(score: {att.get('confidence_score', 'N/A')})")
                if att.get("issues_identified"):
                    for issue in att["issues_identified"]:
                        lines.append(f"    - Issue: {issue}")

        # --- Test Results (structured) ---
        if test_results:
            lines.append(f"\n**Test Verification**:")
            lines.append(f"  - Passed: {test_results.get('passed', 'N/A')}")
            lines.append(f"  - Failed: {test_results.get('failed', 'N/A')}")
            lines.append(f"  - Exit Code: {test_results.get('exit_code', 'N/A')}")

        # --- Committed Files ---
        committed_files = manifest.get("files", [])
        if committed_files:
            lines.append(f"\n**Committed Files**: {', '.join(committed_files)}")

        # --- Artifact Contents ---
        if artifacts:
            lines.append(f"\n**Artifact Contents**:")
            for filename, content in artifacts.items():
                lines.append(f"\n#### `{filename}`")
                lines.append(f"```")
                lines.append(content)
                lines.append(f"```")

        lines.append("")

    return "\n".join(lines)


def _compute_quality_summary(task_data: Dict[str, Any]) -> str:
    """Computes aggregate quality metrics from collected evidence.
    
    Provides the report writer with a concrete quality baseline
    that prevents score inflation.
    """
    lines = []
    
    # Judge score
    judge = task_data.get("judge", {})
    if judge:
        judge_score = judge.get("total_score", 0)
        judge_max = judge.get("max_possible", 60)
        judge_pct = (judge_score / judge_max * 100) if judge_max > 0 else 0
        lines.append(f"**Plan Judge Score**: {judge_score}/{judge_max} ({judge_pct:.0f}%)")
        lines.append(f"**Risk Level**: {judge.get('risk_level', 'unknown')}")
        weaknesses = judge.get("weaknesses", [])
        if weaknesses:
            lines.append("**Judge-Identified Weaknesses**:")
            for w in weaknesses:
                lines.append(f"  - {w}")
    
    # Aggregate reflector metrics
    steps = task_data.get("steps", {})
    confidence_scores = []
    criteria_pass_count = 0
    criteria_total_count = 0
    total_retries = 0
    
    for step_id, step_data in steps.items():
        reflector = step_data.get("reflector_report", {})
        if reflector:
            score = reflector.get("confidence_score")
            if score is not None:
                confidence_scores.append(score)
            
            for ce in reflector.get("criteria_evaluation", []):
                criteria_total_count += 1
                if ce.get("status") == "pass":
                    criteria_pass_count += 1
        
        attempts = step_data.get("attempt_history", [])
        if len(attempts) > 1:
            total_retries += len(attempts) - 1
    
    if confidence_scores:
        avg_confidence = sum(confidence_scores) / len(confidence_scores)
        lines.append(f"**Average Reflector Confidence**: {avg_confidence:.2f}")
    
    if criteria_total_count > 0:
        pass_rate = criteria_pass_count / criteria_total_count
        lines.append(f"**Criteria Pass Rate**: {criteria_pass_count}/{criteria_total_count} ({pass_rate:.0%})")
    
    lines.append(f"**Total Retries**: {total_retries}")
    
    # Compute recommended score ceiling
    if judge and confidence_scores:
        judge_pct = judge.get("total_score", 0) / judge.get("max_possible", 60) * 100
        avg_conf_pct = avg_confidence * 100
        ceiling = min(judge_pct + 15, avg_conf_pct + 10, 100)
        lines.append(f"**Recommended Score Ceiling**: {ceiling:.0f}/100 "
                     f"(based on judge={judge_pct:.0f}% + 15% margin, reflector avg={avg_conf_pct:.0f}% + 10%)")

    return "\n".join(lines) if lines else "No quality metrics available."


def report_generator_node(state: AgentState) -> Dict[str, Any]:
    """
    Final node: Scans the task directory for all committed evidence
    and generates a comprehensive scientific-style report.
    """
    print("\n" + "=" * 60, flush=True)
    print("REPORT GENERATOR: Collecting evidence and generating report...", flush=True)
    print("=" * 60, flush=True)

    workspace = state.get("workspace")
    if isinstance(workspace, dict):
        workspace = Workspace.from_dict(workspace)

    task_dir = Path(workspace.task_directory_rel)

    # 1. Collect all data from disk
    print("DEBUG: Scanning task directory for evidence...", flush=True)
    task_data = _collect_task_data(task_dir)

    task_description = task_data.get("task", {}).get("task_description", "Unknown task")
    step_count = len(task_data["steps"])
    total_attempts = sum(len(s.get("attempt_history", [])) for s in task_data["steps"].values())

    print(f"DEBUG: Task: {task_description}", flush=True)
    print(f"DEBUG: Steps: {step_count}, Total attempts across all steps: {total_attempts}", flush=True)

    # 2. Format context for LLM
    plan_summary = _format_plan_summary(task_data.get("plan", {}))
    step_evidence = _format_step_evidence(task_data.get("steps", {}))

    # 2b. Compute aggregate quality metrics
    quality_summary = _compute_quality_summary(task_data)

    user_prompt = REPORT_USER_PROMPT.format(
        task_description=task_description,
        plan_summary=plan_summary,
        step_evidence=step_evidence,
        quality_summary=quality_summary
    )

    # Log context size for debugging
    total_context = len(REPORT_SYSTEM_PROMPT) + len(user_prompt)
    print(f"DEBUG: Total context size: {total_context:,} chars ({total_context // 4:,} est. tokens)", flush=True)

    # 3. Generate report via LLM
    print("DEBUG: Sending evidence to LLM for report generation...", flush=True)
    llm = get_llm("reflector")  # Low temp for factual scientific writing

    messages = [
        SystemMessage(content=REPORT_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt)
    ]

    import time
    
    # Retry wrapper for transient network errors
    MAX_RETRIES = 3
    RETRY_DELAYS = [10, 20, 40]
    response = None

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = llm.invoke(messages)
            break
        except Exception as e:
            error_name = type(e).__name__
            is_transient = any(keyword in error_name for keyword in [
                "Timeout", "ReadTimeout", "ConnectTimeout", "ConnectionError",
                "RemoteDisconnected", "ConnectionReset"
            ]) or "timed out" in str(e).lower()

            if is_transient and attempt < MAX_RETRIES:
                delay = RETRY_DELAYS[attempt]
                print(f"⚠️  Report Gen transient error ({error_name}), retrying in {delay}s...", flush=True)
                time.sleep(delay)
                continue
            else:
                print(f"DEBUG: FATAL ERROR in report generator: {str(e)}", flush=True)
                raise e
    report_content = _normalize_llm_content(response.content)

    # 4. Save report
    report_path = task_dir / "final_report.md"
    report_path.write_text(report_content, encoding="utf-8")
    print(f"DEBUG: Report saved to {report_path} ({len(report_content):,} chars)", flush=True)

    # 5. Save metadata
    meta = {
        "task_id": task_data.get("task", {}).get("task_id", "unknown"),
        "steps_analyzed": list(task_data["steps"].keys()),
        "total_attempts": total_attempts,
        "report_path": str(report_path),
        "report_size_chars": len(report_content),
        "context_size_chars": total_context,
        "model_used": "reflector",
        "judge_score": task_data.get("judge", {}).get("total_score"),
        "judge_max": 60
    }
    meta_path = task_dir / "report_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\n" + "=" * 60, flush=True)
    print(f"REPORT GENERATOR: Complete! Report saved to {report_path}", flush=True)
    print("=" * 60, flush=True)

    return {
        "phase": "report_generated",
        "progress_reports": [f"Final report generated: {report_path}"]
    }
