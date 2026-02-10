"""Verification: planner complexity ceiling + all architecture fixes."""
import sys
sys.path.insert(0, "/home/jim/code/atom/src")

errors = []

# 1. Planner prompts
try:
    from atom_agent.prompts.planner_prompts import PLANNER_SYSTEM_PROMPT
    assert "COMPLEXITY CEILING" in PLANNER_SYSTEM_PROMPT, "Missing COMPLEXITY CEILING section"
    assert '"low|medium"' in PLANNER_SYSTEM_PROMPT or "'low|medium'" in PLANNER_SYSTEM_PROMPT or "low\" or \"medium" in PLANNER_SYSTEM_PROMPT, "Still allows high complexity"
    assert "high" not in PLANNER_SYSTEM_PROMPT.split("estimated_complexity")[1].split("\n")[0] or "INVALID" in PLANNER_SYSTEM_PROMPT, "High still in estimated_complexity"
    print("OK executor_prompts")
except Exception as e:
    errors.append(f"FAIL planner_prompts: {e}")
    print(errors[-1])

# 2. Executor prompts
try:
    from atom_agent.prompts.executor_prompts import EXECUTOR_SYSTEM_PROMPT, EXECUTOR_USER_PROMPT
    assert "{dependency_context}" in EXECUTOR_USER_PROMPT
    print("OK executor_prompts")
except Exception as e:
    errors.append(f"FAIL executor_prompts: {e}")
    print(errors[-1])

# 3. Reflector prompts
try:
    from atom_agent.prompts.reflector_prompts import REFLECTOR_SYSTEM_PROMPT, REFLECTOR_USER_PROMPT
    assert "{programmatic_warnings}" in REFLECTOR_USER_PROMPT
    print("OK reflector_prompts")
except Exception as e:
    errors.append(f"FAIL reflector_prompts: {e}")
    print(errors[-1])

# 4. Report prompts
try:
    from atom_agent.prompts.report_prompts import REPORT_SYSTEM_PROMPT, REPORT_USER_PROMPT
    assert "{quality_summary}" in REPORT_USER_PROMPT
    print("OK report_prompts")
except Exception as e:
    errors.append(f"FAIL report_prompts: {e}")
    print(errors[-1])

# 5. Reflector node
try:
    from atom_agent.nodes.reflector import _run_programmatic_prechecks
    print("OK reflector.py")
except Exception as e:
    errors.append(f"FAIL reflector.py: {e}")
    print(errors[-1])

# 6. Executor node
try:
    from atom_agent.nodes.executor import _build_dependency_context
    print("OK executor.py")
except Exception as e:
    errors.append(f"FAIL executor.py: {e}")
    print(errors[-1])

# 7. Report generator
try:
    from atom_agent.nodes.report_generator import _compute_quality_summary
    print("OK report_generator.py")
except Exception as e:
    errors.append(f"FAIL report_generator.py: {e}")
    print(errors[-1])

# 8. Planner node
try:
    from atom_agent.nodes.planner import planner_node
    print("OK planner.py")
except Exception as e:
    errors.append(f"FAIL planner.py: {e}")
    print(errors[-1])

print(f"\n{'='*40}")
if errors:
    print(f"FAILED: {len(errors)} error(s)")
    for e in errors:
        print(f"  {e}")
    sys.exit(1)
else:
    print("ALL 8 CHECKS PASSED")
