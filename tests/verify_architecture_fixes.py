"""Quick import verification for all modified modules."""
import sys
sys.path.insert(0, "/home/jim/code/atom/src")

errors = []

try:
    from atom_agent.prompts.executor_prompts import EXECUTOR_SYSTEM_PROMPT, EXECUTOR_USER_PROMPT
    assert "{dependency_context}" in EXECUTOR_USER_PROMPT, "Missing {dependency_context} placeholder"
    assert "HARDCODED OUTPUT" in EXECUTOR_SYSTEM_PROMPT, "Missing anti-hardcoded rules"
    print("✅ executor_prompts: OK")
except Exception as e:
    errors.append(f"❌ executor_prompts: {e}")
    print(errors[-1])

try:
    from atom_agent.prompts.reflector_prompts import REFLECTOR_SYSTEM_PROMPT, REFLECTOR_USER_PROMPT
    assert "{programmatic_warnings}" in REFLECTOR_USER_PROMPT, "Missing {programmatic_warnings} placeholder"
    assert "HARDCODED OUTPUT" in REFLECTOR_SYSTEM_PROMPT, "Missing hardcoded output scoring rule"
    assert "EXISTENCE-ONLY TESTS" in REFLECTOR_SYSTEM_PROMPT, "Missing shallow tests rule"
    assert "CONTENT DEPTH VALIDATION" in REFLECTOR_SYSTEM_PROMPT, "Missing content depth section"
    print("✅ reflector_prompts: OK")
except Exception as e:
    errors.append(f"❌ reflector_prompts: {e}")
    print(errors[-1])

try:
    from atom_agent.prompts.report_prompts import REPORT_SYSTEM_PROMPT, REPORT_USER_PROMPT
    assert "{quality_summary}" in REPORT_USER_PROMPT, "Missing {quality_summary} placeholder"
    assert "Score Calibration" in REPORT_SYSTEM_PROMPT, "Missing score calibration rules"
    assert "Known Limitations" in REPORT_SYSTEM_PROMPT, "Missing Known Limitations rule"
    print("✅ report_prompts: OK")
except Exception as e:
    errors.append(f"❌ report_prompts: {e}")
    print(errors[-1])

try:
    from atom_agent.nodes.reflector import reflector_node, _run_programmatic_prechecks
    print("✅ reflector.py: OK (imports + _run_programmatic_prechecks)")
except Exception as e:
    errors.append(f"❌ reflector.py: {e}")
    print(errors[-1])

try:
    from atom_agent.nodes.executor import executor_node, _build_dependency_context
    print("✅ executor.py: OK (imports + _build_dependency_context)")
except Exception as e:
    errors.append(f"❌ executor.py: {e}")
    print(errors[-1])

try:
    from atom_agent.nodes.report_generator import report_generator_node, _compute_quality_summary
    print("✅ report_generator.py: OK (imports + _compute_quality_summary)")
except Exception as e:
    errors.append(f"❌ report_generator.py: {e}")
    print(errors[-1])

print(f"\n{'='*40}")
if errors:
    print(f"FAILED: {len(errors)} error(s)")
    for e in errors:
        print(f"  {e}")
    sys.exit(1)
else:
    print("ALL CHECKS PASSED ✅")
