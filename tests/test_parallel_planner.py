"""
Verify that _generate_plan_candidates runs in parallel using ThreadPoolExecutor.
If 3 candidates each take 2s, parallel should complete in ~2s, sequential in ~6s.
"""
import time
import sys
import os
from unittest.mock import patch

# Ensure src is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from atom_agent.nodes.planner import _generate_plan_candidates


def _mock_generate_single_candidate(**kwargs):
    """Simulate a 2-second LLM call."""
    directive = kwargs.get("directive", {})
    time.sleep(2.0)
    return {
        "directive": directive.get("id", "unknown"),
        "directive_label": directive.get("label", "unknown"),
        "steps_count": 3,
        "plan": {"steps": []},
    }


def test_parallel_generation():
    directives = [
        {"id": "d1", "label": "Standard"},
        {"id": "d2", "label": "Security-First"},
        {"id": "d3", "label": "Minimal"},
    ]

    with patch(
        "atom_agent.nodes.planner._generate_single_candidate",
        side_effect=_mock_generate_single_candidate,
    ):
        start = time.time()
        results = _generate_plan_candidates(
            directives=directives,
            task_description="test task",
            workspace_context="{}",
            research_context="no research",
            skill_name="test",
            available_skills_summary="none",
            historical_context="none",
            rollback_context="none",
            task_id="t1",
            task_directory_rel="tasks/test",
        )
        elapsed = time.time() - start

    print(f"\n{'=' * 40}")
    print(f"Candidates returned : {len(results)}")
    print(f"Total time          : {elapsed:.2f}s")
    print(f"Expected (parallel) : ~2s")
    print(f"Expected (sequential): ~6s")

    assert len(results) == 3, f"Expected 3 candidates, got {len(results)}"
    assert elapsed < 4.0, f"Took {elapsed:.1f}s — execution was sequential, not parallel!"

    # Verify deterministic sort order
    ids = [r["directive"] for r in results]
    assert ids == sorted(ids), f"Candidates not sorted: {ids}"

    print(f"✅ PASSED: Parallel execution confirmed ({elapsed:.1f}s < 4s)")


if __name__ == "__main__":
    test_parallel_generation()
