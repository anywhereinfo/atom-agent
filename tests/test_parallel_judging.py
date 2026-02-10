"""
Verify that _judge_plans correctly parallelizes candidate scoring.
"""
import time
import sys
import os
import json
from unittest.mock import patch, MagicMock

# Ensure src is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from atom_agent.nodes.planner import _judge_plans


def _mock_score_single_plan(idx, candidate, task_description):
    """Simulate a 2-second LLM call for scoring."""
    time.sleep(2.0)
    return {
        "candidate_index": idx,
        "scores": {"coverage": 8},
        "total_score": 40 + idx,
        "strengths": ["test"],
        "weaknesses": []
    }


def test_parallel_judging():
    candidates = [
        {"directive": "d1", "directive_label": "Standard", "steps_count": 3, "plan": {"steps": []}},
        {"directive": "d2", "directive_label": "Security-First", "steps_count": 3, "plan": {"steps": []}},
        {"directive": "d3", "directive_label": "Minimal", "steps_count": 3, "plan": {"steps": []}},
    ]

    # Mock the aggregator LLM call
    mock_aggregator_response = MagicMock()
    mock_aggregator_response.content = json.dumps({
        "selected_index": 2,
        "selection_reasoning": "Highest score",
        "margin": 1,
        "risk_level": "low",
        "escalation_recommended": False
    })

    with patch("atom_agent.nodes.planner._score_single_plan", side_effect=_mock_score_single_plan), \
         patch("atom_agent.nodes.planner.get_llm") as mock_get_llm:
        
        mock_get_llm.return_value.invoke.return_value = mock_aggregator_response
        
        start = time.time()
        evaluation = _judge_plans(
            candidates=candidates,
            task_description="test task"
        )
        elapsed = time.time() - start

    print(f"\n{'=' * 40}")
    print(f"Total time          : {elapsed:.2f}s")
    print(f"Expected (parallel) : ~2s + aggregate call")
    print(f"Expected (sequential): ~6s + aggregate call")

    assert evaluation["selected_index"] == 2
    assert len(evaluation["evaluations"]) == 3
    assert elapsed < 4.0, f"Took {elapsed:.1f}s — scoring was sequential, not parallel!"

    print(f"✅ PASSED: Parallel judging confirmed ({elapsed:.1f}s < 4s)")


if __name__ == "__main__":
    test_parallel_judging()
