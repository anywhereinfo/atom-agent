"""
Tests for parallel plan scoring hardening:
  1. Parallel generation (pre-existing)
  2. Disk-backed candidate storage
  3. Scorer retry logic
  4. Index remapping when candidates fail
"""
import json
import time
import sys
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

# Ensure src is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from atom_agent.nodes.planner import (
    _generate_plan_candidates,
    _persist_candidates_to_disk,
    _load_candidate_plan,
    _score_single_plan,
    _judge_plans,
)


# ── Helpers ──

def _make_candidate(idx: int, label: str = "Test") -> dict:
    """Create a candidate dict with a minimal plan."""
    return {
        "directive": f"directive_{idx}",
        "directive_label": f"{label}-{idx}",
        "steps_count": 2,
        "plan": {
            "task_id": "t1",
            "task_directory_rel": "tasks/test",
            "steps": [
                {
                    "step_id": f"step_{idx}_1",
                    "title": f"Step 1 for candidate {idx}",
                    "estimated_complexity": "medium",
                    "dependencies": [],
                    "acceptance_criteria": ["done"],
                    "can_run_in_parallel": False,
                    "max_attempts": 3,
                },
                {
                    "step_id": f"step_{idx}_2",
                    "title": f"Step 2 for candidate {idx}",
                    "estimated_complexity": "low",
                    "dependencies": [f"step_{idx}_1"],
                    "acceptance_criteria": ["done"],
                    "can_run_in_parallel": False,
                    "max_attempts": 3,
                },
            ],
        },
    }


def _make_score(candidate_index: int, total: int = 45) -> dict:
    return {
        "candidate_index": candidate_index,
        "directive_label": f"Test-{candidate_index}",
        "total_score": total,
        "scores": {"coverage": 8, "decomposition": 7, "dependency_structure": 8,
                   "complexity_balance": 7, "parallelism": 7, "risk_management": 8},
        "strengths": ["solid"],
        "weaknesses": [],
    }


# ── Test 1: Disk-backed persistence ──

def test_disk_backed_storage():
    """Candidates are persisted to disk and can be loaded back."""
    with tempfile.TemporaryDirectory() as tmpdir:
        task_dir = Path(tmpdir)
        candidates = [_make_candidate(i) for i in range(3)]

        refs = _persist_candidates_to_disk(task_dir, candidates)

        assert len(refs) == 3
        for i, ref in enumerate(refs):
            # Ref should NOT contain plan
            assert "plan" not in ref
            assert "plan_path" in ref
            assert ref["directive_label"] == f"Test-{i}"

            # Plan should be loadable from disk
            plan = _load_candidate_plan(ref["plan_path"])
            assert "steps" in plan
            assert len(plan["steps"]) == 2

    print("✅ PASSED: Disk-backed storage")


# ── Test 2: Scorer retry ──

def test_scorer_retry():
    """Scorer retries once before giving up."""
    with tempfile.TemporaryDirectory() as tmpdir:
        task_dir = Path(tmpdir)
        candidates = [_make_candidate(0)]
        refs = _persist_candidates_to_disk(task_dir, candidates)

        call_count = 0

        def mock_invoke(messages):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("ReadTimeout: simulated")
            # Second attempt succeeds
            return MagicMock(content=json.dumps(_make_score(0)))

        with patch("atom_agent.nodes.planner.get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.invoke = mock_invoke
            mock_get_llm.return_value = mock_llm

            result = _score_single_plan(0, refs[0], "test task")

        assert result is not None, "Scorer should have succeeded on retry"
        assert call_count == 2, f"Expected 2 LLM calls (1 fail + 1 retry), got {call_count}"
        assert result["candidate_index"] == 0

    print("✅ PASSED: Scorer retry")


def test_scorer_permanent_failure():
    """Scorer returns None after exhausting retries."""
    with tempfile.TemporaryDirectory() as tmpdir:
        task_dir = Path(tmpdir)
        candidates = [_make_candidate(0)]
        refs = _persist_candidates_to_disk(task_dir, candidates)

        def mock_invoke(messages):
            raise ConnectionError("permanent failure")

        with patch("atom_agent.nodes.planner.get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.invoke = mock_invoke
            mock_get_llm.return_value = mock_llm

            result = _score_single_plan(0, refs[0], "test task")

        assert result is None, "Scorer should return None after permanent failure"

    print("✅ PASSED: Scorer permanent failure")


# ── Test 3: Index remapping ──

def test_index_remapping_with_failure():
    """When candidate 1 fails, aggregator sees contiguous indices and
    the selected_index is remapped back to the original candidate."""
    with tempfile.TemporaryDirectory() as tmpdir:
        task_dir = Path(tmpdir)
        candidates = [_make_candidate(i) for i in range(3)]
        refs = _persist_candidates_to_disk(task_dir, candidates)

        # Scorer mock: candidate 1 fails, 0 and 2 succeed
        def mock_score(candidate_index, candidate_ref, task_description):
            if candidate_index == 1:
                return None  # simulate failure
            score = _make_score(candidate_index, total=40 + candidate_index * 5)
            return score

        # Aggregator mock: selects contiguous index 1 (which maps to original candidate 2)
        aggregator_response = json.dumps({
            "selected_index": 1,  # contiguous index 1 = original candidate 2
            "selection_reasoning": "Candidate 2 has higher score",
            "margin": 5,
            "risk_level": "low",
            "escalation_recommended": False,
        })

        with patch("atom_agent.nodes.planner._score_single_plan", side_effect=mock_score), \
             patch("atom_agent.nodes.planner.get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = MagicMock(content=aggregator_response)
            mock_get_llm.return_value = mock_llm

            result = _judge_plans(refs, "test task")

        # The remapped index should be 2 (original candidate 2), not 1
        assert result["selected_index"] == 2, \
            f"Expected selected_index=2 (remapped), got {result['selected_index']}"
        assert len(result["evaluations"]) == 2, \
            f"Expected 2 valid evaluations, got {len(result['evaluations'])}"

    print("✅ PASSED: Index remapping with failure")


# ── Test 4: Parallel generation (pre-existing) ──

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

    assert len(results) == 3, f"Expected 3 candidates, got {len(results)}"
    assert elapsed < 4.0, f"Took {elapsed:.1f}s — execution was sequential, not parallel!"
    print(f"✅ PASSED: Parallel execution confirmed ({elapsed:.1f}s < 4s)")


# ── Test 5: Escalation Persistence Fix ──

def test_escalation_persistence_fix():
    """Verify that escalation persistence does not overwrite initial candidates."""
    with tempfile.TemporaryDirectory() as tmpdir:
        task_dir = Path(tmpdir)
        
        # Phase 1: Initial candidates (0, 1)
        initial_candidates = [_make_candidate(0), _make_candidate(1)]
        refs1 = _persist_candidates_to_disk(task_dir, initial_candidates)
        
        # Verify initial files exist
        candidates_dir = task_dir / "state" / "candidates"
        assert (candidates_dir / "candidate_0.json").exists()
        assert (candidates_dir / "candidate_1.json").exists()
        
        # Phase 2: Escalation candidates (2, 3)
        escalation_candidates = [_make_candidate(2), _make_candidate(3)]
        
        # This call SHOULD start indexing at len(refs1) = 2
        refs2 = _persist_candidates_to_disk(task_dir, escalation_candidates, start_index=len(refs1))
        
        # Verify escalation files exist and have correct names
        assert (candidates_dir / "candidate_2.json").exists()
        assert (candidates_dir / "candidate_3.json").exists()
        
        # CRITICAL: Verify initial files STILL exist (weren't overwritten/renamed)
        assert (candidates_dir / "candidate_0.json").exists(), "candidate_0.json disappeared!"
        assert (candidates_dir / "candidate_1.json").exists(), "candidate_1.json disappeared!"

    print("✅ PASSED: Escalation persistence fix (no overwrites)")


if __name__ == "__main__":
    test_disk_backed_storage()
    test_scorer_retry()
    test_scorer_permanent_failure()
    test_index_remapping_with_failure()
    test_parallel_generation()
    test_escalation_persistence_fix()
    print(f"\n{'=' * 40}")
    print("ALL TESTS PASSED ✅")
