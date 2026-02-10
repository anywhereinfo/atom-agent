import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path.cwd() / "src"))

from atom_agent.nodes.executor import _prepare_executor_messages
from atom_agent.state import AgentState, PlanStep
from langchain_core.messages import SystemMessage, HumanMessage

def test_executor_prompt():
    print("Testing executor prompt construction...")
    
    workspace = {
        "task_id": "test_task",
        "task_directory_rel": "tasks/test_task",
        "paths": {
            "plan_path": "state/plan.json"
        }
    }
    
    current_step: PlanStep = {
        "step_id": "step_01",
        "title": "Initial Step",
        "description": "Do something cool",
        "acceptance_criteria": ["Criteria 1", "Criteria 2"],
        "max_attempts": 3,
        "estimated_complexity": "low",
        "dependencies": [],
        "status": "pending",
        "uses_skills": [],
        "skill_instructions": None,
        "can_run_in_parallel": False,
        "current_attempt": 2
    }
    
    state: AgentState = {
        "task_description": "Test task",
        "task_context": {},
        "workspace": workspace,
        "implementation_plan": [current_step],
        "current_step_index": 0,
        "available_skills": [],
        "skills_to_use": [],
        "loaded_skill_content": {},
        "is_skill_learning_task": False,
        "task_id": "test_task"
    }
    
    messages = _prepare_executor_messages(state, current_step)
    
    assert len(messages) == 2
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)
    
    user_content = messages[1].content
    print("\nGenerated User Message Content:")
    print("-" * 30)
    print(user_content)
    print("-" * 30)
    
    # Assertions
    assert "Attempt ID: a02" in user_content
    assert "Step ID: step_01" in user_content
    assert '"task_id": "test_task"' in user_content
    assert "- Criteria 1" in user_content
    
    print("\n✅ Verification SUCCESSFUL: Prompt correctly generated with JSON context and attempt_id.")

if __name__ == "__main__":
    try:
        test_executor_prompt()
    except Exception as e:
        print(f"\n❌ Verification FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
