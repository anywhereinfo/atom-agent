from typing import List
from ..state import AgentState, PlanStep

def plan_viewer_node(state: AgentState) -> dict:
    """Prints the implementation plan in execution order."""
    plan = state.get("implementation_plan", [])
    if not plan:
        print("\n" + "!"*60)
        print("      WARNING: NO IMPLEMENTATION PLAN FOUND")
        print("!"*60 + "\n")
        return {}

    # Topological Sort to find execution order based on dependencies
    ordered_steps = _topological_sort(plan)

    print("\n" + "="*60)
    print("      IMPLEMENTATION PLAN: EXECUTION ORDER")
    print("="*60)

    for i, step in enumerate(ordered_steps, 1):
        parallel_tag = "🚀 [CAN RUN IN PARALLEL]" if step.get("can_run_in_parallel") else "🔒 [SEQUENTIAL]"
        print(f"\n{i}. {step.get('title', 'No Title')} ({step.get('step_id', 'unknown')})")
        print(f"   Status: {parallel_tag}")
        print(f"   Complexity: {step.get('estimated_complexity', 'U').upper()}")
        print(f"   Description: {step.get('description', 'No description provided.')}")
        
        deps = step.get("dependencies", [])
        if deps:
            print(f"   Dependencies: {', '.join(deps)}")
        
        skills = step.get("uses_skills", [])
        if skills:
            print(f"   Skills to load: {', '.join(skills)}")
            
        print(f"   Acceptance Criteria:")
        for criteria in step.get("acceptance_criteria", []):
            print(f"     ✅ {criteria}")

        instructions = step.get("skill_instructions") or {}
        if instructions:
            print(f"   Skill Instructions:")
            for skill, instr in instructions.items():
                if skill:
                    print(f"     🛠️ {skill}: {instr}")
                    
        print(f"   Executable Contract:")
        print(f"     📄 File: {step.get('implementation_path')}")
        print(f"     🧪 Test: {step.get('test_path')}")
        print(f"     ⚙️  Run:  {step.get('run_command')}")
        outputs = step.get("outputs", [])
        if outputs:
            print(f"     📦 Outputs: {', '.join(outputs)}")

    print("\n" + "="*60 + "\n")

    return {
        "implementation_plan": plan, # Ensure plan preserved in state
    }

def _topological_sort(steps: List[PlanStep]) -> List[PlanStep]:
    """Helper to sort steps based on their dependency graph."""
    step_map = {s['step_id']: s for s in steps}
    visited = set()
    temp_visited = set()
    ordered = []

    def visit(step_id):
        if step_id in temp_visited:
            # Cycle detected or dependency missing, skip in this simple implementation
            return
        if step_id in visited:
            return
            
        step = step_map.get(step_id)
        if not step:
            return

        temp_visited.add(step_id)
        # Visit dependencies first
        for dep_id in step.get("dependencies", []):
            visit(dep_id)
            
        temp_visited.remove(step_id)
        visited.add(step_id)
        ordered.append(step)

    for step in steps:
        visit(step['step_id'])

    return ordered
