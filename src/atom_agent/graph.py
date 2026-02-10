from langgraph.graph import StateGraph, END
from .state import AgentState
from .nodes.setup import task_setup_node
from .nodes.planner import planner_node
from .nodes.executor import executor_node
from .nodes.reflector import reflector_node
from .nodes.commit import commit_node
from .nodes.report_generator import report_generator_node

def route_reflection(state: AgentState) -> str:
    """
    Implements the Minimal Routing Contract.
    Routes to executor, planner, or commit based on Reflector decision.
    """
    review = state.get("reflector_review", {})
    decision = review.get("decision", "refine")
    
    plan = state.get("implementation_plan", [])
    current_step_idx = state.get("current_step_index", 0)
    
    if current_step_idx < 0 or current_step_idx >= len(plan):
        return "end"

    current_step = plan[current_step_idx]
    attempt_num = current_step.get("current_attempt", 1)
    max_attempts = current_step.get("max_attempts", 3)

    # 1. Guardrail: Max attempts exhausted (for refine/rollback paths)
    if attempt_num > max_attempts and decision != "proceed":
        print(f"DEBUG ROUTER: Attempts exhausted ({attempt_num-1}/{max_attempts}). Forcing rollback.", flush=True)
        return "planner"

    # 2. Reflector Decisions
    if decision == "proceed":
        return "commit" 
    elif decision == "rollback":
        return "planner"
    else: # refine
        return "executor"

def create_graph():
    graph = StateGraph(AgentState)

    graph.add_node("setup", task_setup_node)
    graph.add_node("planner", planner_node)
    graph.add_node("executor", executor_node)
    graph.add_node("reflector", reflector_node)
    graph.add_node("commit", commit_node)
    graph.add_node("report_generator", report_generator_node)

    graph.set_entry_point("setup")
    graph.add_edge("setup", "planner")
    graph.add_edge("planner", "executor")
    graph.add_edge("executor", "reflector")
    
    # Routing from Reflector
    graph.add_conditional_edges(
        "reflector",
        route_reflection,
        {
            "executor": "executor",
            "planner": "planner",
            "commit": "commit"
        }
    )

    # Logic after Commit: Move to next step (executor) or END
    def route_post_commit(state: AgentState) -> str:
        plan = state.get("implementation_plan", [])
        idx = state.get("current_step_index", 0)
        if idx >= len(plan):
            return "end"
        return "executor"

    graph.add_conditional_edges(
        "commit",
        route_post_commit,
        {
            "executor": "executor",
            "end": "report_generator"
        }
    )

    graph.add_edge("report_generator", END)

    return graph.compile()


