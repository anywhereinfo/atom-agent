import os
import sys
from dotenv import load_dotenv

# Add src to path so we can import atom_agent
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from atom_agent.graph import create_graph

def main():
    # Load environment variables (like GOOGLE_API_KEY)
    load_dotenv()
    
    if not os.getenv("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY environment variable not set.")
        print("Please set it in a .env file or export it in your terminal.")
        return

    # Create the graph
    app = create_graph()

    # Initial state
    initial_state = {
        "task_description": "Formal Separation and Empirical Evaluation of Planning Algorithms vs Control Architectures in Autonomous LLM Agents.",
        "skill_name": "Agentic_Framework_Evaluator",
        "available_skills": [
            {
                "name": "file_operations",
                "description": "Read, write, and list files or directories.",
                "capabilities": ["write_file", "read_file", "list_dir"]
            },
            {
                "name": "python_executor",
                "description": "Execute Python code snippets and run pytest suites.",
                "capabilities": ["execute_python_code", "run_pytest"]
            },
            {
                "name": "web_search",
                "description": "Search the web for real-time information and documentation using Tavily.",
                "capabilities": ["tavily_search"]
            },
            {
                "name": "memory_retrieval",
                "description": "Retrieve message history and reports from previously committed tasks/steps.",
                "capabilities": ["get_committed_step_history"]
            }
        ],
        "is_skill_learning_task": True,
        "task_id": "task_001"
    }

    print("\n--- Starting Skill Learning Agent ---\n")
    
    # Run the graph
    # We use stream to see the outputs from each node
    for output in app.stream(initial_state):
        # output is a dict with node names as keys
        for node_name, state_update in output.items():
            print(f"Finished node: {node_name}")
            # The viewer node already prints the plan, so we don't need to do much here

    print("\n--- Agent Execution Finished ---")

if __name__ == "__main__":
    main()
