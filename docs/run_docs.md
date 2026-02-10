# run.py Documentation

## Overall Flow
`run.py` is the entry point for the Atom Agent application. It sets up the environment, initializes the agent graph, defines the initial state, and runs the agent loop.

The flow is as follows:
1.  **Environment Setup**: Loads environment variables using `python-dotenv` and adds the `src` directory to the system path.
2.  **Validation**: Checks for the existence of `GOOGLE_API_KEY`.
3.  **Graph Creation**: Calls `create_graph()` from `atom_agent.graph` to build the LangGraph application.
4.  **Initial State Definition**: constructs the `initial_state` dictionary, including:
    -   `task_description`: The main goal of the agent (currently hardcoded for evaluation).
    -   `skill_name`: The context of the skill being learned/evaluated.
    -   `available_skills`: A list of predefined skills (file_operations, python_executor, web_search, memory_retrieval).
    -   `is_skill_learning_task`: Boolean flag.
    -   `task_id`: Unique identifier for the task.
5.  **Execution Loop**: Iterates through the output of `app.stream(initial_state)`, printing the name of each finished node.

## Use Cases
-   **Starting the Agent**: This is the script you run to start the agentic workflow.
    -   Command: `python run.py`
-   **Configuration**: You can modify the `initial_state` dictionary to change the task description or available skills.

## Edge Cases
-   **Missing API Key**: If `GOOGLE_API_KEY` is not set, the script prints an error message and exits immediately.
-   **Import Errors**: Relies on `src` being correctly structured relative to `run.py`.

## Method Documentation

### `main()`
The primary function of the script.

-   **Description**: Orchestrates the startup and execution of the agent.
-   **Arguments**: None.
-   **Returns**: None.
-   **Logic**:
    -   Loads .env.
    -   Checks API key.
    -   Builds graph via `create_graph()`.
    -   Defines `initial_state`.
    -   Streams execution output to console.
