# atom_agent.config Documentation

## Overall Flow
`src/atom_agent/config.py` acts as the centralized configuration hub for Large Language Models (LLMs) used throughout the application. It defines the specific models and parameters (like temperature) for different agent components (planner, executor, reflector).

The flow is simple:
1.  **Configuration Dictionary**: Defines `MODEL_CONFIG`, a dictionary mapping component names to their model settings.
2.  **Factory Function**: `get_llm(component_name)` retrieves the configuration and initializes a `ChatGoogleGenerativeAI` instance.

## Use Cases
-   **Model Switching**: Easily switch all components to a newer model version (e.g., `gemini-1.5-pro` -> `gemini-2.0-flash`) by changing `MODEL_CONFIG`.
-   **Parameter Tuning**: Adjust temperature settings for specific roles (e.g., lower temperature for planner, higher for creative tasks).

## Edge Cases
-   **Unknown Component**: `get_llm` raises a `ValueError` if requested for a component not in `MODEL_CONFIG`.
-   **Missing API Key**: The underlying `ChatGoogleGenerativeAI` initialization will fail if `GOOGLE_API_KEY` is not in the environment (execution time error).

## Method Documentation

### `MODEL_CONFIG` (Variable)
A dictionary mapping component names to their configuration:
-   **Reasoning Tier** (`planner`, `plan_judge`, `reflector`): Uses `gemini-3-pro-preview` for deep analysis and architectural decision-making.
-   **Execution Tier** (`plan_generator`, `executor`): Uses `gemini-3-flash-preview` for speed and implementation tasks.
-   `temperature`: Creativity setting (0.0 - 1.0).

### `get_llm(component_name: str) -> ChatGoogleGenerativeAI`
Factory function to get a configured LLM instance.

-   **Arguments**:
    -   `component_name` (str): The role requiring an LLM (must be a key in `MODEL_CONFIG`).
-   **Returns**: An instance of `ChatGoogleGenerativeAI`.
-   **Raises**: `ValueError` if `component_name` is invalid.
