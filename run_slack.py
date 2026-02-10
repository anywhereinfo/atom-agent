"""
Slack entry point for Atom Agent.

Usage:
    python run_slack.py

Required environment variables:
    GOOGLE_API_KEY     - Google AI API key (for Gemini)
    SLACK_BOT_TOKEN    - Slack Bot User OAuth Token (xoxb-...)
    SLACK_APP_TOKEN    - Slack App-Level Token for Socket Mode (xapp-...)
    SLACK_CHANNEL_ID   - Channel ID to listen in (e.g., C0123456789)
"""

import os
import sys
from dotenv import load_dotenv

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from atom_agent.graph import create_graph
from atom_agent.slack_bot import SlackAtomBot


def build_initial_state(task_description: str) -> dict:
    """Build the initial state dict from a Slack message."""
    return {
        "task_description": task_description,
        "skill_name": None,  # Auto-derived from task description
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
    }


def main():
    load_dotenv()

    # Validate required env vars
    required = {
        "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
        "SLACK_BOT_TOKEN": os.getenv("SLACK_BOT_TOKEN"),
        "SLACK_APP_TOKEN": os.getenv("SLACK_APP_TOKEN"),
        "SLACK_CHANNEL_ID": os.getenv("SLACK_CHANNEL_ID"),
    }

    missing = [k for k, v in required.items() if not v]
    if missing:
        print(f"❌ Missing required environment variables: {', '.join(missing)}")
        print("Set them in a .env file or export them in your terminal.")
        sys.exit(1)

    bot = SlackAtomBot(
        bot_token=required["SLACK_BOT_TOKEN"],
        app_token=required["SLACK_APP_TOKEN"],
        channel_id=required["SLACK_CHANNEL_ID"],
    )

    print("🚀 Starting Atom Agent Slack Bot...")
    print(f"   Channel: {required['SLACK_CHANNEL_ID']}")
    print(f"   Press Ctrl+C to stop\n")

    bot.start(
        create_graph_fn=create_graph,
        build_initial_state_fn=build_initial_state,
    )


if __name__ == "__main__":
    main()
