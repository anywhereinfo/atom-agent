"""
Slack integration for Atom Agent using Socket Mode.

Listens to messages in a configured channel and triggers agent runs,
posting progress updates and results as thread replies.
"""

import os
import threading
import time
import traceback
from typing import Optional, Callable

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler


class SlackAtomBot:
    """Slack bot that listens in a channel and triggers Atom agent runs."""

    def __init__(self, bot_token: str, app_token: str, channel_id: str):
        self.channel_id = channel_id
        self.app_token = app_token
        self._active_runs = {}  # thread_ts -> threading.Thread

        # Initialize Slack Bolt app
        self.app = App(token=bot_token)

        # Register event listener
        self.app.event("message")(self._handle_message)

        # Store the graph factory (set via .start())
        self._create_graph = None
        self._build_initial_state = None

    def start(self, create_graph_fn, build_initial_state_fn):
        """
        Start listening for messages.

        Args:
            create_graph_fn: Callable that returns a compiled LangGraph app
            build_initial_state_fn: Callable(task_description: str) -> dict
                                    that builds the initial_state for the graph
        """
        self._create_graph = create_graph_fn
        self._build_initial_state = build_initial_state_fn

        print(f"🤖 Atom Agent Slack Bot starting...", flush=True)
        print(f"📡 Listening in channel: {self.channel_id}", flush=True)

        handler = SocketModeHandler(self.app, self.app_token)
        handler.start()  # Blocks until Ctrl+C

    def _handle_message(self, event, say, client):
        """Handle incoming messages in the target channel."""
        # Filter: only process messages in our channel
        if event.get("channel") != self.channel_id:
            return

        # Ignore bot messages, edits, deletes, thread replies
        if event.get("bot_id"):
            return
        if event.get("subtype") in ("message_changed", "message_deleted", "bot_message"):
            return
        if event.get("thread_ts"):
            return  # Ignore thread replies — only top-level messages trigger runs

        text = event.get("text", "").strip()
        if not text:
            return

        user_id = event.get("user", "unknown")
        message_ts = event.get("ts")

        print(f"\n{'═' * 60}", flush=True)
        print(f"📩 New task from <@{user_id}>: {text[:100]}...", flush=True)
        print(f"{'═' * 60}", flush=True)

        # Acknowledge the message (try/except for missing permissions)
        try:
            client.reactions_add(
                channel=self.channel_id,
                timestamp=message_ts,
                name="brain",  # 🧠 emoji
            )
        except Exception as e:
            print(f"⚠️  Could not add reaction (check 'reactions:write' scope): {e}", flush=True)

        # Post initial reply in a thread
        reply = client.chat_postMessage(
            channel=self.channel_id,
            thread_ts=message_ts,
            text="🧠 *Agent starting...*\nI'll post progress updates in this thread.",
        )

        # Run the agent in a background thread
        thread = threading.Thread(
            target=self._run_agent,
            args=(text, message_ts, client),
            daemon=True,
        )
        thread.start()
        self._active_runs[message_ts] = thread

    def _post_thread(self, client, thread_ts: str, text: str):
        """Post a message as a thread reply."""
        try:
            client.chat_postMessage(
                channel=self.channel_id,
                thread_ts=thread_ts,
                text=text,
            )
        except Exception as e:
            print(f"⚠️  Failed to post to Slack: {e}", flush=True)

    def _run_agent(self, task_description: str, thread_ts: str, client):
        """Execute the full agent pipeline in a background thread."""
        try:
            # Build the graph and initial state
            app = self._create_graph()
            initial_state = self._build_initial_state(task_description)

            self._post_thread(client, thread_ts, f"📋 *Task:* {task_description}")

            # Stream graph execution and post progress
            last_node = None
            for output in app.stream(initial_state):
                for node_name, state_update in output.items():
                    last_node = node_name
                    self._post_node_update(client, thread_ts, node_name, state_update)

            # Final success message (try/except for missing permissions)
            try:
                client.reactions_add(
                    channel=self.channel_id,
                    timestamp=thread_ts,
                    name="white_check_mark",  # ✅
                )
            except Exception:
                pass  # Ignore reaction errors
            
            self._post_thread(client, thread_ts, "✅ *Agent run complete!*")

        except Exception as e:
            error_msg = f"❌ *Agent failed:*\n```{str(e)[:500]}```"
            self._post_thread(client, thread_ts, error_msg)
            print(f"ERROR in agent run: {traceback.format_exc()}", flush=True)

            # Add failure reaction (try/except for missing permissions)
            try:
                client.reactions_add(
                    channel=self.channel_id,
                    timestamp=thread_ts,
                    name="x",  # ❌
                )
            except Exception:
                pass

        finally:
            self._active_runs.pop(thread_ts, None)

    def _post_node_update(self, client, thread_ts: str, node_name: str, state_update: dict):
        """Post a progress update based on which graph node just finished."""

        if node_name == "setup":
            task_ctx = state_update.get("task_context", {})
            task_id = task_ctx.get("task_id", "unknown")
            self._post_thread(client, thread_ts,
                f"⚙️ *Setup complete*\nTask ID: `{task_id}`")

        elif node_name == "planner":
            plan = state_update.get("implementation_plan", [])
            step_count = len(plan)
            step_list = "\n".join(
                f"  {i+1}. {s.get('title', s.get('step_id', '?'))} "
                f"[{s.get('estimated_complexity', '?')}]"
                for i, s in enumerate(plan)
            )
            self._post_thread(client, thread_ts,
                f"🧠 *Plan generated* — {step_count} steps:\n{step_list}")

        elif node_name == "executor":
            idx = state_update.get("current_step_index")
            plan = state_update.get("implementation_plan", [])
            if plan and idx is not None and 0 <= idx < len(plan):
                step = plan[idx]
                title = step.get("title", step.get("step_id", "?"))
                attempt = step.get("current_attempt", 1)
                self._post_thread(client, thread_ts,
                    f"⚙️ *Executing:* {title} (attempt {attempt})")

        elif node_name == "reflector":
            review = state_update.get("reflector_review", {})
            decision = review.get("decision", "?")
            score = review.get("score", "?")
            emoji = {"proceed": "✅", "refine": "🔄", "rollback": "⛔"}.get(decision, "❓")
            self._post_thread(client, thread_ts,
                f"{emoji} *Reflector:* {decision} (score: {score})")

        elif node_name == "commit":
            self._post_thread(client, thread_ts, "💾 *Step committed*")

        elif node_name == "report_generator":
            # Extract report path from progress reports
            progress = state_update.get("progress_reports", [])
            report_path = None
            for p in progress:
                # Look for "Final report generated: <path>"
                if "Final report generated:" in p:
                    try:
                        report_path = p.split("Final report generated:")[1].strip()
                    except IndexError:
                        pass
            
            if report_path and os.path.exists(report_path):
                self._post_thread(client, thread_ts, f"📄 *Final report generated:* `{report_path}`\nUploading artifact...")
                self._upload_file(client, thread_ts, report_path)
            else:
                self._post_thread(client, thread_ts, "📄 *Final report generated* (path not found)")

    def _upload_file(self, client, thread_ts, file_path):
        """Upload a file to the Slack thread."""
        try:
            client.files_upload_v2(
                channel=self.channel_id,
                thread_ts=thread_ts,
                file=file_path,
                title=os.path.basename(file_path),
                initial_comment="Here is your requested report 📊"
            )
        except Exception as e:
            print(f"⚠️  Failed to upload file to Slack (check 'files:write' scope): {e}", flush=True)
            self._post_thread(client, thread_ts, f"⚠️  Could not upload report artifact (missing permissions?)\nPath: `{file_path}`")
