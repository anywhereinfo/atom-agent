import json
from pathlib import Path
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langgraph.prebuilt import create_react_agent

from ..state import AgentState
from ..workspace import Workspace
from ..config import get_llm
from ..prompts.executor_prompts import EXECUTOR_SYSTEM_PROMPT, EXECUTOR_USER_PROMPT
from ..memory import MemoryManager
from ..tools.memory_tools import create_memory_tools
from ..tools.file_tools import create_file_tools
from ..tools.code_tools import create_code_tools
from ..tools.search_tools import create_search_tools

# Stub monitor for pulse tracking
class SimpleMonitor:
    def pulse(self, msg: str):
        print(f"DEBUG MONITOR: {msg}", flush=True)
    def stop(self):
        pass

monitor = SimpleMonitor()

def _clean_message(msg: dict) -> dict:
    """
    Strips internal Gemini metadata, top-level IDs, and content signatures
    to keep history.json clean and token-efficient for agent memory.
    """
    # 1. Strip internal LangChain/Gemini metadata
    if "additional_kwargs" in msg:
        msg["additional_kwargs"].pop("__gemini_function_call_thought_signatures__", None)
        msg["additional_kwargs"].pop("usage_metadata", None)
        
    # 2. Strip top-level IDs and usage metadata (noise)
    msg.pop("id", None)
    msg.pop("usage_metadata", None)
    
    # 3. Strip empty response metadata
    if msg.get("response_metadata") == {}:
        msg.pop("response_metadata", None)
        
    # 4. Strip internal signatures from content parts (common in Gemini)
    content = msg.get("content")
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and "extras" in part:
                 part["extras"].pop("signature", None)
                 if not part["extras"]:
                     part.pop("extras")

    return msg

def _load_executor_tools(current_step: dict, workspace: Workspace) -> list:
    """Returns the list of executable tool instances for the agent, bound to the workspace."""
    file_tools = create_file_tools(workspace)
    code_tools = create_code_tools(workspace)
    
    # Tier 3 Memory Tools
    memory_tools = create_memory_tools(workspace)
    
    # Web Search Tools
    search_tools = create_search_tools()
    
    return file_tools + code_tools + memory_tools + search_tools

def _build_dependency_context(state: AgentState, current_step: dict) -> str:
    """Builds a context string listing committed artifact paths from dependency steps.
    
    This gives the executor explicit knowledge of which files to read_file() from,
    preventing hallucinated or hardcoded content.
    """
    workspace = state.get("workspace")
    if not workspace:
        return "No workspace available — cannot resolve dependency artifacts."

    deps = current_step.get("dependencies", [])
    if not deps:
        return "This step has no dependencies. Generate all content via tools and computation."

    plan = state.get("implementation_plan", [])
    step_map = {s.get("step_id"): s for s in plan}
    task_dir = Path(workspace.task_directory_rel)

    sections = []
    for dep_id in deps:
        dep_step = step_map.get(dep_id)
        if not dep_step:
            sections.append(f"### {dep_id}\n⚠ Dependency step not found in plan.")
            continue

        try:
            committed_dir_template = workspace.get_path("committed_dir", step_id=dep_id)
            committed_path = task_dir / committed_dir_template
        except KeyError:
            sections.append(f"### {dep_id}\n⚠ Cannot resolve committed path.")
            continue

        if not committed_path.exists():
            sections.append(f"### {dep_id}\n⚠ Committed directory does not exist yet (step may not have completed).")
            continue

        # List committed artifacts
        artifacts_dir = committed_path / "artifacts"
        artifact_list = []
        if artifacts_dir.exists():
            for f in sorted(artifacts_dir.iterdir()):
                if f.is_file():
                    rel_path = str(f)
                    artifact_list.append(f"  - `{rel_path}` ({f.stat().st_size} bytes)")
        
        # Also check for committed impl.py / manifest
        for fname in ["manifest.json", "impl.py"]:
            fpath = committed_path / fname
            if fpath.exists():
                artifact_list.append(f"  - `{str(fpath)}` ({fpath.stat().st_size} bytes)")

        section = f"### {dep_id} — \"{dep_step.get('title', 'Untitled')}\"\n"
        section += f"Description: {dep_step.get('description', 'N/A')}\n"
        if artifact_list:
            section += f"Committed artifacts (use read_file to access):\n" + "\n".join(artifact_list)
        else:
            section += "No committed artifacts found."
        
        sections.append(section)

    return "\n\n".join(sections) if sections else "No dependency artifacts resolved."


def _prepare_executor_messages(state: AgentState, current_step: dict) -> list:
    """Constructs tiered messages for the executor agent using the filesystem index."""
    workspace = state.get("workspace")
    task_dir_rel = workspace.task_directory_rel if workspace else "."
    step_id = current_step.get("step_id", "unknown")
    attempt = current_step.ensure_attempt() if hasattr(current_step, 'ensure_attempt') else None
    attempt_id = attempt.attempt_id if attempt else f"a{current_step.get('current_attempt', 1):02d}"

    # 1. Resolve staging paths via attempt workspace
    if attempt and attempt.workspace:
        staging_paths = attempt.workspace.get_staging_paths()
    elif workspace:
        staging_paths = workspace.get_staging_paths(step_id, attempt_id)
    else:
        staging_paths = {}

    # 2. Acceptance criteria
    criteria_list = "\n".join([f"- {c}" for c in current_step.get("acceptance_criteria", [])])

    # 3. Build dependency context (committed artifact paths from dep steps)
    dependency_context = _build_dependency_context(state, current_step)

    # 4. Format the primary User Instruction
    user_msg_content = EXECUTOR_USER_PROMPT.format(
        step_id=step_id,
        attempt_id=attempt_id,
        step_title=current_step.get("title", "Untitled Step"),
        step_description=current_step.get("description", "No description provided."),
        acceptance_criteria=criteria_list,
        staged_impl_path=staging_paths.get("impl", "unknown"),
        staged_test_path=staging_paths.get("test", "unknown"),
        staged_artifacts_dir=staging_paths.get("artifacts_dir", "unknown"),
        dependency_context=dependency_context
    )

    if current_step.get("skill_instructions"):
        inst_map = current_step["skill_instructions"]
        inst_str = "\n".join([f"  - *{skill}*: {text}" for skill, text in inst_map.items()]) if isinstance(inst_map, dict) else inst_map
        user_msg_content += f"\n\nSkill Usage Instructions:\n{inst_str}"

    # 5. Use MemoryManager to reconstruct the tiered context (Tier 0-2)
    # This pulls reports from previous steps (ranked by relevance) and existing history for THIS attempt.
    messages = MemoryManager.reconstruct_context(
        workspace=workspace,
        plan=state.get("implementation_plan", []),
        current_step_idx=state.get("current_step_index", 0),
        system_prompt=EXECUTOR_SYSTEM_PROMPT,
        user_prompt=user_msg_content,
        query=current_step.get("description"), # Explicit query for Cross-Encoder ranking
        top_k=3,
        min_score=0.0 # Stricter threshold for higher precision
    )

    return messages

def _process_executor_result(result: dict, current_step: dict, history_start_idx: int, workspace: Workspace) -> dict:
    """Processes the React agent result into the standard state format and persists history."""
    
    messages = result.get("messages", [])
    if not messages:
        last_msg_content = "No output produced"
    else:
        last_item = messages[-1]
        if hasattr(last_item, 'content'):
            last_msg_content = last_item.content
        elif isinstance(last_item, dict):
            last_msg_content = last_item.get('content', "No content in message dict")
        else:
            last_msg_content = str(last_item)
    
    # Handle potential multimodal content (list of dicts) or non-string content
    if isinstance(last_msg_content, list):
        text_parts = []
        for part in last_msg_content:
            if isinstance(part, dict) and part.get("type") == "text":
                text_parts.append(part.get("text", ""))
            elif isinstance(part, str):
                text_parts.append(part)
        last_msg = " ".join(text_parts)
    else:
        last_msg = str(last_msg_content)

    # Determine if agent is specifically asking for input or just finished
    is_asking = any(word in last_msg.lower() for word in ["please provide", "waiting for", "what is the", "tell me"])
    
    # Persistence according to workspace contract
    task_dir_rel = workspace.task_directory_rel if workspace else "."
    step_id = current_step.get("step_id", "unknown")
    attempt = current_step.current_attempt() if hasattr(current_step, 'current_attempt') else None
    attempt_num = attempt.attempt_number if attempt else current_step.get("current_attempt", 1)
    attempt_id = attempt.attempt_id if attempt else f"a{attempt_num:02d}"

    # Resolve messages dir and staging files via step/attempt workspace
    step_ws = current_step.step_workspace if hasattr(current_step, 'step_workspace') else None
    if step_ws:
        msg_dir_rel = step_ws.get_messages_dir()
    elif workspace:
        msg_dir_rel = workspace.get_path("step_messages_dir", step_id=step_id)
    else:
        msg_dir_rel = f"steps/{step_id}/messages/"

    if attempt and attempt.workspace:
        staging_paths = attempt.workspace.get_staging_paths()
    elif workspace:
        staging_paths = workspace.get_staging_paths(step_id, attempt_id)
    else:
        staging_paths = {}
    impl_path_rel = staging_paths.get("impl", "")
    test_path_rel = staging_paths.get("test", "")
    errors_dir_rel = staging_paths.get("errors_dir", "")

    msg_dir_abs = Path(task_dir_rel) / msg_dir_rel
    impl_path_abs = Path(task_dir_rel) / impl_path_rel
    test_path_abs = Path(task_dir_rel) / test_path_rel
    errors_dir_abs = Path(task_dir_rel) / errors_dir_rel

    # 1. Persist Message History
    try:
        msg_dir_abs.mkdir(parents=True, exist_ok=True)
        history_path = msg_dir_abs / "history.json"
        
        # 1. Load existing history ONLY if we are continuing (attempt > 1)
        existing_history = []
        if attempt_num > 1 and history_path.exists():
            try:
                with open(history_path, "r", encoding="utf-8") as f:
                    existing_history = json.load(f)
            except Exception as e:
                 print(f"DEBUG: Failed to load existing history: {str(e)}", flush=True)
        elif attempt_num == 1:
            print(f"DEBUG: Fresh start for step {step_id}. Overwriting any legacy history.", flush=True)

        # Prepare new messages
        new_messages = []
        for m in messages[history_start_idx:]:
            if hasattr(m, 'model_dump'):
                msg_dict = m.model_dump()
                new_messages.append(_clean_message(msg_dict))
            else:
                msg_dict = {"type": type(m).__name__, "content": str(m.content)}
                new_messages.append(_clean_message(msg_dict))
        
        # Combine and persist (overwrite legacy if it's attempt 1)
        full_history = existing_history + new_messages
        
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(full_history, f, indent=2)
        print(f"DEBUG: Attempt history persisted to {history_path}", flush=True)
        
        # FORCE return the Full persisted history to the state
        # preventing the reflector from seeing only a slice
        serializable_history = full_history

    except Exception as e:
        print(f"DEBUG: Failed to persist attempt history: {str(e)}", flush=True)
        serializable_history = [] # Fallback

    return {
        "awaiting_input": is_asking,
        "full_history": serializable_history
    }

def executor_node(state: AgentState) -> dict:
    print("DEBUG: executor_node entered", flush=True)

    current_step_idx = state["current_step_index"]
    plan = state["implementation_plan"]

    if current_step_idx >= len(plan):
        print("DEBUG: All steps completed. Moving to reflecting phase.", flush=True)
        return {"phase": "reflecting"}

    current_step = plan[current_step_idx]

    workspace = state.get("workspace")
    # Robustly ensure Workspace object
    if isinstance(workspace, dict):
        workspace = Workspace.from_dict(workspace)

    all_tools = _load_executor_tools(current_step, workspace)

    all_messages = _prepare_executor_messages(state, current_step)

    # Initialize LLM for the agent
    llm = get_llm("executor")


    class ToolLogHandler(BaseCallbackHandler):
        def on_llm_start(self, *args, **kwargs):
            print(f"-- LLM Thought", flush=True)
        def on_tool_start(self, serialized, input_str, **kwargs):
            print(f"-- Tool Call: {serialized.get('name')} | Input: {input_str[:100]}...", flush=True)
            monitor.pulse(f"Tool {serialized.get('name')} start")
        def on_tool_end(self, *args, **kwargs):
            print(f"-- LLM Thought Ended", flush=True)

    import time
    start_time = time.time()

    # Retry wrapper for transient network errors (timeouts, connection resets)
    MAX_RETRIES = 3
    RETRY_DELAYS = [30, 60, 120]  # exponential backoff

    dynamic_agent = create_react_agent(llm, all_tools)
    result = None
    last_error = None

    for attempt in range(MAX_RETRIES + 1):
        try:
            result = dynamic_agent.invoke(
                {"messages": all_messages},
                config={"callbacks": [ToolLogHandler()]}
            )
            monitor.stop()
            break  # success
        except Exception as e:
            error_name = type(e).__name__
            is_transient = any(keyword in error_name for keyword in [
                "Timeout", "ReadTimeout", "ConnectTimeout", "ConnectionError",
                "RemoteDisconnected", "ConnectionReset"
            ]) or "timed out" in str(e).lower()

            if is_transient and attempt < MAX_RETRIES:
                delay = RETRY_DELAYS[attempt]
                print(f"⚠️  Transient error ({error_name}), retrying in {delay}s... (attempt {attempt + 1}/{MAX_RETRIES})", flush=True)
                time.sleep(delay)
                all_messages = all_messages  # preserve context
                continue
            else:
                monitor.stop()
                print(f"DEBUG: FATAL ERROR in agent executor: {str(e)}", flush=True)
                raise e

    if result is None:
        monitor.stop()
        raise RuntimeError("Executor failed after all retry attempts")
    
    # 4. Result Processing
    processed = _process_executor_result(result, current_step, len(all_messages), workspace)
    
    return {
        "implementation_plan": plan,
        "current_step_index": current_step_idx,
        "messages": processed["full_history"],
        "awaiting_user_input": processed["awaiting_input"],
        "phase": "reflecting" if not processed["awaiting_input"] else "awaiting_user_input"
    }

