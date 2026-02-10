from .workspace import Workspace
from pathlib import Path
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage, ToolMessage
from sentence_transformers import CrossEncoder
from typing import List, Dict, Any, Optional
import json

class MemoryManager:
    """
    Manages tiered memory retrieval by using AgentState as an index into the filesystem.
    Includes Cross-Encoder ranking for Tier 2 memory.
    """
    _encoder = None

    @classmethod
    def _get_encoder(cls):
        if cls._encoder is None:
            # Using a lightweight, efficient cross-encoder model
            cls._encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        return cls._encoder

    @staticmethod
    def load_step_history(workspace: Workspace, step_id: str) -> List[BaseMessage]:
        """Loads full message history from the step-level directory."""
        task_dir_rel = workspace.task_directory_rel

        # Use workspace contract for path resolution
        msg_dir_rel = workspace.get_path("step_messages_dir", step_id=step_id)
        history_path = Path(task_dir_rel) / msg_dir_rel / "history.json"

        if not history_path.exists():
            return []

        try:
            with open(history_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            messages = []
            for m in data:
                m_type = m.get("type", "HumanMessage")
                content = m.get("content", "")
                
                if m_type == "HumanMessage":
                    messages.append(HumanMessage(content=content))
                elif m_type == "SystemMessage":
                    messages.append(SystemMessage(content=content))
                elif m_type == "AIMessage":
                    tool_calls = m.get("additional_kwargs", {}).get("tool_calls", [])
                    messages.append(AIMessage(content=content, tool_calls=tool_calls))
                elif m_type == "ToolMessage":
                    messages.append(ToolMessage(content=content, tool_call_id=m.get("tool_call_id", "")))
                else:
                    messages.append(HumanMessage(content=f"[{m_type}] {content}"))
            return messages
        except Exception as e:
            print(f"DEBUG: Error loading history from {history_path}: {str(e)}", flush=True)
            return []

    @staticmethod
    def load_previous_step_reports(workspace: Workspace, plan: List[Dict[str, Any]], current_step_idx: int, query: str, top_k: int = 3, min_score: float = 0.0) -> str:
        """
        Gathers lean reports from successfully completed steps in the 'committed/' folder,
        ranks them using a Cross-Encoder, and returns the top K that meet the min_score.
        """
        task_dir_rel = workspace.task_directory_rel
        reports = []

        for i in range(current_step_idx):
            step = plan[i]
            step_id = step.get("step_id")
            
            committed_dir_template = workspace.get_path("committed_dir", step_id=step_id)
            report_path = Path(task_dir_rel) / committed_dir_template / "report.json"
            
            status = step.get("status", "completed")
            content_str = ""
            
            if report_path.exists():
                try:
                    with open(report_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    
                    content_str = (
                        f"Step: {step_id}\n"
                        f"Description: {data.get('description', step.get('description', ''))}\n"
                        f"Status: {status}\n"
                        f"Final Output: {data.get('output_summary', 'No summary available.')}"
                    )
                except Exception:
                    content_str = f"Step: {step_id}\nDescription: {step.get('description', '')}\nStatus: {status}"
            else:
                content_str = f"Step: {step_id}\nDescription: {step.get('description', '')}\nStatus: {status}"

            reports.append({
                "id": step_id,
                "content": content_str
            })

        if not reports:
            return "No previous step reports available in committed history."

        # Rank and filter reports using Cross-Encoder
        try:
            encoder = MemoryManager._get_encoder()
            # Pairs: (query, report_content)
            pairs = [(query, r["content"]) for r in reports]
            scores = encoder.predict(pairs)
            
            # Filter by score threshold AND sort by score descending
            scored_reports = []
            for score, report in zip(scores, reports):
                if score >= min_score:
                    scored_reports.append((score, report))
            
            # Sort by score descending and take top K
            ranked_reports = [r for _, r in sorted(scored_reports, key=lambda x: x[0], reverse=True)]
            reports = ranked_reports[:top_k]
        except Exception as e:
            print(f"DEBUG: Error during Cross-Encoder ranking: {str(e)}", flush=True)
            # Fallback to last K reports if ranking fails (no thresholding here as we don't have scores)
            reports = reports[-top_k:]

        if not reports:
            return f"No reports met the relevancy threshold of {min_score}."

        formatted_reports = [f"### Historical Context: {r['id']}\n{r['content']}" for r in reports]
        return "\n\n---\n\n".join(formatted_reports)

    @staticmethod
    def reconstruct_context(
        workspace: Workspace, 
        plan: List[Dict[str, Any]], 
        current_step_idx: int,
        system_prompt: str,
        user_prompt: str,
        query: Optional[str] = None,
        top_k: int = 3,
        min_score: float = 0.0
    ) -> List[BaseMessage]:
        """
        Constructs the full message list for the LLM by selectively populating 
        tiers of context from the filesystem with Cross-Encoder ranking and thresholding.
        """
        current_step = plan[current_step_idx]
        step_id = current_step.get("step_id")
        
        # Use step description as default query if none provided
        if not query:
            query = current_step.get("description", user_prompt)

        # 1. System Prompt (Tier 0)
        messages: List[BaseMessage] = [SystemMessage(content=system_prompt)]

        # 2. Gather Global Context (Tier 2: Ranked/Filtered Previous Steps)
        global_context = MemoryManager.load_previous_step_reports(workspace, plan, current_step_idx, query, top_k, min_score)
        context_msg = (
            "## GLOBAL TASK CONTEXT (Ranked Relevant History)\n"
            "Below are summaries of relevant previous steps. If you need to see the full implementation "
            "details, tool calls, or reasoning for any of these steps, use the `get_committed_step_history(step_id)` tool.\n\n"
            f"{global_context}"
        )
        messages.append(SystemMessage(content=context_msg))

        # 3. Load Active Step History (Tier 1: Local Context - aggregated across attempts)
        # CRITICAL: Only load history if we are CONTINUING a strategy (attempt > 1).
        # If it's attempt 1, we start fresh to avoid poisoning by legacy history from failed rollbacks.
        attempt_num = current_step.get("current_attempt", 1)
        if attempt_num > 1:
            step_history = MemoryManager.load_step_history(workspace, step_id)
            if step_history:
                messages.extend(step_history)
        else:
            print(f"DEBUG MEMORY: Fresh start for step {step_id} (Attempt 1). Ignoring legacy history.", flush=True)
            step_history = []

        # 4. Add the Current User Instruction (Tier 0)
        # Only add if it doesn't look like we've already started this conversation
        # (Heuristic: if step_history is empty, we must add user prompt. 
        # If it's not empty, the user prompt might be in there, but usually we append it 
        # to ensure the LLM knows what to do 'now' or if it's a re-prompt.
        # However, typically in RL loops, the user prompt is the 'seed' of the loop.
        # If persistence works correctly, the initial user prompt is saved in history.
        # Let's check for HumanMessage presence.)
        if not any(isinstance(m, HumanMessage) for m in step_history):
            messages.append(HumanMessage(content=user_prompt))

        return messages
