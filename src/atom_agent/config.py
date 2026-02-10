from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Dict, Any
import time

# Centralized Model Configuration
# Pro: deep reasoning (planning, judging, reflection)
# Flash: fast execution (candidate generation, code implementation)
MODEL_CONFIG = {
    "planner": {
        "model": "gemini-3-pro-preview",
        "temperature": 0.1,
    },
    "plan_generator": {
        "model": "gemini-3-flash-preview",
        "temperature": 0.7,
    },
    "plan_judge": {
        "model": "gemini-3-pro-preview",
        "temperature": 0.0,
    },
    "executor": {
        "model": "gemini-3-flash-preview",
        "temperature": 0.4,
    },
    "reflector": {
        "model": "gemini-3-pro-preview",
        "temperature": 0.1,
    }
}

# Rate Limiting Configuration (reactive)
_call_counter = 0
_last_call_time = 0.0
_backoff_seconds = 0.0  # Grows only after hitting rate limits
_BASE_DELAY = 0.1       # 100ms between calls (always)
_BACKOFF_STEP = 10      # Seconds added per consecutive rate limit hit
_MAX_BACKOFF = 120      # Cap backoff at 2 minutes

def _apply_rate_limit():
    """Applies a small delay between LLM calls. Escalates if rate limited."""
    global _call_counter, _last_call_time, _backoff_seconds

    _call_counter += 1
    now = time.time()
    elapsed = now - _last_call_time

    total_delay = _BASE_DELAY + _backoff_seconds
    if elapsed < total_delay and _last_call_time > 0:
        wait = total_delay - elapsed
        if _backoff_seconds > 0:
            print(f"DEBUG: Rate limiter — backoff active, waiting {wait:.1f}s (call #{_call_counter})...", flush=True)
        time.sleep(wait)

    _last_call_time = time.time()

def on_rate_limit_hit():
    """Call this when a 429/RESOURCE_EXHAUSTED error is received."""
    global _backoff_seconds
    _backoff_seconds = min(_backoff_seconds + _BACKOFF_STEP, _MAX_BACKOFF)
    print(f"⚠️  Rate limit hit! Backoff increased to {_backoff_seconds}s", flush=True)

def on_rate_limit_clear():
    """Call this when a successful LLM response is received."""
    global _backoff_seconds
    if _backoff_seconds > 0:
        _backoff_seconds = max(0, _backoff_seconds - _BACKOFF_STEP)
        print(f"✅ Rate limit clearing. Backoff reduced to {_backoff_seconds}s", flush=True)

def get_llm(component_name: str) -> ChatGoogleGenerativeAI:
    """
    Returns an initialized LLM instance for the specified component.
    Applies rate limiting to prevent quota exhaustion.
    """
    _apply_rate_limit()

    config = MODEL_CONFIG.get(component_name)
    if not config:
        raise ValueError(f"Unknown component: {component_name}")
    
    return ChatGoogleGenerativeAI(
        model=config["model"],
        temperature=config["temperature"],
        max_retries=6,
        request_timeout=600
    )
