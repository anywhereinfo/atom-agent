from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Dict, Any

# Centralized Model Configuration
MODEL_CONFIG = {
    "planner": {
        "model": "gemini-3-pro-preview",
        "temperature": 0.1,
    },
    "executor": {
        "model": "gemini-3-pro-preview", # Consistency: use pro for lead execution
        "temperature": 0.4,
    },
    "reflector": {
        "model": "gemini-3-pro-preview", # Flash is good for quick evaluations
        "temperature": 0.1,
    }
}

def get_llm(component_name: str) -> ChatGoogleGenerativeAI:
    """
    Returns an initialized LLM instance for the specified component.
    """
    config = MODEL_CONFIG.get(component_name)
    if not config:
        raise ValueError(f"Unknown component: {component_name}")
    
    return ChatGoogleGenerativeAI(
        model=config["model"],
        temperature=config["temperature"]
    )
