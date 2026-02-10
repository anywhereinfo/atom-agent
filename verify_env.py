import sys
try:
    import huggingface_hub
    from huggingface_hub import is_offline_mode
    print(f"huggingface_hub: {huggingface_hub.__version__} - OK")
except ImportError as e:
    print(f"huggingface_hub: FAILED - {e}")

try:
    import transformers
    print(f"transformers: {transformers.__version__} - OK")
except ImportError as e:
    print(f"transformers: FAILED - {e}")

try:
    import autogen
    print(f"autogen: OK")
except ImportError as e:
    print(f"autogen: FAILED - {e}")

try:
    import langgraph
    print(f"langgraph: OK")
except ImportError as e:
    print(f"langgraph: FAILED - {e}")

try:
    import crewai
    print(f"crewai: OK")
except ImportError as e:
    print(f"crewai: FAILED - {e}")
