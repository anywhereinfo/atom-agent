import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from atom_agent.nodes.setup import slugify

test_cases = [
    ("Hello World", "hello-world"),
    ("Framework Maturity & Capabilities!!!", "framework-maturity-capabilities"),
    ("  Too   Many   Hyphens  ---  ", "too-many-hyphens"),
    ("A" * 100, "a" * 48), # Trim test
    ("123-abc_ABC", "123-abc-abc"), # Non-alnum
]

for inp, expected in test_cases:
    result = slugify(inp)
    assert result == expected, f"Failed: '{inp}' -> '{result}' (expected '{expected}')"

print("Slugify tests passed!")
