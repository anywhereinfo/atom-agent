import sys
import os
import time
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# Load environment variables from .env
load_dotenv()

def check_pro_rate_limit(num_calls=5):
    print(f"Testing rate limit for gemini-3-pro-preview ({num_calls} serial calls)...")
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ ERROR: GOOGLE_API_KEY not found in environment or .env file.")
        return
    else:
        print(f"✅ GOOGLE_API_KEY found (length: {len(api_key)})")

    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-3-pro-preview",
            temperature=0.1,
            max_retries=0
        )
    except Exception as e:
        print(f"❌ FAILED to initialize LLM")
        print(f"Error: {e}")
        return
    
    for i in range(num_calls):
        start_time = time.time()
        print(f"Call {i+1}/{num_calls}...", end=" ", flush=True)
        try:
            response = llm.invoke([HumanMessage(content="Explain 1+1 in one word.")])
            duration = time.time() - start_time
            content = response.content
            if isinstance(content, list):
                text = " ".join([str(c.get("text", c)) if isinstance(c, dict) else str(c) for c in content])
            else:
                text = str(content)
            print(f"✅ OK ({duration:.2f}s): {text.strip()}")
        except Exception as e:
            duration = time.time() - start_time
            print(f"❌ FAILED ({duration:.2f}s)")
            import traceback
            traceback.print_exc()
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e).upper():
                print("\n🚨 RATE LIMIT DETECTED (429 Resource Exhausted)")
            return

    print("\n🎉 No rate limit hit for these calls.")

if __name__ == "__main__":
    check_pro_rate_limit()
