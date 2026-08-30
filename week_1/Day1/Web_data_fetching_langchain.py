# 🤖 TWO-STAGE AGENTIC PIPELINE WITH PYDANTIC & LANGCHAIN

# Stage 1: An AI Agent uses web tools to fetch raw web text.
# Stage 2: A Pydantic Structured Model parses the raw text into a strict Python object.

from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.tools import tool

# Imports for web requests
import urllib.request
import urllib.error

from pydantic import BaseModel, Field

# STEP 1: DEFINE PYDANTIC OUTPUT SCHEMA
# Pydantic models define the exact fields, types, and descriptions we expect from the LLM.
class AIEngineeringGuide(BaseModel):
    title: str = Field(description="Title of the article.")
    key_steps: list[str] = Field(description="List of steps to become an AI Engineer.")
    skills_needed: list[str] = Field(description="Key technical skills required to become an AI Engineer.")

# 2 . Web Fetching Tool
def fetch_text_from_url(url: str) -> str:
    """Fetch the document text from a URL."""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; quickstart-research/1.0)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            raw = resp.read()
    except Exception as e:
        return f"Fetch failed: {e}"
        
    text = raw.decode("utf-8", errors="replace")
    return text[:1000]  # Truncate to first 1000 characters for speed

# STEP 2: INITIALIZE THE LLM & STRUCTURED MODEL

# Initialize the base Groq chat model
model = init_chat_model(
    model='openai/gpt-oss-20b',  # Groq model identifier with higher TPM limit
    model_provider='groq',
    temperature=0.5,
    max_tokens=1000
)

# Create structured output model for Stage 2
structured_model = model.with_structured_output(AIEngineeringGuide)

# STEP 3: DEFINE SYSTEM PROMPT & AGENT TOOL
SYSTEM_PROMPT = """You are a helpful data assistant who speaks in bullet points always and very short.
Always use the `fetch_text_from_url` tool to retrieve document text from the given URL before answering."""

# STEP 4: CREATE THE AGENT (Stage 1: Tool calling)
agent = create_agent(
    model=model,
    tools=[fetch_text_from_url],
    system_prompt=SYSTEM_PROMPT
)


if __name__ == "__main__":
    prompt = "Fetch the details of how to become genearative ai engineer . Use this url to get the details https://www.datacamp.com/blog/how-to-become-an-ai-engineer"

    print("[Stage 1] Fetching web page text...")
    # --- STAGE 1: AGENT EXECUTION ---
    agent_result = agent.invoke({
        "messages": [
            {
                'role': 'user',
                'content': prompt
            }
        ]
    })

    # Extract raw text content from the agent's final message
    raw_agent_text = agent_result['messages'][-1].content

    print("[Stage 2] Parsing into Pydantic structured schema...")
    # --- STAGE 2: PYDANTIC PARSING ---
    structured_data: AIEngineeringGuide = structured_model.invoke(raw_agent_text)

    # --- OUTPUT RESULTS ---
    print("\n=================== PYDANTIC STRUCTURED OUTPUT ===================")
    print(f"TITLE: {structured_data.title}\n")
    print("STEPS TO BECOME AN GENERATIVE AI ENGINEER:")
    for step in structured_data.key_steps:
        print(f"  - {step}")
        
    print("\nREQUIRED SKILLS:")
    for skill in structured_data.skills_needed:
        print(f"  - {skill}")
    print("==================================================================")
