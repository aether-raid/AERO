"""
Experiment Design Utilities (init_utils.py)
===========================================
This module provides utility functions and configuration for the experiment design tree system, including:
1. Environment and logging setup for LLM and API usage.
2. Asynchronous LLM response handling using OpenAI API.
3. Extraction and structuring of research components (goal, hypotheses, variables, constraints) from user input via LLM.
4. JSON cleaning and parsing utilities for robust LLM output handling.

"""
import os
import json
import re
import asyncio
from openai import AsyncOpenAI
from dotenv import load_dotenv

# --- Load environment variables ---
try:
    load_dotenv('env.example')  # Load from env.example first
    load_dotenv()  # This will override with .env if present
except:
    pass  # dotenv not available, will rely on system environment variables

# Initialize clients and ArXiv processor
primary_client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("BASE_URL", "https://agents.aetherraid.dev")
)

PRIMARY_MODEL = os.getenv("DEFAULT_MODEL", "gpt-4o")
arxiv_processor = None

# --- LLM response ---
async def get_llm_response(messages, temperature=0.2, max_tokens=None):
    """Get LLM response using OpenAI API with cost tracking"""
    
    await asyncio.sleep(0.02)
    
    try:
        kwargs = {"model": PRIMARY_MODEL, "messages": messages, "temperature": temperature}
        if max_tokens:
            kwargs["max_tokens"] = max_tokens

        response = await primary_client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content
        if content is None:
            content = "No response"
                
        return content.strip()
                                    
    except Exception as e:
        return f"Error: API call failed (e: {e})"


# --- Research Plan Understanding ---
async def extract_research_components(user_input):
    """Extract research goal, hypotheses, and relevant information"""
    prompt = f"""
    Extract and structure the following from the research plan:
    - research_goal: Main research objective
    - hypotheses: List of testable hypotheses (as strings)
    - variables: Key independent and dependent variables
    - relevant_info: Supporting information, constraints
    - experiment ideas: Dictionary of potential experiment ideas with brief descriptions/methods

    Returned output should ONLY contain information which have been **EXTRACTED** from the research plan. Return only JSON format.

    Example output format:
    {{
    "research_goal": "...",
    "hypotheses": ["..."],
    "variables": "...",
    "relevant_info": "...",
    "experiment_ideas": [
        {{"name": "...", "details": "..."}},
        {{"name": "...", "details": "..."}}
    ]
    }}
        
    Research Plan: {user_input}
    """
    
    try:
        content = await get_llm_response([
            {"role": "system", "content": "Extract research components. Return only valid JSON with hypotheses as string array."},
            {"role": "user", "content": prompt}
        ], temperature=0.2)
        
        cleaned_content = clean_json_string(content)
        result = json.loads(cleaned_content)
        
        # Ensure all values are strings
        if "hypotheses" in result:
            hypotheses = result["hypotheses"]
            if isinstance(hypotheses, list):
                result["hypotheses"] = [str(h) for h in hypotheses]
            else:
                result["hypotheses"] = [str(hypotheses)]
        
        return result
    except Exception:
        return {"error": "Failed to parse", "hypotheses": [user_input]}

def clean_json_string(text):
    """Clean JSON string by removing control characters and markdown"""
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    return text

if __name__ == "__main__":
    import asyncio

    print("Paste your research plan (end with an empty line):")
    user_input = ""
    while True:
        line = input()
        if line.strip() == "":
            break
        user_input += line + "\n"

    async def test_extract():
        result = await extract_research_components(user_input)
        print("\n--- Extracted Research Components ---")
        print(json.dumps(result, indent=2, ensure_ascii=False))

    asyncio.run(test_extract())
