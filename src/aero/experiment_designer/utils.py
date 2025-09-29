import json
import re
import asyncio
from langgraph.config import get_stream_writer
from aero.utils.llm_client import load_openai_client

# --- LLM response ---
async def get_llm_response(messages, temperature=0.2, max_tokens=None):
    """Get LLM response using OpenAI API with cost tracking"""
    # Initialize clients and ArXiv processor
    primary_client, PRIMARY_MODEL = load_openai_client()
    
    await asyncio.sleep(0.02)
    
    try:
        kwargs = {"model": PRIMARY_MODEL, "messages": messages, "temperature": temperature}
        if max_tokens:
            kwargs["max_tokens"] = max_tokens

        response = primary_client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content
        if content is None:
            content = "No response"
                
        return content.strip()
                                    
    except Exception as e:
        return f"Error: API call failed (e: {e})"

async def stream_writer(message, writer=None, stream_mode="update"):
    payload = {"stream_mode": stream_mode, "data": message}
    if writer:
        await writer(payload)
    else:
        default_writer = get_stream_writer()
        if default_writer is not None:
            # Make sure default_writer is awaitable
            result = default_writer(payload)
            if asyncio.iscoroutine(result):
                await result
        else:
            # Optionally print for debugging, or just pass
            print(f"[STREAM] {payload}")

# --- Decorator: Stream step name ---
def stream_step_name(func):
    async def wrapper(state, *args, **kwargs):
        # Get writer from dict or object
        if isinstance(state, dict):
            writer = state.get("writer")
        else:
            writer = getattr(state, "writer", None)
        step_name = func.__name__
        await stream_writer(step_name, writer=writer, stream_mode="update")
        return await func(state, *args, **kwargs)
    return wrapper


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
            {"role": "system", "content": "Extract research components from possibly unstructured or single-paragraph input. Return only valid JSON with hypotheses as string array."},
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