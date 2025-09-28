import os
import openai

def load_openai_client():
    # Load configuration
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables. Please ensure it is set.")

    base_url = os.getenv("BASE_URL")
    model = os.getenv("DEFAULT_MODEL") or "gemini/gemini-2.5-flash"
    model_cheap = "gemini/gemini-2.5-flash-lite"
    model_expensive = "gemini/gemini-2.5-pro"

    # Initialize dependencies
    try:
        client = openai.OpenAI(api_key=api_key, base_url=base_url)
        return client, model
    except Exception as e:
        raise ValueError(f"Failed to initialize OpenAI client: {str(e)}. Please check your API key and base URL configuration.")
    