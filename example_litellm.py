#!/usr/bin/env python3
"""
Example demonstrating the LiteLLM proxy setup similar to your code snippet
"""

import openai
from openai import OpenAI
from ml_research_assistant import MLResearchAssistant

def simple_example():
    """Simple example matching your provided code structure."""
    print("🚀 Simple LiteLLM Proxy Example")
    print("=" * 40)
    
    # Your configuration (loaded from env.example)
    model = "gemini/gemini-2.5-flash"  # dynamically selected by user
    api_key = "sk-CXR-oonreHJF5sp6sOZmTw"  # loaded from env
    base_url = "https://agents.aetherraid.dev"  # loaded from config
    content = "Hello, how are you?"  # input from user
    
    # Create OpenAI client exactly like your example
    client = openai.OpenAI(
        api_key=api_key,
        base_url=base_url
    )
    
    # Request sent to model set on litellm proxy
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"content": content, "role": "user"}]
        )
        
        print(f"📤 Sent: {content}")
        print(f"📥 Response: {response.choices[0].message.content}")
        
        if hasattr(response, 'usage') and response.usage:
            print(f"📊 Tokens: {response.usage.total_tokens}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def ml_research_example():
    """Example using the ML Research Assistant with your configuration."""
    print("\n🔬 ML Research Assistant Example")
    print("=" * 40)
    
    try:
        # Initialize with your configuration (automatically loads from env.example)
        assistant = MLResearchAssistant()
        
        print(f"🤖 Using model: {assistant.model}")
        print(f"🌐 Base URL: {assistant.base_url}")
        print(f"🔑 API Key: {assistant.api_key[:8]}***{assistant.api_key[-4:]}")
        
        # Generate a quick research plan
        topic = "Graph neural networks for beginners"
        print(f"\n📝 Generating comprehensive research plan for: '{topic}'")
        
        research_plan = assistant.generate_research_plan(topic)
        
        # Show just the first part of the plan
        lines = research_plan['content'].split('\n')[:10]
        preview = '\n'.join(lines)
        
        print(f"\n📄 Research Plan Preview:")
        print("-" * 40)
        print(preview)
        print("...")
        
        print(f"\n📊 Metadata:")
        print(f"   Model: {research_plan['metadata']['model']}")
        print(f"   Base URL: {research_plan['metadata']['base_url']}")
        print(f"   Tokens: {research_plan['metadata']['tokens_used']}")
        print(f"   Cost: ${research_plan['metadata']['cost_estimate']:.4f}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    """Run both examples."""
    print("🎯 LiteLLM Proxy Examples")
    print("=" * 50)
    
    # Example 1: Direct OpenAI client usage (your code style)
    simple_example()
    
    # Example 2: ML Research Assistant usage
    ml_research_example()
    
    print("\n✅ Examples completed!")

if __name__ == "__main__":
    main()
