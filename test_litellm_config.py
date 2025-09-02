#!/usr/bin/env python3
"""
Test script to verify the LiteLLM proxy configuration
"""

from ml_research_assistant import MLResearchAssistant
import os

def test_configuration():
    """Test the LiteLLM proxy configuration."""
    print("🔬 Testing LiteLLM Proxy Configuration")
    print("=" * 50)
    
    try:
        # Initialize the assistant
        assistant = MLResearchAssistant()
        
        print(f"✅ Successfully initialized ML Research Assistant")
        print(f"📍 API Key: {assistant.api_key[:8]}***{assistant.api_key[-4:] if len(assistant.api_key) > 12 else '***'}")
        print(f"🌐 Base URL: {assistant.base_url}")
        print(f"🤖 Model: {assistant.model}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error initializing assistant: {e}")
        return False

def test_simple_request():
    """Test a simple request to verify the connection works."""
    print("\n🔍 Testing Simple Request")
    print("=" * 50)
    
    try:
        assistant = MLResearchAssistant()
        
        # Simple test message
        test_content = "Hello, how are you?"
        
        print(f"📤 Sending test message: '{test_content}'")
        
        response = assistant.client.chat.completions.create(
            model=assistant.model,
            messages=[{"content": test_content, "role": "user"}],
            max_tokens=100
        )
        
        response_content = response.choices[0].message.content
        print(f"📥 Response: {response_content}")
        
        if hasattr(response, 'usage') and response.usage:
            print(f"📊 Tokens used: {response.usage.total_tokens}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error with test request: {e}")
        return False

def main():
    """Run configuration tests."""
    print("🚀 LiteLLM Proxy Configuration Test")
    print("=" * 60)
    
    test1 = test_configuration()
    test2 = test_simple_request() if test1 else False
    
    print("\n" + "=" * 60)
    print("📊 Test Results:")
    
    if test1 and test2:
        print("🎉 All tests passed!")
        print("✅ Your LiteLLM proxy configuration is working correctly")
        print("🚀 Ready to generate ML research plans!")
    elif test1:
        print("⚠️  Configuration loaded but API test failed")
        print("Check your API key and proxy connection")
    else:
        print("❌ Configuration test failed")
        print("Please check your env.example file")

if __name__ == "__main__":
    main()
