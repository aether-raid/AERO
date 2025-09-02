#!/usr/bin/env python3
"""
Test script to verify API key reading functionality
"""

from ml_research_assistant import MLResearchAssistant
import os

def test_api_key_reading():
    """Test that the API key reading from env.example works."""
    print("🔍 Testing API Key Reading Functionality")
    print("=" * 50)
    
    # Test the _get_api_key method directly
    try:
        assistant = MLResearchAssistant.__new__(MLResearchAssistant)  # Create instance without calling __init__
        api_key = assistant._get_api_key()
        
        if api_key:
            # Mask the API key for security
            masked_key = api_key[:8] + '*' * (len(api_key) - 12) + api_key[-4:] if len(api_key) > 12 else '*' * len(api_key)
            print(f"✅ API key successfully read: {masked_key}")
            print(f"📏 Key length: {len(api_key)} characters")
            
            # Check which source it came from
            if os.getenv('OPENAI_API_KEY'):
                print("📍 Source: Environment variable")
            elif os.path.exists('.env'):
                print("📍 Source: .env file")
            elif os.path.exists('env.example'):
                print("📍 Source: env.example file")
            else:
                print("📍 Source: Unknown")
            
            return True
        else:
            print("❌ No API key found")
            return False
            
    except Exception as e:
        print(f"❌ Error reading API key: {e}")
        return False

def test_file_reading():
    """Test reading from env.example file specifically."""
    print("\n🔍 Testing env.example File Reading")
    print("=" * 50)
    
    if not os.path.exists('env.example'):
        print("❌ env.example file not found")
        return False
    
    try:
        assistant = MLResearchAssistant.__new__(MLResearchAssistant)
        api_key = assistant._read_api_key_from_file('env.example')
        
        if api_key:
            masked_key = api_key[:8] + '*' * (len(api_key) - 12) + api_key[-4:] if len(api_key) > 12 else '*' * len(api_key)
            print(f"✅ Successfully read from env.example: {masked_key}")
            return True
        else:
            print("❌ Could not read API key from env.example")
            return False
            
    except Exception as e:
        print(f"❌ Error reading env.example: {e}")
        return False

def main():
    """Run API key reading tests."""
    print("🔬 ML Research Assistant - API Key Reading Test")
    print("=" * 60)
    
    test1 = test_api_key_reading()
    test2 = test_file_reading()
    
    print("\n" + "=" * 60)
    print("📊 Test Results:")
    
    if test1 and test2:
        print("🎉 All API key reading tests passed!")
        print("✅ The ML Research Assistant can successfully read API keys from env.example")
    else:
        print("⚠️  Some tests failed")
        print("Please check your env.example file format")

if __name__ == "__main__":
    main()
