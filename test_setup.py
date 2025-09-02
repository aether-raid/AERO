#!/usr/bin/env python3
"""
Test script to verify the ML Research Assistant setup
"""

def test_imports():
    """Test that all required modules can be imported."""
    try:
        import openai
        from openai import OpenAI
        import os
        import json
        from datetime import datetime
        print("✅ All required modules imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_api_key():
    """Test if API key is available."""
    import os
    
    # Check environment variable first
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        print("✅ OpenAI API key found in environment variable")
        return True
    
    # Check .env file
    if os.path.exists('.env'):
        try:
            with open('.env', 'r') as f:
                for line in f:
                    if line.strip().startswith('OPENAI_API_KEY='):
                        value = line.split('=', 1)[1].strip()
                        if value.startswith(("'", '"')) and value.endswith(("'", '"')):
                            value = value[1:-1]
                        if value and value != 'your-api-key-here':
                            print("✅ OpenAI API key found in .env file")
                            return True
        except Exception:
            pass
    
    # Check env.example file
    if os.path.exists('env.example'):
        try:
            with open('env.example', 'r') as f:
                for line in f:
                    if line.strip().startswith('OPENAI_API_KEY='):
                        value = line.split('=', 1)[1].strip()
                        if value.startswith(("'", '"')) and value.endswith(("'", '"')):
                            value = value[1:-1]
                        if value and value != 'your-api-key-here':
                            print("✅ OpenAI API key found in env.example file")
                            return True
        except Exception:
            pass
    
    print("⚠️  OpenAI API key not found in any source")
    print("The tool will look for API keys in:")
    print("1. Environment variable: OPENAI_API_KEY")
    print("2. .env file")
    print("3. env.example file")
    print("4. Command line argument: --api-key")
    return False

def test_ml_research_assistant():
    """Test the ML Research Assistant module."""
    try:
        from ml_research_assistant import MLResearchAssistant
        print("✅ ML Research Assistant module loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Error loading ML Research Assistant: {e}")
        return False

def main():
    """Run all tests."""
    print("🔬 Testing ML Research Assistant Setup")
    print("=" * 50)
    
    tests = [
        ("Module Imports", test_imports),
        ("API Key Check", test_api_key),
        ("ML Research Assistant", test_ml_research_assistant)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}:")
        result = test_func()
        results.append(result)
    
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"🎉 All tests passed ({passed}/{total})")
        print("\n✅ Your ML Research Assistant is ready to use!")
        print("\nNext steps:")
        print("1. Set your OpenAI API key: $env:OPENAI_API_KEY='your-key-here'")
        print("2. Run: python ml_research_assistant.py --help")
        print("3. Try: python ml_research_assistant.py --interactive")
    else:
        print(f"⚠️  {passed}/{total} tests passed")
        print("Please fix the issues above before using the tool.")

if __name__ == "__main__":
    main()
