#!/usr/bin/env python3
"""
Test script to verify long responses don't get cut off
"""

from ml_research_assistant import MLResearchAssistant

def test_long_response():
    """Test that comprehensive research plans don't get cut off."""
    print("🔬 Testing Long Response Generation")
    print("=" * 50)
    
    try:
        assistant = MLResearchAssistant()
        
        print(f"📊 Configuration:")
        print(f"   Model: {assistant.model}")
        print(f"   Max Tokens: {assistant.max_tokens}")
        print(f"   Base URL: {assistant.base_url}")
        
        # Test with a topic that should generate a long response
        topic = "Deep learning for natural language processing"
        
        print(f"\n📝 Generating comprehensive research plan for: '{topic}'")
        print("⏳ This may take 60-90 seconds due to increased token limit...")
        
        research_plan = assistant.generate_research_plan(topic)
        
        content = research_plan['content']
        
        # Analyze the response
        lines = content.split('\n')
        words = len(content.split())
        characters = len(content)
        
        print(f"\n📈 Response Analysis:")
        print(f"   Lines: {len(lines)}")
        print(f"   Words: {words}")
        print(f"   Characters: {characters}")
        print(f"   Tokens used: {research_plan['metadata']['tokens_used']}")
        print(f"   Max tokens: {assistant.max_tokens}")
        
        # Check if response was cut off
        if research_plan['metadata']['tokens_used'] >= assistant.max_tokens * 0.95:
            print("⚠️  Response may have been cut off (using >95% of max tokens)")
        else:
            print("✅ Response appears complete")
        
        # Check if it ends properly
        if content.strip().endswith(('...', 'etc.', 'and so on')):
            print("⚠️  Response may have been cut off (ends with truncation indicators)")
        else:
            print("✅ Response ends properly")
        
        # Show a preview of the end
        print(f"\n📄 Response Preview (last 500 characters):")
        print("-" * 50)
        print(content[-500:])
        print("-" * 50)
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Run the long response test."""
    print("🚀 Long Response Test")
    print("=" * 60)
    
    success = test_long_response()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 Test completed!")
        print("Check the analysis above to see if responses are being cut off.")
    else:
        print("❌ Test failed!")

if __name__ == "__main__":
    main()
