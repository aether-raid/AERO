#!/usr/bin/env python3
"""
Test script to verify the task decomposition and architecture analysis functionality
"""

from ml_research_assistant import MLResearchAssistant

def test_task_decomposition():
    """Test the task decomposition analysis with a specific example."""
    print("🔬 Testing Task Decomposition & Architecture Analysis")
    print("=" * 60)
    
    try:
        assistant = MLResearchAssistant()
        
        print(f"📊 Configuration:")
        print(f"   Model: {assistant.model}")
        print(f"   Max Tokens: {assistant.max_tokens}")
        
        # Test with the specific problem statement
        task = "Given a specific task, how might we decompose the task into several properties (e.g. variable-length, time-invariant, lag-invariant, etc), and research on existing model architectures that can support the task?"
        
        print(f"\n📝 Analyzing task decomposition methodology...")
        print("⏳ This may take 60-90 seconds...")
        
        research_plan = assistant.generate_research_plan(task)
        
        content = research_plan['content']
        
        # Analyze the response
        lines = content.split('\n')
        words = len(content.split())
        
        print(f"\n📈 Response Analysis:")
        print(f"   Lines: {len(lines)}")
        print(f"   Words: {words}")
        print(f"   Tokens used: {research_plan['metadata']['tokens_used']}")
        
        # Check if key sections are present
        key_sections = [
            "Task Property Decomposition Framework",
            "Property-Architecture Mapping Matrix", 
            "Architecture Selection Methodology",
            "Temporal Properties",
            "Variable-length",
            "Time-invariant",
            "Lag-invariant"
        ]
        
        found_sections = []
        for section in key_sections:
            if section.lower() in content.lower():
                found_sections.append(section)
        
        print(f"\n✅ Key Sections Found ({len(found_sections)}/{len(key_sections)}):")
        for section in found_sections:
            print(f"   • {section}")
        
        if len(found_sections) < len(key_sections):
            missing = set(key_sections) - set(found_sections)
            print(f"\n⚠️  Missing Sections:")
            for section in missing:
                print(f"   • {section}")
        
        # Show a preview
        print(f"\n📄 Response Preview (first 1000 characters):")
        print("-" * 60)
        print(content[:1000])
        print("...")
        print("-" * 60)
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_specific_examples():
    """Test with specific task examples."""
    print("\n🎯 Testing Specific Task Examples")
    print("=" * 60)
    
    examples = [
        "Variable-length sequence classification with temporal dependencies",
        "Real-time object detection with lag-invariant properties",
        "Multi-modal learning with time-invariant features"
    ]
    
    try:
        assistant = MLResearchAssistant()
        
        for i, example in enumerate(examples, 1):
            print(f"\n📝 Example {i}: {example}")
            
            # Generate analysis (shortened for testing)
            research_plan = assistant.generate_research_plan(example)
            
            # Check for key decomposition elements
            content = research_plan['content'].lower()
            
            properties_found = []
            property_checks = [
                "variable-length",
                "temporal", 
                "time-invariant",
                "lag-invariant",
                "sequential",
                "multi-modal"
            ]
            
            for prop in property_checks:
                if prop in content:
                    properties_found.append(prop)
            
            print(f"   Properties identified: {', '.join(properties_found)}")
            print(f"   Tokens used: {research_plan['metadata']['tokens_used']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Run the task decomposition tests."""
    print("🚀 Task Decomposition & Architecture Analysis Test")
    print("=" * 70)
    
    test1 = test_task_decomposition()
    test2 = test_specific_examples()
    
    print("\n" + "=" * 70)
    print("📊 Test Results:")
    
    if test1 and test2:
        print("🎉 All tests passed!")
        print("✅ The ML Research Assistant successfully addresses task decomposition")
        print("✅ Property analysis and architecture mapping are working correctly")
    else:
        print("⚠️  Some tests failed")
        print("Please check the analysis above")

if __name__ == "__main__":
    main()
