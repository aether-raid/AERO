#!/usr/bin/env python3
"""
Example usage of the ML Research Assistant
"""

from ml_research_assistant import MLResearchAssistant
import os

def main():
    """Example demonstrating how to use the ML Research Assistant programmatically."""
    
    # Check if API key is available
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ Please set your OPENAI_API_KEY environment variable")
        print("Example: $env:OPENAI_API_KEY='your-key-here'")
        return
    
    # Initialize the assistant
    try:
        assistant = MLResearchAssistant()
        print("✅ ML Research Assistant initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        return
    
    # Example research topics
    topics = [
        "Transformer models for time series forecasting",
        "Real-time object detection for autonomous vehicles", 
        "Mathematical foundations of attention mechanisms",
        "Recommendation systems for e-commerce platforms"
    ]
    
    print("\n🔬 Generating example research plans...\n")
    
    for i, topic in enumerate(topics, 1):
        print(f"📝 Example {i}: {topic}")
        
        try:
            # Generate comprehensive research plan
            research_plan = assistant.generate_research_plan(topic)
            
            # Save the plan
            filename = assistant.save_research_plan(research_plan)
            
            print(f"✅ Generated and saved: {filename}")
            print(f"📊 Tokens used: {research_plan['metadata']['tokens_used']}")
            print(f"💰 Cost: ${research_plan['metadata']['cost_estimate']:.4f}")
            print("-" * 50)
            
        except Exception as e:
            print(f"❌ Error generating plan for '{topic}': {e}")
        
        # Add delay between requests to be respectful to the API
        import time
        time.sleep(1)
    
    print("\n🎉 All example research plans generated!")
    print("📁 Check the current directory for the generated files")

if __name__ == "__main__":
    main()
