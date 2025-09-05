"""
Test script for the web search validation functionality
"""
import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import our ML researcher module
from ml_researcher_langgraph import MLResearchWorkflow

async def test_web_search_validation():
    """Test the web search validation functionality"""
    
    # Initialize the workflow
    workflow = MLResearchWorkflow()
    
    # Test problem that should have existing research
    test_problem = "Using transformers for image classification"
    
    print(f"Testing web search validation for: {test_problem}")
    print("-" * 50)
    
    # Create a minimal state for testing
    test_state = {
        "problem_statement": test_problem,
        "context": "",
        "research_plan": [],
        "summary": "",
        "next_workflow": "research_planning"
    }
    
    try:
        # Test the validation node directly
        result = await workflow._validate_problem_node(test_state)
        
        print(f"Validation Result:")
        print(f"Problem Statement: {result.get('problem_statement', 'N/A')}")
        print(f"Context: {result.get('context', 'N/A')[:200]}...")
        print(f"Next Workflow: {result.get('next_workflow', 'N/A')}")
        
        return result
        
    except Exception as e:
        print(f"Error during validation: {e}")
        return None

if __name__ == "__main__":
    # Run the test
    result = asyncio.run(test_web_search_validation())
    
    if result:
        print("\n✅ Web search validation test completed successfully!")
    else:
        print("\n❌ Web search validation test failed!")
