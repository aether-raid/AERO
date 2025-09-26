#!/usr/bin/env python3
"""
Test script for the Model Researcher Streaming Workflow.

This script demonstrates how to use the streaming version of the suggest_models function
from the aero.model_researcher module.
"""

import asyncio
import sys
import os
from typing import final
from dotenv import load_dotenv


load_dotenv()
# Add the src directory to the path so we can import aero
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def test_model_suggestion_streaming():
    """Test the model suggestion streaming workflow with a sample prompt."""

    try:
        # Import the streaming workflow function
        from aero.model_researcher.model_suggestion_nodes import run_model_suggestion_workflow

        # Sample prompt for testing
        test_prompt = "I need help with image classification on medical images. I have a dataset of X-ray images and want to classify them into normal/abnormal categories."

        print("🤖 Testing Model Researcher Streaming Workflow")
        print("=" * 60)
        print(f"📝 Test Prompt: {test_prompt}")
        print()

        # FOR STREAMING
        final_result = None
        async for result in await run_model_suggestion_workflow(test_prompt, streaming=True):
            final_result = result  # keep overwriting, so last one wins
            
        result = final_result
                
       
                
        

        # FOR NON STREAMING
        #result = await run_model_suggestion_workflow("Find models for X-rays", streaming=False)
        
        
        #check if we got anything
        if not result:
            print("❌ No results received from streaming workflow")
            return None
        print(type(result))
    
        
        # Check if successful
        if result.get("model_suggestions", {}).get("suggestions_successful"):
            print("✅ Model suggestions generated successfully!")
            print("\n📋 MODEL SUGGESTIONS:")
            print("-" * 30)
            suggestions = result["model_suggestions"].get("model_suggestions", "")
            print(suggestions[:300] + "..." if len(suggestions) > 300 else suggestions)
        else:
            print("❌ Model suggestions failed")
            
        # Show workflow statistics
        print("\n📊 WORKFLOW STATISTICS:")
        print(f"   - Papers analyzed: {len(result.get('arxiv_results', {}).get('papers', []))}")
        print(f"   - Categories detected: {len(result.get('detected_categories', []))}")
        print(f"   - Search iterations: {result.get('search_iteration', 0)}")
        print(f"   - Suggestion iterations: {result.get('suggestion_iteration', 0)}")

        # Show any errors
        if result.get("errors"):
            print(f"\n⚠️ Errors encountered: {len(result['errors'])}")
            for i, error in enumerate(result["errors"][-3:], 1):  # Show last 3 errors
                print(f"  {i}. {error}")

        return result

    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("Make sure you're running this from the project root directory")
        print("and that all dependencies are installed.")
        return None

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Run the test
    result = asyncio.run(test_model_suggestion_streaming())

    if result:
        print("\n🎉 Streaming test completed successfully!")
    else:
        print("\n❌ Streaming test failed!")
        sys.exit(1)