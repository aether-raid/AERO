#!/usr/bin/env python3
"""
Test script for the Paper Writing Streaming Workflow.

This script demonstrates how to use the streaming version of the write_paper function
from the aero.report_writer module.
"""

import asyncio
import sys
import os
from typing import final
from dotenv import load_dotenv


load_dotenv()
# Add the src directory to the path so we can import aero
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

async def test_paper_writing_streaming():
    """Test the paper writing streaming workflow with a sample prompt."""

    try:
        # Import the streaming workflow function
        from aero.report_writer.main import write_paper

        # Sample prompt for testing
        test_prompt = "Write a comprehensive paper about machine learning fundamentals, covering supervised learning, unsupervised learning, and deep learning approaches."

        print("📝 Testing Paper Writing Streaming Workflow")
        print("=" * 60)
        print(f"📝 Test Prompt: {test_prompt}")
        print()

        # FOR STREAMING
        final_result = None
        async for result in await write_paper(test_prompt, streaming=True):
            final_result = result  # keep overwriting, so last one wins
            
        result = final_result
        
        
        # FOR NON STREAMING
        # result = await write_paper("Write a short paper about neural networks", streaming=False)
        # print(result.get("formatted_paper", "No paper generated"))
        
        #check if we got anything
        if not result:
            print("❌ No results received from streaming workflow")
            return None
        
        
        # Check if successful
        if result.get("current_step") == "paper_finalized":
            print("✅ Paper writing completed successfully!")
            print("\n📄 FINAL PAPER:")
            print("-" * 30)
            paper = (result.get("formatted_paper", "") or 
                    result.get("final_outputs", {}).get("paper_content", "") or
                    result.get("final_outputs", {}).get("markdown", ""))
            print(paper[:1000] + "..." if len(paper) > 1000 else paper)
        else:
            print("❌ Paper writing failed")
            
        # Show workflow statistics
        print("\n📊 WORKFLOW STATISTICS:")
        print(f"   - Current step: {result.get('current_step', 'Unknown')}")
        print(f"   - Paper length: {len(str(result.get('formatted_paper', '')))} characters")
        print(f"   - Sections generated: {len(result.get('section_content', {}))}")
        print(f"   - Refinement count: {result.get('refinement_count', 0)}")

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
    result = asyncio.run(test_paper_writing_streaming())

    if result:
        print("\n🎉 Streaming test completed successfully!")
    else:
        print("\n❌ Streaming test failed!")
        sys.exit(1)