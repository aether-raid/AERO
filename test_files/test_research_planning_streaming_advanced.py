#!/usr/bin/env python3
"""
Test script for the Research Planning Streaming Workflow.

This script demonstrates how to use the streaming version of the plan_research function
from the aero.research_planner module.
"""

import asyncio
import sys
import os
from typing import final
from dotenv import load_dotenv


load_dotenv()
# Add the src directory to the path so we can import aero
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def test_research_planning_streaming():
    """Test the research planning streaming workflow with a sample prompt."""

    try:
        # Import the streaming workflow function
        from aero.research_planner.research_planning_nodes import plan_research

        # Sample prompt for testing
        test_prompt = "Develop a comprehensive research plan for investigating the effectiveness of transformer-based models in natural language processing applications."

        print("🔬 Testing Research Planning Streaming Workflow")
        print("=" * 60)
        print(f"📝 Test Prompt: {test_prompt}")
        print()

        # FOR STREAMING
        final_result = None
        async for result in await plan_research(test_prompt, streaming=True):
            final_result = result  # keep overwriting, so last one wins
            
        result = final_result
        
        
        # FOR NON STREAMING
        # result = await plan_research("Create a research plan for machine learning optimization", streaming=False)
        # print(result.get("research_plan", "No research plan generated"))
        
        #check if we got anything
        if not result:
            print("❌ No results received from streaming workflow")
            return None
        
        
        # Check if successful
        if result.get("current_step") == "finalize_plan":
            print("✅ Research planning completed successfully!")
            
            # Extract the complete research plan
            research_plan_data = result.get("research_plan", {})
            
            # Display the research plan content
            if research_plan_data and isinstance(research_plan_data, dict):
                print("\n📋 FINAL RESEARCH PLAN:")
                print("=" * 80)
                
                # Display selected problem if available
                selected_problem = research_plan_data.get("selected_problem", {})
                if selected_problem:
                    print(f"\n🎯 SELECTED RESEARCH PROBLEM:")
                    print(f"   {selected_problem.get('statement', 'Not available')}")
                
                # Display main research plan
                plan_content = research_plan_data.get("research_plan", "")
                if plan_content:
                    print(f"\n📄 RESEARCH PLAN CONTENT:")
                    print(f"   {plan_content}")
                else:
                    print("\n   No detailed plan content found")
                
                # Display finalization metadata
                print(f"\n📊 FINALIZATION DATA:")
                print(f"   • Finalized at: {research_plan_data.get('finalized_at', 'Not available')}")
                print(f"   • Total iterations: {research_plan_data.get('total_iterations', 0)}")
                print(f"   • Total refinements: {research_plan_data.get('total_refinements', 0)}")
                print(f"   • Final critique score: {research_plan_data.get('final_critique_score', 0.0):.1f}/10")
                print(f"   • Problems generated: {research_plan_data.get('problems_generated', 0)}")
                print(f"   • Web search performed: {research_plan_data.get('web_search_performed', False)}")
                
                print("=" * 80)
            else:
                print("\n❌ No research plan data found in result")
                
        elif result.get("current_step") == "plan_finalized":
            print("✅ Research planning completed successfully!")
            print("\n📋 FINAL RESEARCH PLAN:")
            print("-" * 30)
            plan = (result.get("research_plan", {}).get("research_plan", "") or 
                   result.get("research_plan", "") or
                   "No plan content found")
            print(plan[:1000] + "..." if len(str(plan)) > 1000 else plan)
        else:
            print("❌ Research planning failed")
            
        # Show workflow statistics
        print("\n📊 WORKFLOW STATISTICS:")
        print(f"   - Current step: {result.get('current_step', 'Unknown')}")
        plan_content = result.get("research_plan", {})
        if isinstance(plan_content, dict):
            plan_text = plan_content.get("research_plan", "")
        else:
            plan_text = str(plan_content)
        print(f"   - Plan length: {len(str(plan_text))} characters")
        print(f"   - Problems generated: {len(result.get('generated_problems', []))}")
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
    result = asyncio.run(test_research_planning_streaming())

    if result:
        print("\n🎉 Streaming test completed successfully!")
    else:
        print("\n❌ Streaming test failed!")
        sys.exit(1)
