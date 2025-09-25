#!/usr/bin/env python3
"""
Fixed Research Planning Test with Streaming
==========================================

This tests the fixed research planning workflow with proper async functions
and LangGraph streaming.
"""

from dotenv import load_dotenv
import os
import sys
import asyncio

async def main():
    """Test the fixed research planning workflow."""
    
    print("🔧 Loading environment variables...")
    load_dotenv()
    
    # Verify key variables are loaded
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("BASE_URL")
    
    print(f"✅ OPENAI_API_KEY: {'Loaded' if api_key else 'Missing'}")
    print(f"✅ BASE_URL: {base_url or 'Not set'}")
    
    if not api_key:
        print("❌ OPENAI_API_KEY is required! Please check your .env file.")
        return
    
    print("\n📦 Importing research planning library...")
    
    try:
        # Import the library
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'research_planner'))
        from aero.research_planner.research_planning_nodes import plan_research
        
        print("✅ Library imported successfully!")
        
        # Test query
        query = "Improving interpretability of deep neural networks for medical diagnosis"
        print(f"\n📝 Query: {query}")
        
        # Simply use the plan_research function with streaming enabled
        print("\n🚀 Running research planning with streaming...")
        
        result = await plan_research(query, enable_streaming=True)
        
        # Check results
        if result and result.get("research_plan"):
            print("\n✅ Research planning completed successfully!")
            plan = result["research_plan"]
            print(f"� Title: {plan.get('title', 'N/A')}")
            sections = plan.get('sections', [])
            print(f"📊 Sections: {len(sections)} parts")
            
            if result.get("display_method") == "terminal_output":
                print("� Research plan was displayed above in terminal")
        elif result and result.get("error"):
            print(f"\n❌ Research planning failed with error: {result.get('error')}")
        else:
            print("\n❌ Research planning failed or returned no results")
            print(f"   Full result keys: {list(result.keys()) if result else 'None'}")
        
    except ImportError as e:
        print(f"❌ Failed to import library: {e}")
        print("   Make sure you're in the correct directory")
        
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())