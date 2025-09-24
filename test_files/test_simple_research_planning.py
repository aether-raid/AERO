#!/usr/bin/env python3
"""
Simple Research Planning Library Test
====================================

Minimal example showing how to use research planning nodes as a library
with proper python-dotenv environment loading.

This demonstrates the clean library usage pattern.
"""

from dotenv import load_dotenv
import os
import asyncio

async def main():
    """Simple test of research planning library with dotenv."""
    
    print("🔧 Loading environment variables...")
    
    # Load .env file into environment variables
    load_dotenv()
    
    # Verify key variables are loaded
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("BASE_URL")
    model = os.getenv("DEFAULT_MODEL")
    
    print(f"✅ OPENAI_API_KEY: {'Loaded' if api_key else 'Missing'}")
    print(f"✅ BASE_URL: {base_url or 'Not set'}")
    print(f"✅ DEFAULT_MODEL: {model or 'Not set'}")
    
    if not api_key:
        print("❌ OPENAI_API_KEY is required! Please check your .env file.")
        return
    
    print("\n📦 Importing research planning library...")
    
    try:
        # Import the library (environment variables are now available)
        from aero.research_planner.research_planning_nodes import plan_research
        
        print("✅ Library imported successfully!")
        print("\n🚀 Running research planning test...")
        
        # Simple test query
        query = "What are the latest developments in quantum computing applications?"
        
        print(f"📝 Query: {query}")
        
        # Run research planning (await since it's async) with debug info
        print("🔄 Calling plan_research with streaming...")
        
        # Import the workflow builder and client initializer
        from aero.research_planner.research_planning_nodes import build_research_planning_graph, _initialize_clients
        from langchain_core.messages import HumanMessage
        
        # Initialize clients first (important!)
        _initialize_clients()
        print("✅ Clients initialized for streaming workflow")
        
        # Initialize state for the workflow
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "original_prompt": query,
            "uploaded_data": [],
            "current_step": "initialize",
            "errors": [],
            "workflow_type": "research_planning",
            "generated_problems": [],
            "validated_problems": [],
            "current_problem": {},
            "validation_results": {},
            "selected_problem": {},
            "research_plan": {},
            "iteration_count": 0,
            "critique_results": {},
            "critique_score_history": [],
            "refinement_count": 0,
            "previous_plans": [],
            "generation_attempts": 0,
            "rejection_feedback": [],
            "auto_validation_enabled": True,
            "web_sources": [],
            "current_web_search_query": ""
        }
        
        # Build the workflow
        workflow = build_research_planning_graph()
        
        # Use streaming mode as per LangGraph documentation
        print("📡 Starting streaming workflow...")
        result = None
        
        config = {"recursion_limit": 50}
        async for chunk in workflow.astream(initial_state, config=config, stream_mode="custom"):
            print(f"🔄 Stream: {chunk}")
            
        # Get final result
        final_result = await workflow.ainvoke(initial_state, config=config)
        result = final_result
        
        print(f"📊 Result type: {type(result)}")
        print(f"📊 Result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
        
        # Check results
        if result and result.get("research_plan"):
            print("\n✅ Research planning completed successfully!")
            print("📄 Research plan was displayed in terminal above")
        elif result and result.get("error"):
            print(f"\n❌ Research planning failed with error: {result.get('error')}")
        else:
            print("\n❌ Research planning failed or returned no results")
            print(f"   Full result: {result}")
            
    except ImportError as e:
        print(f"❌ Failed to import library: {e}")
        print("   Make sure you're in the correct directory")
        
    except Exception as e:
        print(f"❌ Error during execution: {e}")

if __name__ == "__main__":
    asyncio.run(main())