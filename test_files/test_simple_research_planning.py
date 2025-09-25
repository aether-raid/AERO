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
import sys
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
        print("✅ Library imported successfully!")
        print("\n🚀 Running research planning test...")
        
        # Simple test query
        query = "What are the latest developments in quantum computing applications?"
        
        print(f"📝 Query: {query}")
        
        # Run research planning with streaming support
        print("🔄 Using plan_research with streaming...")
        
        # Import the plan_research function
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'research_planner'))
        from research_planning_nodes import plan_research
        
        # Simply call plan_research with streaming enabled
        result = await plan_research(query, enable_streaming=True)
        
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
        
        # 🎯 RECOMMENDED: Use 'updates' streaming mode
        print("📡 Starting streaming workflow with 'updates' mode...")
        print("=" * 60)
        
        final_result = None
        step_counter = 0
        config = {"recursion_limit": 50}
        
        # Stream with 'updates' mode - shows only changes after each node
        async for chunk in workflow.astream(initial_state, config=config, stream_mode="updates"):
            if chunk:
                step_counter += 1
                
                for node_name, node_updates in chunk.items():
                    print(f"\n� Step {step_counter}: {node_name}")
                    print("-" * 40)
                    
                    # Show key updates based on the node
                    if "current_step" in node_updates:
                        print(f"🔄 Current step: {node_updates['current_step']}")
                    
                    if "current_problem" in node_updates and node_updates["current_problem"]:
                        problem = node_updates["current_problem"]
                        statement = problem.get('statement', 'N/A')
                        print(f"💡 Problem generated: {statement[:100]}{'...' if len(statement) > 100 else ''}")
                    
                    if "validation_results" in node_updates and node_updates["validation_results"]:
                        validation = node_updates["validation_results"]
                        recommendation = validation.get("recommendation", "unknown")
                        print(f"✅ Validation result: {recommendation}")
                        if validation.get("reasoning"):
                            reasoning = validation['reasoning']
                            print(f"   📝 Reason: {reasoning[:150]}{'...' if len(reasoning) > 150 else ''}")
                    
                    if "research_plan" in node_updates and node_updates["research_plan"]:
                        plan = node_updates["research_plan"]
                        title = plan.get('title', 'Untitled')
                        sections = plan.get('sections', [])
                        print(f"📋 Research plan generated: {title}")
                        print(f"   📊 Sections: {len(sections)} parts")
                    
                    if "critique_results" in node_updates and node_updates["critique_results"]:
                        critique = node_updates["critique_results"]
                        score = critique.get("overall_score", 0)
                        recommendation = critique.get("recommendation", "unknown")
                        print(f"📊 Critique score: {score}/10")
                        print(f"📊 Recommendation: {recommendation}")
                    
                    if "errors" in node_updates and node_updates["errors"]:
                        for error in node_updates["errors"]:
                            print(f"❌ Error: {error}")
                    
                    # Store the final result
                    final_result = {**initial_state, **node_updates}
        
        print("\n" + "=" * 60)
        print("🏁 Workflow completed!")
        
        # Use the streamed result
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