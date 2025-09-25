#!/usr/bin/env python3
"""
Streaming Research Planning Example
==================================

Demonstrates different streaming modes for the research planning workflow.
"""

from dotenv import load_dotenv
import os
import sys
import asyncio
import json

async def stream_research_planning_updates(query: str):
    """Stream with 'updates' mode - shows only changes after each node."""
    print("\n🔄 STREAMING MODE: 'updates' (recommended)")
    print("="*60)
    
    # Load environment and setup
    load_dotenv()
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'research_planner'))
    
    from research_planning_nodes import build_research_planning_graph, _initialize_clients
    from langchain_core.messages import HumanMessage
    
    _initialize_clients()
    
    # Initialize state
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
    
    workflow = build_research_planning_graph()
    config = {"recursion_limit": 50}
    
    step_counter = 0
    
    async for chunk in workflow.astream(initial_state, config=config, stream_mode="updates"):
        step_counter += 1
        
        # Extract node name and updates
        if chunk:
            for node_name, node_updates in chunk.items():
                print(f"\n📍 Step {step_counter}: {node_name}")
                print("-" * 40)
                
                # Show key updates based on the node
                if "current_step" in node_updates:
                    print(f"🔄 Current step: {node_updates['current_step']}")
                
                if "current_problem" in node_updates and node_updates["current_problem"]:
                    problem = node_updates["current_problem"]
                    print(f"💡 Problem generated: {problem.get('statement', 'N/A')[:100]}...")
                
                if "validation_results" in node_updates and node_updates["validation_results"]:
                    validation = node_updates["validation_results"]
                    recommendation = validation.get("recommendation", "unknown")
                    print(f"✅ Validation: {recommendation}")
                    if validation.get("reasoning"):
                        print(f"   Reason: {validation['reasoning'][:150]}...")
                
                if "research_plan" in node_updates and node_updates["research_plan"]:
                    plan = node_updates["research_plan"]
                    print(f"📋 Research plan: {plan.get('title', 'Untitled')}")
                    sections = plan.get('sections', [])
                    print(f"   Sections: {len(sections)} parts")
                
                if "critique_results" in node_updates and node_updates["critique_results"]:
                    critique = node_updates["critique_results"]
                    score = critique.get("overall_score", 0)
                    print(f"📊 Critique score: {score}/10")
                
                if "errors" in node_updates and node_updates["errors"]:
                    for error in node_updates["errors"]:
                        print(f"❌ Error: {error}")

async def stream_research_planning_values(query: str):
    """Stream with 'values' mode - shows complete state after each node."""
    print("\n🔄 STREAMING MODE: 'values' (complete state)")
    print("="*60)
    
    # Same setup as above...
    load_dotenv()
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'research_planner'))
    
    from research_planning_nodes import build_research_planning_graph, _initialize_clients
    from langchain_core.messages import HumanMessage
    
    _initialize_clients()
    
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
    
    workflow = build_research_planning_graph()
    config = {"recursion_limit": 50}
    
    step_counter = 0
    
    async for chunk in workflow.astream(initial_state, config=config, stream_mode="values"):
        step_counter += 1
        print(f"\n📍 Step {step_counter}: Complete State")
        print("-" * 40)
        
        # Show key state information (not the entire state as it's too verbose)
        current_step = chunk.get("current_step", "unknown")
        iteration = chunk.get("iteration_count", 0)
        problems_count = len(chunk.get("generated_problems", []))
        
        print(f"🔄 Current step: {current_step}")
        print(f"🔄 Iteration: {iteration}")
        print(f"💡 Problems generated: {problems_count}")
        
        if chunk.get("current_problem"):
            statement = chunk["current_problem"].get("statement", "N/A")
            print(f"💡 Current problem: {statement[:100]}...")
        
        if chunk.get("validation_results"):
            validation = chunk["validation_results"]
            print(f"✅ Validation: {validation.get('recommendation', 'unknown')}")
        
        if chunk.get("research_plan"):
            plan = chunk["research_plan"]
            print(f"📋 Plan: {plan.get('title', 'Untitled')}")

async def stream_research_planning_debug(query: str):
    """Stream with 'debug' mode - detailed execution info."""
    print("\n🔄 STREAMING MODE: 'debug' (execution details)")
    print("="*60)
    
    load_dotenv()
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'research_planner'))
    
    from research_planning_nodes import build_research_planning_graph, _initialize_clients
    from langchain_core.messages import HumanMessage
    
    _initialize_clients()
    
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
    
    workflow = build_research_planning_graph()
    config = {"recursion_limit": 50}
    
    async for chunk in workflow.astream(initial_state, config=config, stream_mode="debug"):
        if chunk:
            print(f"🐛 Debug: {chunk['type']} - {chunk.get('step', 'N/A')}")
            if chunk.get('payload'):
                # Show limited payload info to avoid spam
                payload = chunk['payload']
                if isinstance(payload, dict):
                    keys = list(payload.keys())[:5]  # Show first 5 keys
                    print(f"   Payload keys: {keys}")

async def main():
    """Test different streaming modes."""
    query = "What are the challenges in quantum machine learning?"
    
    print("🧪 Testing Different LangGraph Streaming Modes")
    print("=" * 60)
    
    # Test different modes
    modes = [
        ("updates", stream_research_planning_updates),
        ("values", stream_research_planning_values), 
        ("debug", stream_research_planning_debug)
    ]
    
    for mode_name, stream_func in modes:
        try:
            print(f"\n🚀 Testing '{mode_name}' streaming mode...")
            await stream_func(query)
            print(f"✅ '{mode_name}' mode completed successfully")
            break  # Only run one mode for testing
        except Exception as e:
            print(f"❌ '{mode_name}' mode failed: {e}")
            continue

if __name__ == "__main__":
    asyncio.run(main())