#!/usr/bin/env python3
"""
Research Planning Workflow Nodes
================================

This module contains all the workflow nodes and functions specifically for the 
research planning workflow, extracted from ml_researcher_langgraph.py.

The research planning workflow includes:
- Problem generation and validation
- Research plan creation and structuring  
- Plan critique and iterative refinement
- Document generation utilities

These functions are designed to be imported and used by the main MLResearcherLangGraph class.
"""

import os
import sys
import json
import re
import math
import asyncio
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, TypedDict, Annotated
import urllib.request as libreq
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.config import get_stream_writer
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

# LLM and related imports
import openai

# Web search imports
from tavily import TavilyClient

class BaseState(TypedDict):
     # Clients (stored in state)
    client: Optional[Any]                     # OpenAI client instance
    tavily_client: Optional[Any]              # Tavily client instance
    model: str    
    """Base state for all workflows."""
    messages: Annotated[List[BaseMessage], add_messages]
    original_prompt: str  # Pure user query without uploaded data
    uploaded_data: List[str]  # Uploaded file contents as separate field
    current_step: str
    errors: List[str]

# Node function imports
from .critique import _critique_plan_node, _finalize_plan_node
from .generate_problem import _generate_problem_node
from .plan_generation import _create_research_plan_node
from .rejection_processing import  _process_rejection_feedback_node
from .validate import _validate_problem_node

# Suppress warnings
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import logging
logging.getLogger("openai._base_client").setLevel(logging.WARNING)


# ==================================================================================
# STREAMWRITER HELPER FUNCTION
# ==================================================================================

def _write_stream(message: str, key: str = "status", progress: int = None):
    """Helper function to write to StreamWriter if available."""
    try:
        # Use LangGraph's get_stream_writer() without parameters (proper way)
        writer = get_stream_writer()
        writer({key: message})
    except Exception:
        # Fallback: try to get stream from config (for testing compatibility)
        try:
            # This fallback is for test compatibility only
            import inspect
            frame = inspect.currentframe()
            while frame:
                if 'config' in frame.f_locals and frame.f_locals['config']:
                    config = frame.f_locals['config']
                    stream = config.get("configurable", {}).get("stream")
                    if stream and hasattr(stream, 'write'):
                        stream.write(message)
                        return
                frame = frame.f_back
        except Exception:
            pass
        # Final fallback: silently fail
        pass



# ==================================================================================
# STATE DEFINITIONS
# ==================================================================================

class ResearchPlanningState(BaseState):
    """State object for the research planning workflow."""
    # Clients (stored in state)
    client: Optional[Any]                        # OpenAI client instance
    tavily_client: Optional[Any]                 # Tavily client instance
    model: str                                   # Model name to use

    # Core workflow data
    generated_problems: List[Dict[str, Any]]     # All generated problem statements
    validated_problems: List[Dict[str, Any]]     # Problems verified as unsolved
    current_problem: Dict[str, Any]              # Currently being validated
    validation_results: Dict[str, Any]           # Web search and validation results
    selected_problem: Dict[str, Any]             # User-selected problem for detailed planning
    research_plan: Dict[str, Any]                # Final research plan
    iteration_count: int                         # Track number of iterations
    critique_results: Dict[str, Any]             # Critique agent feedback
    critique_score_history: List[float]          # Track score improvements
    refinement_count: int                        # Number of refinements attempted
    previous_plans: List[Dict[str, Any]]         # Store previous plan versions
    
    # 🔧 STREAMLINED WORKFLOW: Enhanced state tracking
    generation_attempts: int                     # Track total problem generation attempts
    rejection_feedback: List[str]                # Track why problems were rejected
    auto_validation_enabled: bool                # Enable automatic validation flow
    web_sources: List[Dict[str, Any]]           # Web search sources for validation
    current_web_search_query: str               # Current search query being used

# ==================================================================================
# CLIENT INITIALIZATION FUNCTIONS
# ==================================================================================

def _initialize_clients_in_state(state: ResearchPlanningState) -> ResearchPlanningState:
    """Initialize clients and store them in state for workflow use."""
    try:
        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("BASE_URL")
        model = os.getenv("MODEL") or "gemini/gemini-2.5-flash"
        
        if api_key:
            import openai
            client = openai.OpenAI(
                api_key=api_key,
                base_url=base_url
            )
            state["client"] = client
            state["model"] = model
            _write_stream("AI model connected successfully")
        else:
            state["errors"] = state.get("errors", []) + ["OpenAI API key not configured"]
            _write_stream("AI model configuration missing")
            
        # Initialize Tavily client
        tavily_key = os.getenv("TAVILY_API_KEY")
        if tavily_key:
            tavily_client = TavilyClient(api_key=tavily_key)
            state["tavily_client"] = tavily_client
            _write_stream("Web search service connected")
        else:
            state["errors"] = state.get("errors", []) + ["Tavily API key not configured"]
            _write_stream("Web search unavailable - API key not configured")
            
    except Exception as e:
        error_msg = f"Client initialization failed: {str(e)}"
        state["errors"] = state.get("errors", []) + [error_msg]
        _write_stream(f"Client setup error: {error_msg}")
        
    return state

    # ==================================================================================
    # WORKFLOW CONTROL & ROUTING FUNCTIONS
    # ==================================================================================

def _streamlined_validation_decision(state: ResearchPlanningState) -> str:
        """🚀 STREAMLINED DECISION: Route based on validation results for single-problem workflow."""
        validation_results = state.get("validation_results", {})
        current_problem = state.get("current_problem", {})
        generation_attempts = state.get("generation_attempts", 0)
        
        # Check if we have validation results
        if not validation_results:
            return "validate"
        
        recommendation = validation_results.get("recommendation", "reject")
        current_iteration = state.get("iteration_count", 0)
        
        # SAFETY VALVE: If we've tried too many times, accept the current problem to prevent infinite loops
        if generation_attempts >= 10:
            # Force acceptance after 10 attempts
            state["selected_problem"] = current_problem
            state["validated_problems"] = [current_problem]
            # Mark it as a forced acceptance
            current_problem["forced_acceptance"] = True
            current_problem["forced_reason"] = f"Accepted after {generation_attempts} attempts to prevent infinite loop"
            return "create_plan"
        
        # Decision logic for streamlined workflow
        if recommendation == "accept":
            # Problem is validated - set as selected problem and proceed to planning
            state["selected_problem"] = current_problem
            state["validated_problems"] = [current_problem]  # Store in list for compatibility
            return "create_plan"
        elif recommendation == "reject":
            # Problem was rejected - process feedback and try again
            return "process_feedback"
        else:
            # Unknown state - default to regeneration
            return "retry_generation"

def _determine_refinement_path(state: ResearchPlanningState) -> str:
        """Determine whether to refine the plan or finalize it based on critique."""
        critique = state.get("critique_results", {})
        refinement_count = state.get("refinement_count", 0)
        
        if not critique:
            return "finalize"
        
        overall_score = critique.get("overall_score", 5.0)
        recommendation = critique.get("recommendation", "accept")
        major_issues = critique.get("major_issues", [])
        
        # Decision logic
        if recommendation == "accept" or overall_score >= 7.0 or refinement_count >= 3:
            return "finalize_plan"
        elif recommendation == "refine" and refinement_count < 3:
            return "refine_plan"
        else:
            return "finalize_plan"

    # ==================================================================================
    # UTILITY & HELPER FUNCTIONS
    # ==================================================================================

def _display_research_plan_terminal(state: ResearchPlanningState, config: Optional[Dict[str, Any]] = None) -> str:
        """Generate and display a comprehensive research plan in the terminal."""
        try:
            from datetime import datetime
            import os
            
            # Extract plan data
            research_plan = state.get("research_plan", {})
            plan_text = research_plan.get("research_plan", "No plan generated")
            selected_problem = research_plan.get("selected_problem", {})
            
            # Create text content
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            content = f"""
RESEARCH PLAN DOCUMENT
Generated: {timestamp}

SELECTED PROBLEM:
{selected_problem.get('statement', 'N/A')}

RESEARCH PLAN:
{plan_text}

METADATA:
- Total Iterations: {state.get('iteration_count', 0)}
- Refinements: {state.get('refinement_count', 0)}
- Final Score: {state.get('critique_results', {}).get('overall_score', 'N/A')}
"""
            
            # Print the research plan to terminal instead of saving to file
            _write_stream("Final Research Plan:")
            _write_stream("Generated Research Plan")
            _write_stream( "="*80 + "\n")
            _write_stream( content + "\n")
            _write_stream( "="*80 + "\n")
            _write_stream( "📋 END OF RESEARCH PLAN\n")
            _write_stream( "="*80 + "\n")
            
            _write_stream("Research plan display completed")
            _write_stream("Plan Statistics:")
            _write_stream(f"Total length: {len(content)} characters")
            _write_stream(f"Estimated pages: {len(content) // 3000:.1f}")
            _write_stream(f"Workflow iterations: {state.get('iteration_count', 0)}", "info")
            _write_stream(f"Plan refinements: {state.get('refinement_count', 0)}", "info")
            
            return "terminal_display"  # Return indicator instead of file path
            
        except Exception as e:
            _write_stream(f"Failed to display research plan: {str(e)}")
            return None

async def plan_research(prompt: str, uploaded_data: List[str] = None, config: Optional[Dict[str, Any]] = None, streaming: bool = False) -> Dict[str, Any]:
        """Main entry point for research planning workflow."""
        try:
            # Load configuration
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables. Please ensure it is set.")

            base_url = os.getenv("BASE_URL")
            model = os.getenv("DEFAULT_MODEL") or "gemini/gemini-2.5-flash"
            model_cheap = "gemini/gemini-2.5-flash-lite"
            model_expensive = "gemini/gemini-2.5-pro"

        # Initialize dependencies
            try:
                client = openai.OpenAI(api_key=api_key, base_url=base_url)
            
                # Initialize Tavily client
                tavily_api_key = os.getenv("TAVILY_API_KEY")
                if tavily_api_key:
                    try:
                        tavily_client = TavilyClient(api_key=tavily_api_key)
                    except Exception as e:
                        print(f"⚠️ Failed to initialize Tavily client: {e}. Web search will be unavailable.")
                    tavily_client = None
                else:
                    print("⚠️ TAVILY_API_KEY not found. Web search will be unavailable.")
                    tavily_client = None
                
            except Exception as e:
                raise ValueError(f"Failed to initialize OpenAI client: {str(e)}. Please check your API key and base URL configuration.")
        
            # ✅ Main workflow execution starts here
            _write_stream("Starting Research Planning Workflow")
            _write_stream(f"Research Domain: {prompt}")
            
            # Initialize state
            initial_state: ResearchPlanningState = {
                "messages": [HumanMessage(content=prompt)],
                "original_prompt": prompt,
                "uploaded_data": uploaded_data or [],
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
                "current_web_search_query": "",
                # Client fields will be populated by initialize_clients_in_state
                "client": client,
                "tavily_client": tavily_client,
                "model": model
            }
            
            # Build and run the workflow with recursion limit
            workflow = build_research_planning_graph()
            
            # Merge recursion limit into config
            invoke_config = config or {}
            invoke_config.update({"recursion_limit": 100000000})
            
            # 🚀 Non-streaming mode
            if not streaming:
                result = await workflow.ainvoke(initial_state, config=invoke_config)
                return result
            else:
                # 🔄 Streaming mode
                async def _stream():
                    try:
                        final_data = None  # track last update

                        async for chunk in workflow.astream(initial_state, config=invoke_config, stream_mode=["updates","custom"]):
                            stream_mode, data = chunk

                            # Debugging / logging (optional, can remove to stay "silent")
                            if stream_mode == "updates":
                                key = list(data.keys())[0] if data else None
                                print(f"Step: {key}")
                            elif stream_mode == "custom" and data.get("status"):
                                print(f"Updates: {data['status']}")

                            # Stream intermediate updates
                            yield data
                            final_data = data

                        # ✅ After loop ends, yield final state - extract from finalize_plan if available
                        if final_data:
                            # Extract the actual state from finalize_plan node if it exists
                            if 'finalize_plan' in final_data:
                                yield final_data['finalize_plan']
                            else:
                                yield final_data
                    except Exception as e:
                        _write_stream(f"Research planning streaming failed: {str(e)}")
                        yield {
                            "error": str(e),
                            "workflow_type": "research_planning",
                            "original_prompt": prompt
                        }

                return _stream()
        
        except ValueError as e:
            print(f"❌ Configuration Error: {str(e)}")
            return {
                "workflow_successful": False,
                "error": str(e),
                "error_type": "ConfigurationError",
                "original_prompt": prompt
            }
        except Exception as e:
            error_msg = f"Unexpected error during initialization: {str(e)}"
            print(f"❌ {error_msg}")
            return {
                "workflow_successful": False,
                "error": error_msg,
                "error_type": type(e).__name__,
                "original_prompt": prompt
            }
            
            if streaming:
                # Return async generator even in error case for streaming
                async def _error_stream():
                    yield error_result
                return _error_stream()
            else:
                return error_result


# ==================================================================================
# WORKFLOW BUILDING
# ==================================================================================

def build_research_planning_graph() -> StateGraph:
    """🚀 STREAMLINED WORKFLOW: Generate one problem, validate with Tavily, create research plan automatically.
    
    Workflow Steps:
    1. Generate a single research problem statement
    2. Validate with Tavily web search (check if solved/novel)
    3. If valid -> auto-select and proceed to research plan
    4. If invalid -> process feedback and retry generation
    5. Generate comprehensive research plan automatically  
    6. Critique plan and refine if needed
    7. Finalize plan
    
    Key Benefits:
    - No manual problem selection step
    - Single problem focus for efficiency
    - Automatic progression to research planning
    - Tavily validation ensures novelty
    """
    workflow = StateGraph(ResearchPlanningState)
    
    # Simplified workflow: initialize clients first, then generate -> validate -> plan (with refinement support)
    workflow.add_node("initialize_clients", _initialize_clients_in_state)
    workflow.add_node("generate_problem", _generate_problem_node)
    workflow.add_node("validate_problem", _validate_problem_node)
    workflow.add_node("process_rejection_feedback", _process_rejection_feedback_node)  # For rejected problems
    workflow.add_node("create_research_plan", _create_research_plan_node)
    workflow.add_node("critique_plan", _critique_plan_node)
    workflow.add_node("finalize_plan", _finalize_plan_node)
    
    # Entry point: initialize clients first
    workflow.set_entry_point("initialize_clients")
    workflow.add_edge("initialize_clients", "generate_problem")
    workflow.add_edge("generate_problem", "validate_problem")
    
    # After validation: either proceed to research plan or retry with feedback
    workflow.add_conditional_edges(
        "validate_problem",
        _streamlined_validation_decision,
        {
            "create_plan": "create_research_plan",           # Problem validated - proceed directly to plan
            "process_feedback": "process_rejection_feedback", # Problem rejected - get feedback
            "retry_generation": "generate_problem"           # Retry generation with feedback
        }
    )
    
    # After feedback processing, try generating a new problem
    workflow.add_edge("process_rejection_feedback", "generate_problem")
    
    # After plan creation, critique it
    workflow.add_edge("create_research_plan", "critique_plan")
    
    # After critique, decide what to do based on quality
    workflow.add_conditional_edges(
        "critique_plan",
        _determine_refinement_path,
        {
            "finalize_plan": "finalize_plan",         # No major issues - finalize
            "refine_plan": "create_research_plan",    # Has issues - regenerate with critique context
            "retry_problem": "generate_problem"       # Fundamental issues - try new problem
        }
    )
    
    workflow.add_edge("finalize_plan", END)
    
    return workflow.compile()


# ==================================================================================
# EXPORTS
# ==================================================================================

# Export the state class and all node functions for import by main file
__all__ = [
    'ResearchPlanningState',
    '_generate_problem_node',
    '_validate_problem_node', 
    '_process_rejection_feedback_node',
    '_create_research_plan_node',
    '_critique_plan_node',
    '_finalize_plan_node',
    '_streamlined_validation_decision',
    '_determine_refinement_path',
    '_display_research_plan_terminal',
    'plan_research',
    'build_research_planning_graph',
    '_clean_text_for_encoding',
    '_load_from_env_file'
]


async def main():
    """Main entry point for standalone execution."""
    import argparse
    
    print("🔧 Research Planning Workflow - Standalone Runner")
    print("=" * 60)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Research Planning Workflow - Generate research plans automatically",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python research_planning_nodes.py "Improving machine learning model interpretability"
  python research_planning_nodes.py --help
        """
    )
    
    parser.add_argument(
        'prompt',
        nargs='?',
        help='Research prompt/domain to plan research for'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    if not args.prompt:
        print("❌ Error: Please provide a research prompt")
        print("Usage: python research_planning_nodes.py \"Your research topic\"")
        return
    
    print(f"🚀 Starting research planning for: {args.prompt}")
    print("-" * 60)
    
    try:
        # Run the research planning workflow
        result = await plan_research(args.prompt)
        
        if result.get("error"):
            print(f"❌ Workflow failed: {result['error']}")
            return
        
        # Display results
        if result.get("research_plan"):
            print("✅ Research plan generated successfully!")
            plan = result["research_plan"]
            print(f"📋 Title: {plan.get('title', 'N/A')}")
            print(f"📊 Sections: {len(plan.get('sections', []))}")
            
            if result.get("display_method") == "terminal_output":
                print("📄 Research plan displayed above in terminal")
        else:
            print("⚠️  No research plan was generated")
            
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        if args.verbose:
            traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
