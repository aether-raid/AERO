from typing import List, Dict, Any

from dotenv import load_dotenv

from langgraph.graph import StateGraph, END

import os

import asyncio

import traceback

from .shared_defs import ExperimentSuggestionState, _load_text_file_safely

import asyncio

import openai

from ..utils.arxiv_paper_utils import ArxivPaperProcessor

from ..utils.llm_client import load_openai_client

# ==================================================================================
# WORKFLOW GRAPH BUILDER
# ==================================================================================

from .nodes.analyze_data_nodes import _analyze_experiment_findings_node, _validate_analysis_node
from .nodes.research_direction_nodes import _decide_research_direction_node, _validate_research_direction_node
from .nodes.arxiv_search_nodes import _generate_experiment_search_query_node, _search_experiment_papers_node, _validate_experiment_papers_node
from .nodes.experiment_generation_nodes import _distill_paper_methodologies_node, _suggest_experiments_tree_2_node, _validate_experiments_tree_2_node
from .edges.conditional_edges import  _should_proceed_with_analysis, _debug_validation_routing

def _build_analyze_and_suggest_experiment_graph() -> StateGraph:
    """Analyze the results and suggest experiments based on findings."""
    workflow = StateGraph(ExperimentSuggestionState)

    # Add nodes for experiment suggestion workflow
    workflow.add_node("analyze_findings", _analyze_experiment_findings_node)
    workflow.add_node("validate_analysis", _validate_analysis_node)
    workflow.add_node("decide_research_direction", _decide_research_direction_node)
    workflow.add_node("validate_research_direction", _validate_research_direction_node)
    workflow.add_node("generate_experiment_search_query", _generate_experiment_search_query_node)
    workflow.add_node("search_experiment_papers", _search_experiment_papers_node)
    workflow.add_node("validate_experiment_papers", _validate_experiment_papers_node)
    workflow.add_node("distill_paper_methodologies", _distill_paper_methodologies_node)
    
    # NEW CLEAN ARCHITECTURE (RECOMMENDED - no dual edges, no state conflicts)
    workflow.add_node("suggest_experiments_tree_2", _suggest_experiments_tree_2_node)
    workflow.add_node("validate_experiments_tree_2", _validate_experiments_tree_2_node)

    # Define the flow
    workflow.set_entry_point("analyze_findings")
    workflow.add_edge("analyze_findings", "validate_analysis")
    
    # Conditional edge after analysis validation - use next_node field
    workflow.add_conditional_edges(
        "validate_analysis",
        lambda state: state.get("next_node", "decide_research_direction"),
        {
            "decide_research_direction": "decide_research_direction",
            "analyze_findings": "analyze_findings"
        }
    )
    
    workflow.add_edge("decide_research_direction", "validate_research_direction")
    
    # Conditional edge after research direction validation - use next_node field
    workflow.add_conditional_edges(
        "validate_research_direction",
        lambda state: state.get("next_node", "decide_research_direction"),
        {
            "generate_experiment_search_query": "generate_experiment_search_query",  # Direction is valid, continue
            "decide_research_direction": "decide_research_direction"  # Direction needs refinement, iterate
        }
    )
    
    workflow.add_edge("generate_experiment_search_query", "search_experiment_papers")
    workflow.add_edge("search_experiment_papers", "validate_experiment_papers")
    
    # Conditional edge after validation - use next_node field  
    workflow.add_conditional_edges(
        "validate_experiment_papers",
        lambda state: state.get("next_node", "distill_paper_methodologies"),  # DEFAULT: Route to distillation step
        {
            "distill_paper_methodologies": "distill_paper_methodologies",  # NEW: Distill methodologies first
            "suggest_experiments_tree_2": "distill_paper_methodologies",    # Fallback: also route to distillation
            "search_experiment_papers": "search_experiment_papers", # Keep current papers, search for backup
            "generate_experiment_search_query": "generate_experiment_search_query"  # Start fresh with new search query
        }
    )
    
    # Add edge from distillation to experiment suggestion
    workflow.add_edge("distill_paper_methodologies", "suggest_experiments_tree_2")
    workflow.add_conditional_edges(
        "suggest_experiments_tree_2",
        lambda state: state.get("next_node", "validate_experiments_tree_2"),  # Default to validation
        {
            "validate_experiments_tree_2": "validate_experiments_tree_2",
            "END": END
        }
    )
    
    # NEW CLEAN CONDITIONAL EDGE (ACTIVE - no state conflicts)
    workflow.add_conditional_edges(
        "validate_experiments_tree_2",
        lambda state: _debug_validation_routing(state),
        {
            "END": END,  # Experiments are valid, finish workflow
            "suggest_experiments_tree_2": "suggest_experiments_tree_2"  # Loop back with feedback
        }
    )

    return workflow.compile()



# ==================================================================================
# MAIN WORKFLOW RUNNER
# ==================================================================================

async def run_experiment_suggestion_workflow_nonstream(
    user_prompt: str,
    experimental_results: Dict[str, Any] = None,
    uploaded_data: List[str] = None,
    file_path: str = None
) -> Dict[str, Any]:
    """
    Compile and run the complete experiment suggestion workflow.
    
    Args:
        user_prompt: The user's research query/context
        experimental_results: Optional dictionary containing experimental data and results
        uploaded_data: Optional list of uploaded file contents
        file_path: Optional path to a file to read and include as input data
        
    Returns:
        Dictionary containing the final workflow state with results
    """
    # Move all imports and initialization inside the function
   
    try:
        
        try:
            client, model = load_openai_client()
        except ValueError as e:
            raise ValueError(str(e))

        # Initialize dependencies
        arxiv_processor = ArxivPaperProcessor(llm_client=client, model_name=model)
    except ValueError as e:
        print(f"Configuration Error: {str(e)}")
        return {
            "workflow_successful": False,
            "error": str(e),
            "error_type": "ConfigurationError",
            "original_prompt": user_prompt
        }
    except Exception as e:
        error_msg = f"Unexpected error during initialization: {str(e)}"
        print(f"ERROR: {error_msg}")
        return {
            "workflow_successful": False,
            "error": error_msg,
            "error_type": type(e).__name__,
            "original_prompt": user_prompt
        }
    
    print("🧪 Starting Experiment Suggestion Workflow...")
    print(f"📝 User Prompt: {user_prompt}")
    print(f"🔬 Experimental Results: {len(experimental_results) if experimental_results else 0} data points")
    print(f"🤖 Model: {model}")
    if file_path:
        print(f"File Input: {file_path}")
    print("=" * 80)

    # Handle file input if provided
    file_content: List[str] = []
    file_warnings: List[str] = []
    if file_path:
        try:
            print(f"Reading file: {file_path}")
            file_content, file_warnings = _load_text_file_safely(file_path)
            if file_content:
                print(f"File loaded successfully ({len(file_content[0])} characters)")
        except FileNotFoundError:
            print(f"Error: File not found - {file_path}")
            raise
        except Exception as e:
            print(f"Error reading file: {str(e)}")
            raise

        for warning in file_warnings:
            print(warning)

    # Combine uploaded_data with file_content if both are provided
    combined_uploaded_data = (uploaded_data or []) + file_content
    
    try:
        # Build the workflow graph
        workflow_graph = _build_analyze_and_suggest_experiment_graph()
        print("Workflow graph compiled successfully")
        
        # Initialize the state with all required fields
        initial_state = {
            # Core workflow data
            "messages": [],
            "original_prompt": user_prompt,
            "uploaded_data": combined_uploaded_data,
            "current_step": "starting",
            "errors": [],
            "workflow_type": "experiment_suggestion",
            
            # Dependencies
            "client": client,
            "model": model,
            "arxiv_processor": arxiv_processor,
            
            # Input data
            "experimental_results": experimental_results or {},
            "findings_analysis": {},
            "research_context": {},
            
            # Processing state
            "analysis_completed": False,
            "experiment_categories": [],
            "experiment_papers": [],
            "experiment_search_query": "",
            "experiment_search_iteration": 0,
            "experiment_validation_results": {},
            "experiment_paper_validation_decision": "",
            "experiment_validation_decision": "",
            
            # New validation fields for the exact workflow structure
            "analysis_validation_decision": "",
            "direction_validation_decision": "",
            "paper_validation_decision": "",
            "experiments_validation_decision": "",
            "validation_feedback": "",
            
            "experiment_iterations": [],
            "research_direction": {},
            "validated_experiment_papers": [],
            "validated_experiment_papers": [],  # Add this key that the clean architecture uses
            "distilled_methodologies": {},       # Distilled methodology content from papers
            "current_experiment_iteration": 0,
            "iteration_from_state": 0,
            "analysis_iterations": [],  # Track analysis validation iterations (list for history)
            "direction_iterations": [],  # Track research direction validation iterations (list for history)
            
            # Issue tracking
            "past_fixed_issues": [],
            "past_unresolved_issues": [],
            "most_recent_generation_issues": [],
            "cumulative_validation_feedback": [],
            
            # Solved issues tracking
            "solved_issues_history": [],
            "current_solved_issues": [],
            "validation_issue_patterns": {},
            "generation_feedback_context": "",
            
            # Output
            "experiment_suggestions": "",
            "experiment_summary": {},
            "next_node": "",
            "literature_context": "",
            "suggestion_source": "",
            "prioritized_experiments": [],
            "implementation_roadmap": {},
            "final_outputs": {}
        }
        
        print("🔄 Running workflow...")
        
        # Run the workflow with increased recursion limit to prevent infinite loops
        final_state = await workflow_graph.ainvoke(initial_state, config={"recursion_limit": 50})
        
        print("\n" + "=" * 80)
        print("WORKFLOW COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
        # Extract and display key results
        if final_state.get("experiment_suggestions"):
            print("Experiment suggestions generated successfully")
            suggestions = final_state.get("experiment_suggestions", "")
            if suggestions:
                print(f"\n📋 EXPERIMENT SUGGESTIONS PREVIEW:")
                print("-" * 40)
                print(suggestions[:500] + "..." if len(suggestions) > 500 else suggestions)
        else:
            print("Experiment suggestions may be incomplete")
        
        # Display any errors
        if final_state.get("errors"):
            print(f"\nErrors encountered: {len(final_state['errors'])}")
            for i, error in enumerate(final_state["errors"][-3:], 1):  # Show last 3 errors
                print(f"  {i}. {error}")
        
        # Display workflow statistics
        print(f"\nWORKFLOW STATISTICS:")
        print(f"   - Papers found: {len(final_state.get('experiment_papers', []))}")
        print(f"   - Papers validated: {len(final_state.get('validated_papers', []))}")
        print(f"   - Search iterations: {final_state.get('experiment_search_iteration', 0)}")
        print(f"   - Experiment iterations: {final_state.get('current_experiment_iteration', 0)}")
        
        return final_state
        
    except Exception as e:
        print(f"\nWORKFLOW FAILED: {str(e)}")
        print("Full error traceback:")
        traceback.print_exc()
        
        # Return error state
        return {
            "workflow_successful": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "original_prompt": user_prompt,
            "experimental_results": experimental_results
        }





async def run_experiment_suggestion_workflow(
    user_prompt: str,
    experimental_results: Dict[str, Any] = None,
    uploaded_data: List[str] = None,
    file_path: str = None,
    streaming: bool = False
) -> Dict[str, Any]:
    """
    Compile and run the complete experiment suggestion workflow.
    
    Args:
        user_prompt: The user's research query/context
        experimental_results: Optional dictionary containing experimental data and results
        uploaded_data: Optional list of uploaded file contents
        file_path: Optional path to a file to read and include as input data
        
    Returns:
        Dictionary containing the final workflow state with results
    """

    try:
        
        try:
            client, model = load_openai_client()
        except ValueError as e:
            raise ValueError(str(e))

        # Initialize dependencies
        arxiv_processor = ArxivPaperProcessor(llm_client=client, model_name=model)
    except ValueError as e:
        print(f"Configuration Error: {str(e)}")
        return {
            "workflow_successful": False,
            "error": str(e),
            "error_type": "ConfigurationError",
            "original_prompt": user_prompt
        }
    except Exception as e:
        error_msg = f"Unexpected error during initialization: {str(e)}"
        print(f"❌ {error_msg}")
        return {
            "workflow_successful": False,
            "error": error_msg,
            "error_type": type(e).__name__,
            "original_prompt": user_prompt
        }
    
    print("Starting Experiment Suggestion Workflow...")
    print(f"User Prompt: {user_prompt}")
    print(f"Experimental Results: {len(experimental_results) if experimental_results else 0} data points")
    print(f"Model: {model}")
    if file_path:
        print(f"File Input: {file_path}")
    print("=" * 80)

    # Handle file input if provided
    file_content: List[str] = []
    file_warnings: List[str] = []
    if file_path:
        try:
            print(f" Reading file: {file_path}")
            file_content, file_warnings = _load_text_file_safely(file_path)
            if file_content:
                print(f"File loaded successfully ({len(file_content[0])} characters)")
        except FileNotFoundError:
            print(f"Error: File not found - {file_path}")
            raise
        except Exception as e:
            print(f"Error reading file: {str(e)}")
            raise

        for warning in file_warnings:
            print(warning)

    # Combine uploaded_data with file_content if both are provided
    combined_uploaded_data = (uploaded_data or []) + file_content
    
    try:
        # Build the workflow graph
        workflow_graph = _build_analyze_and_suggest_experiment_graph()
        print("Workflow graph compiled successfully")
        
        # Initialize the state with all required fields
        initial_state = {
            # Core workflow data
            "messages": [],
            "original_prompt": user_prompt,
            "uploaded_data": combined_uploaded_data,
            "current_step": "starting",
            "errors": [],
            "workflow_type": "experiment_suggestion",
            
            # Dependencies
            "client": client,
            "model": model,
            "arxiv_processor": arxiv_processor,
            
            # Input data
            "experimental_results": experimental_results or {},
            "findings_analysis": {},
            "research_context": {},
            
            # Processing state
            "analysis_completed": False,
            "experiment_categories": [],
            "experiment_papers": [],
            "experiment_search_query": "",
            "experiment_search_iteration": 0,
            "experiment_validation_results": {},
            "experiment_paper_validation_decision": "",
            "experiment_validation_decision": "",
            
            # New validation fields for the exact workflow structure
            "analysis_validation_decision": "",
            "direction_validation_decision": "",
            "paper_validation_decision": "",
            "experiments_validation_decision": "",
            "validation_feedback": "",
            
            "experiment_iterations": [],
            "research_direction": {},
            "validated_experiment_papers": [],
            "validated_experiment_papers": [],  # Add this key that the clean architecture uses
            "distilled_methodologies": {},       # Distilled methodology content from papers
            "current_experiment_iteration": 0,
            "iteration_from_state": 0,
            "analysis_iterations": [],  # Track analysis validation iterations (list for history)
            "direction_iterations": [],  # Track research direction validation iterations (list for history)
            
            # Issue tracking
            "past_fixed_issues": [],
            "past_unresolved_issues": [],
            "most_recent_generation_issues": [],
            "cumulative_validation_feedback": [],
            
            # Solved issues tracking
            "solved_issues_history": [],
            "current_solved_issues": [],
            "validation_issue_patterns": {},
            "generation_feedback_context": "",
            
            # Output
            "experiment_suggestions": "",
            "experiment_summary": {},
            "next_node": "",
            "literature_context": "",
            "suggestion_source": "",
            "prioritized_experiments": [],
            "implementation_roadmap": {},
            "final_outputs": {}
        }
        
        print("Running workflow...")

        if not streaming:
            final_state = await workflow_graph.ainvoke(initial_state)
            return final_state.get("validate_experiments_tree_2", final_state)
        # Run the workflow with increased recursion limit to prevent infinite loops
        async def _stream():
            final_data = None  # track last update

            async for chunk in workflow_graph.astream(initial_state, stream_mode=["updates","custom"]):
                stream_mode, data = chunk

                # Debugging / logging (optional, can remove to stay "silent")
                if stream_mode == "updates":
                    key = list(data.keys())[0] if data else None
                    print(f"Node Complete: {key}.")
                    print("-" * 20)
                elif stream_mode == "custom" and data.get("status"):
                    print(f"Updates: {data['status']}")

                # Stream intermediate updates
                yield data
                final_data = data

            # After loop ends, yield final state only if it has "validate_experiments_tree_2"
            if final_data and "validate_experiments_tree_2" in final_data:
                yield final_data["validate_experiments_tree_2"]

        return _stream()

    except Exception as e:
        print(f"\nWORKFLOW FAILED: {str(e)}")
        print("Full error traceback:")
        traceback.print_exc()
        
        # Return error state
        return {
            "workflow_successful": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "original_prompt": user_prompt,
            "experimental_results": experimental_results
        }
