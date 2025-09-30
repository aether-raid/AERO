from typing import List, Dict, Any
from .shared_defs import ModelSuggestionState
from ..utils.arxiv_paper_utils import ArxivPaperProcessor
from langgraph.graph import StateGraph, END
from typing import Dict, List, Any
import traceback

# import nodes and edges needed. 
from .nodes.suggestion_nodes import (
    _suggest_models_node as _suggest_models_node2,
    _critique_response_node as _critique_response_node2,
    _revise_suggestions_node as _revise_suggestions_node2
)
from .nodes.arxiv_search_nodes import (_search_arxiv_node as _search_arxiv_node2,
                                _validate_papers_node as _validate_papers_node2,
                                _generate_search_query_node as _generate_search_query_node2)
from .nodes.analyze_properties_nodes import _analyze_properties_and_task_node as _analyze_properties_and_task_node2

from .edges.conditional_edges import (_should_continue_with_papers as _should_continue_with_papers2, 
                               _should_revise_suggestions as _should_revise_suggestions2)
from ..utils.llm_client import load_openai_client

# ==================================================================================
# WORKFLOW GRAPH BUILDER
def _build_model_suggestion_graph() -> StateGraph:
    """Build the model suggestion workflow with critique and revision."""
    workflow = StateGraph(ModelSuggestionState)
    
    # Add nodes for model suggestion pipeline
    workflow.add_node("analyze_properties_and_task", _analyze_properties_and_task_node2)
    workflow.add_node("generate_search_query", _generate_search_query_node2)
    workflow.add_node("search_arxiv", _search_arxiv_node2)
    workflow.add_node("validate_papers", _validate_papers_node2)
    workflow.add_node("suggest_models", _suggest_models_node2)
    workflow.add_node("critique_response", _critique_response_node2)
    workflow.add_node("revise_suggestions", _revise_suggestions_node2)

    # Define the flow
    workflow.set_entry_point("analyze_properties_and_task")
    workflow.add_edge("analyze_properties_and_task", "generate_search_query")
    workflow.add_edge("generate_search_query", "search_arxiv")
    workflow.add_edge("search_arxiv", "validate_papers")
    
    # Conditional edge after validation - decide whether to continue or search again
    workflow.add_conditional_edges(
        "validate_papers",
        _should_continue_with_papers2,
        {
            "continue": "suggest_models",           # Papers are good, continue with model suggestions
            "search_backup": "search_arxiv",       # Keep current papers, search for backup
            "search_new": "generate_search_query"  # Start fresh with new search query
        }
    )
    
    workflow.add_edge("suggest_models", "critique_response")
    
    # Conditional edge after critique - decide whether to revise or finalize
    workflow.add_conditional_edges(
        "critique_response",
        _should_revise_suggestions2,
        {
            "revise": "suggest_models",      # Loop back to suggestions for revision
            "finalize": END                  # If suggestions are good as-is
        }
    )
    
    return workflow.compile()



async def run_model_suggestion_workflow_nonstream(
    user_prompt: str,
    uploaded_data: List[str] = None
) -> Dict[str, Any]:
    
    
    """
    Compile and run the complete model suggestion workflow.
    
    Args:
        user_prompt: The user's research query
        uploaded_data: Optional list of uploaded file contents
        
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
    
    print("Starting Model Suggestion Workflow...")
    print(f"User Prompt: {user_prompt}")
    print(f"Model: {model}")
    print("=" * 80)
    
    
    
    
    try:
        # Build the workflow graph
        workflow_graph = _build_model_suggestion_graph()
        print("Workflow graph compiled successfully")
        
        # Initialize the state with all required fields
        initial_state = {
            # Core workflow data
            "messages": [],
            "original_prompt": user_prompt,
            "uploaded_data": uploaded_data or [],
            "current_step": "starting",
            "errors": [],
            "workflow_type": "model_suggestion",
            
            # Dependencies
            "client": client,
            "model": model,
            "arxiv_processor": arxiv_processor,
            
            # Workflow-specific fields (initialize as empty/default)
            "detected_categories": [],
            "detailed_analysis": {},
            "arxiv_search_query": "",
            "arxiv_results": {},
            "validation_results": {},
            "paper_validation_decision": "",
            "search_iteration": 0,
            "all_seen_paper_ids": set(),
            "arxiv_chunk_metadata": [],
            "model_suggestions": {},
            "critique_results": {},
            "suggestion_iteration": 0,
            "critique_history": [],
            "cumulative_issues": {
                "fixed_issues": [],
                "persistent_issues": [],
                "recurring_issues": []
            }
        }
        
        print("Running workflow...")
        
        # Run the workflow
        final_state = await workflow_graph.ainvoke(initial_state)
        
        print("\n" + "=" * 80)
        print("WORKFLOW COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
        # Extract and display key results
        if final_state.get("model_suggestions", {}).get("suggestions_successful"):
            print("Model suggestions generated successfully")
            suggestions = final_state["model_suggestions"].get("model_suggestions", "")
            if suggestions:
                print(f"\nFINAL RECOMMENDATIONS:")
                print("-" * 40)
                print(suggestions[:500] + "..." if len(suggestions) > 500 else suggestions)
        else:
            print("Model suggestions may have failed or are incomplete")
        
        # Display any errors
        if final_state.get("errors"):
            print(f"\nErrors encountered: {len(final_state['errors'])}")
            for i, error in enumerate(final_state["errors"][-3:], 1):  # Show last 3 errors
                print(f"  {i}. {error}")
        
        # Display workflow statistics
        print(f"\nWORKFLOW STATISTICS:")
        print(f"   - Papers analyzed: {len(final_state.get('arxiv_results', {}).get('papers', []))}")
        print(f"   - Categories detected: {len(final_state.get('detected_categories', []))}")
        print(f"   - Search iterations: {final_state.get('search_iteration', 0)}")
        print(f"   - Suggestion iterations: {final_state.get('suggestion_iteration', 0)}")
        
        return final_state
        
    except Exception as e:
        print(f"WORKFLOW FAILED: {str(e)}")
        print("Full error traceback:")
       
        traceback.print_exc()
        
        # Return error state
        return {
            "workflow_successful": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "original_prompt": user_prompt
        }



async def run_model_suggestion_workflow(
    user_prompt: str,
    uploaded_data: List[str] = None,
    streaming: bool = False,
):
    """
    Compile and run the complete model suggestion workflow.

    Args:
        user_prompt: The user's research query
        uploaded_data: Optional list of uploaded file contents
        streaming: If True, yield updates as they happen. If False, return only final state.

    Returns:
        - If streaming=False: final workflow state (dict)
        - If streaming=True: async generator yielding updates
    """
    
    # Use the same client loading logic as the non-streaming version
    try:
        try:
            client, model = load_openai_client()
        except ValueError as e:
            raise ValueError(str(e))

        # Initialize dependencies
        arxiv_processor = ArxivPaperProcessor(llm_client=client, model_name=model)
    except ValueError as e:
        return {
            "workflow_successful": False,
            "error": str(e),
            "error_type": "ConfigurationError",
            "original_prompt": user_prompt
        }
    except Exception as e:
        error_msg = f"Unexpected error during initialization: {str(e)}"
        return {
            "workflow_successful": False,
            "error": error_msg,
            "error_type": type(e).__name__,
            "original_prompt": user_prompt
        }

    # Build workflow graph
    workflow_graph = _build_model_suggestion_graph()

    initial_state = {
        "messages": [],
        "original_prompt": user_prompt,
        "uploaded_data": uploaded_data or [],
        "current_step": "starting",
        "errors": [],
        "workflow_type": "model_suggestion",
        "client": client,
        "model": model,
        "arxiv_processor": arxiv_processor,
        "detected_categories": [],
        "detailed_analysis": {},
        "arxiv_search_query": "",
        "arxiv_results": {},
        "validation_results": {},
        "paper_validation_decision": "",
        "search_iteration": 0,
        "all_seen_paper_ids": set(),
        "arxiv_chunk_metadata": [],
        "model_suggestions": {},
        "critique_results": {},
        "suggestion_iteration": 0,
        "critique_history": [],
        "cumulative_issues": {
            "fixed_issues": [],
            "persistent_issues": [],
            "recurring_issues": [],
        },
    }

    # Non-streaming mode
    if not streaming:
        final_state = await workflow_graph.ainvoke(initial_state)
        return final_state.get("critique_response", final_state)

    # Streaming mode
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

        # After loop ends, yield final state only if it has "critique_response"
        if final_data and "critique_response" in final_data:
            yield final_data["critique_response"]

    return _stream()

    


    