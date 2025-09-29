from ..shared_defs import ModelSuggestionState, PropertyHit, BaseState, Evidence, _write_stream, _clean_text_for_utf8, ML_RESEARCH_CATEGORIES
# --- MODEL SUGGESTION WORKFLOW CONTROL ---

def _should_continue_with_papers(state: ModelSuggestionState) -> str:
    """Determine whether to continue with current papers or search again."""
    
    # First try the backup decision key, then fall back to validation_results
    decision = state.get("paper_validation_decision")
    if decision is None:
        validation_results = state.get("validation_results", {})
        decision = validation_results.get("decision", "continue")
    
    search_iteration = state.get("search_iteration", 0)
    
    # Safety check: After 3 iterations, force continue to avoid infinite loops
    if search_iteration >= 3:
        _write_stream("Maximum search iterations reached (3), forcing continue...")
        return "continue"
    
    # Clean up decision string
    decision = str(decision).strip().upper()
    
    # Map validation decisions to workflow routing
    _write_stream(f"Validation decision: {decision}")
    if decision == "SEARCH_BACKUP": 
        return "search_backup"
    elif decision == "SEARCH_NEW":
        return "search_new"
    else:
        return "continue"

def _should_revise_suggestions(state: ModelSuggestionState) -> str:
    """Conditional edge function to determine if suggestions need revision."""
    critique_results = state.get("critique_results", {})
    iteration_count = state.get("suggestion_iteration", 0)
    cumulative_issues = state.get("cumulative_issues", {})
    
    # Maximum iterations to prevent infinite loops (matching conversation summary)
    MAX_ITERATIONS = 4
    
    if iteration_count >= MAX_ITERATIONS:
        _write_stream(f"Maximum iterations ({MAX_ITERATIONS}) reached, finalizing suggestions...")
        _write_stream(f"Final Status: {len(cumulative_issues.get('fixed_issues', []))} issues fixed, {len(cumulative_issues.get('recurring_issues', []))} recurring")
        return "finalize"
    
    if not critique_results.get("critique_successful", False):
        return "finalize"  # Skip revision if critique failed
    
    needs_revision = critique_results.get("needs_revision", False)
    recommendation = critique_results.get("recommendation", "accept")
    
    # Check for recurring issues - if we have any recurring issues after 2 iterations, finalize
    recurring_count = len(cumulative_issues.get("recurring_issues", []))
    persistent_count = len(cumulative_issues.get("persistent_issues", []))

    if (recurring_count >= 2 and iteration_count >= 5) or (persistent_count >= 3 and iteration_count >= 5):
        _write_stream(f"Detected {recurring_count} recurring issues and {persistent_count} persistent issues after {iteration_count} iterations - finalizing to prevent infinite loop...")
        return "finalize"
    
    # Revise if explicitly flagged for revision or if recommendation is revise/major_revision
    if needs_revision or recommendation in ["revise", "major_revision"]:
        fixed_count = len(cumulative_issues.get("fixed_issues", []))
        _write_stream(f"Revision needed (iteration {iteration_count + 1}) - {fixed_count} issues already fixed, looping back...")
        return "revise"
    else:
        fixed_count = len(cumulative_issues.get("fixed_issues", []))
        _write_stream(f"Suggestions approved after {iteration_count} iteration(s) - {fixed_count} total issues fixed, finalizing...")
        return "finalize"
