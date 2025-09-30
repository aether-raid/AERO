
from ..shared_defs import ExperimentSuggestionState, _write_stream
# ==================================================================================
# WORKFLOW CONTROL FUNCTIONS
# ==================================================================================

def _debug_validation_routing(state: ExperimentSuggestionState) -> str:
    """Debug routing function for validate_experiments_tree_2 node."""
    next_node = state.get("next_node", "END")
    validation_decision = state.get("experiments_validation_decision", "PASS")
    current_iteration = state.get("current_experiment_iteration", 0)

    # SAFETY CHECK: Prevent infinite loops - align with validation logic (max 5 iterations)
    MAX_ITERATIONS = 5
    if current_iteration >= MAX_ITERATIONS:
        _write_stream(f"Maximum iterations reached ({MAX_ITERATIONS}), forcing workflow END to prevent infinite recursion")
        return "END"

    if next_node == "END" or validation_decision in ["PASS", "FORCE_PASS"]:
        return "END"
    else:
        return "suggest_experiments_tree_2"
    
    
def _should_continue_with_papers(state: ExperimentSuggestionState) -> str:
    """Determine whether to continue with current papers or search again."""
    validation_decision = state.get("experiment_paper_validation_decision", "continue")
    search_iteration = state.get("experiment_search_iteration", 0)

    # Safety check: After 3 iterations, force continue
    if search_iteration >= 3:
        _write_stream("Maximum search iterations reached (3), forcing continue...")
        return "continue"

    # Map validation decisions to workflow routing
    if validation_decision == "search_backup":
        _write_stream("Searching for additional papers...")
        return "search_backup"
    elif validation_decision == "search_new":
        _write_stream("Starting new paper search...")
        return "search_new"
    else:
        _write_stream("Continuing with current papers...")
        return "continue"


def _should_proceed_with_direction(state: ExperimentSuggestionState) -> str:
    """Determine whether to proceed with the research direction."""
    validation_decision = state.get("direction_validation_decision", "PASS")
    direction_iterations = state.get("direction_iterations", [])

    # Safety check: After 3 iterations, force proceed
    if len(direction_iterations) >= 3:
        _write_stream("Maximum direction validation iterations reached (3), forcing proceed...")
        return "proceed"

    if validation_decision == "PASS":
        return "proceed"
    else:
        return "revise_direction"


def _should_proceed_with_analysis(state: ExperimentSuggestionState) -> str:
    """Determine whether to proceed with analysis or revise it."""
    
    next_node = state.get("next_node", "decide_research_direction")
    analysis_iterations = state.get("analysis_iterations", [])
    validation_result = state.get("analysis_validation_decision", "IDK")
    _write_stream(f"DEBUG Analysis validation decision: {validation_result}")
    _write_stream(f"DEBUG Analysis iterations so far: {len(analysis_iterations)}")
    _write_stream(f"DEBUG Next node: {next_node}")
    # Safety check: After 3 iterations, force proceed to prevent infinite loops
    if len(analysis_iterations) >= 3:
        _write_stream("Maximum analysis iterations reached (3), forcing proceed to research direction...")
        return "decide_research_direction"
    elif validation_result == "PASS":
        _write_stream("Analysis validated successfully, proceeding to research direction...")
        return "decide_research_direction"
    else:
        _write_stream("Analysis validation failed, revising analysis...")
        return "analyze_findings"

    return next_node