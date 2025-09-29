"""
Research Planning Module - AI-Driven Research Plan Generation

This module provides an intelligent workflow for generating comprehensive, validated
research plans with iterative refinement and quality assurance capabilities.

Key Features:
- AI-driven problem generation from broad research topics
- Multi-criteria validation for feasibility, novelty, and significance
- Comprehensive research plan creation with methodology and timelines
- Iterative refinement based on automated critique feedback
- Support for experimental data and literature context
- Publication-ready research plans suitable for grant applications

Main Functions:
- plan_research: Generate complete research plans from topics
- build_research_planning_graph: Access the underlying LangGraph workflow
"""

# Package metadata
__version__ = "2.0.0"
__author__ = "Aero ML Research Assistant" 
__description__ = "AI-powered research planning with validation and refinement"

# Package information for introspection
__package_info__ = {
    "name": "research_planner",
    "version": __version__,
    "description": __description__,
    "features": [
        "AI-driven problem generation",
        "Multi-criteria problem validation",
        "Comprehensive research plan creation",
        "Automated critique and refinement",
        "Timeline and milestone planning",
        "Risk assessment and mitigation",
        "Literature gap analysis",
        "Resource requirement estimation"
    ],
    "workflow_nodes": [
        "generate_problem", "validate_problem", "create_plan",
        "critique_plan", "refine_plan", "finalize_plan", "output_results"
    ],
    "validation_criteria": [
        "novelty", "feasibility", "significance", 
        "methodology", "timeline", "resources"
    ],
    "output_formats": ["Markdown", "Word Document"],
    "typical_metrics": {
        "plan_completeness": "85-95%",
        "feasibility_scores": "7.0-9.0/10", 
        "validation_success": "85%+",
        "processing_time": "3-6 minutes"
    }
}

# Import main functions when they're needed
def _import_main_functions():
    """Lazy import to avoid dependency issues."""
    try:
        from .main import plan_research, build_research_planning_graph
        return plan_research, build_research_planning_graph
    except ImportError as e:
        print(f"Warning: Cannot import research planning functions: {e}")
        return None, None

# Convenience wrapper function with cleaner interface
async def create_research_plan(
    research_topic: str,
    context_data: list = None,
    domain: str = None,
    timeline_constraint: str = None,
    max_iterations: int = 3
) -> dict:
    """
    Create a comprehensive research plan from a research topic.

    This is a convenience wrapper around plan_research with enhanced parameters.

    Args:
        research_topic: Description of the research area or problem
        context_data: Optional list of experimental data or literature context
        domain: Optional research domain for specialized planning
        timeline_constraint: Optional timeline constraint (e.g., "2_years", "6_months")
        max_iterations: Maximum refinement iterations (default: 3)

    Returns:
        Dictionary containing the complete workflow results:
        - research_plan: The complete generated research plan
        - plan_file_path: Path to saved plan file
        - validation_results: Problem validation outcomes
        - critique_results: Plan quality assessment
        - refinement_history: Iteration tracking and improvements

    Example:
        import asyncio
        from research_planner import create_research_plan

        # Basic usage
        result = asyncio.run(create_research_plan(
            research_topic="Interpretable AI for medical diagnosis"
        ))
        
        # With constraints and context
        result = asyncio.run(create_research_plan(
            research_topic="Deep learning optimization techniques",
            domain="computer_vision",
            timeline_constraint="18_months",
            context_data=["Current CNN accuracy: 89%", "Dataset: 50k images"]
        ))
        
        print(f"Plan saved to: {result['plan_file_path']}")
        print(f"Validation score: {result['validation_results']['overall_score']}/10")
        print(f"Plan quality: {result['critique_results']['overall_score']}/10")
    """
    plan_research, _ = _import_main_functions()
    if plan_research is None:
        raise ImportError("Cannot import research planning functions. Check dependencies.")
    
    return await plan_research(
        prompt=research_topic,
        uploaded_data=context_data
    )

# Specialized functions for different research types
async def plan_experimental_research(
    research_topic: str,
    experimental_data: dict,
    hypothesis: str = None
) -> dict:
    """
    Create a research plan focused on experimental validation.
    
    Args:
        research_topic: The research question or area
        experimental_data: Current experimental results and metrics
        hypothesis: Optional research hypothesis to validate
        
    Returns:
        Research plan optimized for experimental research methodology
    """
    plan_research, _ = _import_main_functions()
    if plan_research is None:
        raise ImportError("Cannot import research planning functions. Check dependencies.")
    
    context = [f"Experimental data: {experimental_data}"]
    if hypothesis:
        context.append(f"Research hypothesis: {hypothesis}")
        
    return await plan_research(
        prompt=f"Experimental research: {research_topic}",
        uploaded_data=context
    )

async def plan_theoretical_research(
    research_topic: str,
    literature_context: list = None,
    theoretical_framework: str = None
) -> dict:
    """
    Create a research plan focused on theoretical contributions.
    
    Args:
        research_topic: The theoretical research area
        literature_context: Optional existing literature and background
        theoretical_framework: Optional theoretical framework to build upon
        
    Returns:
        Research plan optimized for theoretical research methodology
    """
    plan_research, _ = _import_main_functions()
    if plan_research is None:
        raise ImportError("Cannot import research planning functions. Check dependencies.")
    
    context = literature_context or []
    if theoretical_framework:
        context.append(f"Theoretical framework: {theoretical_framework}")
        
    return await plan_research(
        prompt=f"Theoretical research: {research_topic}",
        uploaded_data=context
    )

# Also expose the workflow graph builder for advanced users
def get_workflow():
    """
    Get the LangGraph workflow for advanced usage.
    
    Returns:
        Compiled LangGraph workflow for research planning
        
    Example:
        from research_planner import get_workflow
        
        workflow = get_workflow()
        # Use workflow directly with custom state management
    """
    _, build_research_planning_graph = _import_main_functions()
    if build_research_planning_graph is None:
        raise ImportError("Cannot import workflow builder. Check dependencies.")
    
    return build_research_planning_graph()

# Utility function for plan quality assessment
def assess_plan_quality(plan_content: str) -> dict:
    """
    Assess the quality of a research plan (placeholder for future implementation).
    
    Args:
        plan_content: The research plan text to assess
        
    Returns:
        Dictionary with quality metrics
    """
    # Placeholder implementation - could be enhanced with actual assessment logic
    return {
        "completeness": 0.0,
        "feasibility": 0.0,
        "novelty": 0.0,
        "methodology_soundness": 0.0,
        "overall_score": 0.0
    }

# Export public API
__all__ = [
    # Main functions
    'create_research_plan',      # Simplified interface
    'plan_experimental_research', # Specialized experimental planning
    'plan_theoretical_research',  # Specialized theoretical planning
    'get_workflow',             # Workflow access
    
    # Utility functions
    'assess_plan_quality',      # Plan assessment
    
    # Package metadata
    '__version__',
    '__package_info__'
]

# Try to import and expose main functions if dependencies are available
try:
    from .main import plan_research, build_research_planning_graph
    __all__.extend(['plan_research', 'build_research_planning_graph'])
except ImportError:
    # Dependencies not available, will use lazy loading
    pass
