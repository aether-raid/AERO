"""
Paper Writing Module - Academic Paper Generation Workflow

This module provides an automated workflow for generating publication-ready academic
papers with citations, critique, and iterative refinement capabilities.

Key Features:
- AI-driven paper structure generation tailored to research topics
- Intelligent citation integration using Tavily web search
- Single-call comprehensive paper generation for optimal narrative flow
- Built-in critique system with iterative refinement (up to 3 iterations)
- Support for experimental data, CSV files, and various document formats
- Publication-ready output with proper academic formatting

Main Functions:
- write_paper: Generate complete academic papers from research topics
- build_paper_writing_graph: Access the underlying LangGraph workflow
"""

# Package metadata
__version__ = "2.0.0"
__author__ = "Aero ML Research Assistant"
__description__ = "AI-powered academic paper writing with citations and critique"

# Package information for introspection
__package_info__ = {
    "name": "report_writing",
    "version": __version__,
    "description": __description__,
    "features": [
        "LLM-driven paper structure generation",
        "Tavily-powered citation integration", 
        "Single-call comprehensive generation",
        "Built-in critique and refinement system",
        "Multi-format input support",
        "Publication-ready academic formatting",
        "Iterative quality improvement",
        "JSON structure parsing with fallbacks"
    ],
    "workflow_nodes": [
        "analyze_results", "setup_paper", "find_sources", 
        "generate_content", "critique_paper", "finalize_paper"
    ],
    "supported_formats": ["CSV", "XLSX", "JSON", "TXT", "DOCX"],
    "output_formats": ["Markdown", "Academic Paper"],
    "typical_metrics": {
        "paper_length": "30,000-40,000 characters",
        "citation_coverage": "85-95%",
        "quality_scores": "6.5-8.5/10",
        "generation_time": "3-5 minutes"
    }
}

# Import main functions when they're needed
def _import_main_functions():
    """Lazy import to avoid dependency issues."""
    try:
        from .main import write_paper, build_paper_writing_graph
        return write_paper, build_paper_writing_graph
    except ImportError as e:
        print(f"Warning: Cannot import paper writing functions: {e}")
        return None, None

# Convenience wrapper function with cleaner interface
async def generate_paper(
    topic: str,
    experimental_data: dict = None,
    uploaded_files: list = None,
    file_paths: list = None,
    target_venue: str = "general"
) -> dict:
    """
    Generate a complete academic research paper.

    This is a convenience wrapper around write_paper with a cleaner interface.

    Args:
        topic: Research topic or paper title
        experimental_data: Optional experimental results and metrics
        uploaded_files: Optional list of pre-formatted file contents
        file_paths: Optional list of file paths to process
        target_venue: Target publication venue (affects formatting)

    Returns:
        Dictionary containing the complete workflow results:
        - formatted_paper: The complete generated paper
        - final_outputs: File paths and metadata
        - critique_results: Quality assessment scores
        - supporting_sources: Citations and references used

    Example:
        import asyncio
        from report_writing import generate_paper

        # Basic usage
        result = asyncio.run(generate_paper(
            topic="Deep learning optimization techniques"
        ))
        
        # With experimental data
        result = asyncio.run(generate_paper(
            topic="Machine learning model comparison",
            experimental_data={"accuracy": 0.95, "f1_score": 0.92},
            target_venue="ICML"
        ))
        
        print(f"Paper saved to: {result['final_outputs']['paper_file']}")
        print(f"Quality score: {result['critique_results']['overall_score']}/10")
    """
    write_paper, _ = _import_main_functions()
    if write_paper is None:
        raise ImportError("Cannot import paper writing functions. Check dependencies.")
    
    return await write_paper(
        user_query=topic,
        experimental_data=experimental_data,
        uploaded_data=uploaded_files,
        file_paths=file_paths,
        target_venue=target_venue
    )

# Also expose the workflow graph builder for advanced users
def get_workflow():
    """
    Get the LangGraph workflow for advanced usage.
    
    Returns:
        Compiled LangGraph workflow for paper writing
        
    Example:
        from report_writing import get_workflow
        
        workflow = get_workflow()
        # Use workflow directly with custom state management
    """
    _, build_paper_writing_graph = _import_main_functions()
    if build_paper_writing_graph is None:
        raise ImportError("Cannot import workflow builder. Check dependencies.")
    
    return build_paper_writing_graph()

# Export public API
__all__ = [
    # Main functions
    'generate_paper',      # Simplified interface
    'get_workflow',        # Workflow access
    
    # Package metadata
    '__version__',
    '__package_info__'
]

# Try to import and expose main functions if dependencies are available
try:
    from .main import write_paper, build_paper_writing_graph
    __all__.extend(['write_paper', 'build_paper_writing_graph'])
except ImportError:
    # Dependencies not available, will use lazy loading
    pass
