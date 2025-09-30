#!/usr/bin/env python3
"""
Paper Writing Workflow Nodes - Standalone Module
===============================================

A modular, standalone implementation of the paper writing workflow.
This module can be imported and used independently of the main ML Researcher tool.

Features:
- LLM-driven paper structure generation
- Tavily-powered citation finding and integration
- Academic paper formatting with proper citations
- Support for experimental results and uploaded data files

Usage:
    from paper_writing_nodes import write_paper

    result = await write_paper(
        user_query="Write a paper about my machine learning experiments",
        experimental_data={"accuracy": 0.95, "f1_score": 0.92},
        uploaded_data=["[CSV: results.csv]\naccuracy,f1_score\n0.95,0.92"]
    )
"""

import asyncio
import os
import json
from typing import Dict, List, Any, Optional, TypedDict
from datetime import datetime
from io import BytesIO
from pathlib import Path

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.config import get_stream_writer
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph.message import add_messages
from typing import Dict, List, Any, Optional, TypedDict, Annotated

# Web search imports
from tavily import TavilyClient

# LLM imports
import openai

# File processing imports
try:
    import pandas as pd  # for CSV/XLSX
except ImportError:
    pd = None
try:
    from docx import Document  # python-docx for DOCX
except ImportError:
    Document = None

# ==================================================================================
# STREAMWRITER HELPER FUNCTION
# ==================================================================================

def _write_stream(message: str, key: str = "status"):
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

class BaseState(TypedDict):
    """Base state for all workflows."""
    messages: Annotated[List[BaseMessage], add_messages]
    original_prompt: str  # Pure user query without uploaded data
    uploaded_data: List[str]  # Uploaded file contents as separate field
    current_step: str
    errors: List[str]
    workflow_type: str  # "model_suggestion" or "research_planning"

class PaperWritingState(BaseState):
    """State object for the paper writing workflow."""
    # Clients (stored in state)
    client: Optional[Any]                     # OpenAI client instance
    tavily_client: Optional[Any]              # Tavily client instance
    model: str                                # Model name to use

    # Input data
    experimental_results: Dict[str, Any]      # Raw experimental data
    research_context: str                     # Background information
    target_venue: str                         # Conference/journal name

    # Generated content
    research_analysis: Dict[str, Any]         # Processed research insights
    paper_structure: Dict[str, Any]           # LLM-generated structure
    template_config: Dict[str, Any]           # Selected template settings
    section_content: Dict[str, str]           # Content by section
    formatted_paper: str                      # Complete formatted paper

    # Source collection and citations (Tavily integration)
    supporting_sources: List[Dict[str, Any]]  # Sources found via Tavily search
    citation_database: Dict[str, Any]         # Organized citations by topic/section
    source_search_queries: List[str]          # Queries used for source discovery
    source_validation_results: Dict[str, Any] # Quality assessment of found sources

    # Quality control and critique system
    critique_results: Dict[str, Any]          # Current critique results
    critique_history: List[Dict[str, Any]]    # Historical critique data
    revision_count: int                       # Track iterations
    quality_score: float                      # Overall quality rating
    refinement_count: int                     # Number of refinement cycles
    critique_score_history: List[float]      # Score progression over iterations
    previous_papers: List[str]               # Previous versions for comparison

    # Output
    final_outputs: Dict[str, str]             # Multiple format versions


def _generate_reference_list(sources: List[Dict[str, Any]]) -> str:
    """Generate a properly formatted reference list from Tavily sources."""
    if not sources:
        return ""

    references = []
    for i, source in enumerate(sources, 1):
        title = source.get('title', 'Unknown Title')
        url = source.get('url', '')
        published_date = source.get('published_date', '')

        # Basic citation format (could be enhanced for specific styles)
        if published_date:
            ref = f"[{i}] {title}. {published_date}. Available: {url}"
        else:
            ref = f"[{i}] {title}. Available: {url}"

        references.append(ref)

    return "## References\n\n" + "\n\n".join(references)

def _initialize_clients_in_state(state: PaperWritingState) -> PaperWritingState:
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

from aero.report_writer.results_analysis import _analyze_results_node
from aero.report_writer.paper_setup import _setup_paper_node
from aero.report_writer.sources import _find_supporting_sources_node
from aero.report_writer.generate_content import _generate_content_node
from aero.report_writer.critique_and_refinement import _critique_paper_node
from aero.report_writer.finalize_paper import _finalize_paper_node
from aero.report_writer.critique_and_refinement import _determine_paper_refinement_path
from aero.report_writer.file_extraction import extract_files_from_paths, extract_files_from_bytes

# ==================================================================================
# WORKFLOW GRAPH BUILDER
# ==================================================================================

def build_paper_writing_graph() -> StateGraph:
    """Build the paper writing workflow for generating research papers with critique system."""
    workflow = StateGraph(PaperWritingState)

    # Add nodes for paper writing pipeline with critique (format_paper node removed)
    workflow.add_node("analyze_results", _analyze_results_node)
    workflow.add_node("setup_paper", _setup_paper_node)
    workflow.add_node("find_sources", _find_supporting_sources_node)
    workflow.add_node("generate_content", _generate_content_node)
    workflow.add_node("critique_paper", _critique_paper_node)  # New critique node
    # workflow.add_node("format_paper", _format_paper_node)  # REMOVED - content is already formatted
    workflow.add_node("finalize_paper", _finalize_paper_node)

    # Define the enhanced flow with critique and refinement (skip format_paper)
    workflow.set_entry_point("analyze_results")
    workflow.add_edge("analyze_results", "setup_paper")
    workflow.add_edge("setup_paper", "find_sources")
    workflow.add_edge("find_sources", "generate_content")
    
    # Add conditional edges for critique and refinement
    workflow.add_conditional_edges(
        "generate_content",
        _determine_paper_refinement_path,
        {
            "critique": "critique_paper",  # First time, always critique
            "finalize": "finalize_paper"   # Skip critique if already refined, go direct to finalize
        }
    )
    
    workflow.add_conditional_edges(
        "critique_paper", 
        _determine_paper_refinement_path,
        {
            "refine": "generate_content",  # Need to regenerate content
            "finalize": "finalize_paper"   # Quality sufficient, proceed directly to finalize
        }
    )
    
    workflow.add_edge("finalize_paper", END)

    return workflow.compile()

# ==================================================================================
# MAIN EXECUTION FUNCTION
# ==================================================================================

async def write_paper(
    user_query: str,
    experimental_data: Dict[str, Any] = None,
    uploaded_data: List[str] = None,
    file_paths: List[str] = None,
    files_data: List[Dict[str, Any]] = None,
    target_venue: str = "general",
    streaming: bool = False
) -> Dict[str, Any]:
    """
    Standalone function to write a research paper using the paper writing workflow.

    Args:
        user_query (str): The research topic or request for the paper
        experimental_data (Dict[str, Any], optional): Experimental results data
        uploaded_data (List[str], optional): List of pre-formatted uploaded file contents
        file_paths (List[str], optional): List of file paths to extract content from
        files_data (List[Dict[str, Any]], optional): List of file data dicts with 'content' (bytes) and 'filename' (str)
        target_venue (str, optional): Target publication venue
        streaming (bool, optional): If True, yield updates as they happen. If False, return only final state.

    Returns:
        - If streaming=False: final workflow state (dict)
        - If streaming=True: async generator yielding updates

    Example:
        # Using pre-formatted data
        result = await write_paper(
            user_query="Write a paper about my machine learning experiments",
            experimental_data={"accuracy": 0.95, "f1_score": 0.92},
            uploaded_data=["[CSV: results.csv]\naccuracy,f1_score\n0.95,0.92"]
        )
        
        # Using file paths
        result = await write_paper(
            user_query="Analyze my research data",
            file_paths=["./data/results.csv", "./reports/analysis.docx"]
        )
        
        # Using file bytes data
        result = await write_paper(
            user_query="Process uploaded files",
            files_data=[
                {"content": file_bytes, "filename": "data.xlsx"}
            ]
        )
    """

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
        
    except ValueError as e:
        print(f"❌ Configuration Error: {str(e)}")
        return {
            "workflow_successful": False,
            "error": str(e),
            "error_type": "ConfigurationError",
            "original_prompt": user_query
        }
    except Exception as e:
        error_msg = f"Unexpected error during initialization: {str(e)}"
        print(f"❌ {error_msg}")
        return {
            "workflow_successful": False,
            "error": error_msg,
            "error_type": type(e).__name__,
            "original_prompt": user_query
        }

   

    # Process files if provided
    processed_uploaded_data = uploaded_data or []
    
    # Extract content from file paths
    if file_paths:
        extracted_from_paths = extract_files_from_paths(file_paths)
        processed_uploaded_data.extend(extracted_from_paths)
    
    # Extract content from file bytes data
    if files_data:
        extracted_from_bytes = extract_files_from_bytes(files_data)
        processed_uploaded_data.extend(extracted_from_bytes)
    
    if processed_uploaded_data:
        pass  # Files processed successfully

    # Build the workflow
    graph = build_paper_writing_graph()

    # Prepare initial state
    initial_state = PaperWritingState(
        messages=[],
        original_prompt=user_query,
        uploaded_data=processed_uploaded_data,
        experimental_results=experimental_data or {},
        target_venue=target_venue,
        current_step="initialized",
        errors=[],
        workflow_type="paper_writing",

        # Initialize Clients (will be set by _initialize_clients_in_state)
        client=client,                 # OpenAI client instance
        tavily_client=tavily_client,   # Tavily client instance
        model=model,                   # Model name

        # Initialize other required fields
        research_analysis={},
        paper_structure={},
        template_config={},
        section_content={},
        formatted_paper="",
        supporting_sources=[],
        citation_database={},
        source_search_queries=[],
        source_validation_results={},
        
        # Critique and refinement fields
        critique_results={},
        critique_history=[],
        revision_count=0,
        quality_score=0.0,
        refinement_count=0,
        critique_score_history=[],
        previous_papers=[],
        
        final_outputs={},
        
     
    )

    # Initialize clients in the state
    initial_state = _initialize_clients_in_state(initial_state)

    # 🚀 Non-streaming mode
    if not streaming:
        final_state = await graph.ainvoke(initial_state)
        return final_state

    # 🚀 Streaming mode
    async def _stream():
        final_data = None  # track last update

        async for chunk in graph.astream(initial_state, stream_mode=["updates","custom"]):
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

        # ✅ After loop ends, yield final state - extract from finalize_paper if available
        if final_data:
            # Extract the actual state from finalize_paper node if it exists
            if 'finalize_paper' in final_data:
                yield final_data['finalize_paper']

    return _stream()

# ==================================================================================
# COMMAND LINE INTERFACE
# ==================================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python paper_writing_nodes.py \"Your research topic\" [file1.csv] [file2.docx] ...")
        print("Examples:")
        print("  python paper_writing_nodes.py \"Write a paper about machine learning model interpretability\"")
        print("  python paper_writing_nodes.py \"Analyze my research data\" data.csv results.xlsx")
        print("  python paper_writing_nodes.py \"Write about experimental results\" experiment_log.docx data.csv")
        print("\nSupported file types: .csv, .xlsx, .xls, .docx, .txt, .md")
        sys.exit(1)

    # Get the research topic from command line
    research_topic = sys.argv[1]
    
    # Get any additional file paths
    file_paths = sys.argv[2:] if len(sys.argv) > 2 else None
    
    if file_paths:
        pass  # Files will be processed silently
    
    # Run the paper writing workflow with file processing
    asyncio.run(write_paper(research_topic, file_paths=file_paths))
