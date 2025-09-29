#!/usr/bin/env python3
import asyncio
import os
import json
from typing import Dict, List, Any, Optional, TypedDict, Annotated
from datetime import datetime
from io import BytesIO
from pathlib import Path

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.config import get_stream_writer
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph.message import add_messages

# Web search imports
from tavily import TavilyClient

# File processing imports
try:
    import pandas as pd  # for CSV/XLSX
except ImportError:
    pd = None
try:
    from docx import Document  # python-docx for DOCX
except ImportError:
    Document = None

# LLM imports
import openai

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
        # Clients (stored in state)
    client: Optional[Any]                     # OpenAI client instance
    tavily_client: Optional[Any]              # Tavily client instance
    model: str                                # Model name to use

    """Base state for all workflows."""
    messages: Annotated[List[BaseMessage], add_messages]
    original_prompt: str  # Pure user query without uploaded data
    uploaded_data: List[str]  # Uploaded file contents as separate field
    current_step: str
    errors: List[str]
    workflow_type: str  # "model_suggestion" or "research_planning"

class PaperWritingState(BaseState):
    """State object for the paper writing workflow."""
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

# ==================================================================================

async def _format_paper_node(state: PaperWritingState) -> PaperWritingState:
    """Node for formatting the paper according to template requirements with enhanced citation support."""

    try:
        section_content = state.get("section_content", {})
        template_config = state.get("template_config", {})
        research_analysis = state.get("research_analysis", {})
        supporting_sources = state.get("supporting_sources", [])
        source_validation = state.get("source_validation_results", {})

        # Combine all sections into a complete paper
        paper_parts = []

        # Add title with enhanced metadata
        title = f"Research Paper: {research_analysis.get('research_context', 'Untitled Research')[:80]}"
        paper_parts.append(f"# {title}\n")

        # Add venue info and citation summary
        venue = template_config.get("venue", "General")
        format_type = template_config.get("format", "academic")
        page_limit = template_config.get("page_limit", 8)

        paper_parts.append(f"**Target Venue**: {venue}")
        paper_parts.append(f"**Format**: {format_type}")
        paper_parts.append(f"**Page Limit**: {page_limit} pages")

        # Add source summary
        if supporting_sources:
            paper_parts.append(f"**Citations**: {len(supporting_sources)} sources integrated")
            paper_parts.append(f"**Research Coverage**: {source_validation.get('search_success_rate', 0):.1%}")

        paper_parts.append("\n\n")

        # Get the actual paper structure from setup node (instead of hardcoded order)
        paper_structure = state.get("paper_structure", {})
        sections = paper_structure.get("sections", [])
        
        if sections:
            # Use the ACTUAL structure order from setup node
            for section in sections:
                section_name = section.get("name", "Unknown")
                if section_name in section_content:
                    paper_parts.append(f"## {section_name}\n\n")
                    paper_parts.append(section_content[section_name])
                    paper_parts.append("\n\n")
            
            # Add any remaining sections not in the structure (like References)
            structure_section_names = [s.get("name", "") for s in sections]
            for section_name, content in section_content.items():
                if section_name not in structure_section_names:
                    paper_parts.append(f"## {section_name}\n\n")
                    paper_parts.append(content)
                    paper_parts.append("\n\n")
        else:
            # Fallback to hardcoded order if no structure available
            section_order = ["Abstract", "Introduction", "Related Work", "Methods", "Results", "Discussion", "Conclusion", "References"]
            for section_name in section_order:
                if section_name in section_content:
                    paper_parts.append(f"## {section_name}\n\n")
                    paper_parts.append(section_content[section_name])
                    paper_parts.append("\n\n")

        # Add source metadata footer
        if supporting_sources:
            paper_parts.append("---\n\n")
            paper_parts.append("## Source Metadata\n\n")
            paper_parts.append(f"This paper was enhanced with {len(supporting_sources)} sources ")
            paper_parts.append(f"found through Tavily web search across {len(source_validation.get('sections_covered', []))} research areas.\n\n")

            # Add search queries used
            search_queries = state.get("source_search_queries", [])
            if search_queries:
                paper_parts.append("**Search Queries Used:**\n")
                for i, query in enumerate(search_queries, 1):
                    paper_parts.append(f"{i}. {query}\n")
                paper_parts.append("\n")

        formatted_paper = "".join(paper_parts)

        total_citations = len([line for line in formatted_paper.split('\n') if '[' in line and ']' in line])

        return {
            **state,
            "formatted_paper": formatted_paper,
            "current_step": "paper_formatted"
        }

    except Exception as e:
        return {
            **state,
            "errors": state.get("errors", []) + [f"Formatting error: {str(e)}"],
            "current_step": "formatting_error"
        }
