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

async def _analyze_results_node(state: PaperWritingState) -> PaperWritingState:
    """Node for analyzing experimental results and research context, including uploaded file data."""
    
    _write_stream("📊 Analyzing experimental results and research context")

    try:
        # Validate client is available
        if not state.get("client"):
            _write_stream("❌ OpenAI client not available")
            return {**state, "errors": state.get("errors", []) + ["Analysis error: OpenAI client unavailable"]}

        # Extract research context from the original prompt and any provided data
        original_prompt = state.get("original_prompt", "")
        experimental_results = state.get("experimental_results", {})
        uploaded_data = state.get("uploaded_data", [])

        # Process uploaded file data
        uploaded_context = ""
        data_analysis = ""

        if uploaded_data:
            _write_stream(f"📎 Processing {len(uploaded_data)} uploaded files")
            uploaded_context = "\n\nUPLOADED FILE DATA:\n"

            for i, file_content in enumerate(uploaded_data, 1):
                # Extract file info from the formatted content
                if file_content.startswith('[CSV:'):
                    file_info = file_content.split('\n')[0]
                    csv_data = '\n'.join(file_content.split('\n')[1:])
                    uploaded_context += f"\n{file_info}\n"

                    # Analyze CSV data structure
                    lines = csv_data.strip().split('\n')
                    if len(lines) > 1:
                        headers = lines[0].split(',')
                        data_rows = len(lines) - 1
                        uploaded_context += f"Headers: {', '.join(headers[:10])}{'...' if len(headers) > 10 else ''}\n"
                        uploaded_context += f"Data rows: {data_rows}\n"
                        uploaded_context += f"Sample data:\n{chr(10).join(lines[1:4])}\n"

                        data_analysis += f"CSV file with {len(headers)} columns, {data_rows} rows. "

                elif file_content.startswith('[XLSX:'):
                    file_info = file_content.split('\n')[0]
                    xlsx_data = '\n'.join(file_content.split('\n')[1:])
                    uploaded_context += f"\n{file_info}\n"

                    # Analyze Excel data
                    lines = xlsx_data.strip().split('\n')
                    if len(lines) > 1:
                        headers = lines[0].split(',')
                        data_rows = len(lines) - 1
                        uploaded_context += f"Headers: {', '.join(headers[:10])}{'...' if len(headers) > 10 else ''}\n"
                        uploaded_context += f"Data rows: {data_rows}\n"

                        data_analysis += f"Excel sheet with {len(headers)} columns, {data_rows} rows. "

                elif file_content.startswith('[DOCX:'):
                    file_info = file_content.split('\n')[0]
                    doc_text = '\n'.join(file_content.split('\n')[1:])
                    uploaded_context += f"\n{file_info}\n"
                    uploaded_context += f"Document content preview:\n{doc_text[:500]}{'...' if len(doc_text) > 500 else ''}\n"

                    data_analysis += f"Document with {len(doc_text)} characters. "

                else:
                    uploaded_context += f"\nFile {i}: {file_content[:200]}{'...' if len(file_content) > 200 else ''}\n"
                    data_analysis += f"Additional file data. "
        else:
            pass

        analysis_prompt = f"""
        Analyze the following experimental results and research context to prepare for paper writing:

        Original Request: "{original_prompt}"

        Experimental Results: {experimental_results if experimental_results else "No structured experimental data provided"}

        {uploaded_context}

        Data Analysis Summary: {data_analysis if data_analysis else "No uploaded data files"}

        Please analyze and extract:
        1. Research Type: (experimental, theoretical, survey, case study)
        2. Domain: (machine learning, computer vision, NLP, etc.)
        3. Key Findings: Main experimental results and insights from uploaded data
        4. Data Types: (tables, figures, metrics, code, datasets, documents)
        5. Contributions: Novel aspects and significance based on data
        6. Research Context: Background and motivation
        7. Data Description: Summary of uploaded files and their relevance
        8. Methodology: Approach used based on available data

        If uploaded files contain experimental data (CSV/Excel), extract specific metrics, results, and findings.
        If uploaded documents exist, summarize their research content and relevance.

        Respond with a JSON object containing this analysis.
        """

        _write_stream("🧠 Analyzing research context with AI")
        
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: state["client"].chat.completions.create(
                model=state["model"],
                messages=[{"role": "user", "content": analysis_prompt}],
                temperature=0.3
            )
        )

        # Parse the analysis
        analysis_text = response.choices[0].message.content.strip()

        # Try to extract JSON from response
        try:
            import json
            # Look for JSON in the response
            start = analysis_text.find('{')
            end = analysis_text.rfind('}') + 1
            if start != -1 and end != -1:
                analysis_json = json.loads(analysis_text[start:end])
            else:
                # Fallback: create basic analysis with uploaded data context
                data_types = ["text"]
                if uploaded_data:
                    if any('[CSV:' in data for data in uploaded_data):
                        data_types.append("tables")
                    if any('[XLSX:' in data for data in uploaded_data):
                        data_types.append("spreadsheets")
                    if any('[DOCX:' in data for data in uploaded_data):
                        data_types.append("documents")

                analysis_json = {
                    "research_type": "experimental" if uploaded_data else "theoretical",
                    "domain": "machine learning",
                    "key_findings": data_analysis if data_analysis else "Experimental results analysis",
                    "data_types": data_types,
                    "contributions": ["Novel approach"],
                    "research_context": original_prompt,
                    "data_description": f"Analysis of {len(uploaded_data)} uploaded files" if uploaded_data else "No uploaded files",
                    "methodology": "Data-driven analysis" if uploaded_data else "Theoretical approach"
                }
        except:
            # Fallback analysis with uploaded data awareness
            data_types = ["text"]
            if uploaded_data:
                if any('[CSV:' in data for data in uploaded_data):
                    data_types.append("tables")
                if any('[XLSX:' in data for data in uploaded_data):
                    data_types.append("spreadsheets")
                if any('[DOCX:' in data for data in uploaded_data):
                    data_types.append("documents")

            analysis_json = {
                "research_type": "experimental" if uploaded_data else "theoretical",
                "domain": "machine learning",
                "key_findings": data_analysis if data_analysis else "Experimental results analysis",
                "data_types": data_types,
                "contributions": ["Novel approach"],
                "research_context": original_prompt,
                "data_description": f"Analysis of {len(uploaded_data)} uploaded files" if uploaded_data else "No uploaded files",
                "methodology": "Data-driven analysis" if uploaded_data else "Theoretical approach"
            }

        research_type = analysis_json.get("research_type", "unknown")
        domain = analysis_json.get("domain", "unknown")
        _write_stream(f"✅ Analysis complete - Type: {research_type}, Domain: {domain}")

        return {
            **state,
            "research_analysis": analysis_json,
            "research_context": analysis_json.get("research_context", original_prompt),  # Ensure research_context is available for Tavily
            "current_step": "results_analyzed"
        }

    except Exception as e:
        return {
            **state,
            "errors": state.get("errors", []) + [f"Analysis error: {str(e)}"],
            "current_step": "analysis_error"
        }