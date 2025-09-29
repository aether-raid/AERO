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

def _parse_critique_json(critique_response: str) -> Dict[str, Any]:
    """
    Robust JSON parser for critique responses with multiple fallback strategies.
    
    Args:
        critique_response: The raw response from the LLM
        
    Returns:
        Dictionary containing the parsed critique data
    """
    import json
    import re
    
    # Strategy 1: Clean up common formatting issues
    cleaned_response = critique_response.strip()
    
    # Remove markdown code blocks
    if cleaned_response.startswith("```json"):
        cleaned_response = cleaned_response[7:]
    elif cleaned_response.startswith("```"):
        cleaned_response = cleaned_response[3:]
    
    if cleaned_response.endswith("```"):
        cleaned_response = cleaned_response[:-3]
    
    cleaned_response = cleaned_response.strip()
    
    # Strategy 2: Try direct JSON parsing
    try:
        return json.loads(cleaned_response)
    except json.JSONDecodeError as e:
        pass
    
    # Strategy 3: Find JSON object boundaries more aggressively
    try:
        # Look for the first { and last }
        start_idx = cleaned_response.find('{')
        end_idx = cleaned_response.rfind('}')
        
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_str = cleaned_response[start_idx:end_idx + 1]
            return json.loads(json_str)
    except (json.JSONDecodeError, ValueError) as e:
        pass
    
    # Strategy 4: Try to fix common JSON issues
    try:
        # Fix trailing commas
        fixed_json = re.sub(r',\s*}', '}', cleaned_response)
        fixed_json = re.sub(r',\s*]', ']', fixed_json)
        
        # Fix unescaped quotes in strings
        fixed_json = re.sub(r'(?<!\\)"(?=.*":)', '\\"', fixed_json)
        
        return json.loads(fixed_json)
    except (json.JSONDecodeError, ValueError) as e:
        pass
    
    # Strategy 5: Extract key values using regex patterns
    try:
        extracted_data = {}
        
        # Extract overall score
        score_match = re.search(r'"overall_score":\s*([0-9.]+)', cleaned_response)
        if score_match:
            extracted_data["overall_score"] = float(score_match.group(1))
        else:
            extracted_data["overall_score"] = 6.0
        
        # Extract recommendation
        rec_match = re.search(r'"recommendation":\s*"([^"]*)"', cleaned_response)
        if rec_match:
            extracted_data["recommendation"] = rec_match.group(1)
        else:
            extracted_data["recommendation"] = "revise"
        
        # Extract major issues array
        major_issues = []
        major_issues_pattern = r'"major_issues":\s*\[(.*?)\]'
        major_match = re.search(major_issues_pattern, cleaned_response, re.DOTALL)
        if major_match:
            issues_str = major_match.group(1)
            # Extract individual quoted strings
            issue_matches = re.findall(r'"([^"]*)"', issues_str)
            major_issues = issue_matches
        
        extracted_data["major_issues"] = major_issues
        
        # Extract specific improvements
        improvements = []
        improvements_pattern = r'"specific_improvements":\s*\[(.*?)\]'
        imp_match = re.search(improvements_pattern, cleaned_response, re.DOTALL)
        if imp_match:
            imp_str = imp_match.group(1)
            imp_matches = re.findall(r'"([^"]*)"', imp_str)
            improvements = imp_matches
        
        extracted_data["specific_improvements"] = improvements
        
        # Extract section scores if possible
        section_scores = {}
        score_patterns = {
            "abstract_intro": r'"abstract_intro":\s*([0-9.]+)',
            "methodology": r'"methodology":\s*([0-9.]+)',
            "results_analysis": r'"results_analysis":\s*([0-9.]+)',
            "writing_quality": r'"writing_quality":\s*([0-9.]+)',
            "citations": r'"citations":\s*([0-9.]+)'
        }
        
        for section, pattern in score_patterns.items():
            match = re.search(pattern, cleaned_response)
            if match:
                section_scores[section] = float(match.group(1))
            else:
                section_scores[section] = extracted_data["overall_score"]
        
        extracted_data["section_scores"] = section_scores
        
        return extracted_data
        
    except Exception as e:
        pass
    
    # Strategy 6: Ultimate fallback with reasonable defaults
    return {
        "overall_score": 6.0,
        "recommendation": "revise",
        "major_issues": ["JSON parsing failed - manual review needed"],
        "specific_improvements": [
            "Review paper content for completeness",
            "Validate citation format and integration",
            "Check section flow and transitions",
            "Ensure academic writing standards"
        ],
        "section_scores": {
            "abstract_intro": 6.0,
            "methodology": 6.0,
            "results_analysis": 6.0,
            "writing_quality": 6.0,
            "citations": 6.0
        },
        "critical_analysis": {},
        "technical_feedback": {},
        "writing_improvements": {},
        "revision_priority": "medium",
        "estimated_revision_effort": "moderate_rewrite"
    }