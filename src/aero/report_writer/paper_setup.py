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

async def _setup_paper_node(state: PaperWritingState) -> PaperWritingState:
    """Node for LLM-driven template selection and comprehensive paper structure planning."""

    _write_stream("🏗️ Setting up comprehensive paper structure")

    try:
        # Validate client is available
        if not state.get("client"):
            _write_stream("❌ OpenAI client not available")
            return {**state, "errors": state.get("errors", []) + ["Setup error: OpenAI client unavailable"]}

        research_analysis = state.get("research_analysis", {})
        target_venue = state.get("target_venue", "general")
        uploaded_data = state.get("uploaded_data", [])
        original_prompt = state.get("original_prompt", "")

        setup_prompt = f"""
        Create a comprehensive paper structure optimized for single-call generation. This structure will guide an LLM to write a complete, coherent academic paper in one comprehensive response.

        Research Topic: {original_prompt}
        Research Analysis: {research_analysis}
        Target Venue: {target_venue}
        Available Data Files: {len(uploaded_data)} files uploaded

        Design a detailed structure that includes:
        1. Template Configuration (formatting and style requirements)
        2. Comprehensive Paper Structure (sections with detailed descriptions, content requirements, and interconnections)
        3. Content Flow Guidelines (how sections should build on each other for coherent narrative)
        4. Quality Requirements (academic standards and publication readiness criteria)

        **CRITICAL**: This structure must enable a single LLM call to generate a complete, publication-ready paper with excellent flow between sections.

        Respond with a comprehensive JSON object:
        {{
            "template_config": {{
                "venue": "conference_name",
                "page_limit": 8,
                "format": "academic_paper",
                "citation_style": "ACM",
                "target_word_count": 6000,
                "quality_standard": "publication_ready"
            }},
            "paper_structure": {{
                "sections": [
                    {{
                        "name": "Abstract", 
                        "length": "200-250 words", 
                        "focus": "comprehensive_summary",
                        "description": "Concise overview of problem, method, key results, and implications",
                        "content_requirements": ["problem statement", "methodology overview", "key findings", "significance"],
                        "flow_position": "standalone_summary"
                    }},
                    {{
                        "name": "Introduction", 
                        "length": "1-1.5 pages", 
                        "focus": "motivation_and_context",
                        "description": "Establish problem significance, review relevant work, present contributions",
                        "content_requirements": ["problem motivation", "research gap", "related work summary", "clear contributions", "paper organization"],
                        "flow_position": "foundation_setting"
                    }},
                    {{
                        "name": "Methodology", 
                        "length": "2-2.5 pages", 
                        "focus": "technical_approach",
                        "description": "Detailed description of methods, experimental design, and implementation",
                        "content_requirements": ["approach overview", "technical details", "experimental setup", "evaluation metrics", "implementation details"],
                        "flow_position": "technical_foundation"
                    }},
                    {{
                        "name": "Results", 
                        "length": "2-2.5 pages", 
                        "focus": "findings_and_analysis",
                        "description": "Present experimental results, analysis, and interpretation",
                        "content_requirements": ["experimental results", "statistical analysis", "performance comparisons", "result interpretation", "findings discussion"],
                        "flow_position": "evidence_presentation"
                    }},
                    {{
                        "name": "Discussion", 
                        "length": "1 page", 
                        "focus": "implications_and_limitations",
                        "description": "Interpret results, discuss implications, acknowledge limitations",
                        "content_requirements": ["result interpretation", "broader implications", "limitations acknowledgment", "future work suggestions"],
                        "flow_position": "synthesis_and_reflection"
                    }},
                    {{
                        "name": "Conclusion", 
                        "length": "0.5 pages", 
                        "focus": "summary_and_impact",
                        "description": "Summarize contributions and significance",
                        "content_requirements": ["contribution summary", "key findings recap", "broader impact", "final thoughts"],
                        "flow_position": "closure_and_impact"
                    }}
                ]
            }},
            "content_guidelines": {{
                "narrative_flow": "Each section builds logically on previous sections with smooth transitions",
                "citation_strategy": "Integrate sources naturally throughout all sections to support claims",
                "technical_depth": "Balance accessibility with rigor appropriate for target venue",
                "coherence_requirements": "Maintain consistent terminology and argument thread throughout",
                "quality_standards": ["publication_ready", "peer_review_quality", "clear_writing", "proper_citations"],
                "emphasis": ["methodology_rigor", "experimental_validation", "clear_presentation"],
                "tone": "formal_academic",
                "target_audience": "researchers_and_practitioners",
                "writing_style": "clear_concise_authoritative"
            }},
            "single_call_optimization": {{
                "structure_clarity": "Provide clear section boundaries and content expectations",
                "flow_guidance": "Include transition requirements between sections",
                "completeness_requirements": "Ensure each section is self-contained yet interconnected",
                "quality_checkpoints": "Built-in requirements for academic rigor and publication standards"
            }}
        }}

        Focus on creating a structure that will result in a coherent, well-flowing academic paper when generated in a single comprehensive LLM call.
        """

        _write_stream("🤖 Generating optimal paper structure with AI")

        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: state["client"].chat.completions.create(
                model=state["model"],
                messages=[{"role": "user", "content": setup_prompt}],
                temperature=0.1
            )
        )

        _write_stream("📋 Processing paper structure configuration")
        setup_text = response.choices[0].message.content.strip()

        # Try to extract JSON from response with robust parsing
        try:
            import json
            import re
            
            # Extract JSON content
            start = setup_text.find('{')
            end = setup_text.rfind('}') + 1
            
            if start != -1 and end != -1:
                json_str = setup_text[start:end]
                
                # Clean up common issues that cause JSON parsing to fail
                # Remove control characters that aren't allowed in JSON
                json_str = ''.join(char for char in json_str if ord(char) >= 32 or char in '\n\r\t')
                
                # Fix common JSON formatting issues
                json_str = re.sub(r',\s*}', '}', json_str)  # Remove trailing commas before }
                json_str = re.sub(r',\s*]', ']', json_str)  # Remove trailing commas before ]
                json_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', json_str)  # Remove control characters
                
                # Try parsing the cleaned JSON
                try:
                    setup_json = json.loads(json_str)
                except json.JSONDecodeError as json_error:
                    # Try a more aggressive cleaning approach
                    json_str_backup = json_str
                    
                    # Additional cleaning steps
                    json_str = re.sub(r'\\n', ' ', json_str)  # Replace literal \n with space
                    json_str = re.sub(r'\\t', ' ', json_str)  # Replace literal \t with space
                    json_str = re.sub(r'\\r', ' ', json_str)  # Replace literal \r with space
                    json_str = re.sub(r'\s+', ' ', json_str)  # Normalize whitespace
                    
                    try:
                        setup_json = json.loads(json_str)
                    except json.JSONDecodeError:
                        raise Exception("JSON parsing failed after all cleanup attempts")
            else:
                # Enhanced fallback structure optimized for single-call generation
                setup_json = {
                    "template_config": {
                        "venue": target_venue,
                        "page_limit": 8,
                        "format": "academic_paper",
                        "citation_style": "ACM",
                        "target_word_count": 6000,
                        "quality_standard": "publication_ready"
                    },
                    "paper_structure": {
                        "sections": [
                            {
                                "name": "Abstract", 
                                "length": "200-250 words", 
                                "focus": "comprehensive_summary",
                                "description": "Concise overview of problem, method, key results, and implications",
                                "content_requirements": ["problem statement", "methodology overview", "key findings", "significance"],
                                "flow_position": "standalone_summary"
                            },
                            {
                                "name": "Introduction", 
                                "length": "1-1.5 pages", 
                                "focus": "motivation_and_context",
                                "description": "Establish problem significance, review relevant work, present contributions",
                                "content_requirements": ["problem motivation", "research gap", "related work summary", "clear contributions", "paper organization"],
                                "flow_position": "foundation_setting"
                            },
                            {
                                "name": "Methodology", 
                                "length": "2-2.5 pages", 
                                "focus": "technical_approach",
                                "description": "Detailed description of methods, experimental design, and implementation",
                                "content_requirements": ["approach overview", "technical details", "experimental setup", "evaluation metrics", "implementation details"],
                                "flow_position": "technical_foundation"
                            },
                            {
                                "name": "Results", 
                                "length": "2-2.5 pages", 
                                "focus": "findings_and_analysis",
                                "description": "Present experimental results, analysis, and interpretation",
                                "content_requirements": ["experimental results", "statistical analysis", "performance comparisons", "result interpretation", "findings discussion"],
                                "flow_position": "evidence_presentation"
                            },
                            {
                                "name": "Discussion", 
                                "length": "1 page", 
                                "focus": "implications_and_limitations",
                                "description": "Interpret results, discuss implications, acknowledge limitations",
                                "content_requirements": ["result interpretation", "broader implications", "limitations acknowledgment", "future work suggestions"],
                                "flow_position": "synthesis_and_reflection"
                            },
                            {
                                "name": "Conclusion", 
                                "length": "0.5 pages", 
                                "focus": "summary_and_impact",
                                "description": "Summarize contributions and significance",
                                "content_requirements": ["contribution summary", "key findings recap", "broader impact", "final thoughts"],
                                "flow_position": "closure_and_impact"
                            }
                        ]
                    },
                    "content_guidelines": {
                        "narrative_flow": "Each section builds logically on previous sections with smooth transitions",
                        "citation_strategy": "Integrate sources naturally throughout all sections to support claims",
                        "technical_depth": "Balance accessibility with rigor appropriate for target venue",
                        "coherence_requirements": "Maintain consistent terminology and argument thread throughout",
                        "quality_standards": ["publication_ready", "peer_review_quality", "clear_writing", "proper_citations"],
                        "emphasis": ["methodology_rigor", "experimental_validation", "clear_presentation"],
                        "tone": "formal_academic",
                        "target_audience": "researchers_and_practitioners",
                        "writing_style": "clear_concise_authoritative"
                    },
                    "single_call_optimization": {
                        "structure_clarity": "Provide clear section boundaries and content expectations",
                        "flow_guidance": "Include transition requirements between sections",
                        "completeness_requirements": "Ensure each section is self-contained yet interconnected",
                        "quality_checkpoints": "Built-in requirements for academic rigor and publication standards"
                    }
                }
        except Exception as e:
            # Use the enhanced fallback structure from above
            setup_json = {
                "template_config": {
                    "venue": target_venue,
                    "page_limit": 8,
                    "format": "academic_paper",
                    "citation_style": "ACM",
                    "target_word_count": 6000,
                    "quality_standard": "publication_ready"
                },
                "paper_structure": {
                    "sections": [
                        {
                            "name": "Abstract", 
                            "length": "200-250 words", 
                            "focus": "comprehensive_summary",
                            "description": "Concise overview of problem, method, key results, and implications"
                        },
                        {
                            "name": "Introduction", 
                            "length": "1-1.5 pages", 
                            "focus": "motivation_and_context",
                            "description": "Establish problem significance, review relevant work, present contributions"
                        },
                        {
                            "name": "Methodology", 
                            "length": "2-2.5 pages", 
                            "focus": "technical_approach",
                            "description": "Detailed description of methods, experimental design, and implementation"
                        },
                        {
                            "name": "Results", 
                            "length": "2-2.5 pages", 
                            "focus": "findings_and_analysis",
                            "description": "Present experimental results, analysis, and interpretation"
                        },
                        {
                            "name": "Discussion", 
                            "length": "1 page", 
                            "focus": "implications_and_limitations",
                            "description": "Interpret results, discuss implications, acknowledge limitations"
                        },
                        {
                            "name": "Conclusion", 
                            "length": "0.5 pages", 
                            "focus": "summary_and_impact",
                            "description": "Summarize contributions and significance"
                        }
                    ]
                },
                "content_guidelines": {
                    "emphasis": ["methodology_rigor", "experimental_validation", "clear_presentation"],
                    "tone": "formal_academic",
                    "target_audience": "researchers_and_practitioners"
                }
            }

        sections_count = len(setup_json.get("paper_structure", {}).get("sections", []))

        _write_stream(f"✅ Paper structure ready - {sections_count} sections configured")

        return {
            **state,
            "paper_structure": setup_json.get("paper_structure", {}),
            "template_config": setup_json.get("template_config", {}),
            "current_step": "paper_setup_complete"
        }

    except Exception as e:
        return {
            **state,
            "errors": state.get("errors", []) + [f"Setup error: {str(e)}"],
            "current_step": "setup_error"
        }