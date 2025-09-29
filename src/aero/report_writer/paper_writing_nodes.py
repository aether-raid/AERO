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

class BaseState(TypedDict):
    """Base state for all workflows."""
    messages: Annotated[List[BaseMessage], add_messages]
    original_prompt: str  # Pure user query without uploaded data
    uploaded_data: List[str]  # Uploaded file contents as separate field
    current_step: str
    errors: List[str]

# ==================================================================================
# STATE DEFINITIONS
# ==================================================================================

class BaseState(TypedDict):
    """Base state for all workflows."""
    messages: List[BaseMessage]
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
# CLIENT INITIALIZATION
# ==================================================================================

# Global client variables (initialized once)
client = None
tavily_client = None
model = None

def _initialize_clients():
    """Initialize OpenAI and Tavily clients for standalone operation."""
    global client, tavily_client, model

    if client is None:
        # Load configuration from env.example file
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("BASE_URL")
        model = os.getenv("DEFAULT_MODEL") or "gemini/gemini-2.5-flash"

        if not api_key:
            raise ValueError("OPENAI_API_KEY not found")

        from openai import OpenAI
        client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )

    if tavily_client is None:
        # Initialize Tavily client
        tavily_api_key = os.getenv("TAVILY_API_KEY")  # Using the key from the example file
        try:
            tavily_client = TavilyClient(api_key=tavily_api_key)
        except Exception as e:
            tavily_client = None

# ==================================================================================
# FILE EXTRACTION UTILITIES
# ==================================================================================

def _docx_extract_full(doc_bytes: bytes) -> str:
    """Extract full text from DOCX, including tables."""
    if Document is None:
        return "(python-docx not available - install with: pip install python-docx)"
    
    try:
        doc = Document(BytesIO(doc_bytes))
        parts: List[str] = []
        
        # Extract paragraphs
        for p in doc.paragraphs:
            if p.text is not None:
                parts.append(p.text)
        
        # Extract tables
        for t in getattr(doc, 'tables', []):
            for row in t.rows:
                cells = [c.text.replace('\n', ' ').strip() for c in row.cells]
                parts.append("\t".join(cells))
        
        return "\n".join([p for p in parts if p and p.strip()])
    except Exception as e:
        return f"(DOCX extraction error: {e})"

def extract_files_from_paths(file_paths: List[str]) -> List[str]:
    """
    Extract content from file paths and return formatted strings.
    
    Args:
        file_paths: List of file paths to process
        
    Returns:
        List of formatted strings with file content
    """
    parsed_contexts: List[str] = []
    
    for file_path in file_paths:
        try:
            path = Path(file_path)
            if not path.exists():
                parsed_contexts.append(f"[{path.name}] (file not found)")
                continue
            
            # Read file content
            content = path.read_bytes()
            name = path.name
            lower = name.lower()
            
            # Process based on file type
            if lower.endswith((".csv",)) and pd is not None:
                try:
                    df = pd.read_csv(BytesIO(content))
                    csv_full = df.to_csv(index=False)
                    parsed_contexts.append(f"[CSV:{name}]\n{csv_full}")
                except Exception as e:
                    parsed_contexts.append(f"[CSV:{name}] (parse error: {e})")
                    
            elif lower.endswith((".xlsx", ".xls")) and pd is not None:
                try:
                    xls = pd.ExcelFile(BytesIO(content))
                    for sheet in xls.sheet_names:
                        df = xls.parse(sheet)
                        csv_full = df.to_csv(index=False)
                        parsed_contexts.append(f"[XLSX:{name}:{sheet}]\n{csv_full}")
                except Exception as e:
                    parsed_contexts.append(f"[XLSX:{name}] (parse error: {e})")
                    
            elif lower.endswith((".docx",)) and Document is not None:
                text = _docx_extract_full(content)
                parsed_contexts.append(f"[DOCX:{name}]\n{text}")
                
            elif lower.endswith((".doc",)):
                parsed_contexts.append(f"[DOC:{name}] (binary .doc format not supported, use .docx)")
                
            elif lower.endswith((".txt", ".md")):
                try:
                    text = content.decode('utf-8')
                    parsed_contexts.append(f"[TXT:{name}]\n{text}")
                except Exception as e:
                    parsed_contexts.append(f"[TXT:{name}] (encoding error: {e})")
                    
            else:
                parsed_contexts.append(f"[{name}] (unsupported file type)")
                
        except Exception as ex:
            parsed_contexts.append(f"[{Path(file_path).name}] (processing error: {ex})")
    
    return parsed_contexts

def extract_files_from_bytes(files_data: List[Dict[str, Any]]) -> List[str]:
    """
    Extract content from file data (bytes + filename) and return formatted strings.
    
    Args:
        files_data: List of dicts with 'content' (bytes) and 'filename' (str) keys
        
    Returns:
        List of formatted strings with file content
    """
    parsed_contexts: List[str] = []
    
    for file_data in files_data:
        try:
            content = file_data.get('content')
            name = file_data.get('filename', 'unknown_file')
            
            if not isinstance(content, bytes):
                parsed_contexts.append(f"[{name}] (invalid content format - expected bytes)")
                continue
            
            lower = name.lower()
            
            # Process based on file type
            if lower.endswith((".csv",)) and pd is not None:
                try:
                    df = pd.read_csv(BytesIO(content))
                    csv_full = df.to_csv(index=False)
                    parsed_contexts.append(f"[CSV:{name}]\n{csv_full}")
                except Exception as e:
                    parsed_contexts.append(f"[CSV:{name}] (parse error: {e})")
                    
            elif lower.endswith((".xlsx", ".xls")) and pd is not None:
                try:
                    xls = pd.ExcelFile(BytesIO(content))
                    for sheet in xls.sheet_names:
                        df = xls.parse(sheet)
                        csv_full = df.to_csv(index=False)
                        parsed_contexts.append(f"[XLSX:{name}:{sheet}]\n{csv_full}")
                except Exception as e:
                    parsed_contexts.append(f"[XLSX:{name}] (parse error: {e})")
                    
            elif lower.endswith((".docx",)) and Document is not None:
                text = _docx_extract_full(content)
                parsed_contexts.append(f"[DOCX:{name}]\n{text}")
                
            elif lower.endswith((".doc",)):
                parsed_contexts.append(f"[DOC:{name}] (binary .doc format not supported, use .docx)")
                
            elif lower.endswith((".txt", ".md")):
                try:
                    text = content.decode('utf-8')
                    parsed_contexts.append(f"[TXT:{name}]\n{text}")
                except Exception as e:
                    parsed_contexts.append(f"[TXT:{name}] (encoding error: {e})")
                    
            else:
                parsed_contexts.append(f"[{name}] (unsupported file type)")
                
        except Exception as ex:
            filename = file_data.get('filename', 'unknown_file')
            parsed_contexts.append(f"[{filename}] (processing error: {ex})")
    
    return parsed_contexts

def create_file_data_from_paths(file_paths: List[str]) -> List[Dict[str, Any]]:
    """
    Convenience function to create files_data from file paths.
    
    Args:
        file_paths: List of file paths
        
    Returns:
        List of file data dicts suitable for files_data parameter
    """
    files_data = []
    for file_path in file_paths:
        try:
            path = Path(file_path)
            if path.exists():
                content = path.read_bytes()
                files_data.append({
                    'content': content,
                    'filename': path.name
                })
        except Exception as e:
            pass  # Skip files that can't be read
    
    return files_data

# ==================================================================================
# PAPER WRITING WORKFLOW NODES
# ==================================================================================

async def _analyze_results_node(state: PaperWritingState) -> PaperWritingState:
    """Node for analyzing experimental results and research context, including uploaded file data."""
    
    _write_stream("📊 Analyzing experimental results and research context")

    try:
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
            lambda: client.chat.completions.create(
                model=model,
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

async def _setup_paper_node(state: PaperWritingState) -> PaperWritingState:
    """Node for LLM-driven template selection and comprehensive paper structure planning."""

    _write_stream("🏗️ Setting up comprehensive paper structure")

    try:
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
            lambda: client.chat.completions.create(
                model=model,
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

async def _find_supporting_sources_node(state: PaperWritingState) -> PaperWritingState:
    """🔍 Find supporting sources and citations using Tavily web search, enhanced with uploaded file context."""

    _write_stream("🔍 Finding supporting sources and citations")

    try:
        # Check if Tavily client is available
        if tavily_client is None:
            _write_stream("⚠️ Web search unavailable - skipping source finding")
            return {
                **state,
                "supporting_sources": [],
                "citation_database": {},
                "source_search_queries": [],
                "current_step": "sources_skipped"
            }

        # Extract research context and analysis
        research_analysis = state.get("research_analysis", {})
        experimental_results = state.get("experimental_results", {})
        paper_structure = state.get("paper_structure", {})
        research_context = state.get("research_context", "")
        uploaded_data = state.get("uploaded_data", [])

        # Extract file-specific context for enhanced search queries
        file_context = ""
        if uploaded_data:
            _write_stream("📄 Analyzing uploaded files for search context")

            # Extract key terms and context from uploaded files
            csv_keywords = []
            doc_keywords = []

            for file_content in uploaded_data:
                if '[CSV:' in file_content or '[XLSX:' in file_content:
                    # Extract column headers as potential search terms
                    lines = file_content.split('\n')
                    if len(lines) > 1:
                        headers = lines[1].split(',') if len(lines) > 1 else []
                        csv_keywords.extend([h.strip().replace('"', '') for h in headers[:10]])

                elif '[DOCX:' in file_content:
                    # Extract key phrases from document content
                    doc_text = '\n'.join(file_content.split('\n')[1:])
                    # Simple keyword extraction (could be enhanced)
                    words = doc_text.lower().split()
                    # Look for technical terms (longer words, mixed case)
                    doc_keywords.extend([w for w in words if len(w) > 6 and any(c.isupper() for c in w)])

            if csv_keywords:
                file_context += f" Data includes metrics: {', '.join(csv_keywords[:8])}"
            if doc_keywords:
                file_context += f" Document keywords: {', '.join(doc_keywords[:8])}"

        _write_stream("🔍 Generating search queries from research analysis")
        # Generate search queries based on research content
        key_findings = research_analysis.get("key_findings", [])
        methodology = research_analysis.get("methodology", "")
        domain_context = research_analysis.get("domain_analysis", {})
        data_description = research_analysis.get("data_description", "")

        # Create targeted search queries for citations
        search_queries = []

        # Query 1: Background and related work (enhanced with file context)
        domain = domain_context.get("primary_domain") if isinstance(domain_context, dict) else research_analysis.get("domain", "machine learning")
        background_query = f"{domain} recent advances state of the art{file_context}"
        search_queries.append({
            "query": background_query,
            "purpose": "background_literature",
            "section": "introduction_related_work"
        })

        # Query 2: Methodology and techniques (enhanced with data types)
        method_context = methodology or research_analysis.get("research_type", "experimental")
        if uploaded_data:
            method_context += f" {data_description}"
        method_query = f"{method_context} methodology techniques recent papers"
        search_queries.append({
            "query": method_query,
            "purpose": "methodology_validation",
            "section": "methodology"
        })

        # Query 3: Results validation and comparison (enhanced with findings)
        findings_text = key_findings if isinstance(key_findings, str) else ' '.join(key_findings) if key_findings else research_analysis.get("key_findings", "")
        if file_context:
            findings_text += file_context
        findings_query = f"{findings_text} results comparison evaluation"
        search_queries.append({
            "query": findings_query,
            "purpose": "results_validation",
            "section": "results_discussion"
        })

        # Query 4: Data-specific search (if files uploaded)
        if uploaded_data:
            data_query = f"{research_context} {data_description} dataset analysis"
            search_queries.append({
                "query": data_query,
                "purpose": "data_validation",
                "section": "data_analysis"
            })
        else:
            # General domain research (fallback)
            context_query = f"{research_context[:100]} recent research papers"
            search_queries.append({
                "query": context_query,
                "purpose": "general_context",
                "section": "general"
            })

        # Perform Tavily searches
        all_sources = []
        citation_database = {}

        _write_stream(f"🌐 Performing {len(search_queries[:4])} web searches for sources")

        for i, search_item in enumerate(search_queries[:4]):  # Limit to 4 searches
            query = search_item["query"]
            purpose = search_item["purpose"]
            section = search_item["section"]

            _write_stream(f"🔍 Search {i+1}: {query[:50]}...")

            try:

                # Execute Tavily search
                search_response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda q=query: tavily_client.search(q, max_results=8)
                )

                if search_response and "results" in search_response:
                    sources_found = 0
                    _write_stream(f"📄 Processing {len(search_response['results'])} results")
                    for result in search_response["results"]:
                        # Extract citation information
                        source_info = {
                            "title": result.get("title", ""),
                            "url": result.get("url", ""),
                            "content": result.get("content", "")[:500],  # Limit content
                            "published_date": result.get("published_date", ""),
                            "purpose": purpose,
                            "section": section,
                            "relevance_score": result.get("score", 0.5),
                            "search_query": query
                        }

                        # Filter for academic/research sources
                        url_lower = source_info["url"].lower()
                        title_lower = source_info["title"].lower()

                        # Prioritize academic sources
                        is_academic = any(domain in url_lower for domain in [
                            'arxiv.org', 'doi.org', 'ieee.org', 'acm.org', 'springer.com',
                            'elsevier.com', 'nature.com', 'science.org', 'plos.org'
                        ])

                        # Check if content seems research-related
                        has_research_keywords = any(keyword in title_lower for keyword in [
                            'research', 'study', 'analysis', 'method', 'algorithm',
                            'evaluation', 'experiment', 'approach', 'framework'
                        ])

                        if is_academic or has_research_keywords:
                            all_sources.append(source_info)
                            sources_found += 1

                            # Organize by section for easy citation
                            if section not in citation_database:
                                citation_database[section] = []
                            citation_database[section].append(source_info)

            except Exception as e:
                continue

        # Summary
        total_sources = len(all_sources)
        sections_with_sources = len(citation_database)

        _write_stream(f"✅ Found {total_sources} sources across {sections_with_sources} sections")

        return {
            **state,
            "supporting_sources": all_sources,
            "citation_database": citation_database,
            "source_search_queries": [sq["query"] for sq in search_queries],
            "source_validation_results": {
                "total_sources": total_sources,
                "sections_covered": list(citation_database.keys()),
                "search_success_rate": sections_with_sources / len(search_queries) if search_queries else 0
            },
            "current_step": "sources_found"
        }

    except Exception as e:
        return {
            **state,
            "errors": state.get("errors", []) + [f"Source finding error: {str(e)}"],
            "supporting_sources": [],
            "citation_database": {},
            "current_step": "source_finding_error"
        }

async def _generate_content_node(state: PaperWritingState) -> PaperWritingState:
    """Node for generating the entire paper content in a single comprehensive LLM call."""
    
    _write_stream("🚀 Starting paper content generation")
    
    # Check if this is a refinement iteration
    refinement_count = state.get("refinement_count", 0)
    critique_results = state.get("critique_results", {})
    
    if refinement_count > 0:
        _write_stream(f"🔄 Refining paper based on critique (iteration {refinement_count})")
        major_issues = critique_results.get("major_issues", [])
        suggestions = critique_results.get("suggestions", [])
    else:
        _write_stream("✍️ Generating complete paper with citations")

    try:
        _write_stream("📋 Preparing paper structure and context")
        
        research_analysis = state.get("research_analysis", {})
        paper_structure = state.get("paper_structure", {})
        experimental_results = state.get("experimental_results", {})
        uploaded_data = state.get("uploaded_data", [])
        original_prompt = state.get("original_prompt", "")

        # Get Tavily-sourced citations and supporting sources
        supporting_sources = state.get("supporting_sources", [])
        citation_database = state.get("citation_database", {})
        source_validation = state.get("source_validation_results", {})

        _write_stream("📚 Processing citations and sources")

        sections = paper_structure.get("sections", [])

        # Prepare comprehensive citations context for the entire paper
        citations_context = ""
        if supporting_sources:
            _write_stream(f"🔗 Integrating {len(supporting_sources)} citations")
            citations_context = "\n\n📚 COMPREHENSIVE SOURCE DATABASE FOR CITATION:\n"
            citations_context += "=" * 60 + "\n"
            for i, source in enumerate(supporting_sources[:15], 1):  # Use up to 15 sources
                citations_context += f"\n[{i}] {source.get('title', 'Unknown Title')}\n"
                citations_context += f"    URL: {source.get('url', 'No URL')}\n"
                citations_context += f"    Content: {source.get('content', '')}...\n"
                citations_context += f"    Relevance: {source.get('purpose', 'general')}\n"
                citations_context += f"    Domain: {source.get('url', '').split('//')[1].split('/')[0] if '//' in source.get('url', '') else 'Unknown'}\n"
                citations_context += "-" * 40 + "\n"

            citations_context += f"""
📝 CITATION INTEGRATION INSTRUCTIONS:
- Use [1], [2], [3], etc. format for in-text citations
- Integrate citations naturally throughout ALL sections
- Support claims, methods, comparisons, and background with appropriate sources
- Aim for 2-4 citations per major section
- Use diverse sources across different aspects of the research
- Prioritize academic and research sources for credibility
- Ensure citations are contextually relevant to the content they support
"""

        # Prepare detailed paper structure for comprehensive generation
        structure_context = "\n\n📋 DETAILED PAPER STRUCTURE TO FOLLOW:\n"
        structure_context += "=" * 60 + "\n"
        for i, section in enumerate(sections, 1):
            section_name = section.get("name", "Unknown")
            section_focus = section.get("focus", "general")
            section_length = section.get("length", "1 page")
            section_description = section.get("description", "No description")
            
            structure_context += f"\n{i}. {section_name.upper()}\n"
            structure_context += f"   Focus: {section_focus}\n"
            structure_context += f"   Length: {section_length}\n"
            structure_context += f"   Content: {section_description}\n"
            structure_context += "-" * 40 + "\n"

        # Build refinement context if this is a revision
        refinement_context = ""
        if refinement_count > 0 and critique_results:
            major_issues = critique_results.get("major_issues", [])
            specific_improvements = critique_results.get("specific_improvements", [])
            critical_analysis = critique_results.get("critical_analysis", {})
            technical_feedback = critique_results.get("technical_feedback", {})
            writing_improvements = critique_results.get("writing_improvements", {})
            
            refinement_context = f"""
🔄 COMPREHENSIVE REFINEMENT REQUIREMENTS (Iteration {refinement_count}):
{"=" * 80}

🎯 **PREVIOUS CRITIQUE SUMMARY:**
- Overall Score: {critique_results.get('overall_score', 0):.1f}/10
- Recommendation: {critique_results.get('recommendation', 'N/A')}
- Revision Priority: {critique_results.get('revision_priority', 'medium')}

🚨 **CRITICAL ISSUES TO ADDRESS THROUGHOUT THE PAPER:**
{chr(10).join([f"❌ {issue}" for issue in major_issues[:6]])}

🎯 **SPECIFIC IMPROVEMENTS TO IMPLEMENT:**
{chr(10).join([f"✅ {improvement}" for improvement in specific_improvements[:8]])}

🔬 **TECHNICAL IMPROVEMENTS NEEDED:**
"""
            methodology_gaps = technical_feedback.get("methodology_gaps", [])
            experimental_weaknesses = technical_feedback.get("experimental_weaknesses", [])
            for gap in methodology_gaps[:3]:
                refinement_context += f"   🔧 {gap}\n"
            for weakness in experimental_weaknesses[:3]:
                refinement_context += f"   ⚡ {weakness}\n"

            clarity_issues = writing_improvements.get("clarity_issues", [])
            if clarity_issues:
                refinement_context += f"\n✍️ WRITING CLARITY IMPROVEMENTS:\n"
                refinement_context += chr(10).join([f"   📝 {issue}" for issue in clarity_issues[:4]])

            refinement_context += f"""

🎯 **PRIORITY ACTION ITEMS FOR COMPLETE PAPER REVISION:**
1. Address ALL critical issues listed above throughout the paper
2. Implement comprehensive improvements in every section
3. Maintain academic rigor and proper citation integration
4. Ensure logical flow and coherent narrative throughout
5. Write clearly and concisely for the target venue
6. Significantly improve quality from previous iteration

**CRITICAL**: This is iteration {refinement_count} - the entire paper must show substantial improvements addressing ALL critique points.
"""

        # Prepare uploaded data context
        data_context = ""
        if uploaded_data:
            data_context = "\n\n📊 UPLOADED DATA AND FILES TO INTEGRATE:\n"
            data_context += "=" * 60 + "\n"
            for i, data in enumerate(uploaded_data[:5], 1):  # Limit to first 5 files for context
                data_preview = data[:1000] + "..." if len(data) > 1000 else data
                data_context += f"\nFile {i}:\n{data_preview}\n"
                data_context += "-" * 40 + "\n"
            if len(uploaded_data) > 5:
                data_context += f"\n... and {len(uploaded_data) - 5} additional files to consider\n"

        # Create the comprehensive prompt for the entire paper
        comprehensive_prompt = f"""
Write a complete academic research paper based on the following requirements. Generate the ENTIRE paper in a single response with all sections flowing coherently together.

🎯 **RESEARCH TOPIC:** {original_prompt}

📊 **RESEARCH ANALYSIS:** {research_analysis}

🧪 **EXPERIMENTAL RESULTS:** {experimental_results}

{data_context}

{structure_context}

{citations_context}

{refinement_context}

📝 **COMPREHENSIVE PAPER GENERATION INSTRUCTIONS:**

1. **OVERALL STRUCTURE:** Generate a complete academic paper following the exact structure provided above
2. **COHERENT FLOW:** Ensure smooth transitions between sections and consistent narrative throughout
3. **CITATION INTEGRATION:** Integrate citations naturally throughout ALL sections using the provided sources
4. **ACADEMIC RIGOR:** Use formal academic tone, proper methodology, and rigorous analysis
5. **COMPREHENSIVE COVERAGE:** Address all aspects of the research topic thoroughly
6. **LOGICAL PROGRESSION:** Build arguments logically from introduction through conclusions
7. **CONSISTENCY:** Maintain consistent terminology, style, and quality across all sections
8. **PUBLICATION-READY:** Generate content suitable for academic publication

📋 **FORMATTING REQUIREMENTS:**
- Use clear section headers (# for main sections, ## for subsections)
- Include in-text citations in [1], [2], etc. format
- Maintain academic writing style throughout
- Ensure proper paragraph structure and transitions
- Include specific details from experimental results and uploaded data
- Make each section substantive and well-developed

🎯 **CRITICAL SUCCESS FACTORS:**
- The paper must read as a coherent, unified work, not separate sections
- All claims must be properly supported with citations
- The research must be presented clearly and convincingly
- The paper must demonstrate academic rigor and originality
{f"- Address ALL critique points comprehensively throughout the paper" if refinement_count > 0 else ""}

Generate the complete academic research paper now:
        """

        _write_stream("🤖 Generating complete paper with AI")

        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert academic writer specializing in comprehensive research papers with integrated citations. Generate complete, publication-ready papers with excellent flow and coherence."},
                    {"role": "user", "content": comprehensive_prompt}
                ],
                temperature=0.2,  # Lower temperature for more consistent academic writing
                max_tokens=16000  # Increased token limit for comprehensive generation
            )
        )

        _write_stream("📝 Processing generated paper content")
        complete_paper = response.choices[0].message.content.strip()
        
        _write_stream("🔍 Parsing paper into sections")
        # Parse the complete paper into sections for compatibility with existing workflow
        section_content = _parse_complete_paper_into_sections(complete_paper, sections)
        
        _write_stream("📚 Generating reference list")
        # Generate reference list from all used sources
        reference_list = _generate_reference_list(supporting_sources)
        if reference_list and "References" not in section_content:
            section_content["References"] = reference_list

        # Count citations used in the complete paper
        total_citations_used = sum(1 for i in range(1, len(supporting_sources) + 1) if f"[{i}]" in complete_paper)

        _write_stream(f"✅ Paper generation complete - {len(complete_paper)} chars, {total_citations_used} citations")

        return {
            **state,
            "section_content": section_content,
            "formatted_paper": complete_paper,  # Store the complete paper as well
            "current_step": "content_generated"
        }

    except Exception as e:
        return {
            **state,
            "errors": state.get("errors", []) + [f"Content generation error: {str(e)}"],
            "current_step": "content_generation_error"
        }
    else:
        pass

    try:
        research_analysis = state.get("research_analysis", {})
        paper_structure = state.get("paper_structure", {})
        experimental_results = state.get("experimental_results", {})

        # Get Tavily-sourced citations and supporting sources
        supporting_sources = state.get("supporting_sources", [])
        citation_database = state.get("citation_database", {})
        source_validation = state.get("source_validation_results", {})

        sections = paper_structure.get("sections", [])
        section_content = {}

        # Generate sections SEQUENTIALLY with context from previous sections
        for section_idx, section in enumerate(sections):
            section_name = section.get("name", "Unknown")
            section_focus = section.get("focus", "general")
            section_length = section.get("length", "1 page")


            # Build context from all previously written sections
            previous_sections_context = ""
            if section_content:
                previous_sections_context = "\n**PREVIOUSLY WRITTEN SECTIONS (for context and continuity):**\n"
                for prev_section, prev_content in section_content.items():
                    # Truncate very long sections for context
                    content_preview = prev_content[:800] + "..." if len(prev_content) > 800 else prev_content
                    previous_sections_context += f"\n### {prev_section}:\n{content_preview}\n"
                
                previous_sections_context += "\n**IMPORTANT CONTINUITY REQUIREMENTS:**\n"
                previous_sections_context += "- Maintain consistent terminology with previous sections\n"
                previous_sections_context += "- Build logically on concepts already introduced\n"
                previous_sections_context += "- Avoid repetition of content already covered\n"
                previous_sections_context += "- Ensure smooth transitions and narrative flow\n"
                previous_sections_context += "- Reference previous sections when appropriate (e.g., 'As mentioned in the Introduction...')\n"
                previous_sections_context += "- Use consistent formatting and style\n\n"

            # Find relevant sources for this section
            section_sources = []
            section_key = section_name.lower().replace(" ", "_")

            # Look for section-specific sources
            for source_key, sources in citation_database.items():
                if (section_key in source_key.lower() or
                    section_focus in source_key.lower() or
                    source_key == "general"):
                    section_sources.extend(sources[:3])  # Limit to 3 sources per category

            # If no specific sources, use general sources
            if not section_sources and supporting_sources:
                section_sources = supporting_sources[:4]  # Use first 4 general sources

            # Prepare citation context for LLM
            citations_context = ""
            if section_sources:
                citations_context = "\n\nAVAILABLE SOURCES FOR CITATION:\n"
                for i, source in enumerate(section_sources[:5], 1):  # Limit to 5 sources max
                    citations_context += f"\n[{i}] {source.get('title', 'Unknown Title')}\n"
                    citations_context += f"    URL: {source.get('url', 'No URL')}\n"
                    citations_context += f"    Content: {source.get('content', '')[:200]}...\n"
                    citations_context += f"    Relevance: {source.get('purpose', 'general')}\n"

                citations_context += "\n📝 CITATION INSTRUCTIONS:\n"
                citations_context += "- Reference sources using [1], [2], etc. format\n"
                citations_context += "- Use citations to support claims, methods, and comparisons\n"
                citations_context += "- Integrate citations naturally into the text\n"
                citations_context += "- Prioritize academic and research sources\n"

            # Build refinement context if this is a revision
            refinement_context = ""
            if refinement_count > 0 and critique_results:
                major_issues = critique_results.get("major_issues", [])
                specific_improvements = critique_results.get("specific_improvements", [])
                critical_analysis = critique_results.get("critical_analysis", {})
                technical_feedback = critique_results.get("technical_feedback", {})
                writing_improvements = critique_results.get("writing_improvements", {})
                
                # Get section-specific feedback
                section_key = section_name.lower().replace(" ", "_").replace("&", "").strip()
                section_feedback = critical_analysis.get(section_key, {})
                section_problems = section_feedback.get("problems", [])
                section_recommendations = section_feedback.get("recommendations", [])
                
                refinement_context = f"""
**🔄 REFINEMENT ITERATION {refinement_count} - CRITICAL IMPROVEMENTS NEEDED:**

� **Previous Critique Summary:**
- Overall Score: {critique_results.get('overall_score', 0):.1f}/10
- Recommendation: {critique_results.get('recommendation', 'N/A')}
- Revision Priority: {critique_results.get('revision_priority', 'medium')}

🚨 **CRITICAL ISSUES TO FIX:**
{chr(10).join([f"❌ {issue}" for issue in major_issues[:4]])}

🎯 **SPECIFIC IMPROVEMENTS FOR THIS PAPER:**
{chr(10).join([f"✅ {improvement}" for improvement in specific_improvements[:5]])}

📝 **{section_name.upper()} SECTION - SPECIFIC PROBLEMS & SOLUTIONS:**
"""
                if section_problems:
                    refinement_context += f"\n🔴 Problems in {section_name}:\n"
                    refinement_context += chr(10).join([f"   • {problem}" for problem in section_problems[:3]])
                
                if section_recommendations:
                    refinement_context += f"\n🔧 Specific Fixes for {section_name}:\n"
                    refinement_context += chr(10).join([f"   → {rec}" for rec in section_recommendations[:3]])

                # Add technical improvements if relevant
                if section_name.lower() in ['methodology', 'methods', 'experimental setup']:
                    methodology_gaps = technical_feedback.get("methodology_gaps", [])
                    experimental_weaknesses = technical_feedback.get("experimental_weaknesses", [])
                    if methodology_gaps or experimental_weaknesses:
                        refinement_context += f"\n🔬 TECHNICAL IMPROVEMENTS NEEDED:\n"
                        for gap in methodology_gaps[:2]:
                            refinement_context += f"   🔧 {gap}\n"
                        for weakness in experimental_weaknesses[:2]:
                            refinement_context += f"   ⚡ {weakness}\n"

                # Add writing improvements
                clarity_issues = writing_improvements.get("clarity_issues", [])
                if clarity_issues:
                    refinement_context += f"\n✍️ WRITING CLARITY IMPROVEMENTS:\n"
                    refinement_context += chr(10).join([f"   📝 {issue}" for issue in clarity_issues[:2]])

                refinement_context += f"""

🎯 **PRIORITY ACTION ITEMS FOR {section_name}:**
1. Address the specific problems listed above
2. Implement the recommended solutions
3. Maintain academic rigor and proper citations
4. Ensure the content flows logically with other sections
5. Write clearly and concisely for the target venue

**IMPROVEMENT FOCUS:** This is iteration {refinement_count} - make substantial improvements to address ALL critique points.
"""

            content_prompt = f"""
Write the {section_name} section for an academic research paper with proper citations.

{previous_sections_context}

{refinement_context}

**Research Context**: {research_analysis}

**Experimental Results**: {experimental_results}

**Section Focus**: {section_focus}
**Target Length**: {section_length}

{citations_context}

**Guidelines:**
- Use formal academic tone appropriate for {section_name}
- Include specific details from the research
- Follow standard academic writing conventions
- Make it publication-ready
- Integrate citations naturally to support claims
- Use [1], [2], etc. format for in-text citations
- Ensure claims are backed by appropriate sources
- **CRITICAL**: Maintain consistency with previously written sections
- **CRITICAL**: Build logically on content already established
- **CRITICAL**: Avoid repeating information from previous sections
- **CRITICAL**: Use smooth transitions that connect to previous sections
{f"- PRIORITY: Address critique feedback and improve quality from previous iteration" if refinement_count > 0 else ""}

Write a complete {section_name} section with integrated citations that flows naturally from the previous sections:
            """

            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are an expert academic writer specializing in research papers with proper citation integration."},
                        {"role": "user", "content": content_prompt}
                    ],
                    temperature=0.3
                )
            )

            section_text = response.choices[0].message.content.strip()
            section_content[section_name] = section_text

            # Track which sources were used
            used_citations = []
            for i, source in enumerate(section_sources[:5], 1):
                if f"[{i}]" in section_text:
                    used_citations.append(source)

        # Generate reference list from all used sources
        reference_list = _generate_reference_list(supporting_sources)
        if reference_list:
            section_content["References"] = reference_list

        total_sources_used = len([s for s in supporting_sources if any(
            f"[{i}]" in content for i, content in enumerate(section_content.values(), 1)
        )])

        return {
            **state,
            "section_content": section_content,
            "current_step": "content_generated"
        }

    except Exception as e:
        return {
            **state,
            "errors": state.get("errors", []) + [f"Content generation error: {str(e)}"],
            "current_step": "content_generation_error"
        }

def _parse_complete_paper_into_sections(complete_paper: str, expected_sections: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    Parse a complete paper into individual sections for compatibility with existing workflow.
    
    Args:
        complete_paper: The complete paper text with section headers
        expected_sections: List of expected section dictionaries with 'name' keys
        
    Returns:
        Dictionary mapping section names to their content
    """
    section_content = {}
    
    # Get expected section names
    expected_names = [section.get("name", "") for section in expected_sections]
    
    # Add common sections that might not be in the structure
    common_sections = ["Abstract", "Introduction", "Related Work", "Methodology", "Results", 
                      "Discussion", "Conclusion", "References", "Acknowledgments"]
    all_expected = expected_names + [s for s in common_sections if s not in expected_names]
    
    # Split the paper by section headers (both # and ## formats)
    import re
    
    # Find all section headers in the paper
    header_pattern = r'^#+\s*(.+?)$'
    headers = re.findall(header_pattern, complete_paper, re.MULTILINE)
    
    if not headers:
        # If no headers found, try to split by expected section names
        for section_name in all_expected:
            # Look for the section name as a standalone line or with formatting
            patterns = [
                f"^{re.escape(section_name)}$",
                rf"^#+\s*{re.escape(section_name)}\s*$",
                f"^{re.escape(section_name.upper())}$",
                rf"^\*\*{re.escape(section_name)}\*\*$"
            ]
            
            for pattern in patterns:
                matches = list(re.finditer(pattern, complete_paper, re.MULTILINE | re.IGNORECASE))
                if matches:
                    break
        
        # If still no clear structure, return the whole paper as a single section
        if not any(re.search(pattern, complete_paper, re.MULTILINE | re.IGNORECASE) 
                  for section_name in all_expected 
                  for pattern in [rf"^#+\s*{re.escape(section_name)}", f"^{re.escape(section_name)}$"]):
            
            # Try to intelligently split the paper
            paragraphs = complete_paper.split('\n\n')
            current_section = "Complete Paper"
            section_content[current_section] = complete_paper
            
            # Try to extract an abstract if it exists
            for i, para in enumerate(paragraphs[:5]):  # Check first 5 paragraphs
                if len(para) > 100 and ('abstract' in para.lower()[:50] or i == 1):
                    section_content["Abstract"] = para.strip()
                    break
            
            return section_content
    
    # Split the paper by actual headers found
    sections = re.split(r'^#+\s*(.+?)$', complete_paper, flags=re.MULTILINE)
    
    # Process the split sections
    current_section = None
    current_content = []
    
    for i, part in enumerate(sections):
        part = part.strip()
        if not part:
            continue
            
        # Check if this part is a header
        is_header = False
        normalized_part = part.lower().strip('*').strip('#').strip()
        
        for expected_name in all_expected:
            if normalized_part == expected_name.lower() or expected_name.lower() in normalized_part:
                # This is a section header
                if current_section and current_content:
                    section_content[current_section] = '\n\n'.join(current_content).strip()
                
                current_section = expected_name
                current_content = []
                is_header = True
                break
        
        if not is_header and current_section:
            # This is content for the current section
            if part:  # Only add non-empty content
                current_content.append(part)
        elif not is_header and not current_section:
            # Content before any headers - might be abstract or introduction
            if 'abstract' in part.lower()[:100]:
                section_content["Abstract"] = part
            elif len(part) > 200:  # Substantial content
                section_content["Introduction"] = part
    
    # Add the last section if any
    if current_section and current_content:
        section_content[current_section] = '\n\n'.join(current_content).strip()
    
    # If we still don't have good sections, create them from the expected structure
    if len(section_content) < 2:
        # Split the paper into roughly equal parts based on expected sections
        paragraphs = [p.strip() for p in complete_paper.split('\n\n') if p.strip()]
        
        if paragraphs and expected_names:
            section_size = max(1, len(paragraphs) // len(expected_names))
            
            for i, section_name in enumerate(expected_names):
                start_idx = i * section_size
                end_idx = (i + 1) * section_size if i < len(expected_names) - 1 else len(paragraphs)
                
                section_text = '\n\n'.join(paragraphs[start_idx:end_idx])
                if section_text:
                    section_content[section_name] = section_text
    
    # Ensure we have at least some content
    if not section_content:
        section_content["Complete Paper"] = complete_paper
    
    return section_content

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

async def _critique_paper_node(state: PaperWritingState) -> PaperWritingState:
    """Node for critiquing the complete paper generated via single LLM call."""

    _write_stream("🔍 Critiquing paper quality and coherence")

    try:
        section_content = state.get("section_content", {})
        formatted_paper = state.get("formatted_paper", "")  # Get the complete paper from single call
        research_analysis = state.get("research_analysis", {})
        supporting_sources = state.get("supporting_sources", [])
        target_venue = state.get("target_venue", "general")
        original_prompt = state.get("original_prompt", "")

        if not section_content and not formatted_paper:
            return {
                **state,
                "critique_results": {"overall_score": 5.0, "recommendation": "accept", "major_issues": []},
                "current_step": "critique_skipped"
            }

        # Initialize critique tracking
        if "critique_score_history" not in state:
            state["critique_score_history"] = []
        if "refinement_count" not in state:
            state["refinement_count"] = 0
        if "previous_papers" not in state:
            state["previous_papers"] = []

        # Use the complete paper if available, otherwise combine sections
        if formatted_paper:
            full_content = formatted_paper
            generation_method = "single-call comprehensive generation"
        else:
            full_content = ""
            for section_name, content in section_content.items():
                full_content += f"\n## {section_name}\n{content}\n"
            generation_method = "section-by-section generation"

        word_count = len(full_content.split())
        char_count = len(full_content)
        
        critique_prompt = f"""
You are a senior academic reviewer with expertise in evaluating research papers generated through advanced AI methods. You are specifically evaluating a paper created using **{generation_method}**.

**IMPORTANT: This paper was generated using a SINGLE COMPREHENSIVE LLM CALL**, which means:
- The entire paper was created holistically for optimal flow and coherence
- Citations were integrated naturally throughout during generation
- Sections were designed to build logically on each other
- Terminology and narrative should be consistent throughout
- The research context and experimental data were considered comprehensively

**PAPER CONTEXT:**
- **Original Request:** {original_prompt}
- **Target Venue:** {target_venue}
- **Research Domain:** {research_analysis.get('domain', 'Unknown')}
- **Research Type:** {research_analysis.get('research_type', 'Unknown')}
- **Number of Citations:** {len(supporting_sources)}
- **Paper Length:** {word_count:,} words, {char_count:,} characters
- **Refinement Iteration:** {state.get('refinement_count', 0)} (0 = first generation, 1+ = refined version)

**COMPLETE PAPER TO EVALUATE:**
{full_content}

**EVALUATION FOCUS FOR SINGLE-CALL GENERATED PAPERS:**

Your evaluation should focus on the strengths and potential weaknesses specific to comprehensive generation:

**STRENGTHS TO ASSESS:**
- **Narrative Coherence**: How well does the paper flow as a unified work?
- **Integrated Argumentation**: Are arguments built consistently throughout?
- **Citation Integration**: Are sources naturally woven into the discourse?
- **Terminological Consistency**: Is terminology used consistently across sections?
- **Logical Progression**: Does each section build appropriately on previous content?
- **Comprehensive Coverage**: Are all aspects of the research adequately addressed?

**POTENTIAL WEAKNESSES TO CHECK:**
- **Depth vs. Breadth**: Does comprehensive generation sacrifice depth for coverage?
- **Section-Specific Expertise**: Are specialized sections (methodology, results) sufficiently detailed?
- **Citation Distribution**: Are citations appropriately distributed rather than clustered?
- **Technical Rigor**: Does the holistic approach maintain technical accuracy?
- **Redundancy**: Are there unnecessary repetitions across sections?

**CRITICAL EVALUATION CRITERIA (Optimized for Single-Call Generation):**

1. **OVERALL COHERENCE & NARRATIVE FLOW (Weight: 35%)**
   - Unified narrative throughout the paper
   - Smooth transitions between sections
   - Consistent argument development
   - Logical progression from introduction to conclusion
   - **This is the PRIMARY STRENGTH of single-call generation**
   - Score (1-10):

2. **TECHNICAL RIGOR & METHODOLOGY (Weight: 25%)**
   - Technical accuracy and depth
   - Experimental design soundness
   - Implementation details sufficiency
   - Integration with provided research context
   - Score (1-10):

3. **CONTENT QUALITY & COMPLETENESS (Weight: 20%)**
   - Comprehensiveness of coverage
   - Appropriate level of detail in each section
   - Balance between sections
   - Effective use of provided experimental data
   - Score (1-10):

4. **CITATION INTEGRATION & ACADEMIC STANDARDS (Weight: 15%)**
   - Natural integration of [1], [2] citation format
   - Appropriate distribution of citations throughout
   - Source relevance and quality
   - Academic writing conventions
   - Score (1-10):

5. **PUBLICATION READINESS (Weight: 5%)**
   - Clarity and readability
   - Professional academic tone
   - Structural organization
   - Venue appropriateness
   - Score (1-10):

**SPECIAL ASSESSMENT FOR SINGLE-CALL GENERATION:**

Evaluate how well this paper demonstrates the advantages of comprehensive generation:
- **Coherence Advantage**: Does it read better than typical section-by-section papers?
- **Integration Quality**: Are citations and concepts better integrated?
- **Flow Quality**: Are transitions smoother and more natural?
- **Consistency**: Is terminology and style more consistent?

**PROVIDE DETAILED FEEDBACK IN THIS JSON FORMAT:**
{{
    "generation_method_assessment": {{
        "coherence_quality": <1-10>,
        "integration_effectiveness": <1-10>,
        "flow_naturalness": <1-10>,
        "consistency_rating": <1-10>,
        "comprehensive_advantages": ["List specific advantages of single-call generation evident in this paper"],
        "potential_depth_tradeoffs": ["Areas where depth might be sacrificed for breadth"]
    }},
    "section_scores": {{
        "overall_coherence": <score>,
        "technical_rigor": <score>,
        "content_quality": <score>,
        "citation_integration": <score>,
        "publication_readiness": <score>
    }},
    "overall_score": <weighted_average>,
    "recommendation": "<accept|revise>",
    "critical_analysis": {{
        "narrative_flow": {{
            "strengths": ["Specific examples of excellent flow"],
            "weaknesses": ["Areas where flow could be improved"],
            "recommendations": ["Specific improvements for narrative coherence"]
        }},
        "technical_content": {{
            "strengths": ["Strong technical aspects"],
            "weaknesses": ["Technical areas needing improvement"],
            "recommendations": ["Specific technical enhancements needed"]
        }},
        "citation_analysis": {{
            "strengths": ["Effective citation usage"],
            "weaknesses": ["Citation integration issues"],
            "recommendations": ["Citation improvements needed"]
        }},
        "content_coverage": {{
            "strengths": ["Well-covered topics"],
            "gaps": ["Missing or underdeveloped content"],
            "recommendations": ["Content additions or expansions needed"]
        }}
    }},
    "major_issues": [
        "ONLY include CRITICAL problems that make the paper unsuitable for publication",
        "Focus on fundamental flaws, not minor improvements",
        "Consider that this was generated comprehensively, not section-by-section"
    ],
    "specific_improvements": [
        "Concrete, actionable improvements for the entire paper",
        "Focus on enhancements that maintain the comprehensive generation advantages",
        "Prioritize improvements that enhance the paper's unified narrative",
        "Consider how changes affect overall flow and coherence"
    ],
    "single_call_specific_feedback": {{
        "generation_effectiveness": "How well did single-call generation work for this paper?",
        "coherence_assessment": "Quality of overall narrative flow and consistency",
        "integration_success": "Effectiveness of citation and concept integration",
        "recommended_adjustments": ["Adjustments that preserve comprehensive generation benefits"],
        "depth_vs_breadth_balance": "Assessment of technical depth vs comprehensive coverage balance"
    }},
    "technical_feedback": {{
        "methodology_assessment": ["Technical rigor evaluation"],
        "experimental_integration": ["How well experimental data is incorporated"],
        "research_context_usage": ["Effectiveness of research context utilization"]
    }},
    "writing_improvements": {{
        "clarity_enhancements": ["Specific clarity improvements"],
        "flow_optimizations": ["Ways to enhance the already good flow"],
        "consistency_fixes": ["Minor consistency improvements"],
        "academic_tone_adjustments": ["Academic writing refinements"]
    }},
    "venue_specific_feedback": {{
        "appropriateness_score": <1-10>,
        "venue_requirements": ["How well does it meet {target_venue} standards?"],
        "competitiveness": ["How does it compare to typical {target_venue} papers?"],
        "publication_readiness": ["Is it ready for submission to {target_venue}?"]
    }},
    "revision_priority": "<low|medium|high>",
    "estimated_revision_effort": "<minor_edits|moderate_rewrite|major_overhaul>",
    "preserve_generation_advantages": ["Key aspects of the single-call generation to preserve during revision"]
}}

**EVALUATION GUIDELINES:**
- **Be constructive and recognize the generation method's strengths**
- **Focus on improvements that enhance rather than disrupt the unified narrative**
- **Evaluate technical depth appropriately for comprehensive generation**
- **Consider the target venue's expectations for paper quality**
- **Provide specific, actionable feedback that maintains coherence advantages**
- **Reserve "major_issues" for truly critical problems only**
- **Acknowledge when single-call generation has produced superior flow and integration**

**RECOMMENDATION CRITERIA:**
- "accept": Overall score ≥ 6.5 OR excellent coherence with only minor issues
- "revise": Only if significant technical or content issues exist that don't disrupt the narrative flow
"""

        _write_stream("🤖 Performing comprehensive AI critique")

        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: client.chat.completions.create(
                model=model,
                temperature=0.1,  # Low temperature for consistent critique
                messages=[
                    {"role": "system", "content": "You are an expert academic reviewer specializing in evaluating AI-generated research papers, with particular expertise in assessing papers created through comprehensive single-call generation methods."},
                    {"role": "user", "content": critique_prompt}
                ]
            )
        )

        _write_stream("📊 Processing critique results")
        critique_response = response.choices[0].message.content.strip()

        try:
            # Use robust JSON parsing with multiple fallback strategies
            critique_data = _parse_critique_json(critique_response)

            # Store critique in history with iteration info
            iteration_count = state.get("refinement_count", 0)
            historical_entry = {
                "iteration": iteration_count,
                "critique_data": critique_data,
                "timestamp": datetime.now().isoformat(),
                "major_issues": critique_data.get("major_issues", []),
                "overall_score": critique_data.get("overall_score", 0.0),
                "generation_method": generation_method,
                "paper_length": word_count
            }

            # Initialize critique_history if it doesn't exist
            if "critique_history" not in state:
                state["critique_history"] = []

            state["critique_history"].append(historical_entry)

            # Store critique results (current format for compatibility)
            state["critique_results"] = critique_data

            # Add to score history
            state["critique_score_history"].append(critique_data.get("overall_score", 0.0))

            # Show generation method specific feedback
            single_call_feedback = critique_data.get("single_call_specific_feedback", {})
            if single_call_feedback:
                generation_effectiveness = single_call_feedback.get("generation_effectiveness", "")
                coherence_assessment = single_call_feedback.get("coherence_assessment", "")

            # Print key feedback with detailed improvements
            major_issues = critique_data.get("major_issues", [])
            specific_improvements = critique_data.get("specific_improvements", [])

            # Show coherence and flow assessment
            critical_analysis = critique_data.get("critical_analysis", {})
            narrative_flow = critical_analysis.get("narrative_flow", {})
            if narrative_flow:
                strengths = narrative_flow.get("strengths", [])
                weaknesses = narrative_flow.get("weaknesses", [])

            # Show technical assessment
            technical_feedback = critique_data.get("technical_feedback", {})

        except json.JSONDecodeError as e:
            # Fallback critique for parsing failures (should rarely happen now)
            critique_data = {
                "overall_score": 6.5,  # Slightly higher default for single-call papers
                "recommendation": "revise",
                "major_issues": ["JSON parsing recovered with fallbacks - review content"],
                "specific_improvements": [
                    "Manual validation recommended for critique accuracy",
                    "Review comprehensive paper flow and coherence",
                    "Validate citation integration throughout paper",
                    "Check technical depth in methodology and results sections"
                ],
                "section_scores": {
                    "overall_coherence": 6.5,
                    "technical_rigor": 6.0,
                    "content_quality": 6.5,
                    "citation_integration": 6.5,
                    "publication_readiness": 6.0
                },
                "single_call_specific_feedback": {
                    "generation_effectiveness": "Unable to assess due to parsing error",
                    "coherence_assessment": "Manual review needed"
                }
            }
            state["critique_results"] = critique_data

        overall_score = critique_data.get("overall_score", 0.0)
        recommendation = critique_data.get("recommendation", "unknown")
        _write_stream(f"✅ Critique complete - Score: {overall_score:.1f}/10, Recommendation: {recommendation}")

        return {
            **state,
            "critique_results": critique_data,
            "quality_score": critique_data.get("overall_score", 0.0),
            "current_step": "paper_critiqued"
        }

    except Exception as e:
        # Fallback to accept if critique fails
        fallback_critique = {
            "overall_score": 6.0,
            "recommendation": "accept",
            "major_issues": [f"Critique error: {str(e)}"],
            "specific_improvements": ["Manual review recommended"],
            "single_call_specific_feedback": {
                "generation_effectiveness": "Could not assess due to error"
            }
        }
        return {
            **state,
            "critique_results": fallback_critique,
            "errors": state.get("errors", []) + [f"Critique error: {str(e)}"],
            "current_step": "critique_error"
        }
def _determine_paper_refinement_path(state: PaperWritingState) -> str:
    """Determine whether to refine the paper or proceed to finalization based on critique."""
    current_step = state.get("current_step", "")
    critique = state.get("critique_results", {})
    refinement_count = state.get("refinement_count", 0)
    
    # If coming from generate_content for the first time, always critique
    if current_step == "content_generated" and refinement_count == 0:
        return "critique"
    
    # If coming from critique_paper, make refinement decision
    if not critique:
        return "finalize"
    
    overall_score = critique.get("overall_score", 5.0)
    recommendation = critique.get("recommendation", "accept")
    major_issues = critique.get("major_issues", [])
    
    # Maximum 3 refinement iterations
    MAX_REFINEMENTS = 3
    
    # If we've hit the maximum refinements, select the best version
    if refinement_count >= MAX_REFINEMENTS:
        # Get score history to find the best version
        score_history = state.get("critique_score_history", [])
        previous_papers = state.get("previous_papers", [])
        current_content = state.get("section_content", {})
        
        if score_history and len(score_history) > 1:
            # Find the iteration with the highest score
            best_score_idx = score_history.index(max(score_history))
            best_score = score_history[best_score_idx]
            
            # If the best version isn't the current one, restore it
            if best_score_idx < len(previous_papers) and best_score_idx != len(score_history) - 1:
                state["section_content"] = previous_papers[best_score_idx]
                # Update critique results to reflect the best version
                critique_history = state.get("critique_history", [])
                if best_score_idx < len(critique_history):
                    state["critique_results"] = critique_history[best_score_idx]["critique_data"]
        
        return "finalize"
    
    # Decision logic for paper refinement (only revise for major issues)
    if recommendation == "accept" or overall_score >= 6.0 or len(major_issues) == 0:
        return "finalize"
    elif recommendation == "revise" and len(major_issues) > 0 and refinement_count < MAX_REFINEMENTS:
        # Store current paper for comparison
        if "previous_papers" not in state:
            state["previous_papers"] = []
        current_content = state.get("section_content", {})
        state["previous_papers"].append(current_content)
        state["refinement_count"] = refinement_count + 1
        return "refine"
    else:
        return "finalize"

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

async def _finalize_paper_node(state: PaperWritingState) -> PaperWritingState:
    """Node for finalizing the paper and creating output files."""

    _write_stream("🎯 Finalizing paper and creating output")

    try:
        formatted_paper = state.get("formatted_paper", "")
        template_config = state.get("template_config", {})
        critique_results = state.get("critique_results", {})
        refinement_count = state.get("refinement_count", 0)
        score_history = state.get("critique_score_history", [])
        
        # Log final paper metrics
        final_score = critique_results.get("overall_score", 0.0)
        final_recommendation = critique_results.get("recommendation", "unknown")
        
        if score_history:
            best_score = max(score_history)
            score_improvement = score_history[-1] - score_history[0] if len(score_history) > 1 else 0

        # Create multiple format outputs
        final_outputs = {}

        # Markdown version (primary)
        final_outputs["markdown"] = formatted_paper

        # Store paper content in outputs but don't save to file
        final_outputs["paper_content"] = formatted_paper
        final_outputs["display_method"] = "terminal_output"

        _write_stream("✅ Paper writing completed successfully!")

        return {
            **state,
            "final_outputs": final_outputs,
            "current_step": "paper_finalized"
        }

    except Exception as e:
        return {
            **state,
            "errors": state.get("errors", []) + [f"Finalization error: {str(e)}"],
            "current_step": "finalization_error"
        }

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

    # Initialize clients
    _initialize_clients()

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
        
        final_outputs={}
    )

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
