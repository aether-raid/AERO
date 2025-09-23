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
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

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

# Local imports
from shared_constants import ML_RESEARCH_CATEGORIES, Evidence, PropertyHit

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
        api_key = _load_from_env_file("OPENAI_API_KEY")
        base_url = _load_from_env_file("BASE_URL") or "https://agents.aetherraid.dev"
        model = _load_from_env_file("DEFAULT_MODEL") or "gemini/gemini-2.5-flash"

        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in env.example")

        from openai import OpenAI
        client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        print("✅ OpenAI client initialized successfully")

    if tavily_client is None:
        # Initialize Tavily client
        tavily_api_key = "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"  # Using the key from the example file
        try:
            tavily_client = TavilyClient(api_key=tavily_api_key)
            print("✅ Tavily web search client initialized successfully")
        except Exception as e:
            print(f"⚠️ Tavily client initialization failed: {e}")
            tavily_client = None

def _load_from_env_file(key: str) -> Optional[str]:
    """Load configuration value from env.example file."""
    try:
        with open("env.example", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    if k.strip() == key:
                        return v.strip().strip('"').strip("'")
    except Exception:
        pass
    return None

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
            else:
                print(f"⚠️ File not found: {file_path}")
        except Exception as e:
            print(f"⚠️ Error reading file {file_path}: {e}")
    
    return files_data

# ==================================================================================
# PAPER WRITING WORKFLOW NODES
# ==================================================================================

async def _analyze_results_node(state: PaperWritingState) -> PaperWritingState:
    """Node for analyzing experimental results and research context, including uploaded file data."""
    print("\n📊 Paper Writing: Analyzing experimental results and research context...")

    try:
        # Extract research context from the original prompt and any provided data
        original_prompt = state.get("original_prompt", "")
        experimental_results = state.get("experimental_results", {})
        uploaded_data = state.get("uploaded_data", [])

        # Process uploaded file data
        uploaded_context = ""
        data_analysis = ""

        if uploaded_data:
            print(f"📎 Processing {len(uploaded_data)} uploaded files...")
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

            print(f"📋 Processed file data: {data_analysis}")
        else:
            print("📝 No uploaded files to process")

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
        print(f"📋 Analysis: {analysis_text[:200]}...")

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

        return {
            **state,
            "research_analysis": analysis_json,
            "research_context": analysis_json.get("research_context", original_prompt),  # Ensure research_context is available for Tavily
            "current_step": "results_analyzed"
        }

    except Exception as e:
        print(f"❌ Error in analyze_results_node: {str(e)}")
        return {
            **state,
            "errors": state.get("errors", []) + [f"Analysis error: {str(e)}"],
            "current_step": "analysis_error"
        }

async def _setup_paper_node(state: PaperWritingState) -> PaperWritingState:
    """Node for LLM-driven template selection and paper structuring."""
    print("\n🏗️ Paper Writing: Setting up paper structure and template...")

    try:
        research_analysis = state.get("research_analysis", {})
        target_venue = state.get("target_venue", "general")

        setup_prompt = f"""
        Create an optimal paper structure for the following research:

        Research Analysis: {research_analysis}
        Target Venue: {target_venue}

        Based on this information, generate:
        1. Template Configuration (formatting requirements)
        2. Paper Structure (sections with focus areas and length allocations)
        3. Content Guidelines (what to emphasize in each section)

        Consider common academic paper structures and venue-specific requirements.

        Respond with a JSON object containing:
        {{
            "template_config": {{
                "venue": "conference_name",
                "page_limit": 8,
                "format": "two_column",
                "citation_style": "ACM"
            }},
            "paper_structure": {{
                "sections": [
                    {{"name": "Abstract", "length": "200-250 words", "focus": "problem_solution_results"}},
                    {{"name": "Introduction", "length": "1-1.5 pages", "focus": "motivation_contributions"}},
                    // ... more sections
                ]
            }},
            "content_guidelines": {{
                "emphasis": ["methodology", "experimental_validation"],
                "tone": "formal_academic",
                "target_audience": "researchers_practitioners"
            }}
        }}
        """

        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": setup_prompt}],
                temperature=0.3
            )
        )

        setup_text = response.choices[0].message.content.strip()
        print(f"📋 Paper setup: {setup_text[:200]}...")

        # Try to extract JSON from response
        try:
            import json
            start = setup_text.find('{')
            end = setup_text.rfind('}') + 1
            if start != -1 and end != -1:
                setup_json = json.loads(setup_text[start:end])
            else:
                # Fallback structure
                setup_json = {
                    "template_config": {
                        "venue": target_venue,
                        "page_limit": 8,
                        "format": "academic",
                        "citation_style": "ACM"
                    },
                    "paper_structure": {
                        "sections": [
                            {"name": "Abstract", "length": "200-250 words", "focus": "summary"},
                            {"name": "Introduction", "length": "1-1.5 pages", "focus": "motivation"},
                            {"name": "Methods", "length": "2-3 pages", "focus": "methodology"},
                            {"name": "Results", "length": "2-3 pages", "focus": "findings"},
                            {"name": "Conclusion", "length": "0.5 pages", "focus": "summary"}
                        ]
                    },
                    "content_guidelines": {
                        "emphasis": ["methodology", "results"],
                        "tone": "formal_academic",
                        "target_audience": "researchers"
                    }
                }
        except:
            # Fallback structure
            setup_json = {
                "template_config": {
                    "venue": target_venue,
                    "page_limit": 8,
                    "format": "academic",
                    "citation_style": "ACM"
                },
                "paper_structure": {
                    "sections": [
                        {"name": "Abstract", "length": "200-250 words", "focus": "summary"},
                        {"name": "Introduction", "length": "1-1.5 pages", "focus": "motivation"},
                        {"name": "Methods", "length": "2-3 pages", "focus": "methodology"},
                        {"name": "Results", "length": "2-3 pages", "focus": "findings"},
                        {"name": "Conclusion", "length": "0.5 pages", "focus": "summary"}
                    ]
                },
                "content_guidelines": {
                    "emphasis": ["methodology", "results"],
                    "tone": "formal_academic",
                    "target_audience": "researchers"
                }
            }

        return {
            **state,
            "paper_structure": setup_json.get("paper_structure", {}),
            "template_config": setup_json.get("template_config", {}),
            "current_step": "paper_setup_complete"
        }

    except Exception as e:
        print(f"❌ Error in setup_paper_node: {str(e)}")
        return {
            **state,
            "errors": state.get("errors", []) + [f"Setup error: {str(e)}"],
            "current_step": "setup_error"
        }

async def _find_supporting_sources_node(state: PaperWritingState) -> PaperWritingState:
    """🔍 Find supporting sources and citations using Tavily web search, enhanced with uploaded file context."""
    print("\n🔍 Step: Finding supporting sources and citations...")

    try:
        # Check if Tavily client is available
        if tavily_client is None:
            print("⚠️ Tavily client not initialized. Skipping source finding.")
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
            print(f"📎 Incorporating {len(uploaded_data)} uploaded files into search strategy...")

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

        print(f"🔍 Performing {len(search_queries)} targeted searches for citations...")

        for i, search_item in enumerate(search_queries[:4]):  # Limit to 4 searches
            query = search_item["query"]
            purpose = search_item["purpose"]
            section = search_item["section"]

            try:
                print(f"   📚 Search {i+1}: {purpose} - {query[:50]}...")

                # Execute Tavily search
                search_response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda q=query: tavily_client.search(q, max_results=8)
                )

                if search_response and "results" in search_response:
                    sources_found = 0
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

                    print(f"   ✅ Found {sources_found} relevant sources for {purpose}")

            except Exception as e:
                print(f"   ❌ Search failed for {purpose}: {str(e)}")
                continue

        # Summary
        total_sources = len(all_sources)
        sections_with_sources = len(citation_database)

        print(f"🎯 Source finding complete:")
        print(f"   📚 Total sources found: {total_sources}")
        print(f"   📋 Sections with sources: {sections_with_sources}")
        print(f"   🔍 Search queries used: {len(search_queries)}")

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
        print(f"❌ Error in find_supporting_sources_node: {str(e)}")
        return {
            **state,
            "errors": state.get("errors", []) + [f"Source finding error: {str(e)}"],
            "supporting_sources": [],
            "citation_database": {},
            "current_step": "source_finding_error"
        }

async def _generate_content_node(state: PaperWritingState) -> PaperWritingState:
    """Node for generating content for each paper section with Tavily-sourced citations."""
    
    # Check if this is a refinement iteration
    refinement_count = state.get("refinement_count", 0)
    critique_results = state.get("critique_results", {})
    
    if refinement_count > 0:
        print(f"\n🔄 Paper Writing: Refining content based on critique (Iteration {refinement_count})...")
        major_issues = critique_results.get("major_issues", [])
        suggestions = critique_results.get("suggestions", [])
        if major_issues:
            print(f"🎯 Addressing {len(major_issues)} major issues from critique")
    else:
        print("\n✍️ Paper Writing: Generating content with citations for each section...")

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

        print(f"📚 Using {len(supporting_sources)} sources for citations across {len(citation_database)} sections")

        for section in sections:
            section_name = section.get("name", "Unknown")
            section_focus = section.get("focus", "general")
            section_length = section.get("length", "1 page")

            print(f"📝 Generating {section_name} section with citations...")

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
{f"- PRIORITY: Address critique feedback and improve quality from previous iteration" if refinement_count > 0 else ""}

Write a complete {section_name} section with integrated citations:
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

            print(f"   ✅ Generated {section_name} ({len(section_text)} chars, {len(used_citations)} citations)")

        # Generate reference list from all used sources
        reference_list = _generate_reference_list(supporting_sources)
        if reference_list:
            section_content["References"] = reference_list
            print(f"📚 Generated References section with {len(supporting_sources)} sources")

        total_sources_used = len([s for s in supporting_sources if any(
            f"[{i}]" in content for i, content in enumerate(section_content.values(), 1)
        )])

        print(f"🎯 Content generation complete:")
        print(f"   📝 Sections written: {len(section_content)}")
        print(f"   📚 Sources integrated: {total_sources_used}/{len(supporting_sources)}")
        print(f"   🔗 Citation coverage: {source_validation.get('search_success_rate', 0):.1%}")

        return {
            **state,
            "section_content": section_content,
            "current_step": "content_generated"
        }

    except Exception as e:
        print(f"❌ Error in generate_content_node: {str(e)}")
        return {
            **state,
            "errors": state.get("errors", []) + [f"Content generation error: {str(e)}"],
            "current_step": "content_generation_error"
        }

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

async def _critique_paper_node(state: PaperWritingState) -> PaperWritingState:
    """Node for critiquing the generated paper content."""
    print("\n🔍 Paper Writing: Critiquing generated paper content...")

    try:
        section_content = state.get("section_content", {})
        research_analysis = state.get("research_analysis", {})
        supporting_sources = state.get("supporting_sources", [])
        target_venue = state.get("target_venue", "general")
        original_prompt = state.get("original_prompt", "")

        if not section_content:
            print("⚠️ No content to critique - skipping critique")
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

        # Combine all sections for critique
        full_content = ""
        for section_name, content in section_content.items():
            full_content += f"\n## {section_name}\n{content}\n"

        critique_prompt = f"""
You are a senior academic reviewer and expert in scientific writing, with extensive experience in peer review for top-tier conferences and journals. Your primary role is to provide CONSTRUCTIVE CRITICISM and SPECIFIC RECOMMENDATIONS to improve the paper quality.

**PAPER CONTEXT:**
- **Original Request:** {original_prompt}
- **Target Venue:** {target_venue}
- **Research Domain:** {research_analysis.get('domain', 'Unknown')}
- **Research Type:** {research_analysis.get('research_type', 'Unknown')}
- **Number of Citations:** {len(supporting_sources)}

**PAPER CONTENT TO EVALUATE:**
{full_content}

**CRITICAL EVALUATION INSTRUCTIONS:**

Your job is to be a constructive but thorough critic. For each section, identify specific problems and provide actionable solutions. Be detailed and specific - avoid generic feedback.

**IMPORTANT: Reserve "major_issues" ONLY for critical problems that would make the paper unsuitable for publication, such as:**
- Fundamental methodological flaws that invalidate results
- Complete absence of essential sections (methodology, results)
- Plagiarism or ethical violations
- Claims completely unsupported by evidence
- Experimental design that cannot support conclusions

**Minor issues like writing style, formatting, or incremental improvements should NOT be classified as major issues.**

**EVALUATION CRITERIA & SCORING:**

1. **ABSTRACT & INTRODUCTION QUALITY (Weight: 20%)**
   - Problem clarity and motivation
   - Contribution statements and significance
   - Research gap identification
   - Score (1-10):

2. **METHODOLOGY & TECHNICAL CONTENT (Weight: 30%)**
   - Technical rigor and soundness
   - Experimental design quality
   - Implementation details
   - Score (1-10):

3. **RESULTS & ANALYSIS (Weight: 25%)**
   - Data presentation and visualization
   - Statistical analysis quality
   - Results interpretation depth
   - Score (1-10):

4. **WRITING QUALITY & STRUCTURE (Weight: 15%)**
   - Logical flow and organization
   - Clarity and readability
   - Grammar and style
   - Score (1-10):

5. **CITATIONS & RELATED WORK (Weight: 10%)**
   - Citation relevance and completeness
   - Related work positioning
   - Reference quality
   - Score (1-10):

**PROVIDE DETAILED FEEDBACK IN THIS JSON FORMAT:**
{{
    "section_scores": {{
        "abstract_intro": <score>,
        "methodology": <score>,
        "results_analysis": <score>,
        "writing_quality": <score>,
        "citations": <score>
    }},
    "overall_score": <weighted_average>,
    "recommendation": "<accept|revise>",
    "critical_analysis": {{
        "abstract": {{
            "problems": ["Specific issue 1", "Specific issue 2"],
            "recommendations": ["Fix X by doing Y", "Improve Z by adding W"]
        }},
        "introduction": {{
            "problems": ["Specific issue 1", "Specific issue 2"],
            "recommendations": ["Fix X by doing Y", "Improve Z by adding W"]
        }},
        "methodology": {{
            "problems": ["Specific issue 1", "Specific issue 2"],
            "recommendations": ["Fix X by doing Y", "Improve Z by adding W"]
        }},
        "results": {{
            "problems": ["Specific issue 1", "Specific issue 2"],
            "recommendations": ["Fix X by doing Y", "Improve Z by adding W"]
        }},
        "discussion": {{
            "problems": ["Specific issue 1", "Specific issue 2"],
            "recommendations": ["Fix X by doing Y", "Improve Z by adding W"]
        }},
        "conclusion": {{
            "problems": ["Specific issue 1", "Specific issue 2"],
            "recommendations": ["Fix X by doing Y", "Improve Z by adding W"]
        }}
    }},
    "major_issues": [
        "ONLY include CRITICAL problems that make the paper unsuitable for publication",
        "Issues that fundamentally undermine the research validity",
        "Problems that would lead to immediate rejection by reviewers"
    ],
    "specific_improvements": [
        "Add specific experimental validation for claim X",
        "Rewrite methodology section to include missing details Y and Z",
        "Restructure results section to better highlight key findings",
        "Strengthen related work by adding citations to recent papers A, B, C",
        "Improve abstract by clearly stating the main contribution"
    ],
    "content_suggestions": {{
        "missing_elements": ["What essential content is missing?"],
        "content_to_expand": ["What sections need more detail?"],
        "content_to_reduce": ["What sections are too verbose?"],
        "structural_changes": ["What organizational changes are needed?"]
    }},
    "technical_feedback": {{
        "methodology_gaps": ["Missing technical details", "Unclear procedures"],
        "experimental_weaknesses": ["Insufficient validation", "Missing baselines"],
        "analysis_improvements": ["Better statistical tests", "More thorough interpretation"]
    }},
    "writing_improvements": {{
        "clarity_issues": ["Confusing sentences to rewrite"],
        "flow_problems": ["Poor transitions between sections"],
        "style_fixes": ["Academic tone improvements needed"]
    }},
    "venue_specific_feedback": {{
        "appropriateness_score": <1-10>,
        "venue_requirements": ["How well does it meet {target_venue} standards?"],
        "competitiveness": ["How does it compare to typical {target_venue} papers?"]
    }},
    "revision_priority": "<low|medium|high>",
    "estimated_revision_effort": "<minor_edits|moderate_rewrite|major_overhaul>"
}}

**RECOMMENDATION CRITERIA:**
- "accept": Overall score ≥ 6.0 OR no major issues identified
- "revise": ONLY if major issues exist that significantly impact paper quality

**FOCUS ON BEING CONSTRUCTIVE:**
- Identify specific problems with concrete examples from the text
- Provide actionable recommendations that can be immediately implemented
- Be thorough but constructive - help the authors improve
- Consider the target venue's standards and expectations
- Prioritize the most impactful improvements
"""

        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: client.chat.completions.create(
                model=model,
                temperature=0.1,  # Low temperature for consistent critique
                messages=[{"content": critique_prompt, "role": "user"}]
            )
        )

        critique_response = response.choices[0].message.content.strip()

        try:
            # Parse critique response
            if critique_response.startswith("```json"):
                critique_response = critique_response[7:]
            if critique_response.endswith("```"):
                critique_response = critique_response[:-3]
            critique_response = critique_response.strip()

            critique_data = json.loads(critique_response)

            # Store critique in history with iteration info
            iteration_count = state.get("refinement_count", 0)
            historical_entry = {
                "iteration": iteration_count,
                "critique_data": critique_data,
                "timestamp": datetime.now().isoformat(),
                "major_issues": critique_data.get("major_issues", []),
                "suggestions": critique_data.get("suggestions", []),
                "strengths": critique_data.get("strengths", []),
                "overall_score": critique_data.get("overall_score", 0.0)
            }

            # Initialize critique_history if it doesn't exist
            if "critique_history" not in state:
                state["critique_history"] = []

            state["critique_history"].append(historical_entry)

            # Store critique results (current format for compatibility)
            state["critique_results"] = critique_data

            # Add to score history
            state["critique_score_history"].append(critique_data.get("overall_score", 0.0))

            print(f"\n🔍 PAPER CRITIQUE COMPLETED:")
            print(f"   📊 Overall Score: {critique_data.get('overall_score', 0):.1f}/10")
            print(f"   📋 Recommendation: {critique_data.get('recommendation', 'unknown')}")
            print(f"   ⚠️ Major Issues: {len(critique_data.get('major_issues', []))}")
            print(f"   💡 Specific Improvements: {len(critique_data.get('specific_improvements', []))}")
            print(f"   🔄 Iteration: {iteration_count}")

            # Print key feedback with detailed improvements
            major_issues = critique_data.get("major_issues", [])
            if major_issues:
                print(f"\n🚨 CRITICAL ISSUES TO ADDRESS:")
                for i, issue in enumerate(major_issues[:3], 1):
                    print(f"   {i}. {issue}")

            specific_improvements = critique_data.get("specific_improvements", [])
            if specific_improvements:
                print(f"\n🎯 SPECIFIC IMPROVEMENTS NEEDED:")
                for i, improvement in enumerate(specific_improvements[:4], 1):
                    print(f"   {i}. {improvement}")

            # Show section-specific problems
            critical_analysis = critique_data.get("critical_analysis", {})
            if critical_analysis:
                print(f"\n📋 SECTION-SPECIFIC FEEDBACK:")
                for section, feedback in list(critical_analysis.items())[:3]:
                    problems = feedback.get("problems", [])
                    if problems:
                        print(f"   📝 {section.title()}: {problems[0] if problems else 'No major issues'}")

            # Show technical feedback
            technical_feedback = critique_data.get("technical_feedback", {})
            if any(technical_feedback.values()):
                print(f"\n🔬 TECHNICAL IMPROVEMENTS:")
                methodology_gaps = technical_feedback.get("methodology_gaps", [])
                if methodology_gaps:
                    print(f"   • Methodology: {methodology_gaps[0]}")
                experimental_weaknesses = technical_feedback.get("experimental_weaknesses", [])
                if experimental_weaknesses:
                    print(f"   • Experiments: {experimental_weaknesses[0]}")

        except json.JSONDecodeError:
            print("⚠️ Could not parse critique JSON, using fallback evaluation")
            # Fallback critique for parsing failures
            critique_data = {
                "overall_score": 6.0,
                "recommendation": "revise",
                "major_issues": ["JSON parsing failed - manual review needed"],
                "suggestions": ["Review and validate paper content manually"],
                "strengths": ["Content generated successfully"],
                "section_scores": {
                    "abstract_intro": 6.0,
                    "methodology": 6.0,
                    "results_analysis": 6.0,
                    "writing_quality": 6.0,
                    "citations": 6.0
                }
            }
            state["critique_results"] = critique_data

        return {
            **state,
            "critique_results": critique_data,
            "quality_score": critique_data.get("overall_score", 0.0),
            "current_step": "paper_critiqued"
        }

    except Exception as e:
        print(f"❌ Error in critique_paper_node: {str(e)}")
        # Fallback to accept if critique fails
        fallback_critique = {
            "overall_score": 5.0,
            "recommendation": "accept",
            "major_issues": [f"Critique error: {str(e)}"],
            "suggestions": ["Manual review recommended"],
            "strengths": ["Paper generated successfully"]
        }
        return {
            **state,
            "critique_results": fallback_critique,
            "errors": state.get("errors", []) + [f"Critique error: {str(e)}"],
            "current_step": "critique_error"
        }

def _determine_paper_refinement_path(state: PaperWritingState) -> str:
    """Determine whether to refine the paper or proceed to formatting based on critique."""
    current_step = state.get("current_step", "")
    critique = state.get("critique_results", {})
    refinement_count = state.get("refinement_count", 0)
    
    # If coming from generate_content for the first time, always critique
    if current_step == "content_generated" and refinement_count == 0:
        print("🔍 First content generation complete - proceeding to critique")
        return "critique"
    
    # If coming from critique_paper, make refinement decision
    if not critique:
        print("🔄 No critique available - proceeding to formatting")
        return "format"
    
    overall_score = critique.get("overall_score", 5.0)
    recommendation = critique.get("recommendation", "accept")
    major_issues = critique.get("major_issues", [])
    
    print(f"🚦 Paper Refinement Decision Point:")
    print(f"   📊 Score: {overall_score:.1f}/10")
    print(f"   📋 Recommendation: {recommendation}")
    print(f"   ⚠️ Major Issues: {len(major_issues)}")
    print(f"   🔄 Refinement Count: {refinement_count}")
    
    # Maximum 3 refinement iterations
    MAX_REFINEMENTS = 3
    
    # If we've hit the maximum refinements, select the best version
    if refinement_count >= MAX_REFINEMENTS:
        print(f"🏁 Maximum refinements ({MAX_REFINEMENTS}) reached - selecting best version")
        
        # Get score history to find the best version
        score_history = state.get("critique_score_history", [])
        previous_papers = state.get("previous_papers", [])
        current_content = state.get("section_content", {})
        
        if score_history and len(score_history) > 1:
            # Find the iteration with the highest score
            best_score_idx = score_history.index(max(score_history))
            best_score = score_history[best_score_idx]
            
            print(f"📊 Score History: {[f'{s:.1f}' for s in score_history]}")
            print(f"🏆 Best Score: {best_score:.1f} at iteration {best_score_idx}")
            
            # If the best version isn't the current one, restore it
            if best_score_idx < len(previous_papers) and best_score_idx != len(score_history) - 1:
                print(f"🔄 Restoring best version from iteration {best_score_idx}")
                state["section_content"] = previous_papers[best_score_idx]
                # Update critique results to reflect the best version
                critique_history = state.get("critique_history", [])
                if best_score_idx < len(critique_history):
                    state["critique_results"] = critique_history[best_score_idx]["critique_data"]
            else:
                print("✅ Current version is the best - keeping it")
        
        return "format"
    
    # Decision logic for paper refinement (only revise for major issues)
    if recommendation == "accept" or overall_score >= 6.0 or len(major_issues) == 0:
        print("✅ Paper accepted - no major issues found or score adequate")
        return "format"
    elif recommendation == "revise" and len(major_issues) > 0 and refinement_count < MAX_REFINEMENTS:
        print(f"🔄 Major issues detected - revision needed (iteration {refinement_count + 1}/{MAX_REFINEMENTS})")
        print(f"   🚨 {len(major_issues)} major issues to address")
        # Store current paper for comparison
        if "previous_papers" not in state:
            state["previous_papers"] = []
        current_content = state.get("section_content", {})
        state["previous_papers"].append(current_content)
        state["refinement_count"] = refinement_count + 1
        return "refine"
    else:
        print("✅ Proceeding to formatting")
        return "format"

async def _format_paper_node(state: PaperWritingState) -> PaperWritingState:
    """Node for formatting the paper according to template requirements with enhanced citation support."""
    print("\n📄 Paper Writing: Formatting paper with citations...")

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

        # Add each section in proper order
        section_order = ["Abstract", "Introduction", "Related Work", "Methods", "Results", "Discussion", "Conclusion", "References"]

        for section_name in section_order:
            if section_name in section_content:
                paper_parts.append(f"## {section_name}\n\n")
                paper_parts.append(section_content[section_name])
                paper_parts.append("\n\n")

        # Add any remaining sections not in the standard order
        for section_name, content in section_content.items():
            if section_name not in section_order:
                paper_parts.append(f"## {section_name}\n\n")
                paper_parts.append(content)
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

        print(f"✅ Paper formatted:")
        print(f"   📄 Total length: {len(formatted_paper)} characters")
        print(f"   📚 Citations integrated: {total_citations}")
        print(f"   📋 Sections included: {len(section_content)}")

        return {
            **state,
            "formatted_paper": formatted_paper,
            "current_step": "paper_formatted"
        }

    except Exception as e:
        print(f"❌ Error in format_paper_node: {str(e)}")
        return {
            **state,
            "errors": state.get("errors", []) + [f"Formatting error: {str(e)}"],
            "current_step": "formatting_error"
        }

async def _finalize_paper_node(state: PaperWritingState) -> PaperWritingState:
    """Node for finalizing the paper and creating output files."""
    print("\n🎯 Paper Writing: Finalizing paper...")

    try:
        formatted_paper = state.get("formatted_paper", "")
        template_config = state.get("template_config", {})
        critique_results = state.get("critique_results", {})
        refinement_count = state.get("refinement_count", 0)
        score_history = state.get("critique_score_history", [])
        
        # Log final paper metrics
        final_score = critique_results.get("overall_score", 0.0)
        final_recommendation = critique_results.get("recommendation", "unknown")
        
        print(f"📊 FINAL PAPER METRICS:")
        print(f"   🏆 Final Score: {final_score:.1f}/10")
        print(f"   📋 Final Recommendation: {final_recommendation}")
        print(f"   🔄 Total Iterations: {refinement_count}")
        
        if score_history:
            best_score = max(score_history)
            score_improvement = score_history[-1] - score_history[0] if len(score_history) > 1 else 0
            print(f"   📈 Score History: {[f'{s:.1f}' for s in score_history]}")
            print(f"   🎯 Best Score Achieved: {best_score:.1f}")
            print(f"   📊 Score Improvement: {score_improvement:+.1f}")

        # Create multiple format outputs
        final_outputs = {}

        # Markdown version (primary)
        final_outputs["markdown"] = formatted_paper

        # Save to file
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"generated_paper_{timestamp}.md"

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(formatted_paper)

        final_outputs["file_path"] = filename

        print(f"✅ Paper saved to: {filename}")
        print(f"📊 Paper statistics:")
        print(f"   - Total length: {len(formatted_paper)} characters")
        print(f"   - Estimated pages: {len(formatted_paper) // 3000:.1f}")
        print(f"   - Sections: {len(state.get('section_content', {}))}")

        return {
            **state,
            "final_outputs": final_outputs,
            "current_step": "paper_finalized"
        }

    except Exception as e:
        print(f"❌ Error in finalize_paper_node: {str(e)}")
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

    # Add nodes for paper writing pipeline with critique
    workflow.add_node("analyze_results", _analyze_results_node)
    workflow.add_node("setup_paper", _setup_paper_node)
    workflow.add_node("find_sources", _find_supporting_sources_node)
    workflow.add_node("generate_content", _generate_content_node)
    workflow.add_node("critique_paper", _critique_paper_node)  # New critique node
    workflow.add_node("format_paper", _format_paper_node)
    workflow.add_node("finalize_paper", _finalize_paper_node)

    # Define the enhanced flow with critique and refinement
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
            "format": "format_paper"       # Skip critique if already refined
        }
    )
    
    workflow.add_conditional_edges(
        "critique_paper", 
        _determine_paper_refinement_path,
        {
            "refine": "generate_content",  # Need to regenerate content
            "format": "format_paper"       # Quality sufficient, proceed to format
        }
    )
    
    workflow.add_edge("format_paper", "finalize_paper")
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
    target_venue: str = "general"
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

    Returns:
        Dict[str, Any]: Complete workflow results including the generated paper

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
    print("📝 Paper Writing Workflow - Standalone Runner")
    print("=" * 60)
    print(f"🎯 Writing paper for: {user_query[:80]}{'...' if len(user_query) > 80 else ''}")
    print("-" * 60)

    # Initialize clients
    _initialize_clients()

    # Process files if provided
    processed_uploaded_data = uploaded_data or []
    
    # Extract content from file paths
    if file_paths:
        print(f"📁 Processing {len(file_paths)} file paths...")
        extracted_from_paths = extract_files_from_paths(file_paths)
        processed_uploaded_data.extend(extracted_from_paths)
        print(f"✅ Extracted content from {len(extracted_from_paths)} files")
    
    # Extract content from file bytes data
    if files_data:
        print(f"📄 Processing {len(files_data)} file data objects...")
        extracted_from_bytes = extract_files_from_bytes(files_data)
        processed_uploaded_data.extend(extracted_from_bytes)
        print(f"✅ Extracted content from {len(extracted_from_bytes)} files")
    
    if processed_uploaded_data:
        print(f"📊 Total processed files: {len(processed_uploaded_data)}")
        for i, data in enumerate(processed_uploaded_data[:3]):  # Show first 3 files
            preview = data.split('\n')[0]  # First line (header)
            print(f"   {i+1}. {preview}")
        if len(processed_uploaded_data) > 3:
            print(f"   ... and {len(processed_uploaded_data) - 3} more files")
        print("-" * 60)

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

    # Execute the workflow
    print("\n🚀 Starting Paper Writing Workflow...")
    try:
        result = await graph.ainvoke(initial_state)

        # Check for successful completion
        if result.get("current_step") == "paper_finalized":
            print("✅ Paper writing completed successfully!")
            print(f"📄 Generated paper saved to: {result.get('final_outputs', {}).get('file_path', 'Unknown')}")

            # Return the complete results
            return dict(result)
        else:
            print(f"❌ Paper writing failed at step: {result.get('current_step')}")
            print(f"Errors: {result.get('errors', [])}")
            return dict(result)

    except Exception as e:
        print(f"❌ Critical error in paper writing workflow: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "current_step": "workflow_error"
        }

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
        print(f"📁 Processing files: {', '.join(file_paths)}")
    
    # Run the paper writing workflow with file processing
    asyncio.run(write_paper(research_topic, file_paths=file_paths))
