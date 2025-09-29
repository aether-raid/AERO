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

async def _find_supporting_sources_node(state: PaperWritingState) -> PaperWritingState:
    """🔍 Find supporting sources and citations using Tavily web search, enhanced with uploaded file context."""

    _write_stream("🔍 Finding supporting sources and citations")

    try:
        # Check if Tavily client is available
        if state.get("tavily_client") is None:
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
                    lambda q=query: state["tavily_client"].search(q, max_results=8)
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