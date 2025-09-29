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