#!/usr/bin/env python3
import os
import sys
import json
import re
import math
import asyncio
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, TypedDict, Annotated
import urllib.request as libreq
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import warnings
import logging

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.config import get_stream_writer
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

# LLM and related imports
import openai

# Web search imports
from tavily import TavilyClient

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
logging.getLogger("openai._base_client").setLevel(logging.WARNING)

# ==================================================================================
# STREAMWRITER HELPER FUNCTION
# ==================================================================================

def _write_stream(message: str, key: str = "status", progress: int = None):
    """Helper function to write to StreamWriter if available."""
    try:
        writer = get_stream_writer()
        writer({key: message})
    except Exception:
        try:
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

# ==================================================================================
# STATE DEFINITIONS
# ==================================================================================

class BaseState(TypedDict):
     # Clients (stored in state)
    client: Optional[Any]                     # OpenAI client instance
    tavily_client: Optional[Any]              # Tavily client instance
    model: str    
    """Base state for all workflows."""
    messages: Annotated[List[BaseMessage], add_messages]
    original_prompt: str  # Pure user query without uploaded data
    uploaded_data: List[str]  # Uploaded file contents as separate field
    current_step: str
    errors: List[str]
class ResearchPlanningState(BaseState):
    """State object for the research planning workflow."""
    generated_problems: List[Dict[str, Any]]     # All generated problem statements
    validated_problems: List[Dict[str, Any]]     # Problems verified as unsolved
    current_problem: Dict[str, Any]              # Currently being validated
    validation_results: Dict[str, Any]           # Web search and validation results
    selected_problem: Dict[str, Any]             # User-selected problem for detailed planning
    research_plan: Dict[str, Any]                # Final research plan
    iteration_count: int                         # Track number of iterations
    critique_results: Dict[str, Any]             # Critique agent feedback
    critique_score_history: List[float]          # Track score improvements
    refinement_count: int                        # Number of refinements attempted
    previous_plans: List[Dict[str, Any]]         # Store previous plan versions
    
    # 🔧 STREAMLINED WORKFLOW: Enhanced state tracking
    generation_attempts: int                     # Track total problem generation attempts
    rejection_feedback: List[str]                # Track why problems were rejected
    auto_validation_enabled: bool                # Enable automatic validation flow
    web_sources: List[Dict[str, Any]]           # Web search sources for validation
    current_web_search_query: str               # Current search query being used

async def _process_rejection_feedback_node(state: ResearchPlanningState, *, config: Optional[Dict[str, Any]] = None) -> ResearchPlanningState:
        """🆕 SMART FEEDBACK: Process rejection feedback and prepare for next generation."""

        _write_stream("Processing feedback for improved problem generation", progress=25)
        state["current_step"] = "process_feedback"
        
        try:
            validation_results = state.get("validation_results", {})
            current_problem = state.get("current_problem", {})
            
            # Get the latest rejection feedback
            rejection_feedback_list = state.get("rejection_feedback", [])
            if not rejection_feedback_list:
                _write_stream("No previous feedback available for analysis")
                return state
            
            latest_feedback = rejection_feedback_list[-1]
            primary_reason = latest_feedback.get("primary_reason", "unknown")
            
            _write_stream(f"Analyzing feedback pattern: {primary_reason}")
            
            # Analyze rejection patterns for adaptive strategy
            rejection_patterns = {}
            for feedback in rejection_feedback_list:
                reason = feedback.get("primary_reason", "unknown")
                rejection_patterns[reason] = rejection_patterns.get(reason, 0) + 1
            
            # Determine adaptive strategy based on patterns
            total_rejections = len(rejection_feedback_list)
            most_common_reason = max(rejection_patterns.items(), key=lambda x: x[1])[0] if rejection_patterns else "unknown"
            
            _write_stream(f"Pattern analysis complete: {total_rejections} attempts, primary issue: {most_common_reason}")
            
            # Create strategic guidance based on patterns
            strategic_guidance = ""
            if most_common_reason == "too_broad":
                strategic_guidance = "Focus on a very specific technical aspect or implementation detail."
            elif most_common_reason == "already_solved":
                strategic_guidance = "Look for novel applications, improvements, or unexplored edge cases."
            elif most_common_reason == "well_studied":
                strategic_guidance = "Consider interdisciplinary approaches or emerging technology combinations."
            elif most_common_reason == "duplicate":
                strategic_guidance = "Try a completely different angle or application domain."
            elif most_common_reason == "unclear":
                strategic_guidance = "Be extremely specific about the problem definition and scope."
            else:
                strategic_guidance = "Consider a different approach or methodology entirely."
            
            # Update feedback context with strategic insights
            current_context = state.get("feedback_context", "")
            enhanced_context = f"""
                {current_context}

                🎯 STRATEGIC ADAPTATION (after {total_rejections} rejections):
                Most frequent issue: {most_common_reason} (occurred {rejection_patterns.get(most_common_reason, 0)} times)
                Strategy: {strategic_guidance}

                🔄 TACTICAL ADJUSTMENTS:
            """
            
            # Add specific tactical adjustments based on latest feedback
            latest_guidance = latest_feedback.get("specific_guidance", "")
            alternative_angles = latest_feedback.get("alternative_angles", [])
            
            if latest_guidance:
                enhanced_context += f"- {latest_guidance}\n"
            
            if alternative_angles:
                enhanced_context += f"- Try these angles: {', '.join(alternative_angles)}\n"
            
            # Add adaptive constraints based on rejection count
            if total_rejections >= 3:
                enhanced_context += "- CONSTRAINT: Be extremely specific and narrow in scope\n"
            if total_rejections >= 5:
                enhanced_context += "- CONSTRAINT: Focus on technical implementation challenges\n"
            if total_rejections >= 7:
                enhanced_context += "- CONSTRAINT: Consider niche applications or edge cases\n"
            
            state["feedback_context"] = enhanced_context
            
            _write_stream(f"Strategy updated: {strategic_guidance}")
            _write_stream("Enhanced context prepared for next generation attempt")
            
            # Add processing message
            state["messages"].append(
                AIMessage(content=f"Processed rejection feedback: {primary_reason} (pattern analysis complete, adaptive strategy updated)")
            )
            
        except Exception as e:
            error_msg = f"Feedback processing failed: {str(e)}"
            state["errors"].append(error_msg)
            _write_stream(f"Feedback processing error: {error_msg}")
            # Continue without enhanced feedback if processing fails
            
        return state

    # --- PHASE 2: RESEARCH PLAN CREATION & STRUCTURING ---

def _clean_text_for_encoding(text: str) -> str:
        """Clean text to avoid encoding issues."""
        if not text:
            return ""
        
        # Replace problematic characters
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        text = text.replace('–', '-').replace('—', '-')
        text = text.replace('…', '...')
        
        # Ensure UTF-8 compatible
        try:
            text.encode('utf-8')
            return text
        except UnicodeEncodeError:
            # Fallback: remove non-ASCII characters
            return ''.join(char for char in text if ord(char) < 128)