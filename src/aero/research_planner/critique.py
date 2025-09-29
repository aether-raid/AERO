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

async def _critique_plan_node(state: ResearchPlanningState, *, config: Optional[Dict[str, Any]] = None) -> ResearchPlanningState:
        """Node for critiquing the generated research plan."""
        
        # Validate clients are available in state
        client = state.get("client")
        if not client:
            state["errors"] = state.get("errors", []) + ["OpenAI client not found in state. Please initialize clients."]
            return state
            
        model = state.get("model", "gpt-4o-mini")

        _write_stream("Evaluating research plan quality", progress=75)
        state["current_step"] = "critique_plan"
        
        try:
            research_plan = state.get("research_plan", {}).get("research_plan", "")
            selected_problem = state.get("selected_problem", {})
            
            if not research_plan:
                raise ValueError("No research plan to critique")
            
            # Initialize critique tracking
            if "critique_score_history" not in state:
                state["critique_score_history"] = []
            if "refinement_count" not in state:
                state["refinement_count"] = 0
            if "previous_plans" not in state:
                state["previous_plans"] = []
            
            
            critique_content = f"""
You are a constructive senior research advisor and peer reviewer with deep expertise in machine learning and academic research. Your primary goal is to provide specific, actionable feedback to help improve a research plan. You are not just scoring it; you are guiding its refinement.

**RESEARCH CONTEXT:**
- **Research Problem Statement:** {selected_problem.get('statement', 'N/A')}

**RESEARCH PLAN TO EVALUATE:**
{research_plan}

**EVALUATION INSTRUCTIONS:**
Evaluate the plan based on its required structure. For each section, assess the corresponding criteria and provide a score (1-10). The final score will be a weighted average.

---
**EVALUATION CRITERIA (by section):**

1.  **WEB-INFORMED PROBLEM ANALYSIS & LITERATURE INTEGRATION (Weight: 20%)**
    -   Does the analysis clearly leverage the web search findings and provided URLs?
    -   Are the identified research gaps genuine and well-supported by the analysis?
    -   Is the problem's relevance and "partially_solved" status well-integrated?
    -   *Score (1-10):*

2.  **PHASES 1-4 (Methodology, Feasibility & Timeline) (Weight: 40%)**
    -   Is the progression through the four phases logical and well-defined?
    -   Are the proposed methods, experiments, and validation frameworks technically sound?
    -   Is the timeline proposed for the phases realistic and appropriate for the project's stated scope (e.g., a PhD project)? # ✏️ MODIFIED: Timeline check is now flexible.
    -   Are the milestones and deliverables clear and measurable?
    -   *Score (1-10):*

3.  **RISK, RESOURCES & MITIGATION (Weight: 15%)**
    -   Are the resource requirements (personnel, tools, data) well-justified and realistic?
    -   Is the risk assessment comprehensive, acknowledging both technical and practical challenges?
    -   Are the mitigation strategies thoughtful and actionable?
    -   *Score (1-10):*

4.  **OUTCOMES, IMPACT & RIGOR (Weight: 25%)**
    -   Are the expected contributions clearly differentiated from existing work?
    -   Is the novelty and potential impact of the research significant?
    -   Is the publication and dissemination strategy ambitious yet credible?
    -   Are the success metrics well-defined and benchmarked against the state-of-the-art?
    -   *Score (1-10):*
---

**RECOMMENDATION GUIDELINES:**
- "finalize": Use when the weighted score is >= 8.5 and there are no major issues.
- "refine_plan": Use when the score is < 8.5 and the issues are fixable.
- "restart": Use when the plan is fundamentally flawed in its core approach or understanding of the problem.

**OUTPUT FORMAT:**
Return only a JSON object with this exact structure. For 'major_issues' and 'suggestions', specify the section of the plan the comment applies to.

{{
    "overall_score": float,  // Weighted average of the 4 section scores
    "dimension_scores": {{
        "problem_analysis_and_literature": float,
        "methodology_and_feasibility": float,
        "risk_and_resources": float,
        "outcomes_and_rigor": float
    }},
    "major_issues": [ // ✏️ MODIFIED: Now an array of objects
        {{
            "section": "The section of the plan with the issue (e.g., PHASE 3)",
            "comment": "Specific description of the major issue."
        }}
    ],
    "suggestions": [ // ✏️ MODIFIED: Now an array of objects
        {{
            "section": "The section of the plan for the suggestion (e.g., RISK MITIGATION)",
            "comment": "Specific, actionable improvement suggestion."
        }}
    ],
    "strengths": [
        "Strength 1 to preserve",
        "Strength 2 to preserve"
    ],
    "recommendation": "finalize|refine_plan|restart",
    "reasoning": "Brief explanation for the overall score and recommendation."
}}
"""

            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: client.chat.completions.create(
                    model=model,
                    temperature=0.1,  # Low temperature for consistent critique
                    messages=[{"content": critique_content, "role": "user"}]
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
                    "timestamp": f"iteration_{iteration_count}",
                    "major_issues": critique_data.get("major_issues", []),
                    "suggestions": critique_data.get("suggestions", []),
                    "strengths": critique_data.get("strengths", []),
                    "overall_score": critique_data.get("overall_score", 0.0)
                }
                
                # Initialize critique_history if it doesn't exist
                if "critique_history" not in state:
                    state["critique_history"] = []
                
                state["critique_history"].append(historical_entry)
                
                # Store critique results (current format for compatibility) - CRITICAL for refinement
                state["critique_results"] = critique_data
                
                # Get major issues for the latest critique
                major_issues = critique_data.get("major_issues", [])
                
                # Ensure critique data persists across refinement iterations
                state["latest_critique"] = {
                    "timestamp": datetime.now().isoformat(),
                    "iteration": state.get("refinement_count", 0),
                    "results": critique_data,
                    "major_issues_count": len(major_issues)
                }
                
                _write_stream("Critique analysis complete - storing results for refinement", progress=80)
                _write_stream("Critique results saved in workflow state")
                _write_stream("Analysis stored with timestamp for tracking")
                _write_stream(f"Ready for refinement iteration {state.get('refinement_count', 0) + 1}", "info")
                
                # Track score history
                overall_score = critique_data.get("overall_score", 0.0)
                state["critique_score_history"].append(overall_score)
                
                # Enhanced critique logging
                llm_recommendation = critique_data.get("recommendation", "unknown")
                
                _write_stream("Critique Results Summary:", progress=82)
                _write_stream(f"Quality Score: {overall_score:.1f}/10.0")
                _write_stream(f"Issues Identified: {len(major_issues)}")
                _write_stream(f"AI Recommendation: {llm_recommendation.upper()}")
                _write_stream(f"Top Issues: {major_issues[:2] if major_issues else 'None identified'}")
                
                if major_issues:
                    _write_stream("Key Issues Requiring Attention:")
                    for i, issue in enumerate(major_issues, 1):
                        _write_stream(f"  {i}. {issue}")
                
                suggestions = critique_data.get("suggestions", [])
                if suggestions:
                    _write_stream("Improvement Suggestions:")
                    for i, suggestion in enumerate(suggestions[:3], 1):
                        _write_stream(f"  {i}. {suggestion}")
                
                strengths = critique_data.get("strengths", [])
                if strengths:
                    _write_stream("Plan Strengths Identified:")
                    for i, strength in enumerate(strengths[:2], 1):
                        _write_stream(f"  {i}. {strength}")
                
                # Clear decision summary
                if len(major_issues) == 0:
                    _write_stream("Excellent! No major issues found - plan ready for finalization")
                elif len(major_issues) <= 2:
                    _write_stream(f"Plan refinement needed - {len(major_issues)} issues to address")
                elif len(major_issues) <= 4:
                    _write_stream(f"Significant issues detected - {len(major_issues)} problems require attention")
                else:
                    _write_stream(f"Major problems identified - {len(major_issues)} fundamental issues need resolution")
                
                state["messages"].append(
                    AIMessage(content=f"Research plan critiqued. Score: {overall_score:.1f}/10, Issues: {len(major_issues)}, Recommendation: {critique_data.get('recommendation', 'unknown')}")
                )
                
            except json.JSONDecodeError as e:
                error_msg = f"Failed to parse critique JSON: {e}"
                _write_stream(f"Critique parsing error: {error_msg}")
                # Default critique for parsing failures
                state["critique_results"] = {
                    "overall_score": 5.0,
                    "recommendation": "refine_plan",
                    "major_issues": ["Critique parsing failed"],
                    "suggestions": ["Manual review recommended"],
                    "reasoning": "Automatic critique failed, defaulting to refinement"
                }
                state["critique_score_history"].append(5.0)
        
        except Exception as e:
            error_msg = f"Critique process failed: {str(e)}"
            state["errors"].append(error_msg)
            _write_stream(f"Critique analysis failed: {error_msg}")
            # Default to accepting plan if critique fails
            state["critique_results"] = {
                "overall_score": 7.0,
                "recommendation": "finalize",
                "major_issues": [],
                "suggestions": [],
                "reasoning": "Critique failed, proceeding with original plan"
            }
            state["critique_score_history"].append(7.0)
        
        return state

async def _finalize_plan_node(state: ResearchPlanningState, *, config: Optional[Dict[str, Any]] = None) -> ResearchPlanningState:
        """Node for finalizing the research plan and preparing outputs."""

        _write_stream("Finalizing research plan and preparing deliverables", progress=95)
        state["current_step"] = "finalize_plan"
        
        try:
            research_plan = state.get("research_plan", {})
            critique = state.get("critique_results", {})
            
            # Add finalization metadata
            finalization_data = {
                "finalized_at": datetime.now().isoformat(),
                "total_iterations": state.get("iteration_count", 0),
                "total_refinements": state.get("refinement_count", 0),
                "final_critique_score": critique.get("overall_score", 0.0),
                "problems_generated": len(state.get("generated_problems", [])),
                "web_search_performed": state.get("validation_results", {}).get("web_search_performed", False)
            }
            
            research_plan.update(finalization_data)
            state["research_plan"] = research_plan
            
            _write_stream("Research plan completed successfully!", progress=100)
            
            # Add finalization message
            state["messages"].append(
                AIMessage(content=f"Research plan finalized with score {critique.get('overall_score', 0.0):.1f}/10 after {state.get('refinement_count', 0)} refinements")
            )
            
        except Exception as e:
            error_msg = f"Plan finalization failed: {str(e)}"
            state["errors"].append(error_msg)
            _write_stream(f"Finalization error: {error_msg}")
        
        return state
