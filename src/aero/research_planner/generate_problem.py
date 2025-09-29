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

async def _generate_problem_node(state: ResearchPlanningState, *, config: Optional[Dict[str, Any]] = None) -> ResearchPlanningState:
        """🚀 STREAMLINED GENERATION: Generate a single research problem for Tavily validation and automatic research planning."""
        
        # Validate clients are available in state
        client = state.get("client")
        if not client:
            state["errors"] = state.get("errors", []) + ["OpenAI client not found in state. Please initialize clients."]
            return state
            
        model = state.get("model", "gpt-4o-mini")
        
        current_iter = state.get("iteration_count", 0) + 1
        state["iteration_count"] = current_iter
        
        # Track generation attempts
        generation_attempts = state.get("generation_attempts", 0) + 1
        state["generation_attempts"] = generation_attempts
        
        _write_stream(f"Generating research problem (attempt {generation_attempts})", progress=20)
        _write_stream("Creating a focused, novel research question from your query")
        state["current_step"] = "generate_problem"
        
        try:
            # Check how many problems we already have
            validated_count = len(state.get("validated_problems", []))
            generated_count = len(state.get("generated_problems", []))
            
            # 🆕 SMART FEEDBACK: Build context from previous rejections
            feedback_context = ""
            rejection_feedback = state.get("rejection_feedback", [])
            
            if rejection_feedback:
                _write_stream(f"Learning from {len(rejection_feedback)} previous attempts to improve results")
                feedback_context = "\n\n🚨 IMPORTANT - LEARN FROM PREVIOUS MISTAKES:\n"
                
                # Group rejection reasons for better learning
                rejection_patterns = {}
                for feedback in rejection_feedback[-5:]:  # Last 5 rejections
                    reason = feedback.get("primary_reason", "unknown")
                    if reason not in rejection_patterns:
                        rejection_patterns[reason] = []
                    rejection_patterns[reason].append(feedback)
                
                for reason, feedbacks in rejection_patterns.items():
                    feedback_context += f"\n❌ AVOID: {reason.upper()} ({len(feedbacks)} rejections)\n"
                    for feedback in feedbacks[-2:]:  # Last 2 examples of this type
                        rejected_problem = feedback.get("rejected_problem", "")
                        specific_issue = feedback.get("specific_guidance", "")
                        feedback_context += f"   • Rejected: \"{rejected_problem[:100]}...\"\n"
                        feedback_context += f"   • Issue: {specific_issue}\n"
                
                feedback_context += f"\n🎯 SPECIFIC GUIDANCE FOR NEXT ATTEMPT:\n{state.get('feedback_context', '')}\n"
            
            # Create context about previously generated problems to avoid repetition
            previous_problems = ""
            if state.get("generated_problems"):
                previous_problems = "\n\nPreviously generated problems (avoid similar ones):\n"
                for i, prob in enumerate(state["generated_problems"][-5:], 1):  # Show last 5
                    status = prob.get("validation", {}).get("recommendation", "unknown")
                    previous_problems += f"{i}. {prob.get('statement', 'Unknown')} [{status}]\n"
            
            # 🆕 ADAPTIVE PROMPTING: Adjust approach based on attempt number
            approach_guidance = ""
            if generation_attempts > 1:
                if generation_attempts <= 3:
                    approach_guidance = "\n🔍 FOCUS: Be more specific and narrow in scope."
                elif generation_attempts <= 5:
                    approach_guidance = "\n🔍 FOCUS: Try a different angle or subfield within the domain."
                else:
                    approach_guidance = "\n🔍 FOCUS: Consider technical implementation challenges or novel applications."

            content = f"""
                You are an expert research problem generator for STREAMLINED RESEARCH PLANNING. Your task is to generate a SINGLE, high-quality research problem that will be automatically validated with Tavily web search and then used for immediate research plan generation.

                Research Domain: {state["original_prompt"]}
                Generation attempt: #{generation_attempts} (iteration {current_iter})
                Workflow: Single Problem → Tavily Validation → Auto Research Planning

                {feedback_context}

                {previous_problems}

                {approach_guidance}

                Requirements for the research problem (STREAMLINED WORKFLOW):
                1. **HIGH QUALITY**: Must be excellent since it will be auto-used for research planning
                2. **SPECIFIC**: Clearly defined scope and objectives (avoid being too broad)
                3. **NOVEL**: Not obviously solved (will be verified by Tavily web search)
                4. **FEASIBLE**: Can realistically be addressed with current technology
                5. **IMPACTFUL**: Would advance the field if solved
                6. **MEASURABLE**: Success can be quantified or evaluated
                7. **CONCISE**: Must be within 400 characters for efficiency
                8. **RESEARCH-READY**: Should immediately lead to actionable research plan

                Generate ONE exceptional research problem that will automatically proceed to research planning:
                - Addresses a concrete, specific gap or limitation
                - Can be formulated as a clear, focused research question  
                - Is narrow enough to be tackled in a research project
                - Is different from any previously generated problems
                - Incorporates lessons learned from previous rejections (if any)
                - Will survive Tavily web validation for novelty

                Respond with a JSON object containing:
                {{
                    "statement": "Clear, specific problem statement (1-2 sentences)",
                    "description": "Brief description of why this is important (2-3 sentences)",
                    "keywords": ["key", "terms", "for", "validation", "search"],
                    "research_question": "Specific research question this addresses",
                    "novelty_claim": "What makes this problem novel and different from existing work",
                    "scope_justification": "Why this scope is appropriate (not too broad/narrow)"
                }}

                Focus on being specific and avoiding overly broad or obviously solved problems.
                Return only the JSON object, no additional text.
            """

            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: client.chat.completions.create(
                    model=model,
                    temperature=0.7,  # Higher temperature for more creative problem generation
                    messages=[{"content": content, "role": "user"}]
                )
            )
            
            # Parse the LLM response
            llm_response = response.choices[0].message.content.strip()
            
            try:
                # Remove any markdown formatting
                if llm_response.startswith("```json"):
                    llm_response = llm_response[7:]
                if llm_response.endswith("```"):
                    llm_response = llm_response[:-3]
                llm_response = llm_response.strip()
                
                problem_data = json.loads(llm_response)
                
                # Add metadata
                problem_data["generated_at"] = current_iter
                problem_data["generation_attempt"] = generation_attempts
                problem_data["status"] = "pending_validation"
                problem_data["learned_from_rejections"] = len(rejection_feedback)
                
                # Store current problem for validation
                state["current_problem"] = problem_data
                
                # Add to generated problems list
                if "generated_problems" not in state:
                    state["generated_problems"] = []
                state["generated_problems"].append(problem_data.copy())
                
                _write_stream("Research problem generated successfully", progress=25)
                _write_stream(f"Problem: {problem_data['statement'][:100]}{'...' if len(problem_data['statement']) > 100 else ''}")
                if rejection_feedback:
                    _write_stream(f"Applied insights from {len(rejection_feedback)} previous attempts")
                
                # Add success message
                state["messages"].append(
                    AIMessage(content=f"Generated research problem #{current_iter} (attempt #{generation_attempts}): {problem_data['statement']}")
                )
                
            except json.JSONDecodeError as e:
                error_msg = f"Failed to parse problem generation JSON response: {e}"
                state["errors"].append(error_msg)
                # Create a fallback problem
                state["current_problem"] = {
                    "statement": f"Investigate novel approaches in {state['original_prompt']}",
                    "description": "Fallback problem due to parsing error",
                    "keywords": ["research", "novel", "approaches"],
                    "research_question": f"How can we advance the state of {state['original_prompt']}?",
                    "generated_at": current_iter,
                    "generation_attempt": generation_attempts,
                    "status": "pending_validation"
                }
                _write_stream(f"{error_msg} - using fallback problem")
        
        except Exception as e:
            error_msg = f"Problem generation failed: {str(e)}"
            state["errors"].append(error_msg)
            # Create a fallback problem
            state["current_problem"] = {
                "statement": f"Research challenges in {state['original_prompt']}",
                "description": "Fallback problem due to generation error",
                "keywords": ["research", "challenges"],
                "research_question": f"What are the key challenges in {state['original_prompt']}?",
                "generated_at": current_iter,
                "generation_attempt": generation_attempts,
                "status": "pending_validation"
            }
            _write_stream(f"{error_msg} - using fallback problem")
        
        return state