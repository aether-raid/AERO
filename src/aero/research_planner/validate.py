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

async def _validate_problem_node(state: ResearchPlanningState, *, config: Optional[Dict[str, Any]] = None) -> ResearchPlanningState:
        """Node for validating if the generated problem is already solved using web search."""
        
        # Validate clients are available in state
        client = state.get("client")
        tavily_client = state.get("tavily_client")
        if not client:
            state["errors"] = state.get("errors", []) + ["OpenAI client not found in state. Please initialize clients."]
            return state
        if not tavily_client:
            state["errors"] = state.get("errors", []) + ["Tavily client not found in state. Please initialize clients."]
            return state
            
        model = state.get("model", "gpt-4o-mini")

        _write_stream("Validating research novelty", progress=30)
        _write_stream("Searching existing research to ensure your problem is novel")
        state["current_step"] = "validate_problem"
        
        try:
            current_problem = state["current_problem"]
            keywords = current_problem.get("keywords", [])
            problem_statement = current_problem.get("statement", "")
            description = current_problem.get("description", "")
            
            # Step 1: Perform web searches to find existing solutions
            _write_stream("Searching academic databases and research papers")
            
            # Check if Tavily client is available - already validated above
            # if not tavily_client:
            #     raise Exception("Tavily client not initialized. Web search unavailable.")
            
            # Construct search queries to find existing solutions
            search_queries = [
                f"{problem_statement} solution",
                f"{problem_statement} solved",
                f"{problem_statement} research paper",
                f"{problem_statement} state of the art",
                " ".join(keywords) + " solution" if keywords else problem_statement,
                " ".join(keywords) + " recent advances" if keywords else f"{problem_statement} advances"
            ]
            
            all_search_results = []
            search_summaries = []
            
            # Perform searches using Tavily
            for query in search_queries[:3]:  # Limit to 3 queries to avoid rate limits
                try:
                    _write_stream(f"Searching: {query[:50]}{'...' if len(query) > 50 else ''}")
                    
                    # Use Tavily search
                    search_response = await asyncio.get_event_loop().run_in_executor(
                        None, 
                        lambda q=query: tavily_client.search(q, max_results=10)
                    )
                    
                    if search_response and "results" in search_response:
                        # Extract URLs and titles from Tavily response for better citations
                        results_info = []
                        for result in search_response["results"]:
                            results_info.append({
                                "url": result.get("url", ""),
                                "title": result.get("title", ""),
                                "query_used": query
                            })
                        
                        urls = [info["url"] for info in results_info]
                        all_search_results.extend(urls)
                        
                        # Store detailed results info for better citations
                        if "detailed_results" not in state:
                            state["detailed_results"] = []
                        state["detailed_results"].extend(results_info)
                        
                        search_summaries.append(f"Query: '{query}' - Found {len(urls)} results")
                        _write_stream(f"Found {len(urls)} relevant papers")
                    else:
                        search_summaries.append(f"Query: '{query}' - No results")
                        _write_stream("No relevant papers found for this query")
                        
                except Exception as search_error:
                    _write_stream(f"Search error: {str(search_error)}")
                    search_summaries.append(f"Query: '{query}' - Error: {str(search_error)}")
            
            # Step 2: Analyze search results with LLM
            _write_stream("Analyzing research findings to assess novelty", progress=35)
            
            # Format search results for analysis
            formatted_results = ""
            if all_search_results:
                formatted_results = "\n".join([f"- {url}" for url in all_search_results[:20]])  # Limit to first 20 URLs
            else:
                formatted_results = "No search results found"
            
            # Create comprehensive analysis prompt with feedback generation
            analysis_content = f"""
You are an expert research validator with SMART FEEDBACK capabilities. Analyze the web search results to determine if a research problem has already been solved AND provide detailed feedback for improvement.

Research Problem: {problem_statement}
Description: {description}
Keywords: {', '.join(keywords)}

Web Search Results:
{formatted_results}

Search Queries Performed:
{chr(10).join(search_summaries)}

Based on the search results, analyze whether this problem:

1. **Is already solved**: Search results show conclusive solutions and implementations
2. **Is well-studied**: Many search results indicate extensive research exists
3. **Is partially solved**: Some solutions exist but search results suggest gaps remain
4. **Is open/novel**: Few or no relevant search results, indicating a research gap

Assessment Criteria:
- Number and relevance of search results
- Presence of solution-oriented URLs and papers
- Academic paper URLs vs general web content
- Recent research activity indicated by search results

🆕 SMART FEEDBACK GENERATION:
If recommending "reject", provide specific guidance for improvement:

Respond with a JSON object:
{{
    "status": "solved" | "well_studied" | "partially_solved" | "open",
    "confidence": 0.0-1.0,
    "reasoning": "Detailed explanation based on search results analysis",
    "existing_solutions": ["URLs or papers that show existing solutions"],
    "research_gaps": ["gaps identified from search analysis"],
    "recommendation": "accept" | "reject",
    "search_evidence": {{
        "total_results": {len(all_search_results)},
        "solution_indicators": "number of results suggesting solutions",
        "academic_sources": "presence of academic/research URLs",
        "recency_indicators": "evidence of recent research activity"
    }},
    "rejection_feedback": {{
        "primary_reason": "too_broad" | "already_solved" | "well_studied" | "duplicate" | "unclear" | "not_novel",
        "specific_issues": [
            "Issue 1: specific problem with the statement",
            "Issue 2: another specific issue"
        ],
        "improvement_suggestions": [
            "Suggestion 1: specific way to improve",
            "Suggestion 2: another specific improvement"
        ],
        "scope_guidance": "How to narrow or adjust the scope",
        "novelty_guidance": "How to make the problem more novel",
        "alternative_angles": ["alternative approach 1", "alternative approach 2"],
        "specific_guidance": "Detailed guidance for the next generation attempt"
    }}
}}

Guidelines for rejection feedback:
- "too_broad": Problem scope is too wide, suggest narrowing to specific aspect
- "already_solved": Existing solutions found, suggest unexplored variations or improvements
- "well_studied": Extensive research exists, suggest novel angles or applications
- "duplicate": Similar to previous attempts, suggest different perspective
- "unclear": Problem statement is vague, suggest specific clarifications
- "not_novel": Limited novelty, suggest innovative aspects or gaps

Provide actionable, specific feedback that helps generate a better problem next time.
Return only the JSON object, no additional text.
"""

            # Get LLM analysis of search results
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: client.chat.completions.create(
                    model=model,
                    temperature=0.3,
                    messages=[{"content": analysis_content, "role": "user"}]
                )
            )
            
            # Parse the validation response
            validation_response = response.choices[0].message.content.strip()
            
            try:
                # Remove any markdown formatting
                if validation_response.startswith("```json"):
                    validation_response = validation_response[7:]
                if validation_response.endswith("```"):
                    validation_response = validation_response[:-3]
                validation_response = validation_response.strip()
                
                validation_data = json.loads(validation_response)
                
                # Check if current problem statement exceeds character limit
                current_statement = state["current_problem"].get("statement", "")
                if len(current_statement) > 400:
                    _write_stream(f"Problem statement too long ({len(current_statement)} characters) - generating shorter version")
                    # Override validation to reject for length
                    validation_data["recommendation"] = "reject"
                    validation_data["status"] = "too_long"
                    validation_data["confidence"] = 0.9  # High confidence in length-based rejection
                    
                    # Add or update rejection feedback for length
                    if "rejection_feedback" not in validation_data:
                        validation_data["rejection_feedback"] = {}
                    
                    validation_data["rejection_feedback"].update({
                        "primary_reason": "too_long",
                        "specific_issues": [f"Problem statement is {len(current_statement)} characters, exceeds 400 character limit"],
                        "improvement_suggestions": [
                            "Reduce statement length to under 400 characters",
                            "Use more concise language while maintaining specificity",
                            "Remove unnecessary phrases or words",
                            "Focus on the core research question"
                        ],
                        "scope_guidance": "Maintain the same scope but express it more concisely",
                        "specific_guidance": f"Current statement has {len(current_statement)} characters. Reduce by at least {len(current_statement) - 400} characters while keeping the core meaning."
                    })
                
                # Store comprehensive validation results including search info
                validation_data["web_search_performed"] = True
                validation_data["search_queries"] = search_queries[:3]
                validation_data["search_results_count"] = len(all_search_results)
                validation_data["total_urls_found"] = len(all_search_results)
                
                # Store relevant URLs for the research plan with titles
                validation_data["relevant_urls"] = all_search_results[:10]  # Store top 10 URLs
                
                # Store detailed source information for better citations
                detailed_sources = state.get("detailed_results", [])
                if detailed_sources:
                    validation_data["detailed_sources"] = detailed_sources[:10]  # Store detailed info for top 10 sources
                    # Create formatted source list for display
                    formatted_sources = []
                    for i, source in enumerate(detailed_sources[:8], 1):
                        title = source.get("title", "No title available")[:100]  # Truncate long titles
                        url = source.get("url", "")
                        formatted_sources.append(f"[{i}] {title} - {url}")
                    validation_data["formatted_sources"] = formatted_sources
                
                # Create a summary of web findings for the research plan
                web_findings_summary = f"Web search found {len(all_search_results)} results. "
                if len(all_search_results) > 10:
                    web_findings_summary += "High activity in this research area suggests established field. "
                elif len(all_search_results) < 3:
                    web_findings_summary += "Limited search results indicate potential research gap. "
                else:
                    web_findings_summary += "Moderate research activity with possible opportunities. "
                
                validation_data["web_findings"] = web_findings_summary
                
                state["validation_results"] = validation_data
                
                # Update current problem with validation info
                state["current_problem"]["validation"] = validation_data
                state["current_problem"]["status"] = "validated"
                
                status = validation_data.get("status", "unknown")
                confidence = validation_data.get("confidence", 0.0)
                recommendation = validation_data.get("recommendation", "reject")
                
                _write_stream(f"Validation Status: {status.upper()}")
                _write_stream(f"Confidence Score: {confidence:.2f}")
                _write_stream(f"Research Papers Found: {len(all_search_results)}")
                _write_stream(f"AI Recommendation: {recommendation.upper()}")
                _write_stream(f"Analysis: {validation_data.get('reasoning', 'No analysis provided')[:150]}{'...' if len(validation_data.get('reasoning', '')) > 150 else ''}", "info")
                
                # 🆕 SMART FEEDBACK: Process rejection feedback for learning
                if recommendation == "reject":
                    _write_stream("Problem requires refinement - collecting feedback for improvement")
                    
                    # Extract and store detailed feedback
                    rejection_feedback = validation_data.get("rejection_feedback", {})
                    if rejection_feedback:
                        feedback_entry = {
                            "timestamp": current_problem.get("generated_at", 0),
                            "generation_attempt": current_problem.get("generation_attempt", 0),
                            "rejected_problem": current_problem.get("statement", ""),
                            "primary_reason": rejection_feedback.get("primary_reason", "unknown"),
                            "specific_issues": rejection_feedback.get("specific_issues", []),
                            "improvement_suggestions": rejection_feedback.get("improvement_suggestions", []),
                            "scope_guidance": rejection_feedback.get("scope_guidance", ""),
                            "novelty_guidance": rejection_feedback.get("novelty_guidance", ""),
                            "alternative_angles": rejection_feedback.get("alternative_angles", []),
                            "specific_guidance": rejection_feedback.get("specific_guidance", ""),
                            "validation_reasoning": validation_data.get("reasoning", ""),
                            "search_evidence": validation_data.get("search_evidence", {})
                        }
                        
                        # Store in rejection feedback list
                        if "rejection_feedback" not in state:
                            state["rejection_feedback"] = []
                        state["rejection_feedback"].append(feedback_entry)
                        
                        # Create focused feedback context for next generation
                        primary_reason = rejection_feedback.get("primary_reason", "unknown")
                        specific_guidance = rejection_feedback.get("specific_guidance", "")
                        
                        feedback_context = f"""
Based on rejection reason '{primary_reason}':
{specific_guidance}

Specific improvements needed:
{chr(10).join(f"- {issue}" for issue in rejection_feedback.get("improvement_suggestions", []))}
"""
                        state["feedback_context"] = feedback_context
                        
                        # Print detailed feedback for user visibility
                        _write_stream(f"Primary Issue: {primary_reason.upper()}")
                        _write_stream(f"Guidance: {specific_guidance[:100]}{'...' if len(specific_guidance) > 100 else ''}")
                        if rejection_feedback.get("alternative_angles"):
                            _write_stream(f"Alternative Approaches: {', '.join(rejection_feedback['alternative_angles'][:2])}", "info")
                    
                    _write_stream("Feedback saved for next iteration")
                else:
                    _write_stream("Research problem validated successfully")
                    # Clear any previous feedback context on success
                    state["feedback_context"] = ""
                
                # Add validation message
                state["messages"].append(
                    AIMessage(content=f"Web-validated problem: {recommendation.upper()} (status: {status}, confidence: {confidence:.2f}, {len(all_search_results)} URLs analyzed)")
                )
                
            except json.JSONDecodeError as e:
                error_msg = f"Failed to parse validation JSON response: {e}"
                state["errors"].append(error_msg)
                # Default to rejection on parsing error
                state["validation_results"] = {
                    "status": "unknown",
                    "confidence": 0.0,
                    "reasoning": "Validation parsing failed",
                    "recommendation": "reject",
                    "web_search_performed": True,
                    "total_urls_found": len(all_search_results)
                }
                state["current_problem"]["validation"] = state["validation_results"]
                _write_stream(f"{error_msg} - using default validation")
        
        except Exception as e:
            error_msg = f"Web search validation failed: {str(e)}"
            state["errors"].append(error_msg)
            _write_stream(f"Validation failed: {error_msg}")
            
            # Fallback to basic LLM validation if web search fails
            _write_stream("Falling back to AI-only validation")
            try:
                fallback_content = f"""
                    Research Problem: {state['current_problem'].get('statement', '')}
                    Based on your knowledge, is this problem already solved? Respond with JSON:
                    {{"status": "solved|open", "confidence": 0.0-1.0, "reasoning": "brief explanation", "recommendation": "accept|reject"}}
                """
                fallback_response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: client.chat.completions.create(
                        model=model,
                        temperature=0.3,
                        messages=[{"content": fallback_content, "role": "user"}]
                    )
                )
                
                fallback_json = json.loads(fallback_response.choices[0].message.content.strip())
                fallback_json["web_search_performed"] = False
                fallback_json["fallback_used"] = True
                
                state["validation_results"] = fallback_json
                state["current_problem"]["validation"] = fallback_json
                _write_stream(f"Fallback validation result: {fallback_json.get('recommendation', 'reject').upper()}", "success")
                
            except Exception as fallback_error:
                _write_stream(f"Fallback validation also failed: {fallback_error}")
                # Final fallback - conservative rejection
                state["validation_results"] = {
                    "status": "unknown", 
                    "confidence": 0.0,
                    "reasoning": "Both web search and LLM validation failed",
                    "recommendation": "reject",
                    "web_search_performed": False,
                    "error": str(e)
                }
                state["current_problem"]["validation"] = state["validation_results"]
        
        return state