#!/usr/bin/env python3
"""
Research Planning Workflow Nodes
================================

This module contains all the workflow nodes and functions specifically for the 
research planning workflow, extracted from ml_researcher_langgraph.py.

The research planning workflow includes:
- Problem generation and validation
- Research plan creation and structuring  
- Plan critique and iterative refinement
- Document generation utilities

These functions are designed to be imported and used by the main MLResearcherLangGraph class.
"""

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

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

# LLM and related imports
import openai

# Web search imports
from tavily import TavilyClient

class BaseState(TypedDict):
    """Base state for all workflows."""
    messages: Annotated[List[BaseMessage], add_messages]
    original_prompt: str  # Pure user query without uploaded data
    uploaded_data: List[str]  # Uploaded file contents as separate field
    current_step: str
    errors: List[str]

# Suppress warnings
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import logging
logging.getLogger("openai._base_client").setLevel(logging.WARNING)


# ==================================================================================
# GLOBAL CLIENTS (for standalone execution)
# ==================================================================================

# Initialize clients globally for standalone use
_client = None
_tavily_client = None
_model = None

def _initialize_clients():
    """Initialize global clients for standalone execution."""
    global _client, _tavily_client, _model
    
    if _client is None:
        try:
            api_key = _load_from_env_file("OPENAI_API_KEY")
            base_url = _load_from_env_file("BASE_URL") or "https://agents.aetherraid.dev"
            _model = _load_from_env_file("MODEL") or "gemini/gemini-2.5-flash"
            
            if api_key:
                import openai
                _client = openai.OpenAI(
                    api_key=api_key,
                    base_url=base_url
                )
                print("✅ OpenAI client initialized successfully")
            else:
                print("⚠️  OPENAI_API_KEY not found, using mock client")
                _client = None
        except ImportError:
            print("⚠️  openai not available, using mock client")
            _client = None
        except Exception as e:
            print(f"⚠️  OpenAI client initialization failed: {e}")
            _client = None
    
    if _tavily_client is None:
        try:
            tavily_key = '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'
            if tavily_key:
                _tavily_client = TavilyClient(api_key=tavily_key)
                print("✅ Tavily web search client initialized successfully")
            else:
                print("⚠️  TAVILY_API_KEY not found, web search disabled")
        except Exception as e:
            print(f"⚠️  Tavily client initialization failed: {e}")
            _tavily_client = None


# ==================================================================================
# STATE DEFINITIONS
# ==================================================================================

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


# ==================================================================================
# UTILITY FUNCTIONS
# ==================================================================================

def _load_from_env_file(key: str) -> Optional[str]:
    """Load environment variable from env.example file."""
    try:
        env_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..\env.example")
        if os.path.exists(env_file_path):
            with open(env_file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith(f"{key}="):
                        return line.split("=", 1)[1].strip().strip('"').strip("'")
        return os.getenv(key)
    except Exception:
        return os.getenv(key)


# ==================================================================================
# NODE FUNCTIONS (for import by main MLResearcherLangGraph class)
# ==================================================================================
        """Build the research planning workflow graph."""
        workflow = StateGraph(ResearchPlanningState)
        
        # Add nodes for the streamlined single-problem workflow  
        workflow.add_node("generate_problem", self._generate_problem_node)
        workflow.add_node("validate_problem", self._validate_problem_node)
        workflow.add_node("process_rejection_feedback", self._process_rejection_feedback_node)
        workflow.add_node("create_plan", self._create_research_plan_node)
        workflow.add_node("critique_plan", self._critique_plan_node)
        workflow.add_node("finalize_plan", self._finalize_plan_node)
        
        # Set entry point
        workflow.set_entry_point("generate_problem")
        
        # Add edges with decision logic
        workflow.add_conditional_edges(
            "generate_problem",
            self._streamlined_validation_decision,
            {
                "validate": "validate_problem",
                "regenerate": "generate_problem"
            }
        )
        
        workflow.add_conditional_edges(
            "validate_problem", 
            self._streamlined_validation_decision,
            {
                "accepted": "create_plan",           # Problem validated as unsolved
                "rejected": "process_rejection_feedback",  # Problem already solved, need new one
                "regenerate": "generate_problem",     # Validation failed, try again
                "error": "process_rejection_feedback"
            }
        )
        
        workflow.add_edge("process_rejection_feedback", "generate_problem")
        
        workflow.add_conditional_edges(
            "create_plan",
            self._determine_refinement_path,
            {
                "critique": "critique_plan",
                "finalize": "finalize_plan"
            }
        )
        
        workflow.add_conditional_edges(
            "critique_plan",
            self._determine_refinement_path,
            {
                "refine": "create_plan",        # Need to improve the plan
                "finalize": "finalize_plan"     # Plan is good enough
            }
        )
        
        workflow.add_edge("finalize_plan", END)
        
        return workflow.compile()
    
    # ==================================================================================
    # RESEARCH PLANNING WORKFLOW NODES  
    # ==================================================================================
    
    # --- PHASE 1: PROBLEM GENERATION & VALIDATION ---

async def _generate_problem_node(state: ResearchPlanningState) -> ResearchPlanningState:
        """🚀 STREAMLINED GENERATION: Generate a single research problem for Tavily validation and automatic research planning."""
        current_iter = state.get("iteration_count", 0) + 1
        state["iteration_count"] = current_iter
        
        # Track generation attempts
        generation_attempts = state.get("generation_attempts", 0) + 1
        state["generation_attempts"] = generation_attempts
        
        print(f"\n🎯 Step {current_iter}: Generating research problem for auto-validation (attempt #{generation_attempts})...")
        print(f"🚀 Streamlined workflow: One problem → Tavily validation → Automatic research planning")
        state["current_step"] = "generate_problem"
        
        try:
            # Check how many problems we already have
            validated_count = len(state.get("validated_problems", []))
            generated_count = len(state.get("generated_problems", []))
            
            # 🆕 SMART FEEDBACK: Build context from previous rejections
            feedback_context = ""
            rejection_feedback = state.get("rejection_feedback", [])
            
            if rejection_feedback:
                print(f"🧠 Learning from {len(rejection_feedback)} previous rejections...")
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
                lambda: _client.chat.completions.create(
                    model=_model,
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
                
                print(f"✅ Generated problem: {problem_data['statement']}")
                print(f"🔍 Keywords for validation: {', '.join(problem_data.get('keywords', []))}")
                if rejection_feedback:
                    print(f"🧠 Incorporated feedback from {len(rejection_feedback)} previous rejections")
                
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
                print(f"⚠️  {error_msg}, using fallback problem")
        
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
            print(f"❌ {error_msg}, using fallback problem")
        
        return state
    
async def _validate_problem_node(state: ResearchPlanningState) -> ResearchPlanningState:
        """Node for validating if the generated problem is already solved using web search."""
        print(f"\n🔍 Validating problem with web search: {state['current_problem']['statement'][:60]}...")
        state["current_step"] = "validate_problem"
        
        try:
            current_problem = state["current_problem"]
            keywords = current_problem.get("keywords", [])
            problem_statement = current_problem.get("statement", "")
            description = current_problem.get("description", "")
            
            # Step 1: Perform web searches to find existing solutions
            print("🌐 Performing web searches for existing solutions...")
            
            # Check if Tavily client is available
            if not _tavily_client:
                raise Exception("Tavily client not initialized. Web search unavailable.")
            
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
                    print(f"🔍 Searching: {query[:50]}...")
                    
                    # Use Tavily search
                    search_response = await asyncio.get_event_loop().run_in_executor(
                        None, 
                        lambda q=query: _tavily_client.search(q, max_results=10)
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
                        print(f"  ✅ Found {len(urls)} results")
                    else:
                        search_summaries.append(f"Query: '{query}' - No results")
                        print(f"  ❌ No results found")
                        
                except Exception as search_error:
                    print(f"  ⚠️  Search error for '{query}': {search_error}")
                    search_summaries.append(f"Query: '{query}' - Error: {str(search_error)}")
            
            # Step 2: Analyze search results with LLM
            print("🧠 Analyzing search results with LLM...")
            
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
                lambda: _client.chat.completions.create(
                    model=_model,
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
                    print(f"⚠️  Problem statement too long ({len(current_statement)} chars), forcing rejection...")
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
                
                print(f"📊 Validation Status: {status.upper()}")
                print(f"🎯 Confidence: {confidence:.2f}")
                print(f"🌐 Search Results: {len(all_search_results)} URLs found")
                print(f"💡 Recommendation: {recommendation.upper()}")
                print(f"🧠 Reasoning: {validation_data.get('reasoning', 'No reasoning provided')[:150]}...")
                
                # 🆕 SMART FEEDBACK: Process rejection feedback for learning
                if recommendation == "reject":
                    print("❌ Problem rejected - storing feedback for learning")
                    
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
                        print(f"🚨 Rejection Reason: {primary_reason.upper()}")
                        print(f"💡 Guidance: {specific_guidance[:100]}...")
                        if rejection_feedback.get("alternative_angles"):
                            print(f"🔄 Suggested Angles: {', '.join(rejection_feedback['alternative_angles'][:2])}")
                    
                    print("🧠 Feedback stored for next generation attempt")
                else:
                    print("✅ Problem validated as open research opportunity")
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
                print(f"⚠️  {error_msg}, defaulting to rejection")
        
        except Exception as e:
            error_msg = f"Web search validation failed: {str(e)}"
            state["errors"].append(error_msg)
            print(f"❌ {error_msg}")
            
            # Fallback to basic LLM validation if web search fails
            print("🔄 Falling back to LLM-only validation...")
            try:
                fallback_content = f"""
                    Research Problem: {state['current_problem'].get('statement', '')}
                    Based on your knowledge, is this problem already solved? Respond with JSON:
                    {{"status": "solved|open", "confidence": 0.0-1.0, "reasoning": "brief explanation", "recommendation": "accept|reject"}}
                """
                fallback_response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: _client.chat.completions.create(
                        model=_model,
                        temperature=0.3,
                        messages=[{"content": fallback_content, "role": "user"}]
                    )
                )
                
                fallback_json = json.loads(fallback_response.choices[0].message.content.strip())
                fallback_json["web_search_performed"] = False
                fallback_json["fallback_used"] = True
                
                state["validation_results"] = fallback_json
                state["current_problem"]["validation"] = fallback_json
                print(f"✅ Fallback validation: {fallback_json.get('recommendation', 'reject').upper()}")
                
            except Exception as fallback_error:
                print(f"❌ Fallback validation also failed: {fallback_error}")
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

async def _process_rejection_feedback_node(state: ResearchPlanningState) -> ResearchPlanningState:
        """🆕 SMART FEEDBACK: Process rejection feedback and prepare for next generation."""
        print(f"\n🧠 Processing rejection feedback for smarter generation...")
        state["current_step"] = "process_feedback"
        
        try:
            validation_results = state.get("validation_results", {})
            current_problem = state.get("current_problem", {})
            
            # Get the latest rejection feedback
            rejection_feedback_list = state.get("rejection_feedback", [])
            if not rejection_feedback_list:
                print("⚠️  No rejection feedback to process")
                return state
            
            latest_feedback = rejection_feedback_list[-1]
            primary_reason = latest_feedback.get("primary_reason", "unknown")
            
            print(f"📊 Analyzing rejection pattern: {primary_reason}")
            
            # Analyze rejection patterns for adaptive strategy
            rejection_patterns = {}
            for feedback in rejection_feedback_list:
                reason = feedback.get("primary_reason", "unknown")
                rejection_patterns[reason] = rejection_patterns.get(reason, 0) + 1
            
            # Determine adaptive strategy based on patterns
            total_rejections = len(rejection_feedback_list)
            most_common_reason = max(rejection_patterns.items(), key=lambda x: x[1])[0] if rejection_patterns else "unknown"
            
            print(f"🔍 Pattern Analysis: {total_rejections} rejections, most common: {most_common_reason}")
            
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
            
            print(f"🎯 Updated strategy: {strategic_guidance}")
            print(f"📝 Enhanced feedback context prepared for next generation")
            
            # Add processing message
            state["messages"].append(
                AIMessage(content=f"Processed rejection feedback: {primary_reason} (pattern analysis complete, adaptive strategy updated)")
            )
            
        except Exception as e:
            error_msg = f"Feedback processing failed: {str(e)}"
            state["errors"].append(error_msg)
            print(f"❌ {error_msg}")
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

def _create_research_plan_node(state: ResearchPlanningState) -> ResearchPlanningState:
        """Node for creating comprehensive research plan based on the selected problem."""
        selected_problem = state.get("selected_problem", {})
        
        # Check if this is a refinement iteration
        is_refinement = state.get("critique_results") is not None and state.get("refinement_count", 0) > 0
        
        if is_refinement:
            print(f"\n� Step: Refining research plan (iteration {state.get('refinement_count', 0) + 1})...")
            # Increment refinement count
            state["refinement_count"] = state.get("refinement_count", 0) + 1
            print(f"🎯 Addressing critique feedback...")
            
            # Verify critique data is available for refinement
            critique = state.get("critique_results", {})
            if not critique:
                print("⚠️  WARNING: No critique results found - this may indicate a state management issue")
                print("⚠️  Proceeding with limited refinement capability")
            else:
                major_issues = critique.get("major_issues", [])
                score = critique.get("overall_score", 0)
                print(f"🎯 REFINEMENT TARGET: Improve score from {score:.1f}/10 by addressing {len(major_issues)} major issues")
        else:
            print(f"\n�📋 Step 4: Generating comprehensive research plan for selected problem...")
            print(f"🎯 Selected Problem: {selected_problem.get('statement', 'N/A')[:100]}...")
            # Initialize refinement tracking
            state["refinement_count"] = 0
            state["previous_plans"] = []
            state["critique_score_history"] = []
        
        state["current_step"] = "create_research_plan"
        
        try:
            # Clean all text inputs to avoid encoding issues
            clean_prompt = _clean_text_for_encoding(state["original_prompt"])
            
            # Validate that we have a proper selected problem
            if not selected_problem or not selected_problem.get('statement'):
                # Try to get from current_problem as fallback
                current_problem = state.get("current_problem", {})
                if current_problem and current_problem.get('statement'):
                    selected_problem = current_problem
                    state["selected_problem"] = current_problem  # Store it properly
                    print(f"✅ Using current_problem as selected_problem: {current_problem.get('statement', 'N/A')[:80]}...")
                else:
                    error_msg = "No valid problem selected for research plan generation"
                    state["errors"].append(error_msg)
                    print(f"❌ {error_msg}")
                    print(f"🔍 Debug - selected_problem: {selected_problem}")
                    print(f"🔍 Debug - current_problem: {current_problem}")
                    return state
            
            # Format the selected problem for the prompt
            problems_text = "\n**Selected Research Problem:**\n"
            problems_text += f"- **Statement:** {selected_problem.get('statement', 'N/A')}\n"
            problems_text += f"- **Description:** {selected_problem.get('description', 'N/A')}\n"
            problems_text += f"- **Research Question:** {selected_problem.get('research_question', 'N/A')}\n"
            problems_text += f"- **Keywords:** {', '.join(selected_problem.get('keywords', []))}\n"
            
            validation = selected_problem.get('validation', {})
            problems_text += f"- **Validation Status:** {validation.get('status', 'unknown')}\n"
            problems_text += f"- **Validation Confidence:** {validation.get('confidence', 0.0):.2f}\n"
            problems_text += f"- **Research Gaps:** {', '.join(validation.get('research_gaps', []))}\n"
            
            # Include web search findings if available
            if validation.get('web_search_performed', False):
                problems_text += f"- **Validation Method:** Web Search Analysis\n"
                problems_text += f"- **Search Results Found:** {validation.get('search_results_count', 0)}\n"
                
                # Include relevant URLs found during validation with better formatting
                relevant_urls = validation.get('relevant_urls', [])
                detailed_sources = validation.get('detailed_sources', [])
                formatted_sources = validation.get('formatted_sources', [])
                
                if formatted_sources:
                    problems_text += f"- **Identified Sources for Literature Review:**\n"
                    for source in formatted_sources:
                        problems_text += f"  {source}\n"
                elif relevant_urls:
                    # Fallback to URLs only if detailed sources not available
                    problems_text += f"- **Identified Sources for Literature Review:**\n"
                    for j, url in enumerate(relevant_urls[:8], 1):
                        problems_text += f"  [{j}] {url}\n"
                
                # Include search queries used for transparency
                search_queries = validation.get('search_queries', [])
                if search_queries:
                    query_list = ', '.join([f"'{q}'" for q in search_queries])
                    problems_text += f"- **Search Strategies Used:** {query_list}\n"
                
                # Include key findings from web search
                web_findings = validation.get('web_findings', '')
                if web_findings:
                    problems_text += f"- **Current Research State:** {web_findings}\n"
                        
                # Include existing solutions found
                existing_solutions = validation.get('existing_solutions', [])
                if existing_solutions:
                    problems_text += f"- **Existing Approaches to Build Upon:** {', '.join(existing_solutions[:5])}\n"
            else:
                problems_text += f"- **Validation Method:** LLM-based (fallback, no web search)\n"
                
                problems_text += "\n"
            
            clean_problems = _clean_text_for_encoding(problems_text)
            
            # Add refinement context if this is a refinement iteration
            refinement_context = ""
            if is_refinement:
                critique = state.get("critique_results", {})
                previous_plan = state.get("research_plan", {}).get("research_plan", "")
                major_issues = critique.get("major_issues", [])
                suggestions = critique.get("suggestions", [])
                strengths = critique.get("strengths", [])
                
                # Format major issues with high priority
                issues_text = ""
                if major_issues:
                    issues_text = "\n".join([f"    ❌ CRITICAL ISSUE {i+1}: {issue}" for i, issue in enumerate(major_issues)])
                
                # Format specific suggestions
                suggestions_text = ""
                if suggestions:
                    suggestions_text = "\n".join([f"    💡 SUGGESTION {i+1}: {suggestion}" for i, suggestion in enumerate(suggestions)])
                
                # Format strengths to preserve
                strengths_text = ""
                if strengths:
                    strengths_text = "\n".join([f"    ✅ PRESERVE: {strength}" for strength in strengths])

                refinement_context = f"""

⚠️⚠️⚠️ CRITICAL REFINEMENT MODE - ITERATION {state['refinement_count']} ⚠️⚠️⚠️

**PREVIOUS PLAN CRITIQUE RESULTS:**
- Overall Score: {critique.get('overall_score', 0):.1f}/10 (NEEDS IMPROVEMENT)
- Number of Major Issues: {len(major_issues)}

**🚨 HIGH PRIORITY: MAJOR ISSUES TO FIX IMMEDIATELY:**
{issues_text}

**💡 SPECIFIC IMPROVEMENT REQUIREMENTS:**
{suggestions_text}

**✅ SUCCESSFUL ELEMENTS TO PRESERVE AND BUILD UPON:**
{strengths_text}

**📋 PREVIOUS RESEARCH PLAN (FOR REFERENCE AND IMPROVEMENT):**
{previous_plan}

**🎯 REFINEMENT INSTRUCTIONS (CRITICAL PRIORITIES):**
1. 🚨 HIGHEST PRIORITY: Address EVERY major issue listed above explicitly
2. 💡 IMPLEMENT: Follow each specific suggestion to enhance the plan
3. ✅ PRESERVE: Maintain and expand upon identified strengths
4. 🔄 IMPROVE: Make substantial improvements to low-scoring sections
5. 📊 ENHANCE: Ensure significantly better quality than previous iteration
6. 🎯 FOCUS: Be more specific, detailed, and academically rigorous

**⚡ CRITICAL SUCCESS CRITERIA:**
- Must address all {len(major_issues)} major issues identified
- Must implement specific suggestions for improvement
- Must significantly improve overall quality and feasibility
- Must maintain academic rigor while being more practical

              """

            task_description = ""
            if is_refinement:
                task_description = f"""
**🚨 PRIMARY TASK: CRITICAL REFINEMENT - ITERATION {state['refinement_count']}**
You MUST significantly improve the previous research plan by addressing all critique feedback. This is NOT a new plan - this is a targeted improvement of an existing plan that scored {critique.get('overall_score', 0):.1f}/10.

**REFINEMENT SUCCESS CRITERIA:**
- Address EVERY major issue explicitly
- Implement ALL improvement suggestions  
- Achieve significantly higher quality than previous iteration
- Maintain successful elements while fixing problems

**SECONDARY TASK:** Create a comprehensive research plan that leverages both the selected problem AND the web search findings."""
            else:
                task_description = """
**YOUR TASK:**
Create a comprehensive research plan that leverages both the selected problem AND the web search findings. The plan should focus deeply on this specific problem, utilizing its research potential, feasibility, and the current state of research as revealed by web analysis."""
            
            content = f"""
                You are an expert research project manager and academic research planner. Your task is to create a comprehensive, actionable research plan based on a specifically selected research problem that has been systematically identified and verified through web search analysis.
                {refinement_context}
                **RESEARCH CONTEXT:**

                **Research Domain/Query:** {clean_prompt}

                **SELECTED RESEARCH PROBLEM (Web-Search Validated):**
                {clean_problems[:4000]}

                **IMPORTANT NOTE:** This problem has been user-selected from multiple validated options and has been verified using real-time web search. Pay special attention to:
                - Web search validation provides current market/research validation
                - Relevant resources and URLs have been identified for immediate follow-up
                - Current research state information is based on actual web findings
                - Use the provided URLs and existing approaches as starting points for literature review

{task_description}

**CITATION AND SOURCE INTEGRATION REQUIREMENTS:**
- Reference the discovered URLs throughout the plan using [1], [2], [3] format
- Include specific sources in each phase where relevant
- Use the identified sources as immediate starting points for literature review
- Build research methodology based on approaches found in these sources
- Ensure proper attribution and citation planning for final publications

                **REQUIRED STRUCTURE:**

                ## EXECUTIVE SUMMARY
                - Brief overview of the research objectives
                - Summary of the selected web-validated research problem
                - Research prioritization strategy based on web search findings
                - Expected timeline and outcomes

## WEB-INFORMED PROBLEM ANALYSIS
- Detailed analysis of the selected research problem
- Current research activity level based on web search insights
- Assessment of research gaps and opportunities identified through source analysis
- Key resources and URLs identified for immediate follow-up (reference specific sources [1], [2], etc.)
- Analysis of existing approaches found in discovered sources
- Identification of potential collaboration opportunities from source authors/institutions
- Competitive landscape analysis based on web-discovered research

## PHASE 1: FOUNDATION & LITERATURE REVIEW (First ~15% of Project Timeline)
Comprehensive literature review strategy starting with sources identified during web validation.

**IMMEDIATE LITERATURE REVIEW SOURCES:**
Use the following web-discovered sources as starting points for literature review:
{f"Sources to prioritize:{chr(10)}{chr(10).join(validation.get('formatted_sources', []))}" if validation.get('formatted_sources') else f"URL-only sources:{chr(10)}{chr(10).join([f'[{i}] {url}' for i, url in enumerate(validation.get('relevant_urls', [])[:8], 1)])}" if validation.get('relevant_urls') else "No specific sources identified - use general literature search"}

**Phase 1 Tasks:**
- Systematic review of identified sources above (Priority 1)
- Follow citation networks from discovered papers (Priority 2)  
- Key papers and research groups to study (expand from web-found sources)
- Knowledge gap validation through the identified web resources
- Initial research question refinement based on current state analysis
- Specific tasks and deliverables for this phase

                ## PHASE 2: PROBLEM FORMULATION & EXPERIMENTAL DESIGN (Next ~10% of Project Timeline)
                Formalize specific research hypotheses for priority problems.

**Building on Web-Discovered Research:**
- Analyze methodologies found in identified sources: {f"[{', '.join([str(i) for i in range(1, min(len(validation.get('formatted_sources', validation.get('relevant_urls', []))), 8) + 1)])}]" if validation.get('formatted_sources') or validation.get('relevant_urls') else "N/A"}
- Design initial experiments or theoretical approaches
- Identify required datasets, tools, and resources (leverage web-found resources)
- Risk assessment for each chosen problem based on challenges identified in literature
- Specific tasks and deliverables for this phase

                ## PHASE 3: ACTIVE RESEARCH & DEVELOPMENT (Core ~50% of Project Timeline)
                Research execution plan for each chosen problem.

**Research Strategy Informed by Current Literature:**
- Experimental design and methodology informed by approaches in sources {f"[{', '.join([str(i) for i in range(1, min(len(validation.get('formatted_sources', validation.get('relevant_urls', []))), 6) + 1)])}]" if validation.get('formatted_sources') or validation.get('relevant_urls') else "N/A"}
- Progress milestones and validation metrics benchmarked against existing work
- Collaboration strategies with research groups identified through web search
- Build upon existing work found through URL analysis
- Expected outcomes and publications plan
- Specific tasks and deliverables for this phase

                ## PHASE 4: EVALUATION, SYNTHESIS & DISSEMINATION (Final ~25% of Project Timeline)
                Results evaluation framework comparing against the current state identified via web search.

**Publication Strategy with Proper Attribution:**
- Validation of research contributions against existing work: {f"compare with sources [{', '.join([str(i) for i in range(1, min(len(validation.get('formatted_sources', validation.get('relevant_urls', []))), 6) + 1)])}]" if validation.get('formatted_sources') or validation.get('relevant_urls') else "general comparison"}
- Publication and dissemination strategy positioning against existing literature
- Proper citation and attribution of foundational work discovered during validation
- Future research directions based on gaps identified through web analysis
- Expected impact assessment relative to the current research landscape
- Specific tasks and deliverables for this phase

                ## WEB-INFORMED RESOURCE REQUIREMENTS
                - Computational resources needed (consider approaches found in web search)
                - Datasets and tools required (prioritize those referenced in found URLs)
                - Personnel requirements with expertise in areas identified through research
                - Budget estimates informed by current research approaches
                - Infrastructure needs based on state-of-the-art identified online

                ## WEB-VALIDATED RISK MITIGATION
                - Challenges identified through analysis of existing research attempts
                - Learn from failures/limitations discovered in web search results
                - Alternative approaches based on diverse methodologies found online
                - Timeline flexibility informed by realistic research durations observed
                - Contingency plans based on common obstacles identified in literature

                ## SUCCESS METRICS BENCHMARKED AGAINST CURRENT RESEARCH
                - Success criteria informed by achievements in existing work
                - Metrics comparing progress against state-of-the-art found through web search
                - Publication targets considering current publication landscape
                - Impact measurement relative to existing research influence

                ## EXPECTED OUTCOMES & CONTRIBUTIONS
                - Contributions positioned relative to current research landscape
                - Expected papers building upon and citing discovered relevant work
                - Potential real-world applications validated through market research
                - Future research enablement informed by current research directions
                - Clear differentiation from existing approaches found through web analysis

            ## REFERENCES & SOURCES
            **Primary Sources Identified During Web Validation:**"""
            
            if validation.get('formatted_sources'):
                sources_text = "The following sources were identified during web search validation and should be prioritized in literature review:\n" + "\n".join(validation.get('formatted_sources', []))
            elif validation.get('relevant_urls'):
                url_list = [f"[{i}] {url}" for i, url in enumerate(validation.get('relevant_urls', [])[:10], 1)]
                sources_text = "Sources found (URLs only):\n" + "\n".join(url_list)
            else:
                sources_text = "No specific sources identified during validation. Standard literature search recommended."
                
            content += f"""
            {sources_text}

            **Search Queries Used for Source Discovery:**"""
            
            if validation.get('search_queries'):
                query_list = ', '.join([f'"{q}"' for q in validation.get('search_queries', [])])
                search_queries_text = f"Search strategies that identified these sources: {query_list}"
            else:
                search_queries_text = "No search queries recorded."
                
            content += f"""
            {search_queries_text}

            **Source Utilization Instructions:**
            - Use the numbered references [1], [2], [3], etc. throughout your research plan
            - Prioritize these sources in your initial literature review
            - Follow citation networks from these foundational sources
            - Contact authors/institutions identified in these sources for potential collaboration

                **Note:** These sources represent the current state of research as discovered through web validation. They provide immediate starting points for literature review and should be supplemented with systematic database searches.

                **RESEARCH FOCUS:** The selected problem shows:
                - Web search validation with {validation.get('search_results_count', 0)} relevant results found
                - {validation.get('status', 'unknown')} status indicating research opportunities
                - Key resources available for immediate literature review via discovered URLs
                - Current research gaps that can be systematically addressed

                Remember: This plan leverages real-time web search validation to ensure relevance, avoid duplication, and build upon existing work. Each phase should incorporate insights from the web search findings, and the URLs discovered should serve as immediate action items for literature review and collaboration outreach.

                **CITATION STRATEGY:** 
                - Reference discovered sources using standard academic format
                - Track additional sources found through citation networks
                - Maintain proper attribution to foundational work identified during validation
                - Use source numbering [1], [2], etc. for easy reference throughout the plan

                Provide a detailed, focused research plan that maximizes impact on this specific validated research problem.
            """

            response = _client.chat.completions.create(
                model=_model,
                messages=[{"content": content, "role": "user"}]
            )
            
            # Clean the response to avoid encoding issues
            research_plan = _clean_text_for_encoding(response.choices[0].message.content)
            
            # Store previous plan if this is a refinement
            if is_refinement:
                current_plan = state.get("research_plan", {})
                critique = state.get("critique_results", {})
                major_issues = critique.get("major_issues", [])
                
                if current_plan and "previous_plans" not in state:
                    state["previous_plans"] = []
                if current_plan:
                    state["previous_plans"].append(current_plan)
                
                print(f"✅ Research plan refined (iteration {state['refinement_count']})")
                print(f"🎯 TARGETED IMPROVEMENTS - Addressed {len(major_issues)} major critique issues:")
                for i, issue in enumerate(major_issues, 1):
                    print(f"   {i}. {issue}")
                print(f"📊 Previous score: {critique.get('overall_score', 0):.1f}/10 - expecting significant improvement")
            else:
                print("✅ Initial research plan generated")
                print(f"📊 Based on selected problem: {selected_problem.get('statement', 'N/A')[:100]}...")
            
            print("\n" + "=" * 80)
            print("📋 COMPREHENSIVE RESEARCH PLAN")
            print("=" * 80)
            print(research_plan)
            print("=" * 80)
            
            state["research_plan"] = {
                "research_plan_successful": True,
                "research_plan": research_plan,
                "model_used": _model,
                "tokens_used": response.usage.total_tokens if response.usage else "unknown",
                "selected_problem": selected_problem,
                "refinement_iteration": state.get("refinement_count", 0),
                "is_refinement": is_refinement,
                "all_validated_problems": state.get("validated_problems", [])
            }
            
            # Add success message
            state["messages"].append(
                AIMessage(content=f"Successfully generated comprehensive research plan for selected problem: {selected_problem.get('statement', 'N/A')[:100]}...")
            )
        
        except Exception as e:
            error_msg = f"Research plan generation failed: {str(e)}"
            state["errors"].append(error_msg)
            state["research_plan"] = {
                "research_plan_successful": False,
                "error": error_msg,
                "research_plan": None,
                "problems_attempted": len(state.get("validated_problems", []))
            }
            print(f"❌ {error_msg}")
        
        return state

    # --- PHASE 3: PLAN CRITIQUE & ITERATIVE REFINEMENT ---

def _critique_plan_node(state: ResearchPlanningState) -> ResearchPlanningState:
        """Node for critiquing the generated research plan."""
        print(f"\n🔍 Step: Critiquing research plan...")
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

            response = _client.chat.completions.create(
                model=_model,
                temperature=0.1,  # Low temperature for consistent critique
                messages=[{"content": critique_content, "role": "user"}]
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
                
                print(f"\n🔍 CRITIQUE STORED FOR REFINEMENT:")
                print(f"   ✅ Critique results stored in state['critique_results']")
                print(f"   ✅ Latest critique stored with timestamp")
                print(f"   ✅ Ready for refinement iteration {state.get('refinement_count', 0) + 1}")
                
                # Track score history
                overall_score = critique_data.get("overall_score", 0.0)
                state["critique_score_history"].append(overall_score)
                
                # Enhanced critique logging
                llm_recommendation = critique_data.get("recommendation", "unknown")
                
                print(f"\n📊 CRITIQUE RESULTS:")
                print(f"   Score: {overall_score:.1f}/10.0")
                print(f"   Major Issues Count: {len(major_issues)}")
                print(f"   LLM Recommendation: {llm_recommendation.upper()}")
                print(f"   Raw Issues List: {major_issues[:2] if major_issues else 'None'}")
                
                if major_issues:
                    print(f"\n⚠️  MAJOR ISSUES TO ADDRESS:")
                    for i, issue in enumerate(major_issues, 1):
                        print(f"   {i}. {issue}")
                
                suggestions = critique_data.get("suggestions", [])
                if suggestions:
                    print(f"\n💡 IMPROVEMENT SUGGESTIONS:")
                    for i, suggestion in enumerate(suggestions[:3], 1):
                        print(f"   {i}. {suggestion}")
                
                strengths = critique_data.get("strengths", [])
                if strengths:
                    print(f"\n✅ IDENTIFIED STRENGTHS:")
                    for i, strength in enumerate(strengths[:2], 1):
                        print(f"   {i}. {strength}")
                
                # Clear decision summary
                if len(major_issues) == 0:
                    print(f"\n🎉 EXCELLENT! No major issues found - plan ready for finalization!")
                elif len(major_issues) <= 2:
                    print(f"\n🔧 REFINEMENT NEEDED: {len(major_issues)} issues to address")
                elif len(major_issues) <= 4:
                    print(f"\n⚠️  SIGNIFICANT ISSUES: {len(major_issues)} problems need attention")
                else:
                    print(f"\n❌ MAJOR PROBLEMS: {len(major_issues)} fundamental issues detected")
                
                state["messages"].append(
                    AIMessage(content=f"Research plan critiqued. Score: {overall_score:.1f}/10, Issues: {len(major_issues)}, Recommendation: {critique_data.get('recommendation', 'unknown')}")
                )
                
            except json.JSONDecodeError as e:
                error_msg = f"Failed to parse critique JSON: {e}"
                print(f"⚠️  {error_msg}")
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
            print(f"❌ {error_msg}")
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

def _finalize_plan_node(state: ResearchPlanningState) -> ResearchPlanningState:
        """Node for finalizing the research plan and preparing outputs."""
        print(f"\n✅ Step 6: Finalizing research plan and preparing deliverables...")
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
            
            print(f"✅ Research plan finalized!")
            print(f"📊 Final Score: {critique.get('overall_score', 0.0):.1f}/10")
            print(f"🔄 Total Iterations: {state.get('iteration_count', 0)}")
            print(f"🎯 Refinements: {state.get('refinement_count', 0)}")
            
            # Add finalization message
            state["messages"].append(
                AIMessage(content=f"Research plan finalized with score {critique.get('overall_score', 0.0):.1f}/10 after {state.get('refinement_count', 0)} refinements")
            )
            
        except Exception as e:
            error_msg = f"Plan finalization failed: {str(e)}"
            state["errors"].append(error_msg)
            print(f"❌ {error_msg}")
        
        return state

    # ==================================================================================
    # WORKFLOW CONTROL & ROUTING FUNCTIONS
    # ==================================================================================

def _streamlined_validation_decision(state: ResearchPlanningState) -> str:
        """🚀 STREAMLINED DECISION: Route based on validation results for single-problem workflow."""
        validation_results = state.get("validation_results", {})
        current_problem = state.get("current_problem", {})
        
        # Check if we have validation results
        if not validation_results:
            print("🔄 No validation results yet - continue to validation")
            return "validate"
        
        recommendation = validation_results.get("recommendation", "reject")
        current_iteration = state.get("iteration_count", 0)
        
        print(f"🚦 Validation Decision Point:")
        print(f"   Recommendation: {recommendation}")
        print(f"   Current Iteration: {current_iteration}")
        
        # Decision logic for streamlined workflow
        if recommendation == "accept":
            # Problem is validated - set as selected problem and proceed to planning
            state["selected_problem"] = current_problem
            state["validated_problems"] = [current_problem]  # Store in list for compatibility
            print("✅ Problem accepted - proceeding to research plan creation")
            return "create_plan"
        elif recommendation == "reject":
            # Problem was rejected - process feedback and try again
            print("❌ Problem rejected - processing feedback for improvement")
            return "process_feedback"
        else:
            # Unknown state - default to regeneration
            print("🔄 Unknown validation state - regenerating problem")
            return "retry_generation"

def _determine_refinement_path(state: ResearchPlanningState) -> str:
        """Determine whether to refine the plan or finalize it based on critique."""
        critique = state.get("critique_results", {})
        refinement_count = state.get("refinement_count", 0)
        
        if not critique:
            print("🔄 No critique available - proceeding to finalization")
            return "finalize"
        
        overall_score = critique.get("overall_score", 5.0)
        recommendation = critique.get("recommendation", "accept")
        major_issues = critique.get("major_issues", [])
        
        print(f"🚦 Refinement Decision Point:")
        print(f"   Score: {overall_score:.1f}/10")
        print(f"   Recommendation: {recommendation}")
        print(f"   Major Issues: {len(major_issues)}")
        print(f"   Refinement Count: {refinement_count}")
        
        # Decision logic
        if recommendation == "accept" or overall_score >= 7.0 or refinement_count >= 3:
            print("✅ Plan quality sufficient - proceeding to finalization")
            return "finalize_plan"
        elif recommendation == "refine" and refinement_count < 3:
            print("🔄 Plan needs refinement - creating improved version")
            return "refine_plan"
        else:
            print("✅ Maximum refinements reached - finalizing current plan")
            return "finalize_plan"

    # ==================================================================================
    # UTILITY & HELPER FUNCTIONS
    # ==================================================================================

def _display_research_plan_terminal(state: ResearchPlanningState) -> str:
        """Generate and display a comprehensive research plan in the terminal."""
        try:
            from datetime import datetime
            import os
            
            # Extract plan data
            research_plan = state.get("research_plan", {})
            plan_text = research_plan.get("research_plan", "No plan generated")
            selected_problem = research_plan.get("selected_problem", {})
            
            # Create text content
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            content = f"""
RESEARCH PLAN DOCUMENT
Generated: {timestamp}

SELECTED PROBLEM:
{selected_problem.get('statement', 'N/A')}

RESEARCH PLAN:
{plan_text}

METADATA:
- Total Iterations: {state.get('iteration_count', 0)}
- Refinements: {state.get('refinement_count', 0)}
- Final Score: {state.get('critique_results', {}).get('overall_score', 'N/A')}
"""
            
            # Print the research plan to terminal instead of saving to file
            print("\n" + "="*80)
            print("📋 GENERATED RESEARCH PLAN")
            print("="*80)
            print(content)
            print("="*80)
            print("📋 END OF RESEARCH PLAN")
            print("="*80)
            
            print(f"✅ Research plan displayed in terminal")
            print(f"📊 Plan statistics:")
            print(f"   - Total length: {len(content)} characters")
            print(f"   - Estimated pages: {len(content) // 3000:.1f}")
            print(f"   - Iterations: {state.get('iteration_count', 0)}")
            print(f"   - Refinements: {state.get('refinement_count', 0)}")
            
            return "terminal_display"  # Return indicator instead of file path
            
        except Exception as e:
            print(f"❌ Failed to display research plan: {str(e)}")
            return None

async def plan_research(prompt: str, uploaded_data: List[str] = None) -> Dict[str, Any]:
        """Main entry point for research planning workflow."""
        try:
            # Initialize clients for standalone execution
            _initialize_clients()
            
            print(f"\n🚀 Starting Research Planning Workflow...")
            print(f"📝 Research Domain: {prompt}")
            
            # Initialize state
            initial_state: ResearchPlanningState = {
                "messages": [HumanMessage(content=prompt)],
                "original_prompt": prompt,
                "uploaded_data": uploaded_data or [],
                "current_step": "initialize",
                "errors": [],
                "workflow_type": "research_planning",
                "generated_problems": [],
                "validated_problems": [],
                "current_problem": {},
                "validation_results": {},
                "selected_problem": {},
                "research_plan": {},
                "iteration_count": 0,
                "critique_results": {},
                "critique_score_history": [],
                "refinement_count": 0,
                "previous_plans": [],
                "generation_attempts": 0,
                "rejection_feedback": [],
                "auto_validation_enabled": True,
                "web_sources": [],
                "current_web_search_query": ""
            }
            
            # Build and run the workflow
            workflow = build_research_planning_graph()
            result = await workflow.ainvoke(initial_state)
            
            # Generate document if successful
            if result.get("research_plan"):
                display_status = _display_research_plan_terminal(result)
                if display_status:
                    result["display_method"] = "terminal_output"
            
            return result
            
        except Exception as e:
            print(f"❌ Research planning workflow failed: {str(e)}")
            return {
                "error": str(e),
                "workflow_type": "research_planning",
                "original_prompt": prompt
            }


# ==================================================================================
# WORKFLOW BUILDING
# ==================================================================================

def build_research_planning_graph() -> StateGraph:
    """🚀 STREAMLINED WORKFLOW: Generate one problem, validate with Tavily, create research plan automatically.
    
    Workflow Steps:
    1. Generate a single research problem statement
    2. Validate with Tavily web search (check if solved/novel)
    3. If valid -> auto-select and proceed to research plan
    4. If invalid -> process feedback and retry generation
    5. Generate comprehensive research plan automatically  
    6. Critique plan and refine if needed
    7. Finalize plan
    
    Key Benefits:
    - No manual problem selection step
    - Single problem focus for efficiency
    - Automatic progression to research planning
    - Tavily validation ensures novelty
    """
    workflow = StateGraph(ResearchPlanningState)
    
    # Simplified workflow: generate -> validate -> plan (with refinement support)
    workflow.add_node("generate_problem", _generate_problem_node)
    workflow.add_node("validate_problem", _validate_problem_node)
    workflow.add_node("process_rejection_feedback", _process_rejection_feedback_node)  # For rejected problems
    workflow.add_node("create_research_plan", _create_research_plan_node)
    workflow.add_node("critique_plan", _critique_plan_node)
    workflow.add_node("finalize_plan", _finalize_plan_node)
    
    # Entry point: generate a single problem
    workflow.set_entry_point("generate_problem")
    workflow.add_edge("generate_problem", "validate_problem")
    
    # After validation: either proceed to research plan or retry with feedback
    workflow.add_conditional_edges(
        "validate_problem",
        _streamlined_validation_decision,
        {
            "create_plan": "create_research_plan",           # Problem validated - proceed directly to plan
            "process_feedback": "process_rejection_feedback", # Problem rejected - get feedback
            "retry_generation": "generate_problem"           # Retry generation with feedback
        }
    )
    
    # After feedback processing, try generating a new problem
    workflow.add_edge("process_rejection_feedback", "generate_problem")
    
    # After plan creation, critique it
    workflow.add_edge("create_research_plan", "critique_plan")
    
    # After critique, decide what to do based on quality
    workflow.add_conditional_edges(
        "critique_plan",
        _determine_refinement_path,
        {
            "finalize_plan": "finalize_plan",         # No major issues - finalize
            "refine_plan": "create_research_plan",    # Has issues - regenerate with critique context
            "retry_problem": "generate_problem"       # Fundamental issues - try new problem
        }
    )
    
    workflow.add_edge("finalize_plan", END)
    
    return workflow.compile()


# ==================================================================================
# EXPORTS
# ==================================================================================

# Export the state class and all node functions for import by main file
__all__ = [
    'ResearchPlanningState',
    '_generate_problem_node',
    '_validate_problem_node', 
    '_process_rejection_feedback_node',
    '_create_research_plan_node',
    '_critique_plan_node',
    '_finalize_plan_node',
    '_streamlined_validation_decision',
    '_determine_refinement_path',
    '_display_research_plan_terminal',
    'plan_research',
    'build_research_planning_graph',
    '_clean_text_for_encoding',
    '_load_from_env_file'
]


async def main():
    """Main entry point for standalone execution."""
    import argparse
    
    print("🔧 Research Planning Workflow - Standalone Runner")
    print("=" * 60)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Research Planning Workflow - Generate research plans automatically",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python research_planning_nodes.py "Improving machine learning model interpretability"
  python research_planning_nodes.py --help
        """
    )
    
    parser.add_argument(
        'prompt',
        nargs='?',
        help='Research prompt/domain to plan research for'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    if not args.prompt:
        print("❌ Error: Please provide a research prompt")
        print("Usage: python research_planning_nodes.py \"Your research topic\"")
        return
    
    print(f"🚀 Starting research planning for: {args.prompt}")
    print("-" * 60)
    
    try:
        # Run the research planning workflow
        result = await plan_research(args.prompt)
        
        if result.get("error"):
            print(f"❌ Workflow failed: {result['error']}")
            return
        
        # Display results
        if result.get("research_plan"):
            print("✅ Research plan generated successfully!")
            plan = result["research_plan"]
            print(f"📋 Title: {plan.get('title', 'N/A')}")
            print(f"📊 Sections: {len(plan.get('sections', []))}")
            
            if result.get("display_method") == "terminal_output":
                print("📄 Research plan displayed above in terminal")
        else:
            print("⚠️  No research plan was generated")
            
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        if args.verbose:
            traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
