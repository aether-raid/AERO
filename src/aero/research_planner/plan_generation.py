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

# Helper function imports
from .rejection_processing import _clean_text_for_encoding

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

async def _create_research_plan_node(state: ResearchPlanningState, *, config: Optional[Dict[str, Any]] = None) -> ResearchPlanningState:
        """Node for creating comprehensive research plan based on the selected problem."""
        
        # Validate clients are available in state
        client = state.get("client")
        if not client:
            state["errors"] = state.get("errors", []) + ["OpenAI client not found in state. Please initialize clients."]
            return state
            
        model = state.get("model", "gpt-4o-mini")

        selected_problem = state.get("selected_problem", {})
        
        # Check if this is a refinement iteration
        is_refinement = state.get("critique_results") is not None and state.get("refinement_count", 0) > 0
        
        if is_refinement:
            _write_stream(f"Refining research plan (iteration {state.get('refinement_count', 0) + 1})", "info", progress=60)
            # Increment refinement count
            state["refinement_count"] = state.get("refinement_count", 0) + 1
            _write_stream("Addressing critique feedback to improve plan quality")
            
            # Verify critique data is available for refinement
            critique = state.get("critique_results", {})
            if not critique:
                _write_stream("No critique feedback available - proceeding with standard refinement")
                _write_stream("This may indicate an issue with the workflow state management")
            else:
                major_issues = critique.get("major_issues", [])
                score = critique.get("overall_score", 0)
                _write_stream(f"Refining research plan (improving from {score:.1f}/10)", progress=70)
        else:
            _write_stream("Creating comprehensive research plan", progress=50)
            _write_stream("Developing detailed methodology and timeline")
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
                    _write_stream(f"Using current problem as basis: {current_problem.get('statement', 'N/A')[:80]}{'...' if len(current_problem.get('statement', '')) > 80 else ''}", "info")
                else:
                    error_msg = "No valid problem selected for research plan generation"
                    state["errors"].append(error_msg)
                    _write_stream(f"Problem statement error: {error_msg}")
                    _write_stream(f"Debug - selected_problem: {selected_problem}")
                    _write_stream(f"Debug - current_problem: {current_problem}")
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

            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: client.chat.completions.create(
                    model=model,
                    messages=[{"content": content, "role": "user"}]
                )
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
                
                _write_stream(f"Research plan refined successfully (iteration {state['refinement_count']})", progress=65)
                _write_stream(f"Addressed {len(major_issues)} key issues from critique:")
                for i, issue in enumerate(major_issues, 1):
                    _write_stream(f"  {i}. {issue}")
                _write_stream(f"Previous quality score: {critique.get('overall_score', 0):.1f}/10 - expecting improvement", "info")
            else:
                _write_stream("Initial research plan generated successfully", progress=55)
                _write_stream(f"Based on problem: {selected_problem.get('statement', 'N/A')[:100]}{'...' if len(selected_problem.get('statement', '')) > 100 else ''}", "info")
            
            _write_stream("Generated Research Plan:", progress=60)
            _write_stream(research_plan)
            
            state["research_plan"] = {
                "research_plan_successful": True,
                "research_plan": research_plan,
                "model_used": model,
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
            _write_stream(f"Research plan generation failed: {error_msg}")
        
        return state
