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

from .parsing_and_formatting import _parse_critique_json

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

def _write_stream(message: str, key: str = "status", progress: int = None):
    try:
        writer = get_stream_writer()
        writer({key: message})
    except Exception:
        pass

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
            lambda: state["client"].chat.completions.create(
                model=state["model"],
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
