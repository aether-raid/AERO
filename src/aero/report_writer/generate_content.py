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

from .critique_and_refinement import _parse_complete_paper_into_sections

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

async def _generate_content_node(state: PaperWritingState) -> PaperWritingState:
    client = state.get("client")
    model = state.get("model")

    """Node for generating the entire paper content in a single comprehensive LLM call."""
    
    _write_stream("🚀 Starting paper content generation")
    
    # Check if this is a refinement iteration
    refinement_count = state.get("refinement_count", 0)
    critique_results = state.get("critique_results", {})
    
    if refinement_count > 0:
        _write_stream(f"🔄 Refining paper based on critique (iteration {refinement_count})")
        major_issues = critique_results.get("major_issues", [])
        suggestions = critique_results.get("suggestions", [])
    else:
        _write_stream("✍️ Generating complete paper with citations")

    try:
        # Validate client is available
        if not state.get("client"):
            _write_stream("❌ OpenAI client not available")
            return {**state, "errors": state.get("errors", []) + ["Content generation error: OpenAI client unavailable"]}

        _write_stream("📋 Preparing paper structure and context")
        
        research_analysis = state.get("research_analysis", {})
        paper_structure = state.get("paper_structure", {})
        experimental_results = state.get("experimental_results", {})
        uploaded_data = state.get("uploaded_data", [])
        original_prompt = state.get("original_prompt", "")

        # Get Tavily-sourced citations and supporting sources
        supporting_sources = state.get("supporting_sources", [])
        citation_database = state.get("citation_database", {})
        source_validation = state.get("source_validation_results", {})

        _write_stream("📚 Processing citations and sources")

        sections = paper_structure.get("sections", [])

        # Prepare comprehensive citations context for the entire paper
        citations_context = ""
        if supporting_sources:
            _write_stream(f"🔗 Integrating {len(supporting_sources)} citations")
            citations_context = "\n\n📚 COMPREHENSIVE SOURCE DATABASE FOR CITATION:\n"
            citations_context += "=" * 60 + "\n"
            for i, source in enumerate(supporting_sources[:15], 1):  # Use up to 15 sources
                citations_context += f"\n[{i}] {source.get('title', 'Unknown Title')}\n"
                citations_context += f"    URL: {source.get('url', 'No URL')}\n"
                citations_context += f"    Content: {source.get('content', '')}...\n"
                citations_context += f"    Relevance: {source.get('purpose', 'general')}\n"
                citations_context += f"    Domain: {source.get('url', '').split('//')[1].split('/')[0] if '//' in source.get('url', '') else 'Unknown'}\n"
                citations_context += "-" * 40 + "\n"

            citations_context += f"""
📝 CITATION INTEGRATION INSTRUCTIONS:
- Use [1], [2], [3], etc. format for in-text citations
- Integrate citations naturally throughout ALL sections
- Support claims, methods, comparisons, and background with appropriate sources
- Aim for 2-4 citations per major section
- Use diverse sources across different aspects of the research
- Prioritize academic and research sources for credibility
- Ensure citations are contextually relevant to the content they support
"""

        # Prepare detailed paper structure for comprehensive generation
        structure_context = "\n\n📋 DETAILED PAPER STRUCTURE TO FOLLOW:\n"
        structure_context += "=" * 60 + "\n"
        for i, section in enumerate(sections, 1):
            section_name = section.get("name", "Unknown")
            section_focus = section.get("focus", "general")
            section_length = section.get("length", "1 page")
            section_description = section.get("description", "No description")
            
            structure_context += f"\n{i}. {section_name.upper()}\n"
            structure_context += f"   Focus: {section_focus}\n"
            structure_context += f"   Length: {section_length}\n"
            structure_context += f"   Content: {section_description}\n"
            structure_context += "-" * 40 + "\n"

        # Build refinement context if this is a revision
        refinement_context = ""
        if refinement_count > 0 and critique_results:
            major_issues = critique_results.get("major_issues", [])
            specific_improvements = critique_results.get("specific_improvements", [])
            critical_analysis = critique_results.get("critical_analysis", {})
            technical_feedback = critique_results.get("technical_feedback", {})
            writing_improvements = critique_results.get("writing_improvements", {})
            
            refinement_context = f"""
🔄 COMPREHENSIVE REFINEMENT REQUIREMENTS (Iteration {refinement_count}):
{"=" * 80}

🎯 **PREVIOUS CRITIQUE SUMMARY:**
- Overall Score: {critique_results.get('overall_score', 0):.1f}/10
- Recommendation: {critique_results.get('recommendation', 'N/A')}
- Revision Priority: {critique_results.get('revision_priority', 'medium')}

🚨 **CRITICAL ISSUES TO ADDRESS THROUGHOUT THE PAPER:**
{chr(10).join([f"❌ {issue}" for issue in major_issues[:6]])}

🎯 **SPECIFIC IMPROVEMENTS TO IMPLEMENT:**
{chr(10).join([f"✅ {improvement}" for improvement in specific_improvements[:8]])}

🔬 **TECHNICAL IMPROVEMENTS NEEDED:**
"""
            methodology_gaps = technical_feedback.get("methodology_gaps", [])
            experimental_weaknesses = technical_feedback.get("experimental_weaknesses", [])
            for gap in methodology_gaps[:3]:
                refinement_context += f"   🔧 {gap}\n"
            for weakness in experimental_weaknesses[:3]:
                refinement_context += f"   ⚡ {weakness}\n"

            clarity_issues = writing_improvements.get("clarity_issues", [])
            if clarity_issues:
                refinement_context += f"\n✍️ WRITING CLARITY IMPROVEMENTS:\n"
                refinement_context += chr(10).join([f"   📝 {issue}" for issue in clarity_issues[:4]])

            refinement_context += f"""

🎯 **PRIORITY ACTION ITEMS FOR COMPLETE PAPER REVISION:**
1. Address ALL critical issues listed above throughout the paper
2. Implement comprehensive improvements in every section
3. Maintain academic rigor and proper citation integration
4. Ensure logical flow and coherent narrative throughout
5. Write clearly and concisely for the target venue
6. Significantly improve quality from previous iteration

**CRITICAL**: This is iteration {refinement_count} - the entire paper must show substantial improvements addressing ALL critique points.
"""

        # Prepare uploaded data context
        data_context = ""
        if uploaded_data:
            data_context = "\n\n📊 UPLOADED DATA AND FILES TO INTEGRATE:\n"
            data_context += "=" * 60 + "\n"
            for i, data in enumerate(uploaded_data[:5], 1):  # Limit to first 5 files for context
                data_preview = data[:1000] + "..." if len(data) > 1000 else data
                data_context += f"\nFile {i}:\n{data_preview}\n"
                data_context += "-" * 40 + "\n"
            if len(uploaded_data) > 5:
                data_context += f"\n... and {len(uploaded_data) - 5} additional files to consider\n"

        # Create the comprehensive prompt for the entire paper
        comprehensive_prompt = f"""
Write a complete academic research paper based on the following requirements. Generate the ENTIRE paper in a single response with all sections flowing coherently together.

🎯 **RESEARCH TOPIC:** {original_prompt}

📊 **RESEARCH ANALYSIS:** {research_analysis}

🧪 **EXPERIMENTAL RESULTS:** {experimental_results}

{data_context}

{structure_context}

{citations_context}

{refinement_context}

📝 **COMPREHENSIVE PAPER GENERATION INSTRUCTIONS:**

1. **OVERALL STRUCTURE:** Generate a complete academic paper following the exact structure provided above
2. **COHERENT FLOW:** Ensure smooth transitions between sections and consistent narrative throughout
3. **CITATION INTEGRATION:** Integrate citations naturally throughout ALL sections using the provided sources
4. **ACADEMIC RIGOR:** Use formal academic tone, proper methodology, and rigorous analysis
5. **COMPREHENSIVE COVERAGE:** Address all aspects of the research topic thoroughly
6. **LOGICAL PROGRESSION:** Build arguments logically from introduction through conclusions
7. **CONSISTENCY:** Maintain consistent terminology, style, and quality across all sections
8. **PUBLICATION-READY:** Generate content suitable for academic publication

📋 **FORMATTING REQUIREMENTS:**
- Use clear section headers (# for main sections, ## for subsections)
- Include in-text citations in [1], [2], etc. format
- Maintain academic writing style throughout
- Ensure proper paragraph structure and transitions
- Include specific details from experimental results and uploaded data
- Make each section substantive and well-developed

🎯 **CRITICAL SUCCESS FACTORS:**
- The paper must read as a coherent, unified work, not separate sections
- All claims must be properly supported with citations
- The research must be presented clearly and convincingly
- The paper must demonstrate academic rigor and originality
{f"- Address ALL critique points comprehensively throughout the paper" if refinement_count > 0 else ""}

Generate the complete academic research paper now:
        """

        _write_stream("🤖 Generating complete paper with AI")

        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert academic writer specializing in comprehensive research papers with integrated citations. Generate complete, publication-ready papers with excellent flow and coherence."},
                    {"role": "user", "content": comprehensive_prompt}
                ],
                temperature=0.2,  # Lower temperature for more consistent academic writing
                max_tokens=16000  # Increased token limit for comprehensive generation
            )
        )

        _write_stream("📝 Processing generated paper content")
        complete_paper = response.choices[0].message.content.strip()
        
        _write_stream("🔍 Parsing paper into sections")
        # Parse the complete paper into sections for compatibility with existing workflow
        section_content = _parse_complete_paper_into_sections(complete_paper, sections)
        
        _write_stream("📚 Generating reference list")
        # Generate reference list from all used sources
        reference_list = _generate_reference_list(supporting_sources)
        if reference_list and "References" not in section_content:
            section_content["References"] = reference_list

        # Count citations used in the complete paper
        total_citations_used = sum(1 for i in range(1, len(supporting_sources) + 1) if f"[{i}]" in complete_paper)

        _write_stream(f"✅ Paper generation complete - {len(complete_paper)} chars, {total_citations_used} citations")

        return {
            **state,
            "section_content": section_content,
            "formatted_paper": complete_paper,  # Store the complete paper as well
            "current_step": "content_generated"
        }

    except Exception as e:
        return {
            **state,
            "errors": state.get("errors", []) + [f"Content generation error: {str(e)}"],
            "current_step": "content_generation_error"
        }
    else:
        pass

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

        # Generate sections SEQUENTIALLY with context from previous sections
        for section_idx, section in enumerate(sections):
            section_name = section.get("name", "Unknown")
            section_focus = section.get("focus", "general")
            section_length = section.get("length", "1 page")

            # Build context from all previously written sections
            previous_sections_context = ""
            if section_content:
                previous_sections_context = "\n**PREVIOUSLY WRITTEN SECTIONS (for context and continuity):**\n"
                for prev_section, prev_content in section_content.items():
                    # Truncate very long sections for context
                    content_preview = prev_content[:800] + "..." if len(prev_content) > 800 else prev_content
                    previous_sections_context += f"\n### {prev_section}:\n{content_preview}\n"
                
                previous_sections_context += "\n**IMPORTANT CONTINUITY REQUIREMENTS:**\n"
                previous_sections_context += "- Maintain consistent terminology with previous sections\n"
                previous_sections_context += "- Build logically on concepts already introduced\n"
                previous_sections_context += "- Avoid repetition of content already covered\n"
                previous_sections_context += "- Ensure smooth transitions and narrative flow\n"
                previous_sections_context += "- Reference previous sections when appropriate (e.g., 'As mentioned in the Introduction...')\n"
                previous_sections_context += "- Use consistent formatting and style\n\n"

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

{previous_sections_context}

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
- **CRITICAL**: Maintain consistency with previously written sections
- **CRITICAL**: Build logically on content already established
- **CRITICAL**: Avoid repeating information from previous sections
- **CRITICAL**: Use smooth transitions that connect to previous sections
{f"- PRIORITY: Address critique feedback and improve quality from previous iteration" if refinement_count > 0 else ""}

Write a complete {section_name} section with integrated citations that flows naturally from the previous sections:
            """

            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: state["client"].chat.completions.create(
                    model=state["model"],
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

        # Generate reference list from all used sources
        reference_list = _generate_reference_list(supporting_sources)
        if reference_list:
            section_content["References"] = reference_list

        total_sources_used = len([s for s in supporting_sources if any(
            f"[{i}]" in content for i, content in enumerate(section_content.values(), 1)
        )])

        return {
            **state,
            "section_content": section_content,
            "current_step": "content_generated"
        }

    except Exception as e:
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