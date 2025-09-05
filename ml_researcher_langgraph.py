#!/usr/bin/env python3
"""
Machine Learning Researcher Tool - LangGraph Compatible Version
==============================================================

A comprehensive tool that uses LangGraph to orchestrate:
1. Task decomposition using LLM via LiteLLM
2. Property extraction from user prompts
3. ArXiv paper search and analysis
4. Model recommendation
5. Open research problem identification
6. Comprehensive research plan generation

This version leverages LangGraph for state management and workflow orchestration.

Usage:
    python ml_researcher_langgraph.py
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

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

# LLM and related imports
import openai

# Local imports
from Report_to_txt import extract_pdf_text
from arxiv import format_search_string


@dataclass
class Evidence:
    snippet: str
    source: str
    score: float


@dataclass
class PropertyHit:
    name: str
    evidence: List[Evidence]
    
    @property
    def confidence(self) -> float:
        """Calculate confidence based on evidence."""
        if not self.evidence:
            return 0.0
        
        # Calculate base confidence using independent signals
        prod = 1.0
        for ev in self.evidence:
            prod *= (1.0 - max(0.0, min(1.0, ev.score)))
        base_confidence = 1.0 - prod
        
        # Apply evidence count bonus with diminishing returns
        evidence_bonus = min(0.05 * math.log(len(self.evidence) + 1), 0.15)
        
        final_confidence = min(1.0, base_confidence + evidence_bonus)
        return round(final_confidence, 3)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "name": self.name,
            "confidence": self.confidence,
            "evidence": [asdict(ev) for ev in self.evidence],
        }


# ML Research Categories for LLM Analysis
ML_RESEARCH_CATEGORIES = {
    "variable_length_sequences": "Data consists of sequences of varying lengths (e.g., text, sensor streams, speech).",
    "fixed_channel_count": "Inputs have a fixed number of channels or features across all samples (e.g., EEG signals, RGB images).",
    "temporal_structure": "Data has inherent time dependencies or ordering that models must capture (e.g., time series forecasting).",
    "reconstruction_objective": "Task requires reconstructing input signals from compressed or corrupted representations (e.g., autoencoders).",
    "latent_embedding_required": "Learning meaningful latent representations is central to the approach (e.g., VAEs, contrastive learning).",
    "shape_preserving_seq2seq": "Output sequences must preserve key structural properties of the input (e.g., translation, speech-to-speech).",
    "classification_objective": "Task involves predicting discrete labels from data (e.g., sentiment analysis, image classification).",
    "regression_objective": "Task involves predicting continuous values (e.g., stock prices, energy consumption).",
    "generation_objective": "Models must produce new data samples from learned distributions (e.g., text generation, image synthesis).",
    "noise_robustness": "System must perform well under noisy, incomplete, or corrupted inputs (e.g., real-world sensor data).",
    "real_time_constraint": "Solution must operate under strict latency or streaming requirements (e.g., real-time detection).",
    "invariance_requirements": "Predictions must remain stable under transformations (e.g., translation, scaling, rotation, time shifts).",
    "sensor_data": "Inputs originate from physical sensors (e.g., IoT, biomedical devices, accelerometers).",
    "multimodal_data": "Task combines multiple data types or modalities (e.g., vision + language, audio + text).",
    "interpretability_required": "Model must provide human-understandable reasoning or explanations (e.g., clinical AI, finance).",
    "high_accuracy_required": "Performance must meet strict accuracy thresholds due to critical application domains (e.g., medical diagnostics).",
    "few_shot_learning": "System must generalize from very few labeled examples (e.g., low-resource languages, rare diseases).",
    "model_selection_query": "Research focuses on choosing or suggesting the most appropriate model for given properties.",
    "text_data": "Inputs are natural language text (e.g., documents, transcripts, chat logs).",
    "multilingual_requirement": "Task involves handling multiple languages or cross-lingual transfer.",
    "variable_document_length": "Document inputs vary significantly in length (e.g., short tweets vs. long research papers)."
}


# LangGraph State Definitions
class BaseState(TypedDict):
    """Base state for all workflows."""
    messages: Annotated[List[BaseMessage], add_messages]
    original_prompt: str
    current_step: str
    errors: List[str]
    workflow_type: str  # "model_suggestion" or "research_planning"

class ModelSuggestionState(BaseState):
    """State object for the model suggestion workflow."""
    detected_categories: List[Dict[str, Any]]
    detailed_analysis: Dict[str, Any]
    arxiv_search_query: str
    arxiv_results: Dict[str, Any]
    model_suggestions: Dict[str, Any]

class ResearchPlanningState(BaseState):
    """State object for the research planning workflow."""
    generated_problems: List[Dict[str, Any]]     # All generated problem statements
    validated_problems: List[Dict[str, Any]]     # Problems verified as unsolved
    current_problem: Dict[str, Any]              # Currently being validated
    validation_results: Dict[str, Any]           # Web search and validation results
    research_plan: Dict[str, Any]                # Final research plan
    iteration_count: int                         # Track number of iterations

class RouterState(TypedDict):
    """State object for the router agent."""
    messages: Annotated[List[BaseMessage], add_messages]
    original_prompt: str
    routing_decision: str  # "model_suggestion" or "research_planning"
    routing_confidence: float
    routing_reasoning: str
    errors: List[str]


class MLResearcherLangGraph:
    """LangGraph-based ML Research Tool with Multi-Workflow Architecture."""
    
    def __init__(self):
        """Initialize the tool with LiteLLM configuration."""
        # Load configuration from env.example file
        self.api_key = self._load_from_env_file("OPENAI_API_KEY")
        self.base_url = self._load_from_env_file("BASE_URL") or "https://agents.aetherraid.dev"
        self.model = self._load_from_env_file("DEFAULT_MODEL") or "gemini/gemini-2.5-flash"
        
        if not self.api_key:
            raise ValueError("API key not found. Check env.example file or set OPENAI_API_KEY environment variable.")
        
        # Initialize OpenAI client with LiteLLM proxy
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        # Build the three workflows
        self.router_graph = self._build_router_graph()
        self.model_suggestion_graph = self._build_model_suggestion_graph()
        self.research_planning_graph = self._build_research_planning_graph()
    
    def _load_from_env_file(self, key: str) -> Optional[str]:
        """Load configuration value from env.example file."""
        try:
            with open('env.example', 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith(f'{key}='):
                        value = line.split('=', 1)[1]
                        # Remove quotes if present
                        if value.startswith(("'", '"')) and value.endswith(("'", '"')):
                            value = value[1:-1]
                        if value and value not in ['your-api-key-here', 'your-base-url-here']:
                            return value
        except Exception:
            pass
        return None
    
    def _build_router_graph(self) -> StateGraph:
        """Build the router workflow to decide which main workflow to use."""
        workflow = StateGraph(RouterState)
        
        # Add router node
        workflow.add_node("route_request", self._route_request_node)
        
        # Simple linear flow
        workflow.set_entry_point("route_request")
        workflow.add_edge("route_request", END)
        
        return workflow.compile()
    
    def _build_model_suggestion_graph(self) -> StateGraph:
        """Build the model suggestion workflow."""
        workflow = StateGraph(ModelSuggestionState)
        
        # Add nodes for model suggestion pipeline
        workflow.add_node("analyze_properties_and_task", self._analyze_properties_and_task_node)
        workflow.add_node("generate_search_query", self._generate_search_query_node)
        workflow.add_node("search_arxiv", self._search_arxiv_node)
        workflow.add_node("suggest_models", self._suggest_models_node)
        
        # Define the flow
        workflow.set_entry_point("analyze_properties_and_task")
        workflow.add_edge("analyze_properties_and_task", "generate_search_query")
        workflow.add_edge("generate_search_query", "search_arxiv")
        workflow.add_edge("search_arxiv", "suggest_models")
        workflow.add_edge("suggest_models", END)
        
        return workflow.compile()
    
    def _build_research_planning_graph(self) -> StateGraph:
        """Build the iterative research planning workflow."""
        workflow = StateGraph(ResearchPlanningState)
        
        # Add nodes for iterative research planning pipeline
        workflow.add_node("generate_problem", self._generate_problem_node)
        workflow.add_node("validate_problem", self._validate_problem_node)
        workflow.add_node("collect_problem", self._collect_problem_node)
        workflow.add_node("create_research_plan", self._create_research_plan_node)
        
        # Define the flow with conditional edges
        workflow.set_entry_point("generate_problem")
        workflow.add_edge("generate_problem", "validate_problem")
        
        # After validation, decide if we should collect the problem
        workflow.add_conditional_edges(
            "validate_problem",
            self._check_completion,
            {
                "collect_problem": "collect_problem",     # Collect if validation passed
                "continue_generation": "generate_problem"  # Skip and generate new if validation failed
            }
        )
        
        # After collecting, decide if we should continue generating or finalize
        workflow.add_conditional_edges(
            "collect_problem", 
            self._should_continue_generating,
            {
                "generate_problem": "generate_problem",    # Generate more if need more problems
                "finalize_plan": "create_research_plan"    # Create plan if enough problems collected
            }
        )
        
        workflow.add_edge("create_research_plan", END)
        
        return workflow.compile()
    
    async def _route_request_node(self, state: RouterState) -> RouterState:
        """Router node to decide which workflow to use based on user prompt."""
        print("\n🤖 Router: Analyzing user request to determine workflow...")
        
        try:
            content = f"""
                You are an expert AI system router. Analyze the user's request and determine which workflow is most appropriate.

                User Request: "{state["original_prompt"]}"

                Available Workflows:
                1. **MODEL_SUGGESTION**: For requests asking about:
                   - "What model should I use for X?"
                   - "Recommend architectures for Y task"
                   - "Best approaches for Z problem"
                   - "Which algorithm is suitable for..."
                   - Model comparison and selection
                   - Architecture recommendations
                   - Technical implementation guidance

                2. **RESEARCH_PLANNING**: For requests asking about:
                   - "What are open problems in X domain?"
                   - "Research opportunities in Y field"
                   - "Future directions for Z"
                   - "Create a research plan for..."
                   - "What should I research in X area?"
                   - Research gap identification
                   - Academic research planning

                Analyze the user's request and respond with a JSON object containing:
                {{
                    "workflow": "MODEL_SUGGESTION" or "RESEARCH_PLANNING",
                    "confidence": 0.0-1.0,
                    "reasoning": "Brief explanation of why this workflow was chosen"
                }}

                Consider the intent and focus of the request. If the user wants practical implementation advice, choose MODEL_SUGGESTION. If they want to understand research gaps and plan academic research, choose RESEARCH_PLANNING.

                Return only the JSON object, no additional text.
            """

            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model=self.model,
                    temperature=0,
                    messages=[{"content": content, "role": "user"}]
                )
            )
            
            # Parse the router decision
            router_response = response.choices[0].message.content.strip()
            
            # Try to extract JSON from the response
            try:
                # Remove any markdown formatting
                if router_response.startswith("```json"):
                    router_response = router_response[7:]
                if router_response.endswith("```"):
                    router_response = router_response[:-3]
                router_response = router_response.strip()
                
                decision_data = json.loads(router_response)
                
                workflow_decision = decision_data.get("workflow", "MODEL_SUGGESTION")
                confidence = decision_data.get("confidence", 0.5)
                reasoning = decision_data.get("reasoning", "Default routing decision")
                
                # Normalize workflow decision
                if workflow_decision.upper() in ["MODEL_SUGGESTION", "MODEL_SUGGESTIONS"]:
                    workflow_decision = "model_suggestion"
                elif workflow_decision.upper() in ["RESEARCH_PLANNING", "RESEARCH_PLAN"]:
                    workflow_decision = "research_planning"
                else:
                    workflow_decision = "model_suggestion"  # Default fallback
                
                state["routing_decision"] = workflow_decision
                state["routing_confidence"] = confidence
                state["routing_reasoning"] = reasoning
                
                print(f"🎯 Router Decision: {workflow_decision.upper()}")
                print(f"📊 Confidence: {confidence:.2f}")
                print(f"💭 Reasoning: {reasoning}")
                
                # Add success message
                state["messages"].append(
                    AIMessage(content=f"Routed to {workflow_decision.upper()} workflow (confidence: {confidence:.2f})")
                )
                
            except json.JSONDecodeError as e:
                error_msg = f"Failed to parse router JSON response: {e}"
                state["errors"].append(error_msg)
                state["routing_decision"] = "model_suggestion"  # Default fallback
                state["routing_confidence"] = 0.5
                state["routing_reasoning"] = "Default due to parsing error"
                print(f"⚠️  {error_msg}, using default routing")
        
        except Exception as e:
            error_msg = f"Router failed: {str(e)}"
            state["errors"].append(error_msg)
            state["routing_decision"] = "model_suggestion"  # Default fallback
            state["routing_confidence"] = 0.5
            state["routing_reasoning"] = "Default due to router error"
            print(f"❌ {error_msg}, using default routing")
        
        return state
    
    async def _analyze_properties_and_task_node(self, state: ModelSuggestionState) -> ModelSuggestionState:
        """Combined node for extracting properties and decomposing task concurrently."""
        print("\n🤖 Step 1: Analyzing properties and decomposing task concurrently...")
        state["current_step"] = "analyze_properties_and_task"
        
        async def extract_properties():
            """Extract properties using LLM analysis."""
            try:
                categories_list = "\n".join([f"- {category}" for category in ML_RESEARCH_CATEGORIES])
                
                content = f"""
                    You are an expert machine learning researcher. Analyze the following research task and determine which of the predefined categories apply.

                    Research Task: {state["original_prompt"]}

                    Categories to analyze:
                    {categories_list}

                    For each category that applies to this research task, provide:
                    1. The category name (exactly as listed above)
                    2. A confidence score between 0.0 and 1.0 (how certain you are this category applies, Refer to the calibration table)
                    3. A brief explanation of why this category applies
                    4. Specific evidence from the task description that supports this categorization

                    Confidence calibration (0.0–1.0):
                    - 0.95–1.00: Category is explicitly stated or entailed by multiple strong cues.
                    - 0.80–0.94: Strong single cue or multiple moderate cues; unlikely to be wrong.
                    - 0.60–0.79: Reasonable inference with at least one clear cue; some uncertainty.
                    - <0.60: Category is highly unlikely to apply, and can be safely ignored.

                    Explanations:
                    - 1–2 sentences, specific and non-generic, referencing how the evidence meets the category's definition.
                    - Avoid restating the evidence verbatim; interpret it.

                    Evidence rules:
                    - "evidence" must be short verbatim quotes or near-verbatim spans from the task (≤ 20 words each). If paraphrase is unavoidable, mark with ~ at start (e.g., "~streaming data implies temporal order").
                    - Provide 1–3 evidence snippets per category, concatenated with " | " if multiple.
                    - No invented facts; no external knowledge.

                    Do not filter categories down to only the applicable ones, you want to always return the full set, but include a confidence score for each (so the tool/user can judge relevance).

                    Format your response as a JSON array like this:
                    [
                    {{
                        "category": "temporal_structure",
                        "confidence": 0.95,
                        "explanation": "The task explicitly mentions time series data which has temporal dependencies",
                        "evidence": "time series forecasting"
                    }},
                    {{
                        "category": "variable_length_sequences", 
                        "confidence": 0.85,
                        "explanation": "Task mentions variable length sequences",
                        "evidence": "variable length sequences"
                    }}
                    ]
                    Always return valid JSON. For any field that may contain multiple values (e.g., evidence), output them as a JSON array of strings instead of separating by commas inside a single string.

                    Return only the JSON array, no additional text.
                """

                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"content": content, "role": "user"}]
                    )
                )
                
                # Parse the LLM response
                llm_response = response.choices[0].message.content.strip()
                
                # Try to extract JSON from the response
                try:
                    # Remove any markdown formatting
                    if llm_response.startswith("```json"):
                        llm_response = llm_response[7:]
                    if llm_response.endswith("```"):
                        llm_response = llm_response[:-3]
                    llm_response = llm_response.strip()
                    
                    properties_data = json.loads(llm_response)
                    
                    # Convert to PropertyHit objects and then to dict
                    property_hits = []
                    for prop_data in properties_data:
                        evidence = [Evidence(
                            snippet=prop_data.get("evidence", ""),
                            source=f"llm_analysis:{prop_data['category']}",
                            score=prop_data.get("confidence", 0.5)
                        )]
                        
                        property_hit = PropertyHit(
                            name=prop_data["category"],
                            evidence=evidence
                        )
                        property_hits.append(property_hit.to_dict())
                    
                    print(f"✅ Property extraction completed: Found {len(property_hits)} properties")
                    return {"success": True, "properties": property_hits}
                    
                except json.JSONDecodeError as e:
                    error_msg = f"Failed to parse LLM JSON response: {e}"
                    print(f"⚠️  {error_msg}")
                    return {"success": False, "error": error_msg, "properties": []}
            
            except Exception as e:
                error_msg = f"LLM property extraction failed: {str(e)}"
                print(f"❌ {error_msg}")
                return {"success": False, "error": error_msg, "properties": []}

        async def decompose_task():
            """Decompose task using LLM analysis."""
            try:
                content = f"""
                    You are an expert machine learning researcher. Analyze the following research task and decompose it into key properties and characteristics.

                    Task: {state["original_prompt"]}

                    Please identify and analyze the following aspects:

                    1. **Data Type**: What kind of data is involved? (text, images, time series, tabular, etc.)
                    2. **Learning Type**: What type of learning is this? (supervised, unsupervised, reinforcement, etc.)
                    3. **Task Category**: What is the main ML task? (classification, regression, generation, clustering, etc.)
                    4. **Architecture Requirements**: What types of models or architectures might be suitable?
                    5. **Key Challenges**: What are the main technical challenges?
                    6. **Data Characteristics**: 
                    - Variable length sequences?
                    - Fixed or variable input dimensions?
                    - Temporal structure?
                    - Multi-modal data?
                    7. **Performance Metrics**: What metrics would be appropriate for evaluation?
                    8. **Domain Specifics**: Any domain-specific considerations?

                    Provide your analysis in a structured JSON format with clear explanations for each identified property.
                """

                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"content": content, "role": "user"}]
                    )
                )
                
                detailed_analysis = {
                    "llm_analysis": response.choices[0].message.content,
                    "model_used": self.model,
                    "tokens_used": response.usage.total_tokens if response.usage else "unknown"
                }
                
                print("✅ Task decomposition completed")
                return {"success": True, "analysis": detailed_analysis}
            
            except Exception as e:
                error_msg = f"LLM decomposition failed: {str(e)}"
                print(f"❌ {error_msg}")
                return {"success": False, "error": error_msg, "analysis": {"error": error_msg, "llm_analysis": None}}

        # Run both tasks concurrently
        print("🔄 Running property extraction and task decomposition in parallel...")
        properties_result, decomposition_result = await asyncio.gather(
            extract_properties(),
            decompose_task(),
            return_exceptions=True
        )
        
        # Handle results
        if isinstance(properties_result, Exception):
            error_msg = f"Property extraction failed: {str(properties_result)}"
            state["errors"].append(error_msg)
            state["detected_categories"] = []
            print(f"❌ {error_msg}")
        elif properties_result["success"]:
            state["detected_categories"] = properties_result["properties"]
            for prop in properties_result["properties"]:
                print(f"  - {prop['name']}: {prop['confidence']:.2f} confidence")
        else:
            state["errors"].append(properties_result["error"])
            state["detected_categories"] = properties_result["properties"]
        
        if isinstance(decomposition_result, Exception):
            error_msg = f"Task decomposition failed: {str(decomposition_result)}"
            state["errors"].append(error_msg)
            state["detailed_analysis"] = {"error": error_msg, "llm_analysis": None}
            print(f"❌ {error_msg}")
        elif decomposition_result["success"]:
            state["detailed_analysis"] = decomposition_result["analysis"]
        else:
            state["errors"].append(decomposition_result["error"])
            state["detailed_analysis"] = decomposition_result["analysis"]
        
        # Add success messages
        if properties_result.get("success") and decomposition_result.get("success"):
            state["messages"].append(
                AIMessage(content=f"Successfully analyzed task properties ({len(properties_result['properties'])} categories) and decomposed task characteristics concurrently.")
            )
        
        return state

    async def _extract_properties_node(self, state: ModelSuggestionState) -> ModelSuggestionState:
        """Node for extracting properties using LLM analysis."""
        print("\n🤖 Step 1: Extracting properties using LLM analysis...")
        state["current_step"] = "extract_properties"
        
        try:
            categories_list = "\n".join([f"- {category}" for category in ML_RESEARCH_CATEGORIES])
            
            content = f"""
                You are an expert machine learning researcher. Analyze the following research task and determine which of the predefined categories apply.

                Research Task: {state["original_prompt"]}

                Categories to analyze:
                {categories_list}

                For each category that applies to this research task, provide:
                1. The category name (exactly as listed above)
                2. A confidence score between 0.0 and 1.0 (how certain you are this category applies, Refer to the calibration table)
                3. A brief explanation of why this category applies
                4. Specific evidence from the task description that supports this categorization

                Confidence calibration (0.0–1.0):
                - 0.95–1.00: Category is explicitly stated or entailed by multiple strong cues.
                - 0.80–0.94: Strong single cue or multiple moderate cues; unlikely to be wrong.
                - 0.60–0.79: Reasonable inference with at least one clear cue; some uncertainty.
                - <0.60: Category is highly unlikely to apply, and can be safely ignored.

                Explanations:
                - 1–2 sentences, specific and non-generic, referencing how the evidence meets the category's definition.
                - Avoid restating the evidence verbatim; interpret it.

                Evidence rules:
                - "evidence" must be short verbatim quotes or near-verbatim spans from the task (≤ 20 words each). If paraphrase is unavoidable, mark with ~ at start (e.g., "~streaming data implies temporal order").
                - Provide 1–3 evidence snippets per category, concatenated with " | " if multiple.
                - No invented facts; no external knowledge.

                Do not filter categories down to only the applicable ones, you want to always return the full set, but include a confidence score for each (so the tool/user can judge relevance).

                Format your response as a JSON array like this:
                [
                {{
                    "category": "temporal_structure",
                    "confidence": 0.95,
                    "explanation": "The task explicitly mentions time series data which has temporal dependencies",
                    "evidence": "time series forecasting"
                }},
                {{
                    "category": "variable_length_sequences", 
                    "confidence": 0.85,
                    "explanation": "Task mentions variable length sequences",
                    "evidence": "variable length sequences"
                }}
                ]
                Always return valid JSON. For any field that may contain multiple values (e.g., evidence), output them as a JSON array of strings instead of separating by commas inside a single string.

                Return only the JSON array, no additional text.
            """

            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"content": content, "role": "user"}]
                )
            )
            
            # Parse the LLM response
            llm_response = response.choices[0].message.content.strip()
            
            # Try to extract JSON from the response
            try:
                # Remove any markdown formatting
                if llm_response.startswith("```json"):
                    llm_response = llm_response[7:]
                if llm_response.endswith("```"):
                    llm_response = llm_response[:-3]
                llm_response = llm_response.strip()
                
                properties_data = json.loads(llm_response)
                
                # Convert to PropertyHit objects and then to dict
                property_hits = []
                for prop_data in properties_data:
                    evidence = [Evidence(
                        snippet=prop_data.get("evidence", ""),
                        source=f"llm_analysis:{prop_data['category']}",
                        score=prop_data.get("confidence", 0.5)
                    )]
                    
                    property_hit = PropertyHit(
                        name=prop_data["category"],
                        evidence=evidence
                    )
                    property_hits.append(property_hit.to_dict())
                
                state["detected_categories"] = property_hits
                
                print(f"✅ Property extraction completed: Found {len(property_hits)} properties")
                for prop in property_hits:
                    print(f"  - {prop['name']}: {prop['confidence']:.2f} confidence")
                
                # Add success message
                state["messages"].append(
                    AIMessage(content=f"Successfully extracted {len(property_hits)} ML categories from the research task.")
                )
                
            except json.JSONDecodeError as e:
                error_msg = f"Failed to parse LLM JSON response: {e}"
                state["errors"].append(error_msg)
                state["detected_categories"] = []
                print(f"⚠️  {error_msg}")
        
        except Exception as e:
            error_msg = f"LLM property extraction failed: {str(e)}"
            state["errors"].append(error_msg)
            state["detected_categories"] = []
            print(f"❌ {error_msg}")
        
        return state
    
    async def _decompose_task_node(self, state: ModelSuggestionState) -> ModelSuggestionState:
        """Node for task decomposition using LLM."""
        print("\n🤖 Step 2: Decomposing task using LLM...")
        state["current_step"] = "decompose_task"
        
        try:
            content = f"""
                You are an expert machine learning researcher. Analyze the following research task and decompose it into key properties and characteristics.

                Task: {state["original_prompt"]}

                Please identify and analyze the following aspects:

                1. **Data Type**: What kind of data is involved? (text, images, time series, tabular, etc.)
                2. **Learning Type**: What type of learning is this? (supervised, unsupervised, reinforcement, etc.)
                3. **Task Category**: What is the main ML task? (classification, regression, generation, clustering, etc.)
                4. **Architecture Requirements**: What types of models or architectures might be suitable?
                5. **Key Challenges**: What are the main technical challenges?
                6. **Data Characteristics**: 
                - Variable length sequences?
                - Fixed or variable input dimensions?
                - Temporal structure?
                - Multi-modal data?
                7. **Performance Metrics**: What metrics would be appropriate for evaluation?
                8. **Domain Specifics**: Any domain-specific considerations?

                Provide your analysis in a structured JSON format with clear explanations for each identified property.
            """

            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"content": content, "role": "user"}]
                )
            )
            
            state["detailed_analysis"] = {
                "llm_analysis": response.choices[0].message.content,
                "model_used": self.model,
                "tokens_used": response.usage.total_tokens if response.usage else "unknown"
            }
            
            print("✅ Task decomposition completed")
            
            # Add success message
            state["messages"].append(
                AIMessage(content="Successfully decomposed the research task into key properties and characteristics.")
            )
        
        except Exception as e:
            error_msg = f"LLM decomposition failed: {str(e)}"
            state["errors"].append(error_msg)
            state["detailed_analysis"] = {
                "error": error_msg,
                "llm_analysis": None
            }
            print(f"❌ {error_msg}")
        
        return state
    
    def _generate_search_query_node(self, state: ModelSuggestionState) -> ModelSuggestionState:
        """Node for generating arXiv search query."""
        print("\n📚 Step 2: Generating arXiv search query...")
        state["current_step"] = "generate_search_query"
        
        try:
            # Extract key properties with high confidence
            high_confidence_props = [prop for prop in state["detected_categories"] if prop.get("confidence", 0) > 0.7]
            prop_names = [prop["name"] for prop in high_confidence_props]
            
            content = f"""
                Based on the following machine learning research task analysis, generate ONE concise arXiv API search query (exactly 4 terms, separated by forward slashes).

                Original Task: {state["original_prompt"]}

                Detected Categories: {', '.join(prop_names)}

                Detailed Analysis: {state["detailed_analysis"].get('llm_analysis', 'Not available')}

                Rules for constructing the query:
                - EXACTLY 4 terms, separated by "/" (no quotes, no extra spaces).
                - Include:
                1) a MODEL keyword (e.g., transformer, ViT, DETR, RT-DETR, Deformable DETR, YOLOS),
                2) the TASK (e.g., object detection, segmentation),
                3) a DEPLOYMENT/CONSTRAINT or TOOLING term if present (e.g., real-time, edge deployment, TensorRT, quantization, INT8).
                4) a DOMAIN or APPLICATION term if relevant (e.g., medical imaging, remote sensing, autonomous vehicles).
                - Prefer task-specific + model-specific terms over generic ones.
                - Avoid vague terms like "deep learning" or "machine learning" unless nothing better fits.
                - Prefer dataset/benchmark anchors (e.g., KITTI, nuScenes, Waymo) OVER broad domain words (e.g., autonomous vehicles). Use the domain ONLY if it is essential and not overly broad.
                - If computer vision is relevant, make the TASK a CV term (e.g., object detection, instance segmentation).
                - Do NOT include arXiv category labels (cs.CV, cs.LG) in the query terms.
                - Return ONLY the query string (no explanation, no punctuation besides "/").

                Good examples:
                - transformer/object detection/real-time/autonomous vehicles
                - RT-DETR/object detection/TensorRT/KITTI
                - Deformable DETR/object detection/KITTI/autonomous driving
                - vision transformer/object detection/edge deployment/medical imaging
            """

            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                messages=[{"content": content, "role": "user"}]
            )
            
            search_query = response.choices[0].message.content.strip()
            state["arxiv_search_query"] = search_query
            
            print(f"Generated search query: '{search_query}'")
            
            # Add success message
            state["messages"].append(
                AIMessage(content=f"Generated arXiv search query: '{search_query}'")
            )
        
        except Exception as e:
            # Fallback to simple keyword extraction with slashes
            keywords = []
            prompt = state["original_prompt"].lower()
            if "neural" in prompt or "deep" in prompt:
                keywords.append("neural network")
            if "time series" in prompt or "temporal" in prompt:
                keywords.append("time series")
            if "classification" in prompt:
                keywords.append("classification")
            if "clustering" in prompt:
                keywords.append("clustering")
            if "anomaly detection" in prompt:
                keywords.append("anomaly detection")
            if "autoencoder" in prompt:
                keywords.append("autoencoder")
            
            search_query = "/".join(keywords) if keywords else "machine learning"
            state["arxiv_search_query"] = search_query
            
            error_msg = f"Search query generation failed, using fallback: {str(e)}"
            state["errors"].append(error_msg)
            print(f"⚠️  {error_msg}")
        
        return state
    
    async def _search_arxiv_node(self, state: ModelSuggestionState) -> ModelSuggestionState:
        """Node for searching arXiv papers using optimized 3-stage workflow."""
        print(f"\n📖 Step 3: Searching arXiv for relevant papers...")
        state["current_step"] = "search_arxiv"
        
        try:
            search_query = state["arxiv_search_query"]
            original_prompt = state["original_prompt"]
            max_results = 15  # Get more papers initially for better selection
            
            print(f"🔍 SEARCHING arXiv: {search_query}")
            print("=" * 80)
            
            # Format the search query
            formatted_query = format_search_string(search_query)
            print(f"Formatted query: {formatted_query}")
            
            # Build the URL
            url = f"http://export.arxiv.org/api/query?search_query={formatted_query}&start=0&max_results={max_results}"
            print(f"🌐 Full URL: {url}")
            
            with libreq.urlopen(url) as response:
                xml_data = response.read()
            
            # Debug: Check XML content
            xml_str = xml_data.decode('utf-8')
            entry_count = xml_str.count('<entry>')
            print(f"🔍 Debug: Found {entry_count} <entry> elements in XML response")
            
            # Parse XML
            root = ET.fromstring(xml_data)
            
            # Namespaces
            ns = {
                'atom': 'http://www.w3.org/2005/Atom',
                'arxiv': 'http://arxiv.org/schemas/atom',
                'opensearch': 'http://a9.com/-/spec/opensearch/1.1/'
            }
            
            # Get total results
            total_results_elem = root.find('opensearch:totalResults', ns)
            total_results = int(total_results_elem.text) if total_results_elem is not None else 0
            
            print(f"Total papers found: {total_results}")
            
            if total_results > 0:
                print("=" * 80)
                
                # Get all paper entries
                entries = root.findall('atom:entry', ns)
                print(f"🔍 Debug: XML parsing found {len(entries)} entries using namespace search")
                
                # Alternative debugging - try without namespace  
                entries_no_ns = root.findall('.//entry')
                print(f"🔍 Debug: Found {len(entries_no_ns)} entries without namespace")
                
                # If no entries found with namespace, try alternative approach
                if len(entries) == 0 and len(entries_no_ns) > 0:
                    print("⚠️ Using entries found without namespace")
                    entries = entries_no_ns
                
                # If we got very few results compared to total, try a simpler query
                if len(entries) < 5 and total_results > 1000:
                    print(f"⚠️ Only found {len(entries)} entries despite {total_results} total results")
                    print("🔄 Attempting fallback with simpler query...")
                    
                    # Try a simpler query by removing the most specific terms
                    query_parts = search_query.split('/')
                    if len(query_parts) > 2:
                        # Keep only the first two most important terms
                        fallback_query = '/'.join(query_parts[:2])
                        formatted_fallback = format_search_string(fallback_query)
                        fallback_url = f"http://export.arxiv.org/api/query?search_query={formatted_fallback}&start=0&max_results={max_results}"
                        print(f"🔄 Fallback query: {fallback_query}")
                        print(f"🌐 Fallback URL: {fallback_url}")
                        
                        try:
                            with libreq.urlopen(fallback_url) as fallback_response:
                                fallback_xml_data = fallback_response.read()
                            
                            fallback_root = ET.fromstring(fallback_xml_data)
                            fallback_entries = fallback_root.findall('atom:entry', ns)
                            
                            if len(fallback_entries) > len(entries):
                                print(f"✅ Fallback found {len(fallback_entries)} entries - using fallback results")
                                entries = fallback_entries
                                xml_data = fallback_xml_data  # Update for consistency
                                root = fallback_root
                            else:
                                print(f"❌ Fallback only found {len(fallback_entries)} entries - keeping original")
                        except Exception as fallback_error:
                            print(f"❌ Fallback query failed: {fallback_error}")
                
                # Stage 1: Extract basic info (title, abstract, metadata) without downloading PDFs
                print(f"📝 Stage 1: Extracting basic info for {len(entries)} papers...")
                
                # Stage 1: Extract basic info (title, abstract, metadata) without downloading PDFs
                print(f"� Stage 1: Extracting basic info for {len(entries)} papers...")
                papers = []
                for i, entry in enumerate(entries, 1):
                    paper_info = self._extract_basic_paper_info(entry, ns, i)
                    papers.append(paper_info)
                    print(f"✅ Basic info extracted for paper #{i}: {paper_info['title'][:50]}...")
                
                # Stage 2: Rank papers by relevance using title + abstract only
                print(f"\n🎯 Stage 2: Ranking papers by relevance (based on title + abstract)...")
                papers = await self._rank_papers_by_relevance(papers, original_prompt)
                
                # Stage 3: Download full content for top 5 papers only
                top_papers = papers[:5]  # Get top 5 papers
                print(f"\n📥 Stage 3: Downloading full PDF content for top {len(top_papers)} papers...")
                
                with ThreadPoolExecutor(max_workers=3) as executor:  # Limit concurrent downloads
                    # Submit download tasks for top papers only
                    future_to_paper = {
                        executor.submit(self._download_paper_content, paper): paper 
                        for paper in top_papers
                    }
                    
                    # Collect results as they complete
                    for future in as_completed(future_to_paper):
                        updated_paper = future.result()
                        # Update the paper in the original list
                        for i, paper in enumerate(papers):
                            if paper['id'] == updated_paper['id']:
                                papers[i] = updated_paper
                                break
                
                print(f"✅ PDF download stage completed. Top 5 papers now have full content.")
                
                # Print final results (now ranked by relevance)
                print("\n" + "=" * 80)
                print("📋 RANKED RESULTS (by relevance):")
                print("=" * 80)
                
                for i, paper in enumerate(papers, 1):
                    relevance_score = paper.get('relevance_score', 0)
                    has_content = paper.get('pdf_downloaded', False)
                    content_status = "📄 FULL CONTENT" if has_content else "📝 TITLE+ABSTRACT"
                    
                    print(f"\n📄 PAPER #{i} ({content_status}) - Relevance: {relevance_score:.1f}/10.0")
                    print("-" * 60)
                    print(f"Title: {paper['title']}")
                    print(f"ID: {paper['id']}")
                    print(f"Published: {paper['published']}")
                    print(f"URL: {paper['url']}")
                    
                    # Show summary for all papers
                    if paper.get('summary'):
                        print(f"Summary: {paper['summary'][:300]}...")
                    
                    # Show content preview only if downloaded
                    if paper.get('content'):
                        print(f"Full Content Preview:\n{paper['content'][:400]}...")
                    elif not has_content and i <= 5:
                        print("Full Content: [Available in top 5 - check PDF download status]")
                    else:
                        print("Full Content: [Not downloaded - not in top 5]")
                    print("-" * 60)
                
                state["arxiv_results"] = {
                    "search_successful": True,
                    "total_results": str(total_results),
                    "papers_returned": len(papers),
                    "papers": papers,
                    "formatted_query": formatted_query,
                    "original_query": search_query
                }
            else:
                print("No papers found")
                state["arxiv_results"] = {
                    "search_successful": False,
                    "total_results": "0",
                    "papers_returned": 0,
                    "papers": [],
                    "formatted_query": formatted_query,
                    "original_query": search_query
                }
            
            # Add success message
            state["messages"].append(
                AIMessage(content=f"ArXiv search completed. Found {total_results} total papers, processed {len(papers) if total_results > 0 else 0} papers.")
            )
                    
        except Exception as e:
            error_msg = f"Error searching arXiv: {e}"
            state["errors"].append(error_msg)
            state["arxiv_results"] = {
                "search_successful": False,
                "error": error_msg,
                "total_results": "0",
                "papers_returned": 0,
                "papers": [],
                "formatted_query": "",
                "original_query": state["arxiv_search_query"]
            }
            print(f"❌ {error_msg}")
        
        return state
    
    def _suggest_models_node(self, state: ModelSuggestionState) -> ModelSuggestionState:
        """Node for suggesting suitable models based on analysis."""
        print(f"\n🤖 Step 4: Analyzing papers and suggesting suitable models...")
        state["current_step"] = "suggest_models"
        
        try:
            # Prepare evidence from arXiv papers
            papers_evidence = ""
            if state["arxiv_results"].get("search_successful") and state["arxiv_results"].get("papers"):
                papers_evidence = "\n--- arXiv Papers Found ---\n"
                for i, paper in enumerate(state["arxiv_results"]["papers"], 1):
                    papers_evidence += f"""
                        Paper {i}: {paper["title"]}
                        Published: {paper["published"]}
                        URL: {paper["url"]}
                        ---
                    """
            else:
                papers_evidence = "\n--- No arXiv Papers Found ---\nNo relevant papers were found in the search, so recommendations will be based on general ML knowledge.\n"
            
            # Prepare detected categories
            categories_text = ", ".join([prop["name"] for prop in state["detected_categories"]])
            
            # Create comprehensive prompt for model suggestion
            content = f"""
                You are an expert machine learning researcher and architect. Based on the following comprehensive analysis, suggest the most suitable machine learning models/architectures for this task and provide detailed justification.

                ## Original Task
                {state["original_prompt"]}

                ## Detected ML Categories
                {categories_text}

                ## Detailed Analysis Summary
                {state["detailed_analysis"].get('llm_analysis', 'Analysis not available')[:1000]}...

                ## Evidence from Recent Research Papers
                {papers_evidence}

                ## Your Task
                Based on ALL the evidence above (task requirements, detected categories, detailed analysis, and recent research papers), provide:

                1. **Top 3 Recommended Models/Architectures** - List the most suitable models in order of preference
                2. **Detailed Justification** - For each model, explain:
                - Each choice MUST be based in truth from the research evidence
                - Why it's suitable for this specific task
                - How it addresses the detected categories/requirements
                - Evidence from the research papers (if available) that supports this choice
                - Specific advantages and potential limitations
                
                3. **Implementation Considerations** - Practical advice for each model:
                - Key hyperparameters to tune
                - Training considerations
                - Expected performance characteristics
                4. **Alternative Approaches** - Brief mention of other viable options and when they might be preferred

                Format your response as a structured analysis that clearly connects your recommendations to the evidence provided.
                Your response MUST be based on the research evidence presented in the prompt and the arXiv papers.
            """

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"content": content, "role": "user"}]
            )
            
            model_suggestions = response.choices[0].message.content
            
            # Print readable summary
            print("✅ Model suggestions generated")
            print("\n" + "=" * 80)
            print("🎯 RECOMMENDED MODELS AND JUSTIFICATION")
            print("=" * 80)
            print(model_suggestions)
            print("=" * 80)
            
            state["model_suggestions"] = {
                "suggestions_successful": True,
                "model_suggestions": model_suggestions,
                "model_used": self.model,
                "tokens_used": response.usage.total_tokens if response.usage else "unknown",
                "papers_analyzed": len(state["arxiv_results"].get("papers", [])),
                "categories_considered": len(state["detected_categories"])
            }
            
            # Add success message
            state["messages"].append(
                AIMessage(content="Successfully generated model recommendations based on research analysis and arXiv papers.")
            )
        
        except Exception as e:
            error_msg = f"Model suggestion failed: {str(e)}"
            state["errors"].append(error_msg)
            state["model_suggestions"] = {
                "suggestions_successful": False,
                "error": error_msg,
                "model_suggestions": None
            }
            print(f"❌ {error_msg}")
        
        return state
    
    async def _generate_problem_node(self, state: ResearchPlanningState) -> ResearchPlanningState:
        """Node for generating a new research problem statement."""
        current_iter = state.get("iteration_count", 0) + 1
        state["iteration_count"] = current_iter
        
        print(f"\n🎯 Step {current_iter}: Generating research problem statement...")
        state["current_step"] = "generate_problem"
        
        try:
            # Check how many problems we already have
            validated_count = len(state.get("validated_problems", []))
            generated_count = len(state.get("generated_problems", []))
            
            # Create context about previously generated problems to avoid repetition
            previous_problems = ""
            if state.get("generated_problems"):
                previous_problems = "\n\nPreviously generated problems (avoid similar ones):\n"
                for i, prob in enumerate(state["generated_problems"][-5:], 1):  # Show last 5
                    previous_problems += f"{i}. {prob.get('statement', 'Unknown')}\n"
            
            content = f"""
                You are an expert research problem generator. Your task is to generate a SINGLE, specific, novel research problem statement in the given domain.

                Research Domain: {state["original_prompt"]}

                Current Progress: {validated_count}/3 validated open problems found
                Generation attempt: {current_iter}

                {previous_problems}

                Requirements for the problem statement:
                1. **SPECIFIC**: Clearly defined scope and objectives
                2. **NOVEL**: Not obviously solved or well-established
                3. **FEASIBLE**: Can realistically be addressed with current technology
                4. **IMPACTFUL**: Would advance the field if solved
                5. **MEASURABLE**: Success can be quantified or evaluated

                Generate ONE specific research problem statement that:
                - Addresses a concrete gap or limitation in the field
                - Can be formulated as a clear research question
                - Is different from previously generated problems
                - Is narrow enough to be tackled in a research project

                Respond with a JSON object containing:
                {{
                    "statement": "Clear, specific problem statement (1-2 sentences)",
                    "description": "Brief description of why this is important (2-3 sentences)",
                    "keywords": ["key", "terms", "for", "validation", "search"],
                    "research_question": "Specific research question this addresses"
                }}

                Focus on being specific and avoiding overly broad or obviously solved problems.
                Return only the JSON object, no additional text.
            """

            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model=self.model,
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
                problem_data["status"] = "pending_validation"
                
                # Store current problem for validation
                state["current_problem"] = problem_data
                
                # Add to generated problems list
                if "generated_problems" not in state:
                    state["generated_problems"] = []
                state["generated_problems"].append(problem_data.copy())
                
                print(f"✅ Generated problem: {problem_data['statement']}")
                print(f"🔍 Keywords for validation: {', '.join(problem_data.get('keywords', []))}")
                
                # Add success message
                state["messages"].append(
                    AIMessage(content=f"Generated research problem #{current_iter}: {problem_data['statement']}")
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
                "status": "pending_validation"
            }
            print(f"❌ {error_msg}, using fallback problem")
        
        return state
    
    async def _validate_problem_node(self, state: ResearchPlanningState) -> ResearchPlanningState:
        """Node for validating if the generated problem is already solved."""
        print(f"\n🔍 Validating problem: {state['current_problem']['statement'][:60]}...")
        state["current_step"] = "validate_problem"
        
        try:
            current_problem = state["current_problem"]
            keywords = current_problem.get("keywords", [])
            
            # Step 1: Web search simulation using LLM knowledge
            search_content = f"""
                You are an expert research validator. Your task is to determine if a research problem has already been solved or is well-established in the literature.

                Research Problem: {current_problem['statement']}
                Description: {current_problem.get('description', '')}
                Keywords: {', '.join(keywords)}
                Research Question: {current_problem.get('research_question', '')}

                Based on your knowledge of the field, analyze whether this problem:

                1. **Is already solved**: Has been conclusively addressed with established solutions
                2. **Is well-studied**: Has extensive research but may have room for improvements
                3. **Is partially solved**: Has some solutions but significant gaps remain
                4. **Is open/novel**: Has minimal research or represents a genuine gap

                Consider:
                - Existing publications and solutions in this area
                - State-of-the-art methods that address this problem
                - Recent breakthroughs or established techniques
                - Whether this is a known challenge in the field

                Respond with a JSON object:
                {{
                    "status": "solved" | "well_studied" | "partially_solved" | "open",
                    "confidence": 0.0-1.0,
                    "reasoning": "Detailed explanation of your assessment",
                    "existing_solutions": ["list", "of", "known", "solutions", "if", "any"],
                    "research_gaps": ["remaining", "gaps", "if", "any"],
                    "recommendation": "accept" | "reject"
                }}

                Use "reject" for problems that are clearly solved or well-established.
                Use "accept" for problems that represent genuine research opportunities.

                Return only the JSON object, no additional text.
            """

            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model=self.model,
                    temperature=0.3,  # Lower temperature for more consistent validation
                    messages=[{"content": search_content, "role": "user"}]
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
                
                # Store validation results
                state["validation_results"] = validation_data
                
                # Update current problem with validation info
                state["current_problem"]["validation"] = validation_data
                state["current_problem"]["status"] = "validated"
                
                status = validation_data.get("status", "unknown")
                confidence = validation_data.get("confidence", 0.0)
                recommendation = validation_data.get("recommendation", "reject")
                
                print(f"📊 Validation Status: {status.upper()}")
                print(f"🎯 Confidence: {confidence:.2f}")
                print(f"💡 Recommendation: {recommendation.upper()}")
                print(f"🧠 Reasoning: {validation_data.get('reasoning', 'No reasoning provided')[:100]}...")
                
                if recommendation == "accept":
                    print("✅ Problem validated as open research opportunity")
                else:
                    print("❌ Problem rejected - already solved or well-established")
                
                # Add validation message
                state["messages"].append(
                    AIMessage(content=f"Validated problem: {recommendation.upper()} (status: {status}, confidence: {confidence:.2f})")
                )
                
            except json.JSONDecodeError as e:
                error_msg = f"Failed to parse validation JSON response: {e}"
                state["errors"].append(error_msg)
                # Default to rejection on parsing error
                state["validation_results"] = {
                    "status": "unknown",
                    "confidence": 0.0,
                    "reasoning": "Validation parsing failed",
                    "recommendation": "reject"
                }
                state["current_problem"]["validation"] = state["validation_results"]
                print(f"⚠️  {error_msg}, defaulting to rejection")
        
        except Exception as e:
            error_msg = f"Problem validation failed: {str(e)}"
            state["errors"].append(error_msg)
            # Default to rejection on error
            state["validation_results"] = {
                "status": "unknown", 
                "confidence": 0.0,
                "reasoning": "Validation failed due to error",
                "recommendation": "reject"
            }
            state["current_problem"]["validation"] = state["validation_results"]
            print(f"❌ {error_msg}, defaulting to rejection")
        
        return state

    def _create_research_plan_node(self, state: ResearchPlanningState) -> ResearchPlanningState:
        """Node for creating comprehensive research plan based on validated problems."""
        print(f"\n📋 Step 4: Generating comprehensive research plan from {len(state.get('validated_problems', []))} validated problems...")
        state["current_step"] = "create_research_plan"
        
        try:
            # Clean all text inputs to avoid encoding issues
            clean_prompt = self._clean_text_for_encoding(state["original_prompt"])
            validated_problems = state.get("validated_problems", [])
            
            # Format validated problems for the prompt
            problems_text = ""
            for i, problem in enumerate(validated_problems, 1):
                problems_text += f"\n**Problem {i}:**\n"
                problems_text += f"- **Statement:** {problem.get('statement', 'N/A')}\n"
                problems_text += f"- **Description:** {problem.get('description', 'N/A')}\n"
                problems_text += f"- **Research Question:** {problem.get('research_question', 'N/A')}\n"
                problems_text += f"- **Keywords:** {', '.join(problem.get('keywords', []))}\n"
                
                validation = problem.get('validation', {})
                problems_text += f"- **Validation Status:** {validation.get('status', 'unknown')}\n"
                problems_text += f"- **Validation Confidence:** {validation.get('confidence', 0.0):.2f}\n"
                problems_text += f"- **Research Gaps:** {', '.join(validation.get('research_gaps', []))}\n\n"
            
            clean_problems = self._clean_text_for_encoding(problems_text)
            
            content = f"""
You are an expert research project manager and academic research planner. Your task is to create a comprehensive, actionable research plan based on validated open problems that have been systematically identified and verified.

**RESEARCH CONTEXT:**

**Research Domain/Query:** {clean_prompt}

**VALIDATED OPEN PROBLEMS:**
{clean_problems[:4000]}

**YOUR TASK:**
Create a comprehensive research plan that addresses these validated open problems in a structured, time-bound manner. The plan should prioritize problems based on their research potential and feasibility.

**REQUIRED STRUCTURE:**

## EXECUTIVE SUMMARY
- Brief overview of the research objectives
- Summary of the {len(validated_problems)} validated open problems
- Research prioritization strategy
- Expected timeline and outcomes

## PROBLEM PRIORITIZATION
- Rank the validated problems by research potential and feasibility
- Explain the selection criteria used
- Identify which problems will be primary vs. secondary focus

## PHASE 1: FOUNDATION & LITERATURE REVIEW (Months 1-3)
- Comprehensive literature review strategy for each priority problem
- Key papers and research groups to study
- Knowledge gap validation through literature
- Initial research question refinement
- Specific tasks and deliverables

## PHASE 2: PROBLEM FORMULATION (Months 4-6)
- Formalize specific research hypotheses for priority problems
- Design initial experiments or theoretical approaches
- Identify required datasets/tools/resources
- Risk assessment for each chosen problem
- Specific tasks and deliverables

## PHASE 3: CORE RESEARCH (Months 7-18)
- Execute primary research activities for each problem
- Develop novel solutions/approaches
- Conduct experiments and evaluations
- Address validation gaps identified in problem assessment
- Specific tasks and deliverables for each problem

## PHASE 4: VALIDATION & DISSEMINATION (Months 19-24)
- Comprehensive evaluation of proposed solutions
- Comparison with existing state-of-the-art
- Paper writing and publication strategy
- Conference and journal targets
- Specific tasks and deliverables

## RISK MITIGATION
- Potential challenges for each research problem
- Alternative approaches if primary methods fail
- Backup problems to pursue if main ones prove infeasible

## RESOURCE REQUIREMENTS
- Computational resources needed for each problem
- Data requirements and acquisition strategy
- Potential collaborations and partnerships
- Funding considerations

## SUCCESS METRICS
- How to measure progress for each problem
- Key performance indicators
- Publication and impact goals

## LONG-TERM IMPACT
- How solving these problems could influence the field
- Potential for follow-up research
- Broader implications and applications

Provide a detailed, well-structured research plan that leverages the validated open problems to make significant contributions to the field.
"""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"content": content, "role": "user"}]
            )
            
            # Clean the response to avoid encoding issues
            research_plan = self._clean_text_for_encoding(response.choices[0].message.content)
            
            # Print readable summary
            print("✅ Comprehensive research plan generated")
            print(f"📊 Based on {len(validated_problems)} validated problems")
            print("\n" + "=" * 80)
            print("📋 COMPREHENSIVE RESEARCH PLAN")
            print("=" * 80)
            print(research_plan)
            print("=" * 80)
            
            state["research_plan"] = {
                "research_plan_successful": True,
                "research_plan": research_plan,
                "model_used": self.model,
                "tokens_used": response.usage.total_tokens if response.usage else "unknown",
                "problems_used": len(validated_problems),
                "validated_problems": validated_problems
            }
            
            # Add success message
            state["messages"].append(
                AIMessage(content=f"Successfully generated comprehensive research plan based on {len(validated_problems)} validated open problems.")
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

    async def _collect_problem_node(self, state: ResearchPlanningState) -> ResearchPlanningState:
        """Node for collecting validated problems and deciding next steps."""
        print(f"\n📥 Collecting validated problem...")
        state["current_step"] = "collect_problem"
        
        try:
            current_problem = state.get("current_problem", {})
            validation_results = state.get("validation_results", {})
            
            # Check if problem was accepted
            if validation_results.get("recommendation") == "accept":
                # Add to validated problems
                validated_problems = state.get("validated_problems", [])
                validated_problems.append(current_problem)
                state["validated_problems"] = validated_problems
                
                print(f"✅ Problem collected! Total validated problems: {len(validated_problems)}")
                
                # Add success message
                state["messages"].append(
                    AIMessage(content=f"Problem collected successfully. Total validated problems: {len(validated_problems)}")
                )
            else:
                print("❌ Problem rejected, not collecting")
                state["messages"].append(
                    AIMessage(content="Problem rejected due to validation failure")
                )
            
            # Increment iteration count
            state["iteration_count"] = state.get("iteration_count", 0) + 1
            
            # Clear current problem for next iteration
            state["current_problem"] = {}
            state["validation_results"] = {}
            
            print(f"🔄 Iteration {state['iteration_count']} completed")
            
        except Exception as e:
            error_msg = f"Problem collection failed: {str(e)}"
            state["errors"].append(error_msg)
            print(f"❌ {error_msg}")
        
        return state
    
    def _clean_text_for_encoding(self, text: str) -> str:
        """Clean text to avoid UTF-8 encoding issues with surrogates."""
        if not text:
            return ""
        
        # Remove or replace surrogate characters
        cleaned = text.encode('utf-8', errors='ignore').decode('utf-8')
        
        # Additional cleaning for common problematic characters
        replacements = {
            '\u2018': "'",  # Left single quotation mark
            '\u2019': "'",  # Right single quotation mark
            '\u201c': '"',  # Left double quotation mark
            '\u201d': '"',  # Right double quotation mark
            '\u2013': '-',  # En dash
            '\u2014': '--', # Em dash
            '\u2026': '...', # Horizontal ellipsis
        }
        
        for old, new in replacements.items():
            cleaned = cleaned.replace(old, new)
        
        return cleaned
    
    
    
    # ARXIV handling funtions 
    def _extract_basic_paper_info(self, entry, ns, index):
        """Extract basic paper info without downloading PDF content."""
        try:
            # Extract basic info
            title = entry.find('atom:title', ns).text.strip()
            paper_id = entry.find('atom:id', ns).text.split('/')[-1]
            
            # Get published date
            published = entry.find('atom:published', ns).text[:10] if entry.find('atom:published', ns) is not None else "Unknown"
            
            # Get abstract/summary
            summary_elem = entry.find('atom:summary', ns)
            summary = summary_elem.text.strip() if summary_elem is not None else ""
            
            # Get arXiv URL
            arxiv_url = f"http://export.arxiv.org/api/query?id_list={paper_id}"
            
            # Store paper info without content
            paper_info = {
                "title": title,
                "id": paper_id,
                "published": published,
                "summary": summary,
                "content": None,  # Will be filled later for top papers
                "url": arxiv_url,
                "index": index,
                "pdf_downloaded": False
            }
            
            return paper_info
            
        except Exception as e:
            print(f"❌ Error extracting basic info for paper #{index}: {e}")
            return {
                "title": f"Error processing paper #{index}",
                "id": "error",
                "published": "Unknown",
                "summary": "",
                "content": None,
                "url": "error",
                "index": index,
                "pdf_downloaded": False,
                "error": str(e)
            }

    def _download_paper_content(self, paper_info):
        """Download and extract PDF content for a specific paper."""
        import requests
        import feedparser
        
        try:
            paper_id = paper_info['id']
            arxiv_url = f"http://export.arxiv.org/api/query?id_list={paper_id}"
            
            response = requests.get(arxiv_url)
            feed = feedparser.parse(response.text)
            
            if not feed.entries:
                return paper_info
                
            entry_data = feed.entries[0]
            
            # Find PDF link
            pdf_link = None
            for link in entry_data.links:
                if link.type == 'application/pdf':
                    pdf_link = link.href
                    break
            
            # Extract text from PDF
            if pdf_link:
                print(f"Fetching PDF from: {pdf_link}")
                pdf_txt = extract_pdf_text(pdf_link)
                paper_info['content'] = pdf_txt
                paper_info['pdf_downloaded'] = True
                print(f"✅ Downloaded PDF content for: {paper_info['title'][:50]}...")
            else:
                print(f"⚠️ No PDF link found for: {paper_info['title'][:50]}...")
            
            return paper_info
            
        except Exception as e:
            print(f"❌ Error downloading PDF for {paper_info['title'][:50]}...: {e}")
            return paper_info
    
    def _process_single_paper(self, entry, ns, index):
        """Process a single paper entry and extract its content."""
        import requests
        import feedparser
        
        try:
            # Extract basic info
            title = entry.find('atom:title', ns).text.strip()
            paper_id = entry.find('atom:id', ns).text.split('/')[-1]
            summary = entry.find('atom:summary', ns).text.strip()
            
            # Get published date
            published = entry.find('atom:published', ns).text[:10] if entry.find('atom:published', ns) is not None else "Unknown"
            
            # Get arXiv URL
            arxiv_url = f"http://export.arxiv.org/api/query?id_list={paper_id}"
            
            response = requests.get(arxiv_url)
            feed = feedparser.parse(response.text)
            entry_data = feed.entries[0]
            
            # Find PDF link
            pdf_link = None
            for link in entry_data.links:
                if link.type == 'application/pdf':
                    pdf_link = link.href
                    break
            
            # Extract text from PDF
            pdf_txt = extract_pdf_text(pdf_link) if pdf_link else None
            
            # Store paper info
            paper_info = {
                "title": title,
                "id": paper_id,
                "published": published,
                "content": pdf_txt,
                "url": arxiv_url,
                "summary": summary,
                "index": index  # Keep track of original order
            }
            
            # Print progress
            print(f"✅ PAPER #{index} processed: {title[:60]}...")
            
            return paper_info
            
        except Exception as e:
            print(f"❌ Error processing paper #{index}: {e}")
            return {
                "title": f"Error processing paper #{index}",
                "id": "error",
                "published": "Unknown",
                "content": None,
                "url": "error",
                "index": index,
                "error": str(e)
            }
    
    async def _score_paper_relevance(self, paper_title: str, paper_content: str, original_query: str) -> float:
        """LLM relevance score in [1.0, 10.0]. Returns a float only."""
        # Keep prompts lean; truncate huge inputs to control tokens
        MAX_CHARS = 8000
        title = (paper_title or "").strip()[:512] or "<untitled>"
        content = (paper_content or "").strip()[:MAX_CHARS]
        query = (original_query or "").strip()[:2000]

        user_prompt = f"""
You are an expert ML librarian. Score how relevant the paper is to the user's research query on a 1–10 scale.

OUTPUT FORMAT (STRICT):
- Return EXACTLY one floating-point number with ONE decimal (e.g., 8.7).
- No words, no JSON, no units, no symbols, no explanation.
- Single line only (no leading/trailing spaces or extra lines).

Use ONLY the text provided below (title + content). Do not browse or assume unstated results. If the content is partial, rely on what’s given (title/abstract first).

Scoring rubric — assign four subscores in [0,1]:
- task_match (40%): directly addresses the research task(s).
- method_match (30%): overlap with the architectures/approaches in the query or close variants.
- constraint_match (20%): aligns with constraints/tooling/datasets/hardware (e.g., real-time, FPS/latency, edge/mobile, TensorRT/ONNX, INT8/FP16).
- evidence_match (10%): concrete signals (benchmarks like nuScenes/Waymo/KITTI, metrics, ablations, deployment notes).

Compute:
- Let t,m,c,e ∈ [0,1].
- score = round((0.40*t + 0.30*m + 0.20*c + 0.10*e) * 10, 1).
- If clearly unrelated (all four < 0.15), output 1.0.
- Clip to [1.0, 10.0].
- Be conservative if uncertain.

Research query:
\"\"\"{query}\"\"\"

Paper title:
\"\"\"{title}\"\"\"

Paper content:
\"\"\"{content}\"\"\"
""".strip()


        async def _call_llm(prompt: str) -> str:
            resp = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model=self.model,
                    temperature=0,
                    messages=[
                        {"role": "system", "content": "You are a strict numeric scorer. Reply with ONLY a number between 1.0 and 10.0."},
                        {"role": "user", "content": prompt},
                    ],
                )
            )
            return (resp.choices[0].message.content or "").strip()

        def _to_score(txt: str) -> float:
            # Pull the first numeric token; tolerate minor deviations
            m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", txt)
            if not m:
                return 1.0
            val = float(m.group())
            # clip to [1.0, 10.0]
            if not math.isfinite(val):
                return 1.0
            return max(1.0, min(10.0, val))

        # Retries with backoff for transient failures
        backoff = 0.6
        for attempt in range(3):
            try:
                raw = await _call_llm(user_prompt)
                score = _to_score(raw)
                return score
            except Exception:
                if attempt < 2:
                    await asyncio.sleep(backoff)
                    backoff *= 2
                else:
                    return 1.0

    async def _rank_papers_by_relevance(self, papers: List[Dict], original_query: str) -> List[Dict]:
        """Score and rank papers by relevance using cosine similarity (fast, deterministic)."""
        print("\n🎯 Scoring papers for relevance using cosine similarity...")
        
        # Create scoring tasks for all papers using the new cosine similarity method
        async def score_paper(i, paper):
            print(f"⏳ Scoring paper {i}/{len(papers)}: {paper['title'][:50]}...")
            
            # Use the new cosine similarity scoring method
            relevance_score = await self._score_paper_relevance(
                paper['title'], 
                paper.get('summary', ''),  # Use summary instead of content for initial ranking
                original_query
            )
            
            paper['relevance_score'] = relevance_score
            print(f"   📊 Cosine Similarity Score: {relevance_score:.1f}/10.0")
            return paper
        
        # Run all scoring tasks concurrently
        scoring_tasks = [score_paper(i, paper) for i, paper in enumerate(papers, 1)]
        scored_papers = await asyncio.gather(*scoring_tasks)
        
        # Sort by relevance score (highest first) and return top 5
        ranked_papers = sorted(scored_papers, key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        print(f"\n✅ Papers ranked by cosine similarity to: '{original_query}'")
        return ranked_papers[:5]  # Return only top 5
    
    async def analyze_research_task(self, prompt: str) -> Dict[str, Any]:
        """Main method to analyze a research task using multi-workflow LangGraph architecture."""
        print(f"🔍 Analyzing research task: {prompt}")
        print("=" * 50)
        
        # Step 1: Route the request to determine which workflow to use
        print("\n🚦 STEP 1: ROUTING REQUEST")
        print("=" * 40)
        
        router_state: RouterState = {
            "messages": [HumanMessage(content=prompt)],
            "original_prompt": prompt,
            "routing_decision": "",
            "routing_confidence": 0.0,
            "routing_reasoning": "",
            "errors": []
        }
        
        # Run the router workflow
        final_router_state = await self.router_graph.ainvoke(router_state)
        
        # Step 2: Execute the appropriate main workflow
        workflow_decision = final_router_state["routing_decision"]
        
        if workflow_decision == "model_suggestion":
            print("\n🎯 STEP 2: EXECUTING MODEL SUGGESTION WORKFLOW")
            print("=" * 50)
            
            # Initialize model suggestion state
            model_state: ModelSuggestionState = {
                "messages": [HumanMessage(content=prompt)],
                "original_prompt": prompt,
                "detected_categories": [],
                "detailed_analysis": {},
                "arxiv_search_query": "",
                "arxiv_results": {},
                "model_suggestions": {},
                "current_step": "",
                "errors": [],
                "workflow_type": "model_suggestion"
            }
            
            # Run the model suggestion workflow
            final_model_state = await self.model_suggestion_graph.ainvoke(model_state)
            
            # Compile results
            results = {
                "workflow_type": "model_suggestion",
                "router_decision": {
                    "decision": final_router_state["routing_decision"],
                    "confidence": final_router_state["routing_confidence"],
                    "reasoning": final_router_state["routing_reasoning"]
                },
                "original_prompt": final_model_state["original_prompt"],
                "detected_categories": final_model_state["detected_categories"],
                "detailed_analysis": final_model_state["detailed_analysis"],
                "arxiv_search_query": final_model_state["arxiv_search_query"],
                "arxiv_results": final_model_state["arxiv_results"],
                "model_suggestions": final_model_state["model_suggestions"],
                "errors": final_router_state["errors"] + final_model_state["errors"],
                "summary": {
                    "workflow_used": "Model Suggestion Pipeline",
                    "total_categories_detected": len(final_model_state["detected_categories"]),
                    "high_confidence_categories": len([p for p in final_model_state["detected_categories"] if p.get("confidence", 0) > 0.7]),
                    "detailed_analysis_successful": "error" not in final_model_state["detailed_analysis"],
                    "arxiv_search_successful": final_model_state["arxiv_results"].get("search_successful", False),
                    "papers_found": final_model_state["arxiv_results"].get("papers_returned", 0),
                    "model_suggestions_successful": final_model_state["model_suggestions"].get("suggestions_successful", False),
                    "total_errors": len(final_router_state["errors"]) + len(final_model_state["errors"])
                }
            }
            
        else:  # research_planning
            print("\n📋 STEP 2: EXECUTING RESEARCH PLANNING WORKFLOW")
            print("=" * 50)
            
            # Initialize research planning state
            research_state: ResearchPlanningState = {
                "messages": [HumanMessage(content=prompt)],
                "original_prompt": prompt,
                "generated_problems": [],
                "validated_problems": [],
                "current_problem": {},
                "validation_results": {},
                "iteration_count": 0,
                "research_plan": {},
                "current_step": "",
                "errors": [],
                "workflow_type": "research_planning"
            }
            
            # Run the research planning workflow
            final_research_state = await self.research_planning_graph.ainvoke(research_state)
            
            # Compile results
            results = {
                "workflow_type": "research_planning",
                "router_decision": {
                    "decision": final_router_state["routing_decision"],
                    "confidence": final_router_state["routing_confidence"],
                    "reasoning": final_router_state["routing_reasoning"]
                },
                "original_prompt": final_research_state["original_prompt"],
                "validated_problems": final_research_state.get("validated_problems", []),
                "research_plan": final_research_state["research_plan"],
                "errors": final_router_state["errors"] + final_research_state["errors"],
                "iteration_count": final_research_state.get("iteration_count", 0),
                "summary": {
                    "workflow_used": "Iterative Research Planning Pipeline",
                    "problems_generated": len(final_research_state.get("generated_problems", [])),
                    "problems_validated": len(final_research_state.get("validated_problems", [])),
                    "research_plan_successful": final_research_state["research_plan"].get("research_plan_successful", False),
                    "total_errors": len(final_router_state["errors"]) + len(final_research_state["errors"]),
                    "iterations_completed": final_research_state.get("iteration_count", 0)
                }
            }
        
        return results

    # Conditional edge functions for research planning workflow
    def _should_continue_generating(self, state: ResearchPlanningState) -> str:
        """Determine if we should continue generating problems or move to final planning."""
        iteration_count = state.get("iteration_count", 0)
        validated_problems = state.get("validated_problems", [])
        max_iterations = 10  # Maximum iterations to prevent infinite loops
        target_problems = 3  # Target number of validated problems
        
        print(f"🔄 Checking continuation: {len(validated_problems)} problems, iteration {iteration_count}")
        
        # Check if we have enough problems or hit max iterations
        if len(validated_problems) >= target_problems:
            print(f"✅ Target reached: {len(validated_problems)}/{target_problems} problems")
            return "finalize_plan"
        elif iteration_count >= max_iterations:
            print(f"⏹️  Max iterations reached: {iteration_count}/{max_iterations}")
            return "finalize_plan"
        else:
            print(f"🔄 Continue generating: {len(validated_problems)}/{target_problems} problems, iteration {iteration_count}/{max_iterations}")
            return "generate_problem"

    def _check_completion(self, state: ResearchPlanningState) -> str:
        """Check if problem validation passed and should be collected."""
        validation_results = state.get("validation_results", {})
        recommendation = validation_results.get("recommendation", "reject")
        
        print(f"🎯 Validation result: {recommendation}")
        
        if recommendation == "accept":
            return "collect_problem"
        else:
            return "continue_generation"
    
    async def interactive_mode(self):
        """Run the tool in interactive mode."""
        print("🔬 ML Research Task Analyzer (Multi-Workflow LangGraph Version)")
        print("=" * 60)
        print("Enter your machine learning research task or question.")
        print("The system will automatically route to the appropriate workflow:")
        print("  📊 Model Suggestion: For model/architecture recommendations")
        print("  📋 Research Planning: For open problems and research planning")
        print("Type 'quit' to exit.\n")
        
        while True:
            try:
                prompt = input("🎯 Research Task: ").strip()
                
                if prompt.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not prompt:
                    print("❌ Please enter a research task.")
                    continue
                
                # Analyze the task
                results = await self.analyze_research_task(prompt)
                
                # Display results
                print("\n" + "=" * 60)
                print("📊 ANALYSIS RESULTS")
                print("=" * 60)
                
                # Show workflow routing info
                router_info = results.get("router_decision", {})
                print(f"🚦 Workflow Used: {results.get('workflow_type', 'unknown').upper()}")
                print(f"📊 Router Confidence: {router_info.get('confidence', 0):.2f}")
                print(f"💭 Router Reasoning: {router_info.get('reasoning', 'Not available')}")
                
                # Save results to file
                timestamp = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")
                workflow_type = results.get('workflow_type', 'unknown')
                filename = f"ml_research_analysis_{workflow_type}_langgraph_{timestamp}.json"
                
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                
                print(f"💾 Full results saved to: {filename}")
                print("\n" + "=" * 60 + "\n")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ An error occurred: {str(e)}")


async def main():
    """Main function to run the ML Researcher Tool."""
    try:
        tool = MLResearcherLangGraph()
        
        if len(sys.argv) > 1:
            # Command line mode
            prompt = " ".join(sys.argv[1:])
            results = await tool.analyze_research_task(prompt)
            print("\n" + json.dumps(results, indent=2))
        else:
            # Interactive mode
            await tool.interactive_mode()
    
    except Exception as e:
        print(f"❌ Failed to initialize ML Researcher Tool: {str(e)}")
        print("Make sure your API key is configured in env.example or .env file.")
        print("Also ensure LangGraph is installed: pip install langgraph")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
