#!/usr/bin/env python3
"""
Shared Constants for ML Research Assistant
==========================================

This module contains common constants and data structures shared across
multiple modules to prevent circular import issues.
"""

from typing import TypedDict, List, Dict, Annotated, Any
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from dataclasses import dataclass, asdict
import math


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
    original_prompt: str  # Pure user query without uploaded data
    uploaded_data: List[str]  # Uploaded file contents as separate field
    current_step: str
    errors: List[str]


class ModelSuggestionState(BaseState):
    """State for model suggestion workflow."""
    detected_properties: List[str]
    suggested_models: List[str]
    model_analysis: str
    arxiv_papers: List[Dict]
    similarity_scores: List[float]
    analysis_result: str
    final_recommendation: str
    workflow_complete: bool


class ResearchPlanningState(BaseState):
    """State for research planning workflow."""
    research_direction: str
    identified_problems: List[str]
    literature_review: str
    research_gaps: List[str]
    methodology_suggestions: List[str]
    evaluation_metrics: List[str]
    timeline_estimate: str
    research_plan: str
    workflow_complete: bool


class PaperWritingState(BaseState):
    """State for paper writing workflow."""
    paper_topic: str
    research_questions: List[str]
    literature_review: str
    methodology: str
    experiments: List[Dict]
    results: str
    discussion: str
    conclusion: str
    references: List[str]
    paper_draft: str
    workflow_complete: bool


class ExperimentSuggestionState(BaseState):
    """State for experiment suggestion workflow."""
    research_question: str
    hypothesis: str
    variables: Dict[str, List[str]]
    methodology: str
    data_requirements: List[str]
    analysis_plan: str
    expected_outcomes: List[str]
    potential_limitations: List[str]
    experiment_design: str
    workflow_complete: bool