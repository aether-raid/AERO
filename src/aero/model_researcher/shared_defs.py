from dataclasses import dataclass, asdict
from typing import List, Dict, Any, TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
import math
from langgraph.config import get_stream_writer
import re
import inspect
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

class BaseState(TypedDict):
    """Base state for all workflows."""
    messages: Annotated[List[BaseMessage], add_messages]
    original_prompt: str  # Pure user query without uploaded data
    uploaded_data: List[str]  # Uploaded file contents as separate field
    current_step: str
    errors: List[str]
    faiss = None


class ModelSuggestionState(BaseState):
    """State object for the model suggestion workflow."""
    detected_categories: List[Dict[str, Any]]
    detailed_analysis: Dict[str, Any]
    arxiv_search_query: str
    arxiv_results: Dict[str, Any]
    # Added fields to ensure validation + routing info isn't dropped between nodes
    validation_results: Dict[str, Any]          # Paper validation results structure
    paper_validation_decision: str              # Simple string decision (continue/search_backup/search_new)
    search_iteration: int                       # Iteration counter for search/validation cycles
    all_seen_paper_ids: set                     # For cross-search deduplication
    arxiv_chunk_metadata: List[Dict[str, Any]]  # Chunk-level metadata for semantic retrieval
    model_suggestions: Dict[str, Any]
    critique_results: Dict[str, Any]
    suggestion_iteration: int                    # Track number of suggestion iterations
    critique_history: List[Dict[str, Any]]       # Historical critique results
    cumulative_issues: Dict[str, List[str]]    
    
    # Dependencies needed by workflow nodes
    client: Any                                  # OpenAI client
    model: str                                   # Model name
    arxiv_processor: Any                         # ArxivPaperProcessor instance    


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




def _clean_text_for_utf8(text):
    """Clean text to ensure UTF-8 compatibility by removing surrogate characters."""
    if not isinstance(text, str):
        return str(text)
    
    # Remove surrogate characters that cause UTF-8 encoding issues
    
    # Remove surrogate pairs (Unicode range U+D800-U+DFFF)
    text = re.sub(r'[\ud800-\udfff]', '', text)
    
    # Replace other problematic Unicode characters with safe alternatives
    text = text.encode('utf-8', errors='ignore').decode('utf-8')
    
    # Clean up any remaining control characters except newlines and tabs
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    
    return text
