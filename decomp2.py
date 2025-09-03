"""
Decomposed Property Identifier
--------------------------------
Rule-first extractor that identifies task properties from a free-form query,
then (optionally) maps them to suggested model families.

Design:
- Property ontology with regex + keyword rules
- Deterministic extraction with confidence aggregation
- (Optional) LLM enrichment hook (stubbed for now)
- Model suggestion rules based on detected properties

Usage:
    python decomposed_property_identifier.py

This will run a demo on an example VAE query and print JSON output.
"""
from __future__ import annotations

import re
import json
import math
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Iterable, Tuple, Optional, Set
from abc import ABC, abstractmethod
import functools
from collections import defaultdict


# -------------------------------
# Data classes
# -------------------------------
@dataclass
class Evidence:
    snippet: str
    source: str  # e.g., "regex:shape", "keyword:autoencoder"
    score: float  # confidence contribution 0..1

@dataclass
class PropertyHit:
    name: str
    evidence: List[Evidence]

    @property
    def confidence(self) -> float:
        """Combine evidence scores with weighted aggregation and diminishing returns.
        Uses a more sophisticated scoring that considers evidence diversity.
        """
        if not self.evidence:
            return 0.0
        
        # Group evidence by source type for diversity bonus
        source_types = {}
        for ev in self.evidence:
            source_type = ev.source.split(':')[0]  # e.g., 'keyword', 'heuristic', 'contextual'
            if source_type not in source_types:
                source_types[source_type] = []
            source_types[source_type].append(ev.score)
        
        # Calculate base confidence using independent signals
        prod = 1.0
        for ev in self.evidence:
            prod *= (1.0 - max(0.0, min(1.0, ev.score)))
        base_confidence = 1.0 - prod
        
        # Apply diversity bonus (more source types = higher confidence)
        diversity_bonus = min(0.1 * (len(source_types) - 1), 0.2)
        
        # Apply evidence count bonus with diminishing returns
        evidence_bonus = min(0.05 * math.log(len(self.evidence) + 1), 0.15)
        
        final_confidence = min(1.0, base_confidence + diversity_bonus + evidence_bonus)
        return round(final_confidence, 3)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "confidence": self.confidence,
            "evidence": [asdict(ev) for ev in self.evidence],
        }


@dataclass
class PropertyDefinition:
    """Enhanced property definition with semantic capabilities."""
    name: str
    description: str
    keywords: List[str]
    base_score: float
    semantic_terms: List[str]  # Terms for semantic similarity matching
    domain: str = "general"  # Domain categorization
    parent_properties: List[str] = None  # Inheritance relationships
    conflicts_with: List[str] = None  # Conflicting properties
    boosts: List[str] = None  # Properties this one boosts
    
    def __post_init__(self):
        if self.parent_properties is None:
            self.parent_properties = []
        if self.conflicts_with is None:
            self.conflicts_with = []
        if self.boosts is None:
            self.boosts = []









# -------------------------------
# Ontology
# -------------------------------
# Minimal, extensible property ontology. Add packs per domain as needed.
ONTOLOGY: Dict[str, Dict[str, Any]] = {
    # Shape / length
    "variable_length_sequences": {
        "description": "Inputs can have varying temporal length (T unfixed).",
        "keywords": [
            r"variable[- ]length",
            r"unfixed length",
            r"arbitrary length",
            r"varying length",
            r"any length",
            r"different lengths?",
            r"recordings of different lengths?",
            r"T\s+is\s+(unfixed|variable)",
            r"sequences? of different sizes?",
            r"varying time steps?",
            r"dynamic length",
            r"irregular length",
            r"non[- ]uniform.+length",
            r"heterogeneous.+length",
            r"unequal.+duration",
        ],
        "score": 0.85,
    },
    "fixed_channel_count": {
        "description": "Channel dimension C is fixed.",
        "keywords": [
            r"C\s+is\s+fixed",
            r"fixed\s+channel(s)?",
            r"constant channel dimension",
            r"same number of features",
            r"fixed feature count",
            r"consistent.+features",
            r"uniform.+channels",
        ],
        "score": 0.6,
    },
    "temporal_structure": {
        "description": "Data has an explicit time/sequence axis.",
        "keywords": [
            r"temporal",
            r"time[- ]series",
            r"sequence(s)?",
            r"time steps?",
            r"T\s*[,)\]]",  # mentions T as a dimension
            r"sequential data",
            r"temporal patterns?",
            r"time-based",
            r"chronological",
            r"over time",
            r"time dimension",
            r"temporal.+dependencies",
            r"time.+ordered",
            r"longitudinal",
        ],
        "score": 0.7,
    },

    # Objectives / outputs
    "reconstruction_objective": {
        "description": "Model must reconstruct input (AE-like).",
        "keywords": [
            r"reconstruct",
            r"reconstruction",
            r"autoencoder|AE|VAE",
            r"decoder",
            r"encode[- ]decode",
            r"output\s+same\s+shape",
            r"preserve(s|d)?\s+(length|shape|T|C)",
            r"recreate.+input",
            r"reproduce.+input",
            r"learn.+representations?",
            r"unsupervised.+learning",
            r"self[- ]supervised",
        ],
        "score": 0.8,
    },
    "latent_embedding_required": {
        "description": "Requires latent space z / embedding.",
        "keywords": [
            r"latent( space)?\s*z\b",
            r"latent\s+space",
            r"embedding",
            r"bottleneck",
            r"representation\s+z",
            r"compressed representation",
            r"feature space",
            r"lower[- ]dimensional",
            r"dimensionality reduction",
            r"encode.+features?",
        ],
        "score": 0.9,
    },
    "shape_preserving_seq2seq": {
        "description": "Output shape matches input shape (T, C) → (T, C).",
        "keywords": [
            r"output\s+of\s+shape\s*\(\s*T\s*,\s*C\s*\)",
            r"input\s+and\s+output\s+shapes?\s+match",
            r"preserve\s+T\s+and\s+C",
            r"same.+dimensions?",
            r"maintain.+shape",
            r"keep.+original.+size",
            r"one[- ]to[- ]one mapping",
        ],
        "score": 0.85,
    },
    
    # Classification objectives
    "classification_objective": {
        "description": "Model performs classification/prediction tasks.",
        "keywords": [
            r"classif(y|ication)",
            r"predict(ion)?",
            r"categoriz(e|ation)",
            r"label(s|ing)?",
            r"class(es)?",
            r"activity recognition",
            r"human activity",
            r"behavior detection",
            r"pattern recognition",
            r"identify.+activities",
            r"detect.+patterns",
            r"recognize.+behavior",
            r"distinguish.+between",
            r"discriminate",
            r"which.+models?.+work.+best",
        ],
        "score": 0.8,
    },
    
    # Regression/continuous objectives
    "regression_objective": {
        "description": "Model performs regression/continuous prediction.",
        "keywords": [
            r"regression",
            r"continuous.+prediction",
            r"estimate.+value",
            r"forecast",
            r"predict.+(value|score|rating)",
            r"numeric.+output",
            r"real[- ]valued",
        ],
        "score": 0.8,
    },
    
    # Generation objectives
    "generation_objective": {
        "description": "Model generates new data samples.",
        "keywords": [
            r"generat(e|ion)",
            r"synthesiz(e|ing)",
            r"creat(e|ing).+new",
            r"sample.+from",
            r"GAN|generative",
            r"diffusion",
            r"flow[- ]based",
        ],
        "score": 0.8,
    },

    # Constraints / invariances (examples; not exhaustive)
    "noise_robustness": {
        "description": "Data may be noisy; model should be robust to noise/artifacts.",
        "keywords": [
            r"noise|noisy|denoise|artifact(s)?",
            r"robust.+noise",
            r"noise[- ]tolerant",
            r"clean.+data",
            r"corruption",
            r"outlier(s)?",
            r"missing values?",
        ],
        "score": 0.6,
    },
    "real_time_constraint": {
        "description": "Latency or on-device constraint.",
        "keywords": [
            r"real[- ]time|low[- ]latency|on[- ]device|edge",
            r"fast inference",
            r"mobile deployment",
            r"lightweight model",
            r"efficient inference",
            r"quick response",
            r"streaming",
            r"online processing",
        ],
        "score": 0.6,
    },
    "invariance_requirements": {
        "description": "Shift/lag/scale/rotation invariances are required.",
        "keywords": [
            r"shift[- ]invariant|translation[- ]invariant|lag[- ]invariant|scale[- ]invariant|rotation[- ]invariant",
            r"timing shifts?",
            r"small timing shifts?",
            r"shouldn't change.+prediction",
            r"robust.+shifts?",
            r"invariant.+position",
            r"position[- ]independent",
            r"offset[- ]invariant",
        ],
        "score": 0.6,
    },
    
    # Data types and domains
    "sensor_data": {
        "description": "Sensor or wearable device data.",
        "keywords": [
            r"sensor data",
            r"wearable",
            r"accelerometer",
            r"gyroscope",
            r"IMU",
            r"physiological",
            r"biometric",
            r"heart rate",
            r"motion",
            r"GPS",
            r"fitness.+tracker",
            r"smartwatch",
            r"IoT.+sensor",
            r"vital.+signs",
        ],
        "score": 0.7,
    },
    
    "multimodal_data": {
        "description": "Multiple data types or modalities.",
        "keywords": [
            r"multimodal",
            r"multi[- ]modal",
            r"multiple.+types",
            r"different.+modalities",
            r"fusion",
            r"combine.+sources",
            r"heterogeneous.+data",
            r"cross[- ]modal",
        ],
        "score": 0.7,
    },
    
    # Performance and optimization requirements
    "interpretability_required": {
        "description": "Model needs to be interpretable/explainable.",
        "keywords": [
            r"interpretab(le|ility)",
            r"explainab(le|ility)",
            r"understand.+decision",
            r"explain.+prediction",
            r"transparent",
            r"black[- ]box",
            r"feature.+importance",
            r"attention.+weights",
        ],
        "score": 0.7,
    },
    
    "high_accuracy_required": {
        "description": "High accuracy or precision is critical.",
        "keywords": [
            r"high.+accuracy",
            r"precise",
            r"critical.+performance",
            r"state[- ]of[- ]the[- ]art",
            r"best.+performance",
            r"maximum.+accuracy",
            r"mission[- ]critical",
        ],
        "score": 0.6,
    },
    
    "few_shot_learning": {
        "description": "Limited training data available.",
        "keywords": [
            r"few[- ]shot",
            r"limited.+data",
            r"small.+dataset",
            r"scarce.+labels",
            r"meta[- ]learning",
            r"transfer.+learning",
            r"pre[- ]trained",
        ],
        "score": 0.7,
    },
    
    # Meta-properties for query understanding
    "model_selection_query": {
        "description": "Query is asking for model recommendations.",
        "keywords": [
            r"which.+model",
            r"what.+approach",
            r"recommend.+method",
            r"best.+algorithm",
            r"suitable.+architecture",
            r"work.+best",
        ],
        "score": 0.6,
    },
    
    # Text/NLP specific properties
    "text_data": {
        "description": "Working with text/natural language data.",
        "keywords": [
            r"text",
            r"natural language",
            r"NLP",
            r"linguistic",
            r"document(s)?",
            r"clinical notes",
            r"sentiment",
            r"language model",
            r"tokeniz",
        ],
        "score": 0.8,
    },
    
    "multilingual_requirement": {
        "description": "Needs to work across multiple languages.",
        "keywords": [
            r"multilingual",
            r"multi[- ]language",
            r"cross[- ]lingual",
            r"different languages",
            r"multiple languages",
            r"language[- ]agnostic",
        ],
        "score": 0.8,
    },
    
    "variable_document_length": {
        "description": "Documents have varying lengths.",
        "keywords": [
            r"very short or very long",
            r"variable.+length",
            r"different.+length",
            r"varying.+size",
            r"short.+long.+document",
        ],
        "score": 0.7,
    },
}








class SemanticMatcher:
    """Handles semantic similarity matching using embeddings."""
    
    def __init__(self):
        self._embeddings_cache = {}
        self._model = None  # Will be lazy-loaded
        self._semantic_available = None  # Cache availability check
        
    def _get_model(self):
        """Lazy load sentence transformer model."""
        if self._semantic_available is False:
            return None
        
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer('all-MiniLM-L6-v2')
                self._semantic_available = True
            except ImportError:
                if self._semantic_available is None:  # Only warn once
                    print("Note: sentence-transformers not available, using keyword matching only")
                self._semantic_available = False
                self._model = None
        return self._model
    
    def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding for text with caching."""
        if text in self._embeddings_cache:
            return self._embeddings_cache[text]
        
        model = self._get_model()
        if model is None:
            return None
            
        embedding = model.encode(text, convert_to_numpy=True)
        self._embeddings_cache[text] = embedding
        return embedding
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts."""
        emb1 = self.get_embedding(text1)
        emb2 = self.get_embedding(text2)
        
        if emb1 is None or emb2 is None:
            return 0.0
        
        # Cosine similarity
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return float(similarity)
    
    def find_semantic_matches(self, query: str, property_definitions: Dict[str, PropertyDefinition], 
                            threshold: float = 0.6) -> List[Tuple[str, float, str]]:
        """Find semantic matches between query and property definitions.
        
        Returns: List of (property_name, similarity_score, matched_term)
        """
        matches = []
        query_lower = query.lower()
        
        for prop_name, prop_def in property_definitions.items():
            # Check semantic similarity with description
            desc_similarity = self.compute_similarity(query, prop_def.description)
            if desc_similarity > threshold:
                matches.append((prop_name, desc_similarity, "description"))
            
            # Check similarity with semantic terms
            for term in prop_def.semantic_terms:
                term_similarity = self.compute_similarity(query, term)
                if term_similarity > threshold:
                    matches.append((prop_name, term_similarity, f"semantic_term:{term}"))
                
                # Also check if semantic terms appear in query (exact/fuzzy match)
                if term.lower() in query_lower:
                    matches.append((prop_name, 0.8, f"exact_semantic:{term}"))
        
        return matches


class PropertyRegistry:
    """Registry for managing expandable property ontology."""
    
    def __init__(self):
        self.properties: Dict[str, PropertyDefinition] = {}
        self.domains: Dict[str, Set[str]] = defaultdict(set)
        self._load_base_properties()
    
    def _load_base_properties(self):
        """Load base properties from the original ONTOLOGY."""
        # Convert original ONTOLOGY to new format
        for prop_name, prop_data in ONTOLOGY.items():
            self.register_property(PropertyDefinition(
                name=prop_name,
                description=prop_data["description"],
                keywords=prop_data["keywords"],
                base_score=prop_data["score"],
                semantic_terms=self._extract_semantic_terms(prop_data["description"], prop_data["keywords"]),
                domain=self._infer_domain(prop_name, prop_data["description"])
            ))
    
    def _extract_semantic_terms(self, description: str, keywords: List[str]) -> List[str]:
        """Extract semantic terms from description and clean keywords."""
        terms = [description]
        
        # Clean regex patterns to extract semantic meaning
        for keyword in keywords[:5]:  # Limit to avoid too many terms
            # Remove regex special characters and extract core terms
            clean_term = re.sub(r'[\\()[\]{}|+*?^$.]', '', keyword)
            clean_term = re.sub(r'\s+', ' ', clean_term).strip()
            if len(clean_term) > 3 and not clean_term.isdigit():
                terms.append(clean_term)
        
        return terms
    
    def _infer_domain(self, prop_name: str, description: str) -> str:
        """Infer domain from property name and description."""
        text = f"{prop_name} {description}".lower()
        
        if any(term in text for term in ['text', 'nlp', 'language', 'document', 'clinical']):
            return 'nlp'
        elif any(term in text for term in ['sensor', 'wearable', 'activity', 'physiological']):
            return 'sensor'
        elif any(term in text for term in ['image', 'vision', 'object', 'visual']):
            return 'vision'
        elif any(term in text for term in ['graph', 'network', 'node', 'edge']):
            return 'graph'
        elif any(term in text for term in ['time', 'temporal', 'sequence', 'series']):
            return 'temporal'
        else:
            return 'general'
    
    def register_property(self, prop_def: PropertyDefinition):
        """Register a new property definition."""
        self.properties[prop_def.name] = prop_def
        self.domains[prop_def.domain].add(prop_def.name)
    
    def get_property(self, name: str) -> Optional[PropertyDefinition]:
        """Get property definition by name."""
        return self.properties.get(name)
    
    def get_domain_properties(self, domain: str) -> List[PropertyDefinition]:
        """Get all properties in a domain."""
        return [self.properties[name] for name in self.domains.get(domain, set())]
    
    def add_modern_ml_properties(self):
        """Add modern ML/AI properties to the registry."""
        modern_properties = [
            PropertyDefinition(
                name="transformer_architecture",
                description="Uses transformer architecture with self-attention mechanisms",
                keywords=[
                    r"transformer", r"attention", r"self[- ]attention", r"multi[- ]head",
                    r"BERT", r"GPT", r"T5", r"encoder[- ]decoder"
                ],
                base_score=0.8,
                semantic_terms=["attention mechanism", "transformer model", "self-attention", "encoder-decoder"],
                domain="architecture"
            ),
            PropertyDefinition(
                name="computer_vision",
                description="Computer vision tasks involving image or video data",
                keywords=[
                    r"image", r"vision", r"visual", r"object detection", r"segmentation",
                    r"CNN", r"convolutional", r"video", r"frame"
                ],
                base_score=0.8,
                semantic_terms=["image processing", "computer vision", "visual recognition", "object detection"],
                domain="vision"
            ),
            PropertyDefinition(
                name="graph_neural_networks",
                description="Graph-based data and neural network architectures",
                keywords=[
                    r"graph", r"node", r"edge", r"GNN", r"graph neural network",
                    r"social network", r"network analysis", r"graph structure"
                ],
                base_score=0.8,
                semantic_terms=["graph neural network", "social network analysis", "network topology"],
                domain="graph"
            ),
            PropertyDefinition(
                name="self_supervised_learning",
                description="Self-supervised learning approaches",
                keywords=[
                    r"self[- ]supervised", r"contrastive learning", r"masked language model",
                    r"SimCLR", r"CLIP", r"pre[- ]training", r"foundation model"
                ],
                base_score=0.7,
                semantic_terms=["self-supervised learning", "contrastive learning", "pretext task"],
                domain="learning_paradigm"
            ),
            PropertyDefinition(
                name="diffusion_models",
                description="Diffusion-based generative models",
                keywords=[
                    r"diffusion", r"DDPM", r"DDIM", r"score[- ]based", r"denoising diffusion",
                    r"stable diffusion", r"latent diffusion"
                ],
                base_score=0.8,
                semantic_terms=["diffusion model", "denoising diffusion", "score-based generative model"],
                domain="generation"
            ),
            PropertyDefinition(
                name="federated_learning",
                description="Distributed learning across multiple devices/organizations",
                keywords=[
                    r"federated learning", r"distributed training", r"privacy[- ]preserving",
                    r"decentralized", r"on[- ]device learning"
                ],
                base_score=0.8,
                semantic_terms=["federated learning", "distributed machine learning", "privacy-preserving AI"],
                domain="distributed"
            ),
            PropertyDefinition(
                name="reinforcement_learning",
                description="Reinforcement learning and decision-making tasks",
                keywords=[
                    r"reinforcement learning", r"RL", r"policy", r"reward", r"agent",
                    r"Q[- ]learning", r"actor[- ]critic", r"environment interaction"
                ],
                base_score=0.8,
                semantic_terms=["reinforcement learning", "policy optimization", "agent-environment interaction"],
                domain="rl"
            ),
            PropertyDefinition(
                name="anomaly_detection",
                description="Detecting outliers, anomalies, or unusual patterns",
                keywords=[
                    r"anomaly detection", r"outlier detection", r"unusual pattern",
                    r"fraud detection", r"intrusion detection", r"change detection"
                ],
                base_score=0.8,
                semantic_terms=["anomaly detection", "outlier analysis", "novelty detection"],
                domain="detection"
            ),
            PropertyDefinition(
                name="mlops_requirements",
                description="MLOps, deployment, monitoring, and production concerns",
                keywords=[
                    r"MLOps", r"deployment", r"monitoring", r"production",
                    r"model drift", r"A/B testing", r"continuous integration"
                ],
                base_score=0.7,
                semantic_terms=["machine learning operations", "model deployment", "production ML"],
                domain="ops"
            ),
            PropertyDefinition(
                name="fairness_constraints",
                description="Fairness, bias, and ethical AI considerations",
                keywords=[
                    r"fairness", r"bias", r"ethical", r"discrimination",
                    r"algorithmic bias", r"equitable", r"demographic parity"
                ],
                base_score=0.7,
                semantic_terms=["algorithmic fairness", "bias mitigation", "ethical AI"],
                domain="ethics"
            )
        ]
        
        for prop in modern_properties:
            self.register_property(prop)





# -------------------------------
# PropertyExtractor2 - Enhanced with Semantic Capabilities
# -------------------------------

class PropertyExtractor2:
    """
    Enhanced property extractor with semantic embedding support and dynamic ontology.
    
    Key improvements over PropertyExtractor:
    1. Semantic embedding layer for concept similarity matching
    2. Expandable ontology structure with property relationships
    3. Domain-specific property organization
    4. Enhanced confidence scoring with uncertainty quantification
    """
    
    def __init__(self, use_semantic_matching: bool = True, semantic_threshold: float = 0.6):
        self.property_registry = PropertyRegistry()
        self.semantic_matcher = SemanticMatcher() if use_semantic_matching else None
        self.semantic_threshold = semantic_threshold
        
        # Load modern ML properties
        self.property_registry.add_modern_ml_properties()
        
        # Performance optimizations
        self._embedding_cache_size = 1000
        self._pattern_cache = {}
    
    def extract(self, text: str, domain_focus: Optional[str] = None) -> List[PropertyHit]:
        """
        Extract properties with enhanced semantic understanding.
        
        Args:
            text: Input query text
            domain_focus: Optional domain to focus search on (e.g., 'nlp', 'vision', 'sensor')
        """
        text_norm = self._normalize_enhanced(text)
        hits: Dict[str, PropertyHit] = {}
        
        # Get relevant properties (domain-filtered if specified)
        target_properties = self._get_target_properties(domain_focus)
        
        # 1) Legacy keyword/regex matching (backwards compatibility)
        self._apply_keyword_matching(text_norm, hits, target_properties)
        
        # 2) Semantic embedding matching
        if self.semantic_matcher:
            self._apply_semantic_matching(text_norm, hits, target_properties)
        
        # 3) Enhanced pattern detection
        self._apply_enhanced_pattern_detection(text_norm, hits)
        
        # 4) Compositional reasoning (Phase 1 basic version)
        self._apply_basic_composition(text_norm, hits)
        
        # 5) Adaptive confidence adjustment
        self._apply_adaptive_scoring(text_norm, hits)
        
        # 6) Property relationship processing
        self._process_property_relationships(hits)
        
        # Sort by confidence with tie-breaking
        out = list(hits.values())
        out.sort(key=lambda h: (-h.confidence, h.name))
        return out
    
    def _get_target_properties(self, domain_focus: Optional[str]) -> Dict[str, PropertyDefinition]:
        """Get properties to search, optionally filtered by domain."""
        if domain_focus:
            return {name: prop for name, prop in self.property_registry.properties.items() 
                   if prop.domain == domain_focus or prop.domain == 'general'}
        return self.property_registry.properties
    
    def _normalize_enhanced(self, text: str) -> str:
        """Enhanced normalization with ML/AI-specific preprocessing."""
        # Basic canonicalization
        text = re.sub(r"\s+", " ", text.strip())
        
        # Expanded ML/AI abbreviations
        ml_abbreviations = {
            r'\bML\b': 'machine learning',
            r'\bAI\b': 'artificial intelligence', 
            r'\bDL\b': 'deep learning',
            r'\bNN\b': 'neural network',
            r'\bCNN\b': 'convolutional neural network',
            r'\bRNN\b': 'recurrent neural network',
            r'\bLSTM\b': 'long short term memory',
            r'\bGRU\b': 'gated recurrent unit',
            r'\bVAE\b': 'variational autoencoder',
            r'\bGAN\b': 'generative adversarial network',
            r'\bGNN\b': 'graph neural network',
            r'\bNLP\b': 'natural language processing',
            r'\bCV\b': 'computer vision',
            r'\bRL\b': 'reinforcement learning',
            r'\bSGD\b': 'stochastic gradient descent',
            r'\bAdam\b': 'adaptive moment estimation',
            r'\bBERT\b': 'bidirectional encoder representations from transformers',
            r'\bGPT\b': 'generative pre-trained transformer',
            r'\bViT\b': 'vision transformer',
            r'\bCLIP\b': 'contrastive language image pre-training',
            r'\bYOLO\b': 'you only look once',
            r'\bR-CNN\b': 'region-based convolutional neural network'
        }
        
        for abbrev, expansion in ml_abbreviations.items():
            abbrev_clean = abbrev.strip('\\b')
            text = re.sub(abbrev, f"{expansion} ({abbrev_clean})", text, flags=re.I)
        
        return text
    
    def _apply_keyword_matching(self, text: str, hits: Dict[str, PropertyHit], 
                              target_properties: Dict[str, PropertyDefinition]):
        """Apply traditional keyword/regex matching."""
        for prop_name, prop_def in target_properties.items():
            for pattern in prop_def.keywords:
                for match in re.finditer(pattern, text, flags=re.IGNORECASE | re.MULTILINE):
                    snippet = self._context_snippet(text, match.start(), match.end())
                    evidence = Evidence(
                        snippet=snippet,
                        source=f"keyword:{pattern[:30]}...",
                        score=prop_def.base_score
                    )
                    self._add_hit(hits, prop_name, evidence)
    
    def _apply_semantic_matching(self, text: str, hits: Dict[str, PropertyHit],
                               target_properties: Dict[str, PropertyDefinition]):
        """Apply semantic similarity matching."""
        if not self.semantic_matcher:
            return
        
        # Find semantic matches
        semantic_matches = self.semantic_matcher.find_semantic_matches(
            text, target_properties, self.semantic_threshold
        )
        
        for prop_name, similarity, match_type in semantic_matches:
            evidence = Evidence(
                snippet=f"Semantic similarity: {similarity:.3f}",
                source=f"semantic:{match_type}",
                score=min(similarity * 0.8, 0.9)  # Scale down semantic scores slightly
            )
            self._add_hit(hits, prop_name, evidence)
    
    def _apply_enhanced_pattern_detection(self, text: str, hits: Dict[str, PropertyHit]):
        """Enhanced pattern detection with ML-specific patterns."""
        # Architecture patterns
        architecture_patterns = {
            "transformer_architecture": [
                r"multi[- ]head attention", r"encoder[- ]decoder", r"self[- ]attention",
                r"positional encoding", r"attention mechanism"
            ],
            "convolutional_architecture": [
                r"convolution", r"feature map", r"pooling", r"kernel", r"filter"
            ],
            "recurrent_architecture": [
                r"hidden state", r"cell state", r"sequence modeling", r"temporal memory"
            ]
        }
        
        for prop_name, patterns in architecture_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, re.I):
                    evidence = Evidence(
                        snippet=f"Architecture pattern: {pattern}",
                        source=f"pattern:{pattern}",
                        score=0.7
                    )
                    self._add_hit(hits, prop_name, evidence)
        
        # Research methodology patterns
        if re.search(r"ablation study|hyperparameter|grid search|cross[- ]validation", text, re.I):
            evidence = Evidence(
                snippet="Research methodology detected",
                source="pattern:research_methodology",
                score=0.6
            )
            self._add_hit(hits, "research_methodology", evidence)
        
        # Performance optimization patterns
        if re.search(r"quantization|pruning|distillation|compression|optimization", text, re.I):
            evidence = Evidence(
                snippet="Model optimization mentioned",
                source="pattern:optimization",
                score=0.7
            )
            self._add_hit(hits, "model_optimization", evidence)
    
    def _apply_basic_composition(self, text: str, hits: Dict[str, PropertyHit]):
        """Basic compositional reasoning for property combinations."""
        # Detect common ML task combinations
        combinations = {
            ("text_data", "classification_objective"): "text_classification",
            ("computer_vision", "classification_objective"): "image_classification", 
            ("sensor_data", "temporal_structure", "classification_objective"): "temporal_sensor_classification",
            ("multimodal_data", "classification_objective"): "multimodal_classification",
            ("graph_neural_networks", "classification_objective"): "graph_classification"
        }
        
        detected_props = set(hits.keys())
        
        for prop_combo, composite_prop in combinations.items():
            if all(prop in detected_props for prop in prop_combo):
                evidence = Evidence(
                    snippet=f"Inferred from: {', '.join(prop_combo)}",
                    source="composition:inference",
                    score=0.8
                )
                self._add_hit(hits, composite_prop, evidence)
    
    def _apply_adaptive_scoring(self, text: str, hits: Dict[str, PropertyHit]):
        """Apply adaptive confidence scoring based on context."""
        # Query complexity analysis
        query_complexity = self._analyze_query_complexity(text)
        
        # Adjust scores based on query type
        if query_complexity["is_exploratory"]:
            # Lower threshold for exploratory queries
            for hit in hits.values():
                for evidence in hit.evidence:
                    if evidence.score < 0.5:
                        evidence.score *= 1.2  # Boost weak signals
        
        if query_complexity["is_specific"]:
            # Higher precision for specific queries
            for hit in hits.values():
                for evidence in hit.evidence:
                    if evidence.score > 0.7:
                        evidence.score *= 1.1  # Boost strong signals
    
    def _analyze_query_complexity(self, text: str) -> Dict[str, bool]:
        """Analyze query characteristics for adaptive scoring."""
        return {
            "is_exploratory": bool(re.search(r"explore|investigate|research|study", text, re.I)),
            "is_specific": bool(re.search(r"implement|build|deploy|specific", text, re.I)),
            "has_constraints": bool(re.search(r"requirement|constraint|must|need", text, re.I)),
            "is_comparative": bool(re.search(r"compare|versus|vs|better than", text, re.I))
        }
    
    def _process_property_relationships(self, hits: Dict[str, PropertyHit]):
        """Process property inheritance and relationships."""
        detected_props = set(hits.keys())
        
        # Apply property boosts and conflicts
        for prop_name in list(detected_props):
            prop_def = self.property_registry.get_property(prop_name)
            if not prop_def:
                continue
            
            # Boost related properties
            for boost_prop in prop_def.boosts:
                if boost_prop in detected_props:
                    evidence = Evidence(
                        snippet=f"Boosted by {prop_name}",
                        source=f"relationship:boost",
                        score=0.15
                    )
                    self._add_hit(hits, boost_prop, evidence)
            
            # Handle conflicts
            for conflict_prop in prop_def.conflicts_with:
                if conflict_prop in detected_props:
                    evidence = Evidence(
                        snippet=f"Conflicts with {prop_name}",
                        source=f"relationship:conflict",
                        score=-0.1
                    )
                    self._add_hit(hits, conflict_prop, evidence)
    
    def _context_snippet(self, text: str, start: int, end: int, radius: int = 50) -> str:
        """Enhanced context snippet extraction."""
        s = max(0, start - radius)
        e = min(len(text), end + radius)
        snippet = text[s:e]
        
        # Add ellipsis if truncated
        if s > 0:
            snippet = "..." + snippet
        if e < len(text):
            snippet = snippet + "..."
            
        return snippet
    
    def _add_hit(self, hits: Dict[str, PropertyHit], prop: str, ev: Evidence):
        """Add evidence to property hit with deduplication."""
        if prop not in hits:
            hits[prop] = PropertyHit(name=prop, evidence=[ev])
        else:
            # Avoid duplicate evidence from same source
            existing_sources = {e.source for e in hits[prop].evidence}
            if ev.source not in existing_sources:
                hits[prop].evidence.append(ev)
    
    def get_domain_analysis(self, text: str) -> Dict[str, float]:
        """Analyze which ML domains are most relevant to the query."""
        domain_scores = defaultdict(float)
        
        # Extract properties for domain analysis
        props = self.extract(text)
        
        for prop in props:
            prop_def = self.property_registry.get_property(prop.name)
            if prop_def:
                domain_scores[prop_def.domain] += prop.confidence
        
        # Normalize scores
        if domain_scores:
            max_score = max(domain_scores.values())
            domain_scores = {domain: score/max_score for domain, score in domain_scores.items()}
        
        return dict(domain_scores)
    
    def suggest_missing_properties(self, detected_props: List[PropertyHit]) -> List[str]:
        """Suggest potentially missing properties based on detected ones."""
        detected_names = {p.name for p in detected_props}
        suggestions = []
        
        # Common property gaps in ML projects
        ml_property_groups = {
            "data_properties": ["temporal_structure", "multimodal_data", "sensor_data", "text_data"],
            "objective_properties": ["classification_objective", "reconstruction_objective", "generation_objective"],
            "constraint_properties": ["real_time_constraint", "interpretability_required", "fairness_constraints"],
            "architecture_properties": ["transformer_architecture", "computer_vision", "graph_neural_networks"]
        }
        
        for group_name, group_props in ml_property_groups.items():
            if not any(prop in detected_names for prop in group_props):
                suggestions.append(f"Consider specifying {group_name.replace('_', ' ')}")
        
        return suggestions
    
    def explain_confidence(self, property_hit: PropertyHit) -> Dict[str, Any]:
        """Provide detailed explanation of confidence calculation."""
        evidence_by_type = defaultdict(list)
        for ev in property_hit.evidence:
            evidence_type = ev.source.split(':')[0]
            evidence_by_type[evidence_type].append(ev.score)
        
        return {
            "total_confidence": property_hit.confidence,
            "evidence_types": dict(evidence_by_type),
            "evidence_count": len(property_hit.evidence),
            "score_breakdown": {
                "base_signals": len([e for e in property_hit.evidence if e.score > 0.5]),
                "weak_signals": len([e for e in property_hit.evidence if 0.2 <= e.score <= 0.5]),
                "negative_signals": len([e for e in property_hit.evidence if e.score < 0])
            }
        }









# -------------------------------
# Enhanced Orchestration - DecomposerTool2 (No Model Suggestions)
# -------------------------------







# -------------------------------
# Enhanced Orchestration - DecomposerTool2
# -------------------------------

class DecomposerTool2:
    """
    Enhanced decomposer tool using PropertyExtractor2 with semantic capabilities.
    
    Improvements:
    - Semantic understanding of queries
    - Domain-aware analysis
    - Enhanced confidence explanations
    - Better gap detection and suggestions
    """
    
    def __init__(self, use_semantic_matching: bool = True):
        self.extractor = PropertyExtractor2(use_semantic_matching=use_semantic_matching)
    
    def analyze(self, query: str, domain_focus: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze query with enhanced semantic understanding.
        
        Args:
            query: Input research query
            domain_focus: Optional domain focus (e.g., 'nlp', 'vision', 'sensor')
        """
        # Extract properties with optional domain focus
        props = self.extractor.extract(query, domain_focus=domain_focus)
        
        # Domain analysis
        domain_scores = self.extractor.get_domain_analysis(query)
        
        # Enhanced gap detection
        missing_suggestions = self.extractor.suggest_missing_properties(props)
        missing = self._detect_enhanced_gaps(props, query, domain_scores)
        
        # Confidence explanations
        confidence_explanations = {}
        for prop in props[:5]:  # Top 5 properties
            confidence_explanations[prop.name] = self.extractor.explain_confidence(prop)
        
        return {
            "detected_properties": [p.to_dict() for p in props],
            "missing_or_ambiguous": missing,
            "domain_analysis": domain_scores,
            "missing_property_suggestions": missing_suggestions,
            "confidence_explanations": confidence_explanations,
            "analysis_summary": self._generate_enhanced_summary(props, domain_scores),
            "query_metadata": self._extract_query_metadata(query)
        }
    
    def _detect_enhanced_gaps(self, props: List[PropertyHit], query: str, 
                            domain_scores: Dict[str, float]) -> List[Dict[str, str]]:
        """Enhanced gap detection with domain awareness."""
        missing = []
        prop_names = {p.name for p in props}
        
        # Domain-specific gap detection
        top_domain = max(domain_scores.items(), key=lambda x: x[1])[0] if domain_scores else None
        
        if top_domain == "nlp":
            if "text_data" not in prop_names:
                missing.append({"name": "text_data", "ask": "What type of text data are you working with?"})
            if not any("multilingual" in name for name in prop_names):
                missing.append({"name": "language_requirements", "ask": "Do you need multilingual support?"})
        
        elif top_domain == "vision":
            if "computer_vision" not in prop_names:
                missing.append({"name": "vision_task", "ask": "What specific computer vision task (detection, segmentation, classification)?"})
            if not any("real_time" in name for name in prop_names):
                missing.append({"name": "performance_requirements", "ask": "Are there real-time processing requirements?"})
        
        elif top_domain == "sensor":
            if not any("temporal" in name for name in prop_names):
                missing.append({"name": "temporal_aspects", "ask": "How important are temporal relationships in your data?"})
        
        # General gaps
        if not any(name.endswith("_objective") for name in prop_names):
            missing.append({"name": "task_objective", "ask": "What is the primary learning objective?"})
        
        if not any("constraint" in name or "requirement" in name for name in prop_names):
            missing.append({"name": "constraints", "ask": "Are there performance, resource, or ethical constraints?"})
        
        # Data characteristics
        if not any("data" in name for name in prop_names):
            missing.append({"name": "data_characteristics", "ask": "What are the key characteristics of your data?"})
        
        return missing
    
    def _generate_enhanced_summary(self, props: List[PropertyHit],
                                 domain_scores: Dict[str, float]) -> Dict[str, Any]:
        """Generate enhanced analysis summary."""
        high_conf_props = [p for p in props if p.confidence >= 0.7]
        medium_conf_props = [p for p in props if 0.4 <= p.confidence < 0.7]
        
        # Determine primary task type with better logic
        task_type = self._infer_task_type(high_conf_props)
        
        # Identify primary domain
        primary_domain = max(domain_scores.items(), key=lambda x: x[1])[0] if domain_scores else "general"
        
        # Complexity assessment
        complexity_factors = self._assess_complexity(props)
        
        # Research stage inference
        research_stage = self._infer_research_stage(props)
        
        return {
            "primary_task_type": task_type,
            "primary_domain": primary_domain,
            "domain_confidence": domain_scores.get(primary_domain, 0.0),
            "complexity_factors": complexity_factors,
            "research_stage": research_stage,
            "high_confidence_properties": len(high_conf_props),
            "medium_confidence_properties": len(medium_conf_props),
            "recommended_next_steps": self._get_enhanced_next_steps(high_conf_props, primary_domain),
            "uncertainty_assessment": self._assess_uncertainty(props)
        }
    
    def _infer_task_type(self, high_conf_props: List[PropertyHit]) -> str:
        """Infer task type with enhanced logic."""
        prop_names = {p.name for p in high_conf_props}
        
        if any("classification" in name for name in prop_names):
            if any("text" in name for name in prop_names):
                return "Text Classification"
            elif any("vision" in name or "image" in name for name in prop_names):
                return "Image Classification"
            elif any("sensor" in name for name in prop_names):
                return "Sensor Classification"
            else:
                return "Classification"
        
        elif any("reconstruction" in name for name in prop_names):
            return "Reconstruction/Autoencoder"
        elif any("generation" in name for name in prop_names):
            return "Generation"
        elif any("detection" in name for name in prop_names):
            return "Detection/Recognition"
        elif any("graph" in name for name in prop_names):
            return "Graph Learning"
        else:
            return "General ML Task"
    
    def _assess_complexity(self, props: List[PropertyHit]) -> List[str]:
        """Assess problem complexity factors."""
        complexity_factors = []
        prop_names = {p.name for p in props}
        
        if any("variable_length" in name for name in prop_names):
            complexity_factors.append("Variable-length inputs")
        if any("multimodal" in name for name in prop_names):
            complexity_factors.append("Multimodal data fusion")
        if any("real_time" in name for name in prop_names):
            complexity_factors.append("Real-time constraints")
        if any("federated" in name for name in prop_names):
            complexity_factors.append("Distributed learning")
        if any("fairness" in name or "bias" in name for name in prop_names):
            complexity_factors.append("Fairness considerations")
        if any("few_shot" in name for name in prop_names):
            complexity_factors.append("Limited data regime")
        
        return complexity_factors
    
    def _infer_research_stage(self, props: List[PropertyHit]) -> str:
        """Infer research/development stage."""
        prop_names = {p.name for p in props}
        
        if any("exploration" in name or "research" in name for name in prop_names):
            return "Exploration"
        elif any("optimization" in name or "hyperparameter" in name for name in prop_names):
            return "Optimization"
        elif any("deployment" in name or "production" in name for name in prop_names):
            return "Deployment"
        elif any("comparison" in name or "benchmark" in name for name in prop_names):
            return "Evaluation"
        else:
            return "Development"
    
    def _assess_uncertainty(self, props: List[PropertyHit]) -> Dict[str, Any]:
        """Assess uncertainty in property detection."""
        if not props:
            return {"level": "high", "reason": "no_properties_detected"}
        
        confidence_scores = [p.confidence for p in props]
        avg_confidence = sum(confidence_scores) / len(confidence_scores)
        max_confidence = max(confidence_scores)
        confidence_variance = sum((c - avg_confidence)**2 for c in confidence_scores) / len(confidence_scores)
        
        if max_confidence < 0.5:
            level = "high"
            reason = "low_confidence_scores"
        elif avg_confidence < 0.6:
            level = "medium-high"
            reason = "moderate_confidence"
        elif confidence_variance > 0.1:
            level = "medium"
            reason = "high_variance"
        else:
            level = "low"
            reason = "consistent_high_confidence"
        
        return {
            "level": level,
            "reason": reason,
            "average_confidence": round(avg_confidence, 3),
            "max_confidence": round(max_confidence, 3),
            "confidence_variance": round(confidence_variance, 3)
        }
    
    def _get_enhanced_next_steps(self, high_conf_props: List[PropertyHit], 
                               primary_domain: str) -> List[str]:
        """Generate enhanced next steps."""
        steps = []
        
        if len(high_conf_props) < 3:
            steps.append(f"Clarify {primary_domain} task requirements - provide more specific details")
        
        steps.append("Review detected properties and confidence scores for accuracy")
        
        # Domain-specific steps
        if primary_domain == "nlp":
            steps.append("Consider text preprocessing: tokenization, normalization, handling of special characters")
        elif primary_domain == "vision":
            steps.append("Consider image preprocessing: normalization, augmentation, resolution requirements")
        elif primary_domain == "sensor":
            steps.append("Consider signal processing: filtering, feature extraction, temporal alignment")
        
        steps.append("Design evaluation metrics and validation strategy")
        steps.append("Consider data quality and labeling requirements")
        
        return steps
    
    def _extract_query_metadata(self, query: str) -> Dict[str, Any]:
        """Extract metadata about the query itself."""
        return {
            "length": len(query),
            "word_count": len(query.split()),
            "has_technical_terms": bool(re.search(r'\b(algorithm|model|architecture|neural|learning)\b', query, re.I)),
            "has_performance_metrics": bool(re.search(r'\b(accuracy|precision|recall|F1|AUC|loss)\b', query, re.I)),
            "has_constraints": bool(re.search(r'\b(real[- ]time|latency|memory|budget|constraint)\b', query, re.I)),
            "question_type": "recommendation" if re.search(r'\b(which|what|how|recommend|suggest|best)\b', query, re.I) else "implementation"
        }








if __name__ == "__main__":
      # Test enhanced PropertyExtractor2

    demo_query =   "I want to build a real-time object detection system for autonomous vehicles using transformer architecture with edge deployment constraints."
    print("ENHANCED PropertyExtractor2 Results")
    print("="*50)
    
    try:
        enhanced_tool = DecomposerTool2(use_semantic_matching=True)
        enhanced_result = enhanced_tool.analyze(demo_query)
        
        
        print(f"\nDetected Properties ({len(enhanced_result['detected_properties'])}):")
        for prop in enhanced_result["detected_properties"][:8]:  # Top 8
            print(f"  • {prop['name']} (confidence: {prop['confidence']})")
            if prop['evidence']:
                print(f"    └─ {prop['evidence'][0]['source']}: {prop['evidence'][0]['snippet'][:60]}...")
        
        print(f"\nDomain Analysis:")
        for domain, score in sorted(enhanced_result['domain_analysis'].items(), 
                                  key=lambda x: x[1], reverse=True)[:5]:
            print(f"  • {domain}: {score:.3f}")
        
        print(f"\nAnalysis Summary:")
        summary = enhanced_result['analysis_summary']
        print(f"  • Primary Task: {summary['primary_task_type']}")
        print(f"  • Primary Domain: {summary['primary_domain']}")
        print(f"  • Research Stage: {summary['research_stage']}")
        print(f"  • Complexity Factors: {', '.join(summary['complexity_factors'])}")
        print(f"  • Uncertainty Level: {summary['uncertainty_assessment']['level']}")
        
        print(f"\nMissing Property Suggestions ({len(enhanced_result['missing_property_suggestions'])}):")
        for suggestion in enhanced_result['missing_property_suggestions']:
            print(f"  • {suggestion}")
        
        print(f"\nQuery Metadata:")
        metadata = enhanced_result['query_metadata']
        print(f"  • Length: {metadata['length']} chars, {metadata['word_count']} words")
        print(f"  • Technical Terms: {metadata['has_technical_terms']}")
        print(f"  • Type: {metadata['question_type']}")
        
    except ImportError as e:
        print("\nNote: Enhanced PropertyExtractor2 requires 'sentence-transformers' for full semantic capabilities.")
        print("Install with: pip install sentence-transformers")
        print("Running with keyword matching only...")
        
        # Run without semantic matching
        enhanced_tool = DecomposerTool2(use_semantic_matching=False)
        enhanced_result = enhanced_tool.analyze(demo_query)
        
        print(f"\nDetected Properties ({len(enhanced_result['detected_properties'])} - keyword matching only):")
        for prop in enhanced_result["detected_properties"][:5]:
            print(f"  • {prop['name']} (confidence: {prop['confidence']})")
    
    print("\n" + "="*80)
    print("Key Improvements in PropertyExtractor2:")
    print("- Semantic understanding via embeddings (when available)")
    print("- Expanded ML/AI ontology with modern architectures") 
    print("- Domain-aware analysis and suggestions")
    print("- Enhanced confidence scoring and explanations")
    print("- Better gap detection and next steps")
    print("- Compositional reasoning for property combinations")
    print("="*80)
