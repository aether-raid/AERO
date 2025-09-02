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
import os
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Iterable, Tuple

# Load environment variables
try:
    from dotenv import load_dotenv
    # Load from env.example first, then .env if it exists
    load_dotenv('env.example')
    load_dotenv()  # This will override with .env if present
except ImportError:
    pass  # dotenv not available, will rely on system environment variables


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

# Shape regex patterns (captures common notations like (T, C), (B, T, C))
SHAPE_PATTERNS = [
    re.compile(r"\(([^\)]*)\)")  # naive but effective for common specs
]


# -------------------------------
# Extraction engine
# -------------------------------
class PropertyExtractor:
    def __init__(self, ontology: Dict[str, Dict[str, Any]] | None = None):
        self.ontology = ontology or ONTOLOGY

    def extract(self, text: str) -> List[PropertyHit]:
        text_norm = self._normalize(text)
        hits: Dict[str, PropertyHit] = {}

        # 1) Keyword/rule hits with case-insensitive matching
        for prop, spec in self.ontology.items():
            for patt in spec.get("keywords", []):
                for m in re.finditer(patt, text_norm, flags=re.IGNORECASE | re.MULTILINE):
                    snippet = self._context_snippet(text_norm, m.start(), m.end())
                    self._add_hit(hits, prop, Evidence(snippet=snippet, source=f"keyword:{patt}", score=spec.get("score", 0.5)))

        # 2) Fuzzy semantic matching for better coverage
        self._apply_fuzzy_matching(text_norm, hits)

        # 3) Shape parsing augmentations
        shape_info = self._parse_shapes(text_norm)
        for ev in shape_info:
            self._add_hit(hits, ev[0], Evidence(snippet=ev[1], source=ev[2], score=ev[3]))

        # 4) Enhanced heuristics
        self._apply_enhanced_heuristics(text_norm, hits)

        # 5) (Optional) LLM enricher (stub – returns [])
        for prop, msg, score in self.llm_enricher(text_norm):
            self._add_hit(hits, prop, Evidence(snippet=msg, source="llm", score=score))

        # 6) Post-processing: resolve conflicts and boost related properties
        self._post_process_hits(hits)

        # Sort by confidence
        out = list(hits.values())
        out.sort(key=lambda h: h.confidence, reverse=True)
        return out

    # ---------------------------
    # Helpers
    # ---------------------------
    def _normalize(self, text: str) -> str:
        """Enhanced text normalization with better preprocessing."""
        # Basic canonicalization
        text = re.sub(r"\s+", " ", text.strip())
        
        # Handle common abbreviations and expansions
        abbreviations = {
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
        }
        
        for abbrev, expansion in abbreviations.items():
            abbrev_clean = abbrev.strip('\\b')
            text = re.sub(abbrev, f"{expansion} ({abbrev_clean})", text, flags=re.I)
        
        return text

    def _context_snippet(self, text: str, start: int, end: int, radius: int = 40) -> str:
        s = max(0, start - radius)
        e = min(len(text), end + radius)
        return text[s:e]

    def _add_hit(self, hits: Dict[str, PropertyHit], prop: str, ev: Evidence):
        if prop not in hits:
            hits[prop] = PropertyHit(name=prop, evidence=[ev])
        else:
            hits[prop].evidence.append(ev)

    def _apply_enhanced_heuristics(self, text: str, hits: Dict[str, PropertyHit]):
        """Apply enhanced heuristic rules for better detection."""
        
        # Enhanced VAE/AE detection
        if re.search(r"\b(VAE|variational\s+autoencoder|autoencoder|AE)\b", text, re.I):
            self._add_hit(hits, "reconstruction_objective", Evidence("VAE/AE implies reconstruction", "heuristic:vae", 0.6))
            self._add_hit(hits, "latent_embedding_required", Evidence("VAE/AE implies latent z", "heuristic:vae", 0.7))
        
        # Activity recognition heuristics
        if re.search(r"(human\s+)?activity\s+(recognition|classification|detection)", text, re.I):
            self._add_hit(hits, "classification_objective", Evidence("Activity recognition is classification", "heuristic:activity", 0.8))
            self._add_hit(hits, "temporal_structure", Evidence("Activity data is temporal", "heuristic:activity", 0.7))
        
        # Sensor data heuristics
        if re.search(r"wearable|sensor|accelerometer|gyroscope|IMU", text, re.I):
            self._add_hit(hits, "sensor_data", Evidence("Mentions sensor/wearable devices", "heuristic:sensor", 0.8))
            self._add_hit(hits, "temporal_structure", Evidence("Sensor data is temporal", "heuristic:sensor", 0.6))
        
        # Timing/shift invariance from context
        if re.search(r"(timing\s+)?shifts?.+(shouldn't|should\s+not|don't|doesn't).+(change|affect|impact)", text, re.I):
            self._add_hit(hits, "invariance_requirements", Evidence("Explicitly mentions shift invariance needed", "heuristic:shift-invariant", 0.9))
        
        # Variable length from context clues
        if re.search(r"(users?|recordings?|samples?).+(different|varying|variable).+(length|size|duration)", text, re.I):
            self._add_hit(hits, "variable_length_sequences", Evidence("Different lengths mentioned", "heuristic:var-length", 0.8))
        
        # Real-time needs from context
        if re.search(r"(real[- ]time|streaming|online|live|immediate)", text, re.I):
            self._add_hit(hits, "real_time_constraint", Evidence("Real-time processing mentioned", "heuristic:realtime", 0.7))
        
        # Question-asking patterns suggest need for model recommendation
        if re.search(r"(which|what).+(model|approach|method|algorithm).+(work|best|suitable|recommend)", text, re.I):
            self._add_hit(hits, "model_selection_query", Evidence("Asking for model recommendations", "heuristic:model-query", 0.6))
        
        # Performance requirements from context
        if re.search(r"(high|maximum|best|optimal).+(accuracy|performance|precision|recall)", text, re.I):
            self._add_hit(hits, "high_accuracy_required", Evidence("High performance requirements", "heuristic:performance", 0.7))
        
        # Interpretability needs
        if re.search(r"(understand|explain|interpret).+(model|decision|prediction)", text, re.I):
            self._add_hit(hits, "interpretability_required", Evidence("Interpretability mentioned", "heuristic:interpret", 0.7))
        
        # Data scarcity indicators
        if re.search(r"(limited|small|few|scarce).+(data|samples|examples|labels)", text, re.I):
            self._add_hit(hits, "few_shot_learning", Evidence("Limited data mentioned", "heuristic:few-shot", 0.7))
        
        # Contextual enhancement for better detection
        self._apply_contextual_enhancement(text, hits)
        
        # Text/NLP specific heuristics
        if re.search(r"(sentiment|emotion|opinion|text|document|clinical notes)", text, re.I):
            self._add_hit(hits, "text_data", Evidence("Text/NLP task detected", "heuristic:text", 0.8))
        
        # Multilingual detection
        if re.search(r"multilingual|multi[- ]language|cross[- ]lingual", text, re.I):
            self._add_hit(hits, "multilingual_requirement", Evidence("Multilingual requirement detected", "heuristic:multilingual", 0.8))
        
        # Variable document length
        if re.search(r"(very\s+short|very\s+long|variable.+length|different.+length).+(document|text|note)", text, re.I):
            self._add_hit(hits, "variable_document_length", Evidence("Variable document length mentioned", "heuristic:var-doc-length", 0.8))

    def _apply_contextual_enhancement(self, text: str, hits: Dict[str, PropertyHit]):
        """Apply contextual rules that consider word proximity and sentence structure."""
        
        # Split into sentences for better context analysis
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Detect classification + temporal patterns in same sentence
            if (re.search(r"classif|predict|recogni", sentence, re.I) and 
                re.search(r"time|temporal|sequenc", sentence, re.I)):
                self._add_hit(hits, "temporal_classification", Evidence(
                    f"Temporal classification in: {sentence[:60]}...", 
                    "contextual:temporal-classification", 0.8))
            
            # Detect performance concerns with specific metrics
            if re.search(r"(accuracy|precision|recall|f1).+\d+%", sentence, re.I):
                self._add_hit(hits, "specific_performance_target", Evidence(
                    f"Specific performance target: {sentence[:60]}...", 
                    "contextual:performance-target", 0.8))
            
            # Detect data preprocessing needs
            if re.search(r"(preprocess|normaliz|filter|clean).+(data|signal)", sentence, re.I):
                self._add_hit(hits, "preprocessing_required", Evidence(
                    f"Data preprocessing mentioned: {sentence[:60]}...", 
                    "contextual:preprocessing", 0.7))
        
        # Cross-sentence context analysis
        text_lower = text.lower()
        
        # Detect user study or experimental setup
        if re.search(r"users?.+(participat|study|experiment|trial)", text_lower):
            self._add_hit(hits, "user_study_context", Evidence(
                "User study context detected", "contextual:user-study", 0.6))
        
        # Detect comparison request
        if re.search(r"compar(e|ison).+(model|approach|method)", text_lower):
            self._add_hit(hits, "model_comparison_request", Evidence(
                "Model comparison requested", "contextual:comparison", 0.7))

    def _apply_fuzzy_matching(self, text: str, hits: Dict[str, PropertyHit]):
        """Apply fuzzy/semantic matching for better property detection."""
        
        # Common synonyms and related terms
        semantic_groups = {
            "classification_objective": [
                "identify", "distinguish", "categorize", "sort", "group", 
                "separate", "differentiate", "detect", "recognize"
            ],
            "temporal_structure": [
                "timeline", "chronology", "sequential", "ordered", "series",
                "progression", "temporal", "time-based"
            ],
            "sensor_data": [
                "device", "measurement", "signal", "reading", "monitoring",
                "tracking", "sensing", "instrumentation"
            ],
            "invariance_requirements": [
                "robust", "stable", "consistent", "unaffected", "immune",
                "tolerant", "insensitive", "invariant"
            ]
        }
        
        text_words = set(re.findall(r'\b\w+\b', text.lower()))
        
        for prop, synonyms in semantic_groups.items():
            matches = text_words.intersection(set(synonyms))
            if matches:
                match_count = len(matches)
                confidence = min(0.4 + 0.1 * match_count, 0.7)  # Scale with number of matches
                self._add_hit(hits, prop, Evidence(
                    f"Semantic matches: {', '.join(matches)}", 
                    "semantic:fuzzy-match", confidence))

    def _post_process_hits(self, hits: Dict[str, PropertyHit]):
        """Post-process hits to resolve conflicts and boost related properties."""
        
        # Property relationships and boosts
        property_relationships = {
            "classification_objective": {
                "boosts": ["temporal_structure", "sensor_data"],
                "conflicts": ["reconstruction_objective"],
                "boost_factor": 0.1
            },
            "sensor_data": {
                "boosts": ["temporal_structure", "variable_length_sequences"],
                "boost_factor": 0.15
            },
            "variable_length_sequences": {
                "boosts": ["temporal_structure"],
                "boost_factor": 0.1
            }
        }
        
        present_props = set(hits.keys())
        
        for prop, relations in property_relationships.items():
            if prop in present_props:
                # Boost related properties
                for boost_prop in relations.get("boosts", []):
                    if boost_prop in present_props:
                        boost_factor = relations.get("boost_factor", 0.1)
                        self._add_hit(hits, boost_prop, Evidence(
                            f"Boosted by presence of {prop}",
                            f"boost:{prop}", boost_factor))
                
                # Handle conflicts (reduce confidence)
                for conflict_prop in relations.get("conflicts", []):
                    if conflict_prop in present_props:
                        # Add evidence of conflict for transparency
                        self._add_hit(hits, conflict_prop, Evidence(
                            f"Potential conflict with {prop}",
                            f"conflict:{prop}", -0.1))

    def _parse_shapes(self, text: str) -> List[Tuple[str, str, str, float]]:
        """Return list of (property, snippet, source, score)."""
        emissions: List[Tuple[str, str, str, float]] = []

        for patt in SHAPE_PATTERNS:
            for m in patt.finditer(text):
                shape = m.group(1)
                snippet = self._context_snippet(text, m.start(), m.end())
                dims = [d.strip() for d in shape.split(',')]
                # Look for T, C naming patterns
                if any(d == 'T' for d in dims):
                    emissions.append(("temporal_structure", snippet, "regex:shape:T", 0.6))
                if any(d == 'C' for d in dims):
                    emissions.append(("fixed_channel_count", snippet, "regex:shape:C", 0.4))

                # If text mentions that T is variable/unfixed near the shape
                window = text[max(0, m.start()-120):min(len(text), m.end()+120)]
                if re.search(r"T\s+(is|can be|may be)\s+(unfixed|variable|arbitrary)", window, re.I):
                    emissions.append(("variable_length_sequences", window, "regex:shape:T-variable", 0.9))

                # If output shape equals input shape explicitly
                if re.search(r"output\s+of\s+shape\s*\(\s*T\s*,\s*C\s*\)\s*.*(preserve|same|match).*\b(T|C)\b", window, re.I):
                    emissions.append(("shape_preserving_seq2seq", window, "regex:shape:io-match", 0.8))

        # Generic phrases for IO shape match
        io_match = re.search(r"output\s+.*\b(shape|dimensions?)\b.*(same|match|preserv)", text, re.I)
        if io_match:
            emissions.append(("shape_preserving_seq2seq", self._context_snippet(text, io_match.start(), io_match.end()), "regex:io-same-shape", 0.6))

        return emissions

    def llm_enricher(self, text: str) -> Iterable[Tuple[str, str, float]]:
        """Hook for semantic enrichment via an LLM.
        Implement by calling your provider and yielding (property, rationale, score).
        Default: no-op.
        """
        return []


# -------------------------------
# Model suggestion rules
# -------------------------------
@dataclass
class Suggestion:
    name: str
    rationale: str
    notes: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "rationale": self.rationale, "notes": self.notes}


class ModelSuggester:
    def __init__(self):
        pass

    def suggest(self, props: List[PropertyHit], original_text: str = "") -> List[Suggestion]:
        names = {p.name for p in props if p.confidence >= 0.5}
        suggestions: List[Suggestion] = []

        def has(*need: str) -> bool:
            return all(n in names for n in need)

        def has_any(*options: str) -> bool:
            return any(n in names for n in options)

        # Enhanced suggestions based on detected properties
        
        # Few-shot learning scenarios
        if has("few_shot_learning", "classification_objective"):
            suggestions.append(Suggestion(
                name="Pre-trained Transformer + Few-shot Fine-tuning",
                rationale="Pre-trained models like BERT/RoBERTa provide strong representations for few-shot scenarios.",
                notes=[
                    "Use domain-adaptive pre-training if possible",
                    "Gradient-based meta-learning (MAML) for fast adaptation",
                    "Prototypical networks or matching networks for few-shot classification",
                    "Data augmentation and regularization crucial with limited data",
                ]
            ))
            suggestions.append(Suggestion(
                name="Siamese/Triplet Networks",
                rationale="Learn similarity metrics that generalize well with few examples per class.",
                notes=[
                    "Contrastive learning with positive/negative pairs",
                    "Metric learning for robust embeddings",
                    "Works well with limited labeled data",
                ]
            ))

        # Text classification with variable length documents
        if has("text_data", "classification_objective", "variable_document_length"):
            suggestions.append(Suggestion(
                name="Hierarchical Attention Networks",
                rationale="Handle variable-length documents with word and sentence-level attention mechanisms.",
                notes=[
                    "Word-level and sentence-level encoders",
                    "Attention weights provide interpretability",
                    "Good for documents with varying structure",
                ]
            ))

        # Clinical text processing
        if "clinical" in original_text.lower() and has("text_data", "classification_objective"):
            suggestions.append(Suggestion(
                name="Domain-Adapted BERT (ClinicalBERT)",
                rationale="Pre-trained on clinical text, understands medical terminology and context.",
                notes=[
                    "Use BioBERT, ClinicalBERT, or similar domain models",
                    "Fine-tune on your specific clinical domain",
                    "Consider medical entity recognition preprocessing",
                ]
            ))

        # Multilingual text classification
        if has("multilingual_requirement", "classification_objective"):
            suggestions.append(Suggestion(
                name="Multilingual BERT (mBERT) or XLM-R",
                rationale="Pre-trained on multiple languages, strong cross-lingual transfer capabilities.",
                notes=[
                    "Fine-tune on target languages/domains",
                    "Consider language-specific adaptation layers",
                    "Use cross-lingual data augmentation",
                ]
            ))

        # Activity classification with sensor data
        if has("classification_objective", "sensor_data", "temporal_structure"):
            suggestions.append(Suggestion(
                name="CNN-LSTM for Activity Recognition",
                rationale="CNNs extract local features from sensor data, LSTMs capture temporal dependencies for classification.",
                notes=[
                    "1D CNNs for multi-channel sensor signals",
                    "Bidirectional LSTM for full sequence context",
                    "Add dropout for regularization with wearable data",
                    "Consider data augmentation (time warping, noise injection)",
                ]
            ))
            suggestions.append(Suggestion(
                name="Transformer for Activity Classification",
                rationale="Self-attention captures long-range temporal dependencies in sensor sequences.",
                notes=[
                    "Positional encoding for temporal ordering",
                    "Classification token or global average pooling",
                    "Pre-training on large sensor datasets if available",
                ]
            ))

        # Variable length sequences with classification
        if has("variable_length_sequences", "classification_objective"):
            suggestions.append(Suggestion(
                name="RNN/GRU with Masking",
                rationale="Handles variable-length sequences naturally with attention masking for classification.",
                notes=[
                    "Pack padded sequences for efficiency",
                    "Use sequence masks to ignore padding",
                    "Consider sequence-to-one classification head",
                ]
            ))

        # Invariance requirements
        if has("invariance_requirements"):
            suggestions.append(Suggestion(
                name="Time-Invariant Features + Classifier",
                rationale="Extract shift-invariant features before classification to handle timing variations.",
                notes=[
                    "Use 1D convolutions with stride for downsampling",
                    "Add data augmentation with random time shifts",
                    "Consider spectral features (FFT) for shift invariance",
                    "Use attention mechanisms that focus on relative patterns",
                ]
            ))

        # Original VAE suggestions (keeping existing logic)
        if has("variable_length_sequences", "temporal_structure", "reconstruction_objective", "latent_embedding_required"):
            suggestions.append(Suggestion(
                name="Sequence VAE (RNN/GRU/LSTM)",
                rationale="Handles variable-length temporal inputs with masking/padding; VAE provides latent z and reconstruction.",
                notes=[
                    "Use padding + attention mask for batching variable T",
                    "Pack padded sequences (PyTorch) or RaggedTensors (TF)",
                    "Loss: recon (MSE/MAE) + KL; consider KL annealing",
                ]
            ))
            suggestions.append(Suggestion(
                name="Temporal Convolutional AE / TCN-VAE",
                rationale="Causal, dilation-based convolutions capture long context without recurrence; good for seq2seq reconstruction.",
                notes=[
                    "Maintain output length via same-padding",
                    "Stack dilated residual blocks",
                    "Combine with VAE bottleneck for z",
                ]
            ))
            suggestions.append(Suggestion(
                name="Transformer VAE (masked)",
                rationale="Self-attention with masks handles variable T; decoder reconstructs per time step; VAE head yields z.",
                notes=[
                    "Use causal mask if autoregressive, otherwise full mask",
                    "Positional encodings; tie input/output embeddings if discrete",
                    "Efficient attn (Performer/Flash) if T large",
                ]
            ))

        # If only reconstruction + latent, still suggest generic AE/VAE
        elif has("reconstruction_objective", "latent_embedding_required"):
            suggestions.append(Suggestion(
                name="Vanilla VAE / Denoising AE",
                rationale="Reconstruction with latent z; choose encoder/decoder per data type (1D, 2D, 3D).",
                notes=[
                    "Match conv1d/2d/3d to data domain",
                    "Consider DAE if noise robustness is required",
                ]
            ))

        # Add cross-cutting suggestions
        if "shape_preserving_seq2seq" in names:
            suggestions.append(Suggestion(
                name="Shape-preserving decoder head",
                rationale="Ensure output has same (T, C) as input.",
                notes=[
                    "Use same-padding and per-step projection to C",
                    "Avoid pooling layers that change T unless upsampled",
                ]
            ))

        if "noise_robustness" in names:
            suggestions.append(Suggestion(
                name="Denoising objective",
                rationale="Explicitly model noise with input corruption + reconstruction.",
                notes=[
                    "Gaussian/Masking noise on inputs",
                    "Add spectral loss if frequency content matters",
                ]
            ))

        # Real-time constraints
        if "real_time_constraint" in names:
            suggestions.append(Suggestion(
                name="Lightweight Architecture",
                rationale="Optimize for inference speed and model size.",
                notes=[
                    "Use depthwise separable convolutions",
                    "Model quantization and pruning",
                    "Knowledge distillation from larger models",
                    "Consider MobileNet-style architectures",
                ]
            ))

        return suggestions


# -------------------------------
# LLM-based arXiv query summarizer
# -------------------------------
def generate_arxiv_query_summary(original_query: str, detected_props: list[str]) -> str:
    """
    Combines original query with detected properties to produce a concise arXiv search string.
    Uses a lightweight LLM call (can be replaced with OpenAI API or local model inference).
    """
    import openai

    # Combine the query and properties
    context_text = f"Original query: {original_query}\nDetected properties: {', '.join(detected_props)}"
    
    # Use the exact format that works with LiteLLM proxy
    model = os.getenv('DEFAULT_MODEL', 'gemini/gemini-2.5-flash')  # dynamically selected
    api_key = os.getenv('OPENAI_API_KEY')  # load from env
    base_url = os.getenv('BASE_URL', 'https://agents.aetherraid.dev')  # load from config
    
    # Create the prompt content
    content = f"Create a concise arXiv search string (5-8 keywords) for this research query:\n\n{context_text}\n\nRespond with only the search keywords, no explanation."
    
    client = openai.OpenAI(
        api_key=api_key,
        base_url=base_url
    )
    
    # Request sent to model set on litellm proxy - using your exact format
    response = client.chat.completions.create(
        model=model,
        messages=[{"content": content, "role": "user"}]
    )
    
    # Get the response content
    summary = response.choices[0].message.content
    
    # Debug output
    print(f"DEBUG: LLM Response: '{summary}'")
    
    return summary


# -------------------------------
# Orchestration
# -------------------------------
class DecomposerTool:
    def __init__(self):
        self.extractor = PropertyExtractor()
        self.suggester = ModelSuggester()

    def analyze(self, query: str) -> Dict[str, Any]:
        props = self.extractor.extract(query)
        suggestions = self.suggester.suggest(props, query)

        # Generate arXiv query summary
        detected_prop_names = [p.name for p in props if p.confidence >= 0.5]
        arxiv_summary = None
        try:
            if os.getenv('OPENAI_API_KEY'):
                arxiv_summary = generate_arxiv_query_summary(query, detected_prop_names)
        except Exception as e:
            print(f"Warning: Failed to generate arXiv summary: {e}")

        # Prepare minimal clarifiers for common gaps
        names = {p.name for p in props}
        missing = []
        
        # More comprehensive gap detection
        if "noise_robustness" not in names and not any("noise" in p.name for p in props):
            missing.append({"name": "noise_robustness", "ask": "Are inputs noisy / do you need denoising?"})
        if "invariance_requirements" not in names and not any("invariant" in p.name for p in props):
            missing.append({"name": "invariance_requirements", "ask": "Should the model be shift/lag/scale invariant?"})
        if "real_time_constraint" not in names and not any("real_time" in p.name for p in props):
            missing.append({"name": "real_time_constraint", "ask": "Is there a latency or on-device constraint?"})
        
        # Check for conflicting or unclear objectives
        has_reconstruction = any("reconstruction" in p.name for p in props)
        has_classification = any("classification" in p.name for p in props)
        
        if not has_reconstruction and not has_classification:
            missing.append({"name": "task_objective", "ask": "What is the main task: classification, reconstruction, generation, or other?"})
        
        # Check for data specificity
        if not any(p.name in ["sensor_data", "multimodal_data"] for p in props):
            missing.append({"name": "data_type", "ask": "What type of data are you working with (sensor, text, image, audio, etc.)?"})
        
        # Performance requirements
        if not any("constraint" in p.name or "real_time" in p.name for p in props):
            missing.append({"name": "performance_requirements", "ask": "Are there specific performance requirements (accuracy, speed, memory)?"})

        return {
            "detected_properties": [p.to_dict() for p in props],
            "suggested_models": [s.to_dict() for s in suggestions],
            "arxiv_query_summary": arxiv_summary,
            "missing_or_ambiguous": missing,
            "analysis_summary": self._generate_analysis_summary(props, suggestions),
        }

    def _generate_analysis_summary(self, props: List[PropertyHit], suggestions: List[Suggestion]) -> Dict[str, Any]:
        """Generate a high-level summary of the analysis."""
        high_conf_props = [p for p in props if p.confidence >= 0.7]
        medium_conf_props = [p for p in props if 0.4 <= p.confidence < 0.7]
        
        # Determine primary task type
        task_type = "Unknown"
        if any("classification" in p.name for p in high_conf_props):
            task_type = "Classification"
        elif any("reconstruction" in p.name for p in high_conf_props):
            task_type = "Reconstruction/Autoencoder"
        elif any("generation" in p.name for p in high_conf_props):
            task_type = "Generation"
        
        # Determine data complexity
        complexity_factors = []
        if any("variable_length" in p.name for p in props):
            complexity_factors.append("Variable-length sequences")
        if any("multimodal" in p.name for p in props):
            complexity_factors.append("Multimodal data")
        if any("invariance" in p.name for p in props):
            complexity_factors.append("Invariance requirements")
        if any("real_time" in p.name for p in props):
            complexity_factors.append("Real-time constraints")
        
        return {
            "primary_task_type": task_type,
            "complexity_factors": complexity_factors,
            "high_confidence_properties": len(high_conf_props),
            "medium_confidence_properties": len(medium_conf_props),
            "total_suggestions": len(suggestions),
            "recommended_next_steps": self._get_next_steps(high_conf_props, suggestions),
        }

    def _get_next_steps(self, high_conf_props: List[PropertyHit], suggestions: List[Suggestion]) -> List[str]:
        """Generate actionable next steps based on analysis."""
        steps = []
        
        if len(high_conf_props) < 3:
            steps.append("Clarify task requirements - provide more specific details about your use case")
        
        if suggestions:
            steps.append(f"Evaluate the top {min(3, len(suggestions))} suggested model architectures")
            steps.append("Implement a baseline model first, then iterate with more complex architectures")
        
        if any("sensor" in p.name for p in high_conf_props):
            steps.append("Consider data preprocessing: normalization, filtering, feature engineering")
        
        if any("variable_length" in p.name for p in high_conf_props):
            steps.append("Design batching strategy for variable-length sequences")
        
        steps.append("Define evaluation metrics and validation strategy")
        
        return steps



if __name__ == "__main__":
    demo_query = (
  #"I want to train a multilingual sentiment classifier for clinical notes, where each document can be very short or very long, and I need the model to generalize well in few-shot scenarios with limited labeled data."
  #"I am interested in detecting anomalies in dynamic social networks, where nodes and edges appear or disappear over time, and the model should capture temporal graph structure as well as sudden changes."
  
  "Can you help me design a multilingual sentiment classifier for clinical notes that works well with limited labeled data?"
  #"Can you recommend approaches for predicting stock prices using both historical price data and news sentiment? The sequences have different lengths and I need the model to be interpretable for regulatory compliance."
  #"Need GDPR-compliant model that can be audited and doesn't store personal information."
  #"hi"
    )

    tool = DecomposerTool()
    result = tool.analyze(demo_query)

    print("\n=== Research Query ===")
    print(demo_query)

    print("\n=== arXiv Search Summary ===")
    if result["arxiv_query_summary"]:
        print(f"Suggested arXiv search: {result['arxiv_query_summary']}")
    else:
        print("(arXiv summary not available - check OpenAI API key)")

    print("\n=== Detected Properties ===")
    for prop in result["detected_properties"]:
        print(f"- {prop['name']} (confidence={prop['confidence']})")
        for ev in prop["evidence"]:
            print(f"    · evidence: '{ev['snippet']}' [{ev['source']}, score={ev['score']}]")

    print("\n=== Suggested Model Families ===")
    if result["suggested_models"]:
        for s in result["suggested_models"]:
            print(f"- {s['name']}")
            print(f"    rationale: {s['rationale']}")
            for note in s["notes"]:
                print(f"    · {note}")
    else:
        print("(No direct model suggestions, consider clarifying task further.)")

    print("\n=== Missing / Ambiguous Aspects ===")
    if result["missing_or_ambiguous"]:
        for m in result["missing_or_ambiguous"]:
            print(f"- {m['name']}: {m['ask']}")
    else:
        print("(No major gaps detected.)")
