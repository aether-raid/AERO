#!/usr/bin/env python3
"""
Machine Learning Researcher Tool
===============================

A comprehensive tool that:
1. Takes user prompts for ML research tasks
2. Decomposes tasks into properties using LLM via LiteLLM
3. Summarizes results for arXiv API search preparation
4. Searches arXiv for relevant papers

Usage:
    python ml_researcher_tool.py
"""

import os
import sys
import json
import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional
import openai
import math
import urllib.request as libreq
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
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



class MLResearcherTool:
    """Main tool for ML research task decomposition and analysis."""
    
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
    
    async def extract_properties_llm_based(self, query: str) -> List[PropertyHit]:
        """Extract properties using LLM analysis of predefined categories."""
        
        categories_list = "\n".join([f"- {category}" for category in ML_RESEARCH_CATEGORIES])
        
        content = f"""
            You are an expert machine learning researcher. Analyze the following research task and determine which of the predefined categories apply.

            Research Task: {query}

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
            - 1–2 sentences, specific and non-generic, referencing how the evidence meets the category’s definition.
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

        try:
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
            import json
            try:
                # Remove any markdown formatting
                if llm_response.startswith("```json"):
                    llm_response = llm_response[7:]
                if llm_response.endswith("```"):
                    llm_response = llm_response[:-3]
                llm_response = llm_response.strip()
                
                properties_data = json.loads(llm_response)
                
                # Convert to PropertyHit objects
                property_hits = []
                for prop_data in properties_data:
                    evidence = [Evidence(
                        snippet=prop_data.get("evidence", ""),
                        source=f"llm_analysis:{prop_data['category']}",
                        score=prop_data.get("confidence", 0.5)
                    )]
                    
                    property_hits.append(PropertyHit(
                        name=prop_data["category"],
                        evidence=evidence
                    ))
                
                return property_hits
                
            except json.JSONDecodeError as e:
                print(f"⚠️  Failed to parse LLM JSON response: {e}")
                print(f"Raw response: {llm_response}")
                return []
        
        except Exception as e:
            print(f"❌ LLM property extraction failed: {str(e)}")
            return []
        
        
        
        
    
    async def decompose_task_with_llm(self, prompt: str) -> Dict[str, Any]:
        """Use LLM to decompose the task and identify additional properties."""
        content = f"""
            You are an expert machine learning researcher. Analyze the following research task and decompose it into key properties and characteristics.

            Task: {prompt}

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

        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"content": content, "role": "user"}]
                )
            )
            
            return {
                "llm_analysis": response.choices[0].message.content,
                "model_used": self.model,
                "tokens_used": response.usage.total_tokens if response.usage else "unknown"
            }
        
        except Exception as e:
            return {
                "error": f"LLM decomposition failed: {str(e)}",
                "llm_analysis": None
            }
            
            
            
    
    def generate_arxiv_search_summary(self, prompt: str, detected_properties: List[PropertyHit], llm_analysis: Dict[str, Any]) -> str:
        """Generate a summary suitable for arXiv API search."""
        
        # Extract key properties with high confidence
        high_confidence_props = [prop for prop in detected_properties if prop.confidence > 0.7]
        prop_names = [prop.name for prop in high_confidence_props]
        
        content = f"""
            Based on the following machine learning research task analysis, generate ONE concise arXiv API search query (exactly 3 terms, separated by forward slashes).

            Original Task: {prompt}

            Detected Categories: {', '.join(prop_names)}

            Detailed Analysis: {llm_analysis.get('llm_analysis', 'Not available')}

            Rules for constructing the query:
            - EXACTLY 4 terms, separated by "/" (no quotes, no extra spaces).
            - Include:
            1) a MODEL keyword (e.g., transformer, ViT, DETR, RT-DETR, Deformable DETR, YOLOS),
            2) the TASK (e.g., object detection, segmentation),
            3) a DEPLOYMENT/CONSTRAINT or TOOLING term if present (e.g., real-time, edge deployment, TensorRT, quantization, INT8).
            - Prefer task-specific + model-specific terms over generic ones.
            - Avoid vague terms like "deep learning" or "machine learning" unless nothing better fits.
            - Prefer dataset/benchmark anchors (e.g., KITTI, nuScenes, Waymo) OVER broad domain words (e.g., autonomous vehicles). Use the domain ONLY if it is essential and not overly broad.
            - If computer vision is relevant, make the TASK a CV term (e.g., object detection, instance segmentation).
            - Do NOT include arXiv category labels (cs.CV, cs.LG) in the query terms.
            - Return ONLY the query string (no explanation, no punctuation besides "/").
            4) a DOMAIN or APPLICATION term if relevant (e.g., medical imaging, remote sensing, autonomous vehicles).

            Good examples:
            - transformer/object detection/real-time
            - RT-DETR/object detection/TensorRT
            - Deformable DETR/object detection/KITTI
            - vision transformer/object detection/edge deployment
        """


        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                messages=[{"content": content, "role": "user"}]
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            # Fallback to simple keyword extraction with slashes
            keywords = []
            if "neural" in prompt.lower() or "deep" in prompt.lower():
                keywords.append("neural network")
            if "time series" in prompt.lower() or "temporal" in prompt.lower():
                keywords.append("time series")
            if "classification" in prompt.lower():
                keywords.append("classification")
            if "clustering" in prompt.lower():
                keywords.append("clustering")
            if "anomaly detection" in prompt.lower():
                keywords.append("anomaly detection")
            if "autoencoder" in prompt.lower():
                keywords.append("autoencoder")
            
            return "/".join(keywords) if keywords else "machine learning"
    
    import re, math, time

    async def _score_paper_relevance(self, paper_title: str, paper_content: str, original_query: str) -> float:
        """LLM relevance score in [1.0, 10.0]. Returns a float only."""
        # Keep prompts lean; truncate huge inputs to control tokens
        MAX_CHARS = 8000
        title = (paper_title or "").strip()[:512] or "<untitled>"
        content = (paper_content or "").strip()[:MAX_CHARS]
        query = (original_query or "").strip()[:2000]

        user_prompt = f"""
        You are an expert ML librarian. Score how relevant the paper is to the user's research query on a 1–10 scale.

        Return ONLY a number between 1.0 and 10.0 (one decimal). No words, no JSON, no symbols.

        Research query:
        \"\"\"{original_query}\"\"\"

        Paper title:
        \"\"\"{title}\"\"\"

        Paper content:
        \"\"\"{content}\"\"\"

        Scoring rubric (weighting):
        - task_match (40%): does the paper directly address the task(s)?
        - method_match (30%): overlap with architectures/approaches or close variants.
        - constraint_match (20%): matches constraints/tooling/datasets/hardware (e.g., real-time, edge, TensorRT, INT8).
        - evidence_match (10%): concrete signals (benchmarks, datasets, metrics, ablations, deployment notes).

        Compute: score = round((0.40*task + 0.30*method + 0.20*constraint + 0.10*evidence)*10, 1).
        If clearly unrelated (all four < 0.15), output 1.0.
        Clip to [1.0, 10.0].

        Output: ONLY the final number (e.g., 8.7).
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
        import re, math, time, asyncio
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

    async def _score_paper_relevance2(self, paper_title: str, paper_content: str, original_query: str) -> float:
        """Score paper relevance using cosine similarity with comprehensive feature extraction."""
        import re
        from collections import Counter
        from math import sqrt
        
        def preprocess_text(text: str) -> str:
            """Clean and normalize text for analysis."""
            if not text:
                return ""
            # Convert to lowercase and remove special characters
            text = re.sub(r'[^\w\s]', ' ', text.lower())
            # Remove extra whitespace
            text = ' '.join(text.split())
            return text
        
        def extract_ml_keywords(text: str) -> set:
            """Extract ML-specific keywords and concepts."""
            ml_patterns = {
                # Core ML concepts
                'machine_learning', 'deep_learning', 'neural_network', 'artificial_intelligence',
                'supervised_learning', 'unsupervised_learning', 'reinforcement_learning',
                
                # Architectures
                'transformer', 'attention', 'bert', 'gpt', 'lstm', 'gru', 'rnn', 'cnn', 'convolution',
                'resnet', 'vgg', 'inception', 'mobilenet', 'efficientnet', 'densenet',
                'unet', 'autoencoder', 'variational_autoencoder', 'gan', 'generative',
                'vision_transformer', 'vit', 'detr', 'yolo', 'faster_rcnn', 'mask_rcnn',
                
                # Tasks
                'classification', 'regression', 'clustering', 'detection', 'segmentation',
                'object_detection', 'image_classification', 'semantic_segmentation', 
                'instance_segmentation', 'face_recognition', 'natural_language_processing',
                'computer_vision', 'speech_recognition', 'recommendation', 'anomaly_detection',
                
                # Domains
                'autonomous_vehicle', 'medical_imaging', 'robotics', 'finance', 'healthcare',
                'self_driving', 'autonomous_driving', 'medical', 'biomedical', 'clinical',
                'remote_sensing', 'satellite', 'surveillance', 'security',
                
                # Constraints/Deployment
                'real_time', 'edge_computing', 'mobile', 'embedded', 'quantization',
                'pruning', 'tensorrt', 'onnx', 'optimization', 'inference', 'deployment',
                'lightweight', 'efficient', 'low_latency', 'edge_deployment',
                
                # Data types
                'image', 'video', 'text', 'audio', 'speech', 'time_series', 'tabular',
                'multimodal', 'sensor_data', 'lidar', 'radar', 'camera',
                
                # Datasets/Benchmarks
                'imagenet', 'coco', 'kitti', 'nuscenes', 'waymo', 'cityscapes',
                'pascal_voc', 'mnist', 'cifar', 'glue', 'squad'
            }
            
            found_keywords = set()
            text_lower = text.lower()
            
            for keyword in ml_patterns:
                # Check for exact match and variations
                keyword_variants = [
                    keyword,
                    keyword.replace('_', ' '),
                    keyword.replace('_', '-'),
                    keyword.replace('_', '')
                ]
                
                for variant in keyword_variants:
                    if variant in text_lower:
                        found_keywords.add(keyword)
                        break
            
            return found_keywords
        
        def extract_technical_terms(text: str) -> set:
            """Extract technical terms using patterns."""
            technical_terms = set()
            text_lower = text.lower()
            
            # Architecture patterns
            arch_patterns = [
                r'\b(?:transformer|attention|bert|gpt|lstm|gru|rnn|cnn)\b',
                r'\b(?:resnet|vgg|inception|mobilenet|efficientnet)\b',
                r'\b(?:yolo|detr|faster.?rcnn|mask.?rcnn)\b'
            ]
            
            for pattern in arch_patterns:
                matches = re.findall(pattern, text_lower)
                technical_terms.update(matches)
            
            return technical_terms
        
        def calculate_cosine_similarity(vec1: dict, vec2: dict) -> float:
            """Calculate cosine similarity between two feature vectors."""
            # Get all unique features
            all_features = set(vec1.keys()) | set(vec2.keys())
            
            if not all_features:
                return 0.0
            
            # Calculate dot product and magnitudes
            dot_product = sum(vec1.get(f, 0) * vec2.get(f, 0) for f in all_features)
            
            mag1 = sqrt(sum(v**2 for v in vec1.values()))
            mag2 = sqrt(sum(v**2 for v in vec2.values()))
            
            if mag1 == 0 or mag2 == 0:
                return 0.0
            
            return dot_product / (mag1 * mag2)
        
        def create_feature_vector(text: str, weight: float = 1.0) -> dict:
            """Create weighted feature vector from text."""
            if not text:
                return {}
            
            # Extract different types of features
            processed_text = preprocess_text(text)
            ml_keywords = extract_ml_keywords(text)
            tech_terms = extract_technical_terms(text)
            
            # Create feature vector with different weights
            features = {}
            
            # ML keywords (high weight)
            for keyword in ml_keywords:
                features[f"ml_{keyword}"] = weight * 3.0
            
            # Technical terms (medium weight)
            for term in tech_terms:
                features[f"tech_{term}"] = weight * 2.0
            
            # Important words (lower weight)
            important_words = [
                'learning', 'neural', 'network', 'model', 'algorithm',
                'training', 'optimization', 'performance', 'accuracy',
                'detection', 'classification', 'segmentation', 'recognition'
            ]
            
            for word in important_words:
                if word in processed_text:
                    count = processed_text.count(word)
                    features[f"word_{word}"] = weight * count * 1.0
            
            return features
        
        # Clean inputs
        title = (paper_title or "").strip()
        content = (paper_content or "").strip()[:4000]  # Limit content length
        query = (original_query or "").strip()
        
        if not title and not content:
            return 1.0
        
        # Create feature vectors with different weights
        query_vector = create_feature_vector(query, weight=1.0)
        title_vector = create_feature_vector(title, weight=2.0)  # Title more important
        content_vector = create_feature_vector(content, weight=1.0)
        
        # Combine paper vectors (title + content)
        paper_vector = {}
        for vec in [title_vector, content_vector]:
            for feature, value in vec.items():
                paper_vector[feature] = paper_vector.get(feature, 0) + value
        
        # Calculate cosine similarity
        similarity = calculate_cosine_similarity(query_vector, paper_vector)
        
        # Apply scoring rules with domain knowledge
        def apply_scoring_rules(sim_score: float, title: str, content: str, query: str) -> float:
            """Apply domain-specific scoring rules to adjust similarity."""
            base_score = sim_score * 10.0  # Scale to 1-10
            
            # Boost for exact task matches
            query_lower = query.lower()
            title_lower = title.lower()
            content_lower = content.lower()
            
            # Task matching boost
            task_keywords = ['detection', 'classification', 'segmentation', 'recognition', 'tracking']
            for task in task_keywords:
                if task in query_lower and task in (title_lower + ' ' + content_lower):
                    base_score += 1.0
                    break
            
            # Architecture matching boost
            arch_keywords = ['transformer', 'cnn', 'lstm', 'bert', 'yolo', 'detr']
            for arch in arch_keywords:
                if arch in query_lower and arch in (title_lower + ' ' + content_lower):
                    base_score += 0.8
                    break
            
            # Domain matching boost
            domain_keywords = ['autonomous', 'medical', 'robotics', 'vision', 'nlp']
            for domain in domain_keywords:
                if domain in query_lower and domain in (title_lower + ' ' + content_lower):
                    base_score += 0.6
                    break
            
            # Constraint matching boost
            constraint_keywords = ['real.?time', 'edge', 'mobile', 'efficient', 'lightweight']
            for constraint in constraint_keywords:
                if re.search(constraint, query_lower) and re.search(constraint, title_lower + ' ' + content_lower):
                    base_score += 0.5
                    break
            
            # Penalty for completely unrelated domains
            if ('medical' in query_lower and 'autonomous' in title_lower) or \
               ('autonomous' in query_lower and 'medical' in title_lower):
                base_score *= 0.3
            
            # Ensure score is in valid range
            return max(1.0, min(10.0, base_score))
        
        # Apply scoring rules and return final score
        final_score = apply_scoring_rules(similarity, title, content, query)
        
        return round(final_score, 1)
    async def _score_paper_relevance3(self, paper_title: str, paper_content: str, original_query: str) -> float:
        """
        Relevance = cosine(query_embedding, paper_embedding), mapped to [1.0, 10.0].
        No anchor bonuses, no handcrafted keyword bumps—purely based on the query and paper text.

        Fallbacks:
        - If sentence-transformers is unavailable, uses TF-IDF cosine.
        - If inputs are empty, returns 1.0.
        """
        import re
        import numpy as np

        # -------- helpers --------
        def _norm(s: str, n: int) -> str:
            s = (s or "").replace("\x00", " ")
            s = re.sub(r"\s+", " ", s).strip()
            return s[:n]

        def _build_doc(title: str, content: str) -> str:
            # Title is often highly informative—weight it implicitly by repetition
            title = _norm(title, 512) or "<untitled>"
            content = _norm(content, 40000)
            return ((title + " ") * 3 + " " + content).strip()

        def _map_to_score(sim: float) -> float:
            # Ensure similarity in [0,1], then map to [1.0, 10.0]
            sim = float(np.clip(sim, 0.0, 1.0))
            return round(1.0 + 9.0 * sim, 1)

        # -------- inputs --------
        title = paper_title or ""
        content = paper_content or ""
        query = original_query or ""

        if not query.strip() or (not title.strip() and not content.strip()):
            return 1.0

        doc_text = _build_doc(title, content)

        # -------- try dense embeddings (sentence-transformers) --------
        try:
            from sentence_transformers import SentenceTransformer
            # cache the model on the instance to avoid reloading
            model = getattr(self, "_st_model", None)
            if model is None:
                # bge-small: fast & good; swap to 'BAAI/bge-base-en-v1.5' for a bit more accuracy
                model = self._st_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

            q_vec = model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
            d_vec = model.encode([doc_text], normalize_embeddings=True, show_progress_bar=False)[0]
            cosine = float(np.dot(q_vec, d_vec))  # normalized => cosine
            return _map_to_score(cosine)

        except Exception:
            # -------- fallback: TF-IDF cosine --------
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.metrics.pairwise import cosine_similarity

                vec = TfidfVectorizer(
                    stop_words="english",
                    ngram_range=(1, 2),
                    max_df=0.95,
                    sublinear_tf=True,
                )
                X = vec.fit_transform([query, doc_text])
                cosine = float(cosine_similarity(X[0], X[1])[0, 0])  # [0,1]
                return _map_to_score(cosine)
            except Exception:
                # Last-resort conservative default
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
                paper.get('summary', ''),  # Use summary instead of content
                original_query
            )
            
            paper['relevance_score'] = relevance_score
            print(f"   📊 Cosine Similarity Score: {relevance_score:.1f}/10.0")
            return paper
        
        # Run all scoring tasks concurrently
        scoring_tasks = [score_paper(i, paper) for i, paper in enumerate(papers, 1)]
        scored_papers = await asyncio.gather(*scoring_tasks)
        
        # Sort by relevance score (highest first)
        ranked_papers = sorted(scored_papers, key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        ranked_top5 = sorted(
            scored_papers,
            key=lambda x: (x.get('relevance_score') is None, x.get('relevance_score', 0)),
            reverse=True
        )[:5]
                
        print(f"\n✅ Papers ranked by cosine similarity to: '{original_query}'")
        return ranked_top5

    def _clean_text_for_json(self, text: str) -> str:
        """Clean text to remove invalid Unicode characters that cause JSON serialization issues."""
        if not isinstance(text, str):
            return str(text)
        
        # Remove surrogate characters and other problematic Unicode
        import unicodedata
        
        # First, try to handle common issues
        try:
            # Remove or replace surrogate pairs
            cleaned = text.encode('utf-8', errors='ignore').decode('utf-8')
            
            # Normalize Unicode characters
            cleaned = unicodedata.normalize('NFKD', cleaned)
            
            # Remove any remaining control characters except common ones
            cleaned = ''.join(char for char in cleaned if unicodedata.category(char) != 'Cc' or char in '\n\r\t')
            
            return cleaned
        except Exception:
            # If all else fails, keep only ASCII characters
            return ''.join(char for char in text if ord(char) < 128)

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

    async def search_arxiv(self, search_query: str, original_prompt: str, max_results: int = 20) -> Dict[str, Any]:
        """Search arXiv for papers using the formatted search query."""
        
        print(f"\n🔍 SEARCHING arXiv: {search_query}")
        print("=" * 80)
        
        # Format the search query
        formatted_query = format_search_string(search_query)
        print(f"Formatted query: {formatted_query}")
        
        # Build the URL
        url = f"http://export.arxiv.org/api/query?search_query={formatted_query}&start=0&max_results={max_results}"
        
        try:
            with libreq.urlopen(url) as response:
                xml_data = response.read()
            
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
                
                print(f"� Stage 1: Extracting basic info for {len(entries)} papers...")
                
                # Stage 1: Extract basic info (title, abstract, metadata) without downloading PDFs
                papers = []
                for i, entry in enumerate(entries, 1):
                    paper_info = self._extract_basic_paper_info(entry, ns, i)
                    papers.append(paper_info)
                    print(f"✅ Basic info extracted for paper #{i}: {paper_info['title'][:50]}...")
                
                print(f"\n🎯 Stage 2: Ranking papers by relevance (based on title + abstract)...")
                # Stage 2: Rank papers by relevance using title + abstract only
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
                   
                
                return {
                    "search_successful": True,
                    "total_results": str(total_results),
                    "papers_returned": len(papers),
                    "papers": papers,
                    "formatted_query": formatted_query,
                    "original_query": search_query
                }
            else:
                print("No papers found")
                return {
                    "search_successful": False,
                    "total_results": "0",
                    "papers_returned": 0,
                    "papers": [],
                    "formatted_query": formatted_query,
                    "original_query": search_query
                }
                    
        except Exception as e:
            print(f"Error searching arXiv: {e}")
            return {
                "search_successful": False,
                "error": f"Search error: {e}",
                "total_results": "0",
                "papers_returned": 0,
                "papers": [],
                "formatted_query": "",
                "original_query": search_query
            }
    
    def suggest_models_from_arxiv(self, prompt: str, arxiv_results: Dict[str, Any], detected_categories: List[PropertyHit], detailed_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Use LLM to suggest suitable models based on arXiv search results and analysis."""
        
        print(f"\n🤖 Step 5: Analyzing papers and suggesting suitable models...")
        
        # Prepare evidence from arXiv papers
        papers_evidence = ""
        if arxiv_results.get("search_successful") and arxiv_results.get("papers"):
            papers_evidence = "\n--- arXiv Papers Found ---\n"
            for i, paper in enumerate(arxiv_results["papers"], 1):  # Use top 3 papers
                papers_evidence += f"""
                    Paper {i}: {paper["title"]}
                    Published: {paper["published"]}
                    URL: {paper["url"]}
                    ---
                """
        else:
            papers_evidence = "\n--- No arXiv Papers Found ---\nNo relevant papers were found in the search, so recommendations will be based on general ML knowledge.\n"
        
        # Prepare detected categories
        categories_text = ", ".join([prop.name for prop in detected_categories])
        
        # Create comprehensive prompt for model suggestion
        content = f"""
            You are an expert machine learning researcher and architect. Based on the following comprehensive analysis, suggest the most suitable machine learning models/architectures for this task and provide detailed justification.

            ## Original Task
            {prompt}

            ## Detected ML Categories
            {categories_text}

            ## Detailed Analysis Summary
            {detailed_analysis.get('llm_analysis', 'Analysis not available')[:1000]}...

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

        try:
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
            
            return {
                "suggestions_successful": True,
                "model_suggestions": model_suggestions,
                "model_used": self.model,
                "tokens_used": response.usage.total_tokens if response.usage else "unknown",
                "papers_analyzed": len(arxiv_results.get("papers", [])),
                "categories_considered": len(detected_categories)
            }
        
        except Exception as e:
            error_msg = f"Model suggestion failed: {str(e)}"
            print(f"❌ {error_msg}")
            return {
                "suggestions_successful": False,
                "error": error_msg,
                "model_suggestions": None
            }
    
    def suggest_open_problems(self, prompt: str, detected_categories: List[PropertyHit], 
                             detailed_analysis: Dict[str, Any], arxiv_results: Dict[str, Any], 
                             model_suggestions: Dict[str, Any]) -> Dict[str, Any]:
        """Use LLM to suggest open problem statements based on research evidence and model suggestions."""
        
        print("\n🔬 Step 6: Identifying open research problems...")
        
        # Extract relevant information
        category_names = [prop.name for prop in detected_categories]
        papers_info = ""
        
        if arxiv_results.get("papers"):
            papers_info = "\n".join([
                f"- {paper['title']}"
                for paper in arxiv_results["papers"][:3]  # Limit to first 3 papers
            ])
        else:
            papers_info = "No specific papers found in arXiv search."
        
        content = f"""
You are an expert machine learning researcher specializing in identifying open research problems and future research directions.

**Original Research Task:** {prompt}

**Detected Categories:** {', '.join(category_names)}

**Detailed Analysis:** {detailed_analysis.get('llm_analysis', 'Not available')}

**Suggested Models:** {model_suggestions.get('model_suggestions', 'Not available')}

**Research Papers Found:**
{papers_info}

Based on all this evidence, identify and suggest **open research problems** and **future research directions** that are:

1. **Specific to the suggested models** - Problems that specifically relate to the models recommended in Step 5
2. **Grounded in current research** - Based on gaps or limitations mentioned in the arXiv papers
3. **Technically feasible** - Realistic problems that can be addressed with current technology
4. **Impactful** - Would advance the field if solved

For each open problem, provide:

1. **Problem Statement** - Clear, concise description of the research gap
2. **Connection to Suggested Models** - How this problem specifically relates to the recommended models
3. **Research Evidence** - What evidence from the papers supports this being an open problem
4. **Potential Impact** - Why solving this would be valuable
5. **Suggested Approach** - High-level methodology for addressing this problem
6. **Timeline Estimate** - Realistic timeframe for investigation (e.g., 6 months, 1-2 years)

**Categories of open problems to consider:**
- Model architecture improvements
- Training methodology enhancements  
- Evaluation metric development
- Dataset limitations and needs
- Computational efficiency challenges
- Interpretability and explainability gaps
- Real-world deployment issues
- Scalability concerns

Provide 3-5 well-justified open problems that would make meaningful contributions to the field.

Format your response as a structured analysis with clear sections for each problem.
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"content": content, "role": "user"}]
            )
            
            open_problems = response.choices[0].message.content
            
            # Print readable summary
            print("✅ Open research problems identified")
            print("\n" + "=" * 80)
            print("🔬 OPEN RESEARCH PROBLEMS & FUTURE DIRECTIONS")
            print("=" * 80)
            print(open_problems)
            print("=" * 80)
            
            return {
                "problems_identification_successful": True,
                "open_problems": open_problems,
                "model_used": self.model,
                "tokens_used": response.usage.total_tokens if response.usage else "unknown",
                "papers_analyzed": len(arxiv_results.get("papers", [])),
                "categories_considered": len(detected_categories),
                "based_on_models": model_suggestions.get("suggestions_successful", False)
            }
        
        except Exception as e:
            error_msg = f"Open problems identification failed: {str(e)}"
            print(f"❌ {error_msg}")
            return {
                "problems_identification_successful": False,
                "error": error_msg,
                "open_problems": None
            }
    
    def clean_text_for_encoding(self, text: str) -> str:
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
    
    def generate_comprehensive_research_plan(self, prompt: str, detected_categories: List[PropertyHit], 
                                           detailed_analysis: Dict[str, Any], arxiv_results: Dict[str, Any], 
                                           model_suggestions: Dict[str, Any], open_problems: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a comprehensive, time-bound research plan synthesizing all previous analysis."""
        
        print("\n📋 Step 7: Generating comprehensive research plan...")
        
        # Extract and clean all relevant information
        category_names = [prop.name for prop in detected_categories]
        papers_info = ""
        
        if arxiv_results.get("papers"):
            papers_info = "\n".join([
                f"- {paper['title']}"
                for paper in arxiv_results["papers"][:3]
            ])
        else:
            papers_info = "No specific papers found in arXiv search."
        
        # Clean all text inputs to avoid encoding issues
        clean_prompt = self.clean_text_for_encoding(prompt)
        clean_detailed_analysis = self.clean_text_for_encoding(str(detailed_analysis.get('llm_analysis', 'Not available')))
        clean_model_suggestions = self.clean_text_for_encoding(str(model_suggestions.get('model_suggestions', 'Not available')))
        clean_open_problems = self.clean_text_for_encoding(str(open_problems.get('open_problems', 'Not available')))
        clean_papers_info = self.clean_text_for_encoding(papers_info)
        
        content = f"""
You are an expert research project manager and machine learning researcher. Your task is to synthesize ALL the previous analysis into a comprehensive, actionable, and time-bound research plan.

**SYNTHESIS INPUTS:**

**Original Task:** {clean_prompt}

**Detected ML Categories:** {', '.join(category_names)}

**Expert Technical Analysis:** {clean_detailed_analysis[:2000]}...

**Recommended Models:** {clean_model_suggestions[:1500]}...

**Open Research Problems:** {clean_open_problems[:1500]}...

**Current Literature (arXiv papers found):**
{clean_papers_info}

**YOUR TASK:**
Create a comprehensive research plan that synthesizes ALL the above information into an actionable roadmap. The plan should be:

1. **COMPREHENSIVE** - Incorporate insights from all previous steps
2. **TIME-BOUND** - Provide realistic timelines for each phase
3. **ACTIONABLE** - Clear next steps and deliverables
4. **STRUCTURED** - Organized phases with dependencies
5. **RESEARCH-ORIENTED** - Focus on advancing knowledge, not just implementation

**REQUIRED STRUCTURE:**

## EXECUTIVE SUMMARY
- Brief overview of the research objectives
- Key insights from the synthesis
- Expected timeline and outcomes

## PHASE 1: FOUNDATION (Months 1-2)
- Literature review and background research
- Dataset acquisition and preparation
- Initial baseline implementations
- Specific tasks and deliverables

## PHASE 2: CORE DEVELOPMENT (Months 3-5)
- Model development and experimentation
- Based on the recommended models from Step 5
- Address specific technical challenges identified
- Specific tasks and deliverables

## PHASE 3: ADVANCED RESEARCH (Months 6-8)
- Tackle open problems identified in Step 6
- Novel contributions and innovations
- Comparative analysis and evaluation
- Specific tasks and deliverables

## PHASE 4: VALIDATION & DISSEMINATION (Months 9-12)
- Comprehensive evaluation and testing
- Real-world validation
- Paper writing and publication
- Specific tasks and deliverables

## RISK MITIGATION
- Potential challenges and mitigation strategies
- Alternative approaches if primary methods fail

## RESOURCE REQUIREMENTS
- Computational resources needed
- Data requirements
- Potential collaborations

## SUCCESS METRICS
- How to measure progress in each phase
- Key performance indicators

Provide a detailed, well-structured research plan that a graduate student or researcher could follow.
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"content": content, "role": "user"}]
            )
            
            # Clean the response to avoid encoding issues
            research_plan = self.clean_text_for_encoding(response.choices[0].message.content)
            
            # Print readable summary
            print("✅ Comprehensive research plan generated")
            print("\n" + "=" * 80)
            print("📋 COMPREHENSIVE RESEARCH PLAN")
            print("=" * 80)
            print(research_plan)
            print("=" * 80)
            
            return {
                "research_plan_successful": True,
                "research_plan": research_plan,
                "model_used": self.model,
                "tokens_used": response.usage.total_tokens if response.usage else "unknown",
                "synthesis_inputs": {
                    "categories_count": len(detected_categories),
                    "papers_analyzed": len(arxiv_results.get("papers", [])),
                    "models_suggested": model_suggestions.get("suggestions_successful", False),
                    "problems_identified": open_problems.get("problems_identification_successful", False)
                }
            }
        
        except Exception as e:
            error_msg = f"Research plan generation failed: {str(e)}"
            print(f"❌ {error_msg}")
            return {
                "research_plan_successful": False,
                "error": error_msg,
                "research_plan": None
            }
    
    async def analyze_research_task(self, prompt: str) -> Dict[str, Any]:
        """Main method to analyze a research task."""
        print(f"🔍 Analyzing research task: {prompt}")
        print("=" * 50)
        
        # Steps 1 & 2: Run property extraction and task decomposition concurrently
        print("🤖 Steps 1 & 2: Running property extraction and task decomposition concurrently...")
        
        # Create concurrent tasks
        property_task = self.extract_properties_llm_based(prompt)
        decomposition_task = self.decompose_task_with_llm(prompt)
        
        # Run both tasks concurrently
        llm_properties, llm_analysis = await asyncio.gather(property_task, decomposition_task)
        print("Task 1 Complete")
        
        print(f"✅ Property extraction completed: Found {len(llm_properties)} properties")
        for prop in llm_properties:
            print(f"  - {prop.name}: {prop.confidence:.2f} confidence")
        
        if "error" in llm_analysis:
            print(f"❌ Task decomposition failed: {llm_analysis['error']}")
        else:
            print("✅ Task 2 :decomposition completed")
            
            wantsee=False
            if wantsee:
                # Display the detailed analysis in readable format
                print("\n" + "=" * 80)
                print("📋 DETAILED ANALYSIS")
                print("=" * 80)
                
                analysis_text = llm_analysis.get('llm_analysis', 'No analysis available')
                
                # Try to parse and format JSON if it's JSON, otherwise just print as text
                try:
                    # Remove markdown formatting if present
                    if analysis_text.startswith("```json"):
                        analysis_text = analysis_text[7:]
                    if analysis_text.endswith("```"):
                        analysis_text = analysis_text[:-3]
                    analysis_text = analysis_text.strip()
                    
                    # Try to parse as JSON for better formatting
                    analysis_json = json.loads(analysis_text)
                    
                    # Pretty print the analysis
                    def print_analysis_section(key, value, indent=0):
                        prefix = "  " * indent  
                        if isinstance(value, dict):
                            print(f"{prefix}📌 {key.replace('_', ' ').title()}:")
                            for sub_key, sub_value in value.items():
                                print_analysis_section(sub_key, sub_value, indent + 1)
                        elif isinstance(value, list):
                            print(f"{prefix}📌 {key.replace('_', ' ').title()}:")
                            for i, item in enumerate(value, 1):
                                if isinstance(item, dict):
                                    print(f"{prefix}  {i}. {item}")
                                else:
                                    print(f"{prefix}  {i}. {item}")
                        else:
                            if key.lower() in ['type', 'category', 'explanation', 'description']:
                                print(f"{prefix}📌 {key.replace('_', ' ').title()}: {value}")
                            else:
                                print(f"{prefix}• {key.replace('_', ' ').title()}: {value}")
                    
                    # Print each main section
                    for main_key, main_value in analysis_json.items():
                        if main_key == 'analysis' and isinstance(main_value, dict):
                            for section_key, section_value in main_value.items():
                                print(f"\n🔍 {section_key.replace('_', ' ').upper()}")
                                print("-" * 60)
                                print_analysis_section(section_key, section_value, 0)
                        else:
                            print(f"\n🔍 {main_key.replace('_', ' ').upper()}")
                            print("-" * 60)
                            print_analysis_section(main_key, main_value, 0)
                            
                except (json.JSONDecodeError, AttributeError):
                    # If it's not JSON or parsing fails, just print as formatted text
                    print(analysis_text)
                
            print("\n" + "=" * 80)
        
        # Step 3: Generate arXiv search summary
        print("\n📚 Step 3: Generating arXiv search summary...")
        arxiv_search_query = self.generate_arxiv_search_summary(prompt, llm_properties, llm_analysis)
        print(f"Generated search query: '{arxiv_search_query}'")
        
        # Step 4: Search arXiv for relevant papers
        print("\n📖 Step 4: Searching arXiv for relevant papers...")
        arxiv_results = await self.search_arxiv(arxiv_search_query, prompt,  max_results=15)
        
        # Step 5: Suggest models based on all evidence
        model_suggestions = self.suggest_models_from_arxiv(prompt, arxiv_results, llm_properties, llm_analysis)
        
        # Step 6: Identify open research problems based on all evidence and model suggestions
        #open_problems = self.suggest_open_problems(prompt, llm_properties, llm_analysis, arxiv_results, model_suggestions)
        
        # Step 7: Generate comprehensive research plan synthesizing all analysis
        #research_plan = self.generate_comprehensive_research_plan(prompt, llm_properties, llm_analysis, arxiv_results, model_suggestions, open_problems)
        
        # Compile results
        results = {
            "original_prompt": prompt,
            "detected_categories": [prop.to_dict() for prop in llm_properties],
            "detailed_analysis": llm_analysis,
            "arxiv_search_query": arxiv_search_query,
            "arxiv_results": arxiv_results,
            "model_suggestions": model_suggestions,
            #"open_problems": open_problems,
            #"research_plan": research_plan,
            "summary": {
                "total_categories_detected": len(llm_properties),
                "high_confidence_categories": len([p for p in llm_properties if p.confidence > 0.7]),
                "detailed_analysis_successful": "error" not in llm_analysis,
                "arxiv_search_successful": arxiv_results.get("search_successful", False),
                "papers_found": arxiv_results.get("papers_returned", 0),
                "model_suggestions_successful": model_suggestions.get("suggestions_successful", False),
                #"open_problems_successful": open_problems.get("problems_identification_successful", False),
                #"research_plan_successful": research_plan.get("research_plan_successful", False)
            }
        }
        
        return results
    
    async def interactive_mode(self):
        """Run the tool in interactive mode."""
        print("🔬 ML Research Task Analyzer")
        print("=" * 30)
        print("Enter your machine learning research task or question.")
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
                print("\n" + "=" * 50)
                print("📊 ANALYSIS RESULTS")
                print("=" * 50)
            
                
                # Save results to file
                timestamp = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"ml_research_analysis_{timestamp}.json"
                
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                
                print(f"💾 Full results saved to: {filename}")
                print("\n" + "=" * 50 + "\n")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ An error occurred: {str(e)}")


async def main():
    """Main function to run the ML Researcher Tool."""
    try:
        tool = MLResearcherTool()
        
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
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())



