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
    
    def extract_properties_llm_based(self, query: str) -> List[PropertyHit]:
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
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"content": content, "role": "user"}]
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
        
        
        
        
    
    def decompose_task_with_llm(self, prompt: str) -> Dict[str, Any]:
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
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"content": content, "role": "user"}]
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

    def _process_single_paper(self, entry, ns, index):
        """Process a single paper entry and extract its content."""
        import requests
        import feedparser
        
        try:
            # Extract basic info
            title = entry.find('atom:title', ns).text.strip()
            paper_id = entry.find('atom:id', ns).text.split('/')[-1]
            
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

    def search_arxiv(self, search_query: str, max_results: int = 20) -> Dict[str, Any]:
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
                
                print(f"🚀 Processing {len(entries)} papers in parallel...")
                
                # Process papers in parallel
                
                papers = []
                with ThreadPoolExecutor(max_workers=5) as executor:  # Limit to 5 concurrent downloads
                    # Submit all tasks
                    future_to_index = {
                        executor.submit(self._process_single_paper, entry, ns, i): i 
                        for i, entry in enumerate(entries, 1)
                    }
                    
                    # Collect results as they complete
                    for future in as_completed(future_to_index):
                        paper_info = future.result()
                        papers.append(paper_info)
                
                # Sort papers back to original order
                papers.sort(key=lambda x: x.get('index', 999))
                
                # Print final results
                print("\n" + "=" * 80)
                print("📋 FINAL RESULTS:")
                print("=" * 80)
                
                for paper in papers:
                    i = paper.get('index', 0)
                    print(f"\n📄 PAPER #{i}")
                    print("-" * 60)
                    print(f"Title: {paper['title']}")
                    print(f"ID: {paper['id']}")
                    print(f"Published: {paper['published']}")
                    print(f"URL: {paper['url']}")
                    
                    if paper['content']:
                        print(f"Content:\n{paper['content'][:500]}")
                    else:
                        print("Content: [No content extracted]")
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
    
    def analyze_research_task(self, prompt: str) -> Dict[str, Any]:
        """Main method to analyze a research task."""
        print(f"🔍 Analyzing research task: {prompt}")
        print("=" * 50)
        
        # Step 1: LLM-based property extraction
        print("🤖 Step 1: Extracting properties using LLM analysis...")
        llm_properties = self.extract_properties_llm_based(prompt)
        
        print(f"Found {len(llm_properties)} properties:")
        for prop in llm_properties:
            print(f"  - {prop.name}: {prop.confidence:.2f} confidence")
        
        # Step 2: Detailed LLM-based task decomposition
        print("\n🔬 Step 2: Performing detailed task decomposition...")
        llm_analysis = self.decompose_task_with_llm(prompt)
        
        if "error" in llm_analysis:
            print(f"❌ LLM analysis failed: {llm_analysis['error']}")
        else:
            print("✅ LLM analysis completed")
            
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
        arxiv_results = self.search_arxiv(arxiv_search_query, max_results=20)
        
        # Step 5: Suggest models based on all evidence
        model_suggestions = self.suggest_models_from_arxiv(prompt, arxiv_results, llm_properties, llm_analysis)
        
        # Compile results
        results = {
            "original_prompt": prompt,
            "detected_categories": [prop.to_dict() for prop in llm_properties],
            "detailed_analysis": llm_analysis,
            "arxiv_search_query": arxiv_search_query,
            "arxiv_results": arxiv_results,
            "model_suggestions": model_suggestions,
            "summary": {
                "total_categories_detected": len(llm_properties),
                "high_confidence_categories": len([p for p in llm_properties if p.confidence > 0.7]),
                "detailed_analysis_successful": "error" not in llm_analysis,
                "arxiv_search_successful": arxiv_results.get("search_successful", False),
                "papers_found": arxiv_results.get("papers_returned", 0),
                "model_suggestions_successful": model_suggestions.get("suggestions_successful", False)
            }
        }
        
        return results
    
    def interactive_mode(self):
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
                results = self.analyze_research_task(prompt)
                
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


def main():
    """Main function to run the ML Researcher Tool."""
    try:
        tool = MLResearcherTool()
        
        if len(sys.argv) > 1:
            # Command line mode
            prompt = " ".join(sys.argv[1:])
            results = tool.analyze_research_task(prompt)
            print("\n" + json.dumps(results, indent=2))
        else:
            # Interactive mode
            tool.interactive_mode()
    
    except Exception as e:
        print(f"❌ Failed to initialize ML Researcher Tool: {str(e)}")
        print("Make sure your API key is configured in env.example or .env file.")
        sys.exit(1)


if __name__ == "__main__":
    main()



