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
from Report_to_txt import extract_pdf_text


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
ML_RESEARCH_CATEGORIES = [
    "variable_length_sequences",
    "fixed_channel_count", 
    "temporal_structure",
    "reconstruction_objective",
    "latent_embedding_required",
    "shape_preserving_seq2seq",
    "classification_objective",
    "regression_objective", 
    "generation_objective",
    "noise_robustness",
    "real_time_constraint",
    "invariance_requirements",
    "sensor_data",
    "multimodal_data",
    "interpretability_required",
    "high_accuracy_required",
    "few_shot_learning",
    "model_selection_query",
    "text_data",
    "multilingual_requirement",
    "variable_document_length"
]


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
2. A confidence score between 0.0 and 1.0 (how certain you are this category applies)
3. A brief explanation of why this category applies
4. Specific evidence from the task description that supports this categorization

Only include categories that clearly apply to the task. If a category doesn't apply or you're uncertain, don't include it.

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
Based on the following machine learning research task analysis, generate a concise search query suitable for arXiv API search.

Original Task: {prompt}

Detected Categories: {', '.join(prop_names)}

Detailed Analysis: {llm_analysis.get('llm_analysis', 'Not available')}

Create a focused search query (2-4 key terms) that would effectively find relevant research papers on arXiv. The query should be:
- Specific enough to find relevant papers
- General enough to not be too restrictive
- Include key ML concepts and techniques
- Use forward slashes (/) to separate different terms or concepts
- Group related words together as single terms (e.g., neural network, time series, machine learning)
- Avoid overly technical jargon

IMPORTANT: Use forward slashes (/) to separate terms, not quotes or spaces.

Examples of good search queries:
- neural network/time series/forecasting
- deep learning/anomaly detection/sensor data
- transformer model/natural language processing
- autoencoder/reconstruction/noise robustness

Return only the search query without explanation.
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
    
    def format_search_string(self, input_string: str) -> str:
        """Convert string to arXiv search format handling slash-separated terms.
        
        Input: "deep learning/time series/forecasting/variable length"
        Output: More flexible search that's likely to find papers
        """
        # Split by forward slashes
        terms = input_string.strip().split('/')
        
        if not terms:
            return "all:machine+learning"
        
        # For better results, use OR between terms instead of AND
        # This makes the search less restrictive
        parts = []
        
        for term in terms:
            term = term.strip()
            if not term:
                continue
            
            # If term has spaces, treat it as a phrase
            if ' ' in term:
                # Replace spaces with + and add URL encoding for quotes
                formatted_term = term.replace(' ', '+')
                parts.append(f'all:%22{formatted_term}%22')
            else:
                # Single word, no quotes needed
                parts.append(f'all:{term}')
        
        # Join with OR for broader results (instead of AND)
        # But limit to first 3 terms to avoid too broad search
        main_parts = parts[:3]
        return '+OR+'.join(main_parts) if main_parts else "all:machine+learning"
    
    def search_arxiv(self, search_query: str, max_results: int = 5) -> Dict[str, Any]:
        """Search arXiv for papers using the formatted search query."""
        
        print(f"\n🔍 SEARCHING arXiv: {search_query}")
        print("=" * 80)
        
        # Try multiple search strategies if first one fails
        search_strategies = [
            ("Full query", search_query),
            ("First two terms", "/".join(search_query.split('/')[:2])),
            ("Main term only", search_query.split('/')[0])
        ]
        
        for strategy_name, query in search_strategies:
            print(f"Trying {strategy_name}: {query}")
            
            # Format the search query
            formatted_query = self.format_search_string(query)
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
                
                # If we found papers, proceed with this strategy
                if total_results > 0:
                    print("=" * 80)
                    
                    # Get all paper entries
                    entries = root.findall('atom:entry', ns)
                    papers = []
                    
                    for i, entry in enumerate(entries, 1):
                        # Extract basic info
                        title = entry.find('atom:title', ns).text.strip()
                        paper_id = entry.find('atom:id', ns).text.split('/')[-1]
                        

                        
                        # Get published date
                        published = entry.find('atom:published', ns).text[:10] if entry.find('atom:published', ns) is not None else "Unknown"
                        
                        # Get arXiv URL
                        arxiv_url = f"http://export.arxiv.org/api/query?id_list={paper_id}"
                        pdf_txt = extract_pdf_text(arxiv_url)
                        # Store paper info
                        paper_info = {
                            "title": title,
                            "id": paper_id,
                            "published": published,
                            #"authors": authors,
                            #"abstract": summary,
                            "content": pdf_txt,
                            "url": arxiv_url
                        }
                        papers.append(paper_info)
                        
                        # Print formatted output
                        print(f"\n📄 PAPER #{i}")
                        print("-" * 60)
                        print(f"Title: {title}")
                        print(f"ID: {paper_id}")
                        print(f"Published: {published}")
                       
                        print(f"URL: {arxiv_url}")
                        print(f"Content:\n{pdf_txt[500:]}")
                        print("-" * 60)
                       
                    
                    return {
                        "search_successful": True,
                        "total_results": str(total_results),
                        "papers_returned": len(papers),
                        "papers": papers,
                        "formatted_query": formatted_query,
                        "original_query": search_query,
                        "strategy_used": strategy_name
                    }
                else:
                    print(f"No papers found with {strategy_name}, trying next strategy...")
                    print("-" * 40)
                    
            except Exception as e:
                print(f"Error with {strategy_name}: {e}")
                print("-" * 40)
                continue
        
        # If all strategies failed
        print("❌ No papers found with any search strategy")
        return {
            "search_successful": False,
            "error": "No papers found with any search strategy",
            "papers": [],
            "formatted_query": "",
            "original_query": search_query
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
        arxiv_results = self.search_arxiv(arxiv_search_query, max_results=5)
        
        # Compile results
        results = {
            "original_prompt": prompt,
            "detected_categories": [prop.to_dict() for prop in llm_properties],
            "detailed_analysis": llm_analysis,
            "arxiv_search_query": arxiv_search_query,
            "arxiv_results": arxiv_results,
            "summary": {
                "total_categories_detected": len(llm_properties),
                "high_confidence_categories": len([p for p in llm_properties if p.confidence > 0.7]),
                "detailed_analysis_successful": "error" not in llm_analysis,
                "arxiv_search_successful": arxiv_results.get("search_successful", False),
                "papers_found": arxiv_results.get("papers_returned", 0)
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
