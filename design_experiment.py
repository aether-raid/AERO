#!/usr/bin/env python3
"""
Two-Level Tree-Based Experimental Design System (Enhanced)
=========================================================

This script implements a comprehensive workflow:
1. User input → breakdown into research goal, hypothesis and relevant information (OpenAI + Mistral fallback)
2. For each hypothesis: generate parent nodes (general experimental strategies) 
3. Expand promising parents → child nodes (full implementation steps with details)
4. Score and prune at each level using AB-MCTS-A adaptive thresholds
5. Output complete tree structure to markdown for supervision

Configuration:
    Reads from environment variables or `.env` file:
        - OPENAI_API_KEY
        - MISTRAL_API_KEY  
        - BASE_URL (default: https://agents.aetherraid.dev)
        - DEFAULT_MODEL (default: gemini/gemini-2.5-flash)
"""

import os
import json
import re
import asyncio
import logging
from datetime import datetime
from dotenv import load_dotenv
from openai import AsyncOpenAI

# --- Load environment variables ---
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Initialize clients
primary_client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("BASE_URL", "https://agents.aetherraid.dev")
)
PRIMARY_MODEL = os.getenv("DEFAULT_MODEL", "gemini/gemini-2.5-flash")

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_MODEL = "mistral-large-latest"
use_fallback = False

def clean_json_string(text):
    """Clean JSON string by removing control characters and markdown"""
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    return text

async def get_llm_response(messages, temperature=0.2, max_tokens=None):
    """Get LLM response with automatic fallback - both async"""
    global use_fallback
    
    # Try primary first
    if not use_fallback:
        try:
            kwargs = {"model": PRIMARY_MODEL, "messages": messages, "temperature": temperature}
            if max_tokens:
                kwargs["max_tokens"] = max_tokens
            response = await primary_client.chat.completions.create(**kwargs)
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"Primary API failed, switching to fallback: {e}")
            use_fallback = True
    
    # Fallback to Mistral (async)
    if MISTRAL_API_KEY:
        try:
            fallback_client = AsyncOpenAI(api_key=MISTRAL_API_KEY, base_url="https://api.mistral.ai/v1")
            kwargs = {"model": MISTRAL_MODEL, "messages": messages, "temperature": temperature}
            if max_tokens:
                kwargs["max_tokens"] = max_tokens
            response = await fallback_client.chat.completions.create(**kwargs)
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Both primary and fallback APIs failed: {e}")
            raise Exception("Both primary and fallback APIs failed")
    else:
        raise Exception("Mistral API key not found")

class ExperimentNode:
    """Represents a node in the two-level experiment tree"""
    def __init__(self, description, depth=0, node_type="strategy"):
        self.description = description
        self.depth = depth
        self.node_type = node_type  # "strategy", "implementation"
        self.feasibility_score = 0.0
        self.impact_score = 0.0
        self.combined_score = 0.0
        self.children = []
        
    def calculate_score(self, w_feasibility=0.6, w_impact=0.4):
        """Calculate weighted combination score"""
        self.combined_score = w_feasibility * self.feasibility_score + w_impact * self.impact_score
        return self.combined_score

async def extract_research_components(user_input):
    """Extract research goal, hypotheses, and relevant information"""
    prompt = f"""
    Extract and structure the following from the research plan:
    - research_goal: Main research objective
    - hypotheses: List of testable hypotheses (as strings)
    - relevant_info: Supporting information, constraints, variables
    
    Return only JSON format.
    
    Research Plan: {user_input}
    """
    
    try:
        content = await get_llm_response([
            {"role": "system", "content": "Extract research components. Return only valid JSON with hypotheses as string array."},
            {"role": "user", "content": prompt}
        ], temperature=0.2)
        
        cleaned_content = clean_json_string(content)
        result = json.loads(cleaned_content)
        
        # Ensure all values are strings
        if "hypotheses" in result:
            hypotheses = result["hypotheses"]
            if isinstance(hypotheses, list):
                result["hypotheses"] = [str(h) for h in hypotheses]
            else:
                result["hypotheses"] = [str(hypotheses)]
        
        return result
    except Exception:
        return {"error": "Failed to parse", "hypotheses": [user_input]}

async def generate_nodes(parent_description, node_type, hypothesis, relevant_info, count=3):
    """Generate child nodes based on parent and type"""
    
    if node_type == "strategy":
        prompt = f"""Generate {count} DISTINCT high-level experimental strategies to test: {hypothesis}

Requirements:
- 1-2 sentences each, concise and clear
- Fundamentally different approaches (controlled experiments, observational studies, simulations, surveys, meta-analysis)
- Focus on overall approach, not implementation details

Context: {relevant_info}

Format your response as plain text with each strategy separated by "---":

Strategy 1 description here

---

Strategy 2 description here

---

Strategy 3 description here"""

    else:  # implementation
        prompt = f"""Generate {count} COMPLETELY DIFFERENT experimental designs that all follow this general strategy approach: 

STRATEGY APPROACH: {parent_description}

Each experimental design should be a STANDALONE, COMPLETE experiment that uses the same general strategy type but with entirely different approaches.

Requirements for each experimental design:
- Each must be a COMPLETE, INDEPENDENT experimental study 
- Focus PRIMARY on detailed step-by-step experimental procedures
- Keep background information concise
- Include specific methodologies, tools, datasets, and evaluation approaches

Structure each experimental design as follows:

**Experimental Design Title:** [Clear, descriptive title]

**Brief Background:** [1-2 sentences on objective and hypothesis]

**Step-by-Step Experimental Protocol:**
Step 1: [Detailed description]
Step 2: [Detailed description]
Step 3: [Detailed description]
[Continue with as many steps as needed]

**Key Tools:** [Brief list of essential tools/software]
**Success Criteria:** [Brief measurable outcomes]

Context: {relevant_info}
Hypothesis: {hypothesis}

Format your response as plain text with each experimental design separated by "---":

Experimental Design 1: [Title]
[Complete experimental design with focus on steps]

---

Experimental Design 2: [Title]  
[Complete experimental design with focus on steps]

---

Experimental Design 3: [Title]
[Complete experimental design with focus on steps]"""
    
    try:
        content = await get_llm_response([
            {"role": "system", "content": f"Generate {count} COMPLETELY INDEPENDENT experimental {node_type}. For implementations, each should be a STANDALONE experimental study that uses the same strategy category but with entirely different approaches, data sources, tools, and methodologies. Think of each as a separate research paper. Return plain text format with sections separated by '---'."},
            {"role": "user", "content": prompt}
        ], temperature=0.8)
        
        # Parse plain text sections separated by "---"
        sections = content.split("---")
        
        # Clean and validate each section
        valid_nodes = []
        for section in sections:
            section = section.strip()
            
            # Enhanced validation to filter out JSON fragments and short content
            if len(section) < 100:  # Too short to be a complete experimental design
                continue
                
            valid_nodes.append(section)
        
        return valid_nodes[:count] if valid_nodes else []
        
    except Exception as e:
        logger.warning(f"Failed to generate {node_type} nodes: {e}")
        return []

async def score_node(description, hypothesis, depth):
    """Score node for feasibility and impact, with special consideration for implementation detail"""
    depth_context = {0: "experimental strategy", 1: "implementation plan"}
    
    if depth == 1:  # Implementation node - include detail assessment
        prompt = f"""
        Score this implementation plan for testing: "{hypothesis}"
        
        Implementation Plan: {description}
        
        Evaluate (0-10 scale):
        1. FEASIBILITY: Resource requirements, complexity, time, expertise needed (consider data availability, computational cost, technical complexity)
        2. IMPACT: Combination of:
           - Experimental rigor and hypothesis testing power
           - Level of implementation detail (step-by-step clarity, specific tools/methods, reproducibility)
           - Scientific value and insight potential
        
        For IMPACT scoring of implementations:
        - 9-10: Complete experimental procedures with all steps, specific tools, clear metrics, high scientific rigor
        - 7-8: Good detail with most steps specified, some tools mentioned, adequate rigor
        - 5-6: Basic implementation outline, missing some critical details
        - 3-4: Vague or incomplete procedures, major gaps in methodology
        - 1-2: Fragment or insufficient detail for actual implementation
        
        Return ONLY: {{"feasibility": X, "impact": Y}} where X,Y are integers 0-10.
        """
    else:  # Strategy node
        prompt = f"""
        Score this experimental strategy for testing: "{hypothesis}"
        
        Strategy: {description}
        
        Evaluate (0-10 scale):
        1. FEASIBILITY: Resource requirements, complexity, time, expertise needed
        2. IMPACT: Ability to test hypothesis, scientific rigor, statistical power
        
        Return ONLY: {{"feasibility": X, "impact": Y}} where X,Y are integers 0-10.
        """
    
    try:
        content = await get_llm_response([
            {"role": "system", "content": "Provide objective scores. Return ONLY valid JSON with integer scores 0-10."},
            {"role": "user", "content": prompt}
        ], temperature=0.1)
        
        cleaned_content = clean_json_string(content)
        scores = json.loads(cleaned_content)
        
        feasibility = max(0, min(10, int(scores.get("feasibility", 5))))
        impact = max(0, min(10, int(scores.get("impact", 5))))
        
        return feasibility, impact
        
    except Exception:
        return 5, 5  # Fallback scores

def apply_adaptive_threshold(nodes, base_threshold=0.6, max_nodes=None):
    """Apply AB-MCTS-A adaptive thresholding (Sakana-style)"""
    if not nodes:
        return []
    
    scores = [node.combined_score for node in nodes]
    mean_score = sum(scores) / len(scores)
    max_score = max(scores)
    
    if len(scores) > 1:
        variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
        std_dev = variance ** 0.5
        adaptive_threshold = max(
            base_threshold * max_score,
            mean_score + 0.5 * std_dev
        )
    else:
        adaptive_threshold = base_threshold * max_score
    
    promising_nodes = [node for node in nodes if node.combined_score >= adaptive_threshold]
    
    # Ensure at least top 2 nodes if any exist
    if not promising_nodes and nodes:
        sorted_nodes = sorted(nodes, key=lambda x: x.combined_score, reverse=True)
        promising_nodes = sorted_nodes[:2]
    
    # Apply maximum limit if specified
    if max_nodes and len(promising_nodes) > max_nodes:
        sorted_nodes = sorted(promising_nodes, key=lambda x: x.combined_score, reverse=True)
        promising_nodes = sorted_nodes[:max_nodes]
    
    return promising_nodes

async def build_experiment_tree(hypothesis, relevant_info):
    """Build two-level experiment tree using AB-MCTS-A with parallel processing"""
    hypothesis_str = str(hypothesis)
    
    print(f"  🌱 Generating strategies for: {hypothesis_str[:60]}...")
    
    # Level 1: Generate strategies (parent nodes) 
    strategy_descriptions = await generate_nodes("", "strategy", hypothesis_str, relevant_info, count=4)
    if not strategy_descriptions:
        return []
    
    # Score all strategies in parallel
    strategy_tasks = []
    for desc in strategy_descriptions:
        node = ExperimentNode(desc, depth=0, node_type="strategy")
        task = score_node(desc, hypothesis_str, 0)
        strategy_tasks.append((node, task))
    
    # Wait for all strategy scoring to complete
    strategy_nodes = []
    for node, task in strategy_tasks:
        node.feasibility_score, node.impact_score = await task
        node.calculate_score()
        strategy_nodes.append(node)
    
    promising_strategies = apply_adaptive_threshold(strategy_nodes, base_threshold=0.5)
    print(f"  📊 Selected {len(promising_strategies)}/{len(strategy_nodes)} strategies")
    
    # Level 2: Generate implementation plans (child nodes) - parallel processing
    impl_tasks = []
    for strategy_node in promising_strategies:
        task = generate_and_score_implementations(strategy_node, hypothesis_str, relevant_info)
        impl_tasks.append(task)
    
    # Wait for all implementation generation and scoring
    await asyncio.gather(*impl_tasks)
    
    return promising_strategies

async def generate_and_score_implementations(strategy_node, hypothesis_str, relevant_info):
    """Generate and score implementation nodes for a strategy"""
    print(f"    ⚙️ Generating implementations...")
    
    impl_descriptions = await generate_nodes(
        strategy_node.description, "implementation", hypothesis_str, relevant_info, count=2
    )
    
    if not impl_descriptions:
        return
    
    # Score implementations in parallel
    impl_tasks = []
    for desc in impl_descriptions:
        impl_node = ExperimentNode(desc, depth=1, node_type="implementation")
        task = score_node(desc, hypothesis_str, 1)
        impl_tasks.append((impl_node, task))
    
    # Wait for all implementation scoring
    for impl_node, task in impl_tasks:
        try:
            impl_node.feasibility_score, impl_node.impact_score = await task
            impl_node.calculate_score()
            strategy_node.children.append(impl_node)
        except Exception as e:
            logger.warning(f"Failed to score implementation: {e}")
    
    # Apply adaptive threshold
    promising_impls = apply_adaptive_threshold(strategy_node.children, base_threshold=0.5, max_nodes=3)
    strategy_node.children = promising_impls
    print(f"    📊 Selected {len(promising_impls)}/{len(impl_descriptions)} implementations")

def serialize_tree(nodes):
    """Convert tree to serializable format"""
    def serialize_node(node):
        return {
            "description": node.description,
            "type": node.node_type,
            "depth": node.depth,
            "feasibility_score": node.feasibility_score,
            "impact_score": node.impact_score,
            "combined_score": node.combined_score,
            "children": [serialize_node(child) for child in node.children]
        }
    return [serialize_node(node) for node in nodes]

def create_tree_markdown(hypothesis_results, processing_time):
    """Create comprehensive markdown output"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"experiment_tree_{timestamp}.md"
    
    md_content = f"""# Two-Level Experiment Design Tree
*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
*Processing Time: {processing_time:.2f} seconds*
*System: AB-MCTS-A Two-Level Tree Expansion*

## Overview
Complete experimental design tree with two levels:
- **Level 1 (Strategies)**: High-level experimental approaches
- **Level 2 (Implementations)**: Complete step-by-step execution plans with details

**Scoring:** Feasibility (F) and Impact (I) from 0-10, Combined = 0.6×F + 0.4×I

---

"""
    
    for hyp_idx, hyp_result in enumerate(hypothesis_results, 1):
        md_content += f"""## Hypothesis {hyp_idx}
**{hyp_result['hypothesis']}**

"""
        
        tree_structure = hyp_result.get('tree_structure', [])
        
        for strategy_idx, strategy_node in enumerate(tree_structure, 1):
            md_content += f"""### {strategy_idx}. Strategy (Level 1)
**Description:** {strategy_node['description']}
**Scores:** F={strategy_node['feasibility_score']}, I={strategy_node['impact_score']}, Combined={strategy_node['combined_score']:.2f}

"""
            
            for impl_idx, impl_node in enumerate(strategy_node.get('children', []), 1):
                md_content += f"""#### {strategy_idx}.{impl_idx}. Implementation (Level 2)
**Description:** {impl_node['description']}
**Scores:** F={impl_node['feasibility_score']}, I={impl_node['impact_score']}, Combined={impl_node['combined_score']:.2f}

"""
        
        md_content += "---\n\n"
    
    # Summary statistics
    total_strategies = sum(len(hyp_result.get('tree_structure', [])) for hyp_result in hypothesis_results)
    total_implementations = sum(
        len(strategy.get('children', [])) 
        for hyp_result in hypothesis_results 
        for strategy in hyp_result.get('tree_structure', [])
    )
    
    md_content += f"""## Summary Statistics
- **Total Hypotheses:** {len(hypothesis_results)}
- **Total Strategies (Level 1):** {total_strategies}
- **Total Implementations (Level 2):** {total_implementations}
- **Processing Time:** {processing_time:.2f} seconds

## Supervision Notes
1. **Strategy Diversity:** Review Level 1 nodes for comprehensive coverage
2. **Implementation Completeness:** Verify Level 2 nodes contain detailed execution steps
3. **Score Validation:** Assess if scores align with domain expertise
4. **Resource Assessment:** Evaluate implementation feasibility

---
*Generated by Enhanced Two-Level Tree-Based Experiment Design System*
"""
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    return filename

async def process_single_hypothesis(hypothesis, relevant_info):
    """Process a single hypothesis and return results"""
    try:
        tree_nodes = await build_experiment_tree(hypothesis, relevant_info)
        
        return {
            "hypothesis": hypothesis,
            "tree_structure": serialize_tree(tree_nodes)
        }
        
    except Exception as e:
        logger.error(f"Error processing hypothesis '{hypothesis}': {e}")
        return {
            "hypothesis": hypothesis,
            "tree_structure": [],
            "error": str(e)
        }

async def main():
    """Main workflow entry point"""
    print("🧪 Enhanced Two-Level Tree-Based Experiment Design System")
    print("=" * 70)    
    user_input = input("\n📝 Enter your research plan: ").strip()
    if not user_input:
        print("❌ No input provided. Exiting.")
        return
    
    # Step 1: Extract research components
    print("\n📋 Step 1: Extracting research components...")
    components = await extract_research_components(user_input)
    
    if "error" in components:
        hypotheses = [user_input]
        relevant_info = ""
        research_goal = "Research goal extraction failed"
    else:
        hypotheses = components.get("hypotheses", [user_input])
        relevant_info = str(components.get("relevant_info", ""))
        research_goal = str(components.get("research_goal", "No specific goal identified"))
    
    print(f"  ✅ Research Goal: {research_goal}")
    print(f"  ✅ Found {len(hypotheses)} hypothesis(es)")
    print(f"  ✅ Context: {relevant_info}")
    
    # Steps 2-4: Build two-level experiment trees in parallel
    print(f"\n🌳 Steps 2-4: Building two-level experiment trees...")
    start_time = datetime.now()
    
    # Process all hypotheses in parallel
    hypothesis_tasks = []
    for i, hypothesis in enumerate(hypotheses, 1):
        print(f"📍 Processing Hypothesis {i}/{len(hypotheses)}: {hypothesis}")
        task = process_single_hypothesis(hypothesis, relevant_info)
        hypothesis_tasks.append(task)
    
    # Wait for all hypothesis processing to complete
    hypothesis_results = await asyncio.gather(*hypothesis_tasks)
    
    processing_time = (datetime.now() - start_time).total_seconds()
    
    # Step 5: Generate markdown output
    print(f"\n📝 Step 5: Creating comprehensive markdown output...")
    markdown_file = create_tree_markdown(hypothesis_results, processing_time)
    
    # Display summary
    print(f"\n" + "=" * 70)
    print("🎯 TWO-LEVEL EXPERIMENT DESIGN SUMMARY")
    print("=" * 70)
    
    total_strategies = sum(len(result.get('tree_structure', [])) for result in hypothesis_results)
    total_implementations = sum(
        len(strategy.get('children', [])) 
        for result in hypothesis_results 
        for strategy in result.get('tree_structure', [])
    )
    
    for i, result in enumerate(hypothesis_results, 1):
        print(f"\n📋 Hypothesis {i}: {result['hypothesis']}")
        tree_structure = result.get('tree_structure', [])
        if tree_structure:
            strategies = len(tree_structure)
            implementations = sum(len(s.get('children', [])) for s in tree_structure)
            print(f"  🌳 Generated {strategies} strategies → {implementations} implementations")
        else:
            print(f"  ❌ No experiments generated")
    
    print(f"\n📊 OVERALL STATISTICS")
    print(f"  • Total Hypotheses: {len(hypothesis_results)}")
    print(f"  • Total Strategies (Level 1): {total_strategies}")
    print(f"  • Total Implementations (Level 2): {total_implementations}")
    print(f"  • Processing Time: {processing_time:.2f} seconds")
    
    print(f"\n✅ WORKFLOW COMPLETE")
    print(f"📄 Complete two-level tree structure saved to: {markdown_file}")
    print(f"📖 Open this file to review and supervise all generated experiments.")

# --- Main workflow ---
if __name__ == "__main__":
    asyncio.run(main())
