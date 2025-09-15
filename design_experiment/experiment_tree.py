"""
Experiment Design Tree System (experiment_tree.py)
==================================================

This module implements an automated experiment design tree search system that leverages literature retrieval and LLM-based node generation. The workflow:

1. Extracts research context and hypotheses from user input.
2. Retrieves and formats relevant literature with numbered citations for LLM prompting.
3. Generates experiment tree nodes (strategy, methodology, implementation) using LLMs, referencing the literature context.
4. Scores each node on multiple criteria (clarity, feasibility, novelty, soundness) with weighted aggregation.
5. Tracks citations and references for each node, outputting a references section with arXiv links.
6. Performs tree search (via treequest) to explore and evaluate experiment designs.
7. Identifies and displays the highest-scoring implementation-level experiment design.

"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import treequest as tq
from search import build_experiment_search_workflow
from init_utils import get_llm_response, extract_research_components

import sys
from io import StringIO

class FilteredStringIO(StringIO):
    def write(self, s):
        # Filter out sampling messages
        if "Sampling:" in s or "INFO - Sampling:" in s:
            return
        super().write(s)

class SuppressOutput:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = FilteredStringIO()
        sys.stderr = FilteredStringIO()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr

@dataclass
class ExperimentState:
    level: str
    content: str
    score: float | None = None
    citations: list[str] = field(default_factory=list)
    references: Dict[int, Dict[str, str]] = field(default_factory=dict)  # {citation_num: {title, url}}
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'level': self.level,
            'content': self.content,
            'score': self.score,
            'citations': self.citations,
            'references': self.references
        }

class ExperimentTreeSystem:
    def __init__(self, user_input: str):
        self.user_input = user_input
        self.literature_context = []
        self.research_context = {}
        
    async def initialize(self):
        """Initialize search workflow and retrieve literature context"""
        print("🔍 Initializing literature search and context...")
        
        # Extract research components from user input
        self.research_context = await extract_research_components(self.user_input)
        
        # Run search workflow for each hypothesis to get literature context
        workflow = await build_experiment_search_workflow()
        app = workflow.compile()
        
        all_literature = []
        hypotheses = self.research_context.get('hypotheses', [self.user_input])
        
        for hypothesis in hypotheses:
            search_state = {'hypothesis': hypothesis}
            result = await app.ainvoke(search_state)
            literature_results = result.get('results', [])
            all_literature.extend(literature_results)
        
        self.literature_context = all_literature
        
        return self
    
    def format_literature_context(self, max_chunks: int = 5) -> str:
        """Format literature context for prompts with numbered citations"""
        if not self.literature_context:
            return "No relevant literature found."
        
        context_text = "=== RELEVANT LITERATURE ===\n"
        for i, chunk in enumerate(self.literature_context[:max_chunks]):
            title = chunk.get('paper_title', 'Unknown Paper')
            source_url = chunk.get('source_url', 'No URL')
            text = chunk.get('text', '')[:500]  # Truncate for prompt efficiency
            similarity = chunk.get('cosine_similarity', 0.0)
            
            context_text += f"\n[{i+1}] {title}\n"
            context_text += f"Source: {source_url}\n"
            context_text += f"Relevance: {similarity:.3f}\n"
            context_text += f"Content: {text}...\n"
            context_text += "-" * 80 + "\n"
        
        return context_text

    async def generate_strategy(self, hypothesis: str) -> ExperimentState:
        """Generate high-level experimental strategy"""
        literature_context = self.format_literature_context(max_chunks=5)
        references = self.create_citation_mapping(self.literature_context[:5])
        
        prompt = f"""You are an expert researcher. Generate a high-level experimental strategy based on the hypothesis and relevant literature.

            HYPOTHESIS: {hypothesis}

            RESEARCH CONTEXT:
            - Goal: {self.research_context.get('research_goal', 'Not specified')}
            - Variables: {self.research_context.get('variables', 'Not specified')}
            - Additional info: {self.research_context.get('relevant_info', 'None')}

            {literature_context}

            Generate a HIGH-LEVEL STRATEGY that:
            1. Identifies 2-3 broad experimental avenues/approaches
            2. References relevant literature using numbered citations [1], [2], etc.
            3. Outlines general methodology directions
            4. Does NOT include specific implementation details

            When referencing literature, use numbered citations in format: [1], [2], [3], etc. ONLY based on the literature list above.

            Format as clear, structured text (not JSON).
        """

        content = await get_llm_response([{"role": "user", "content": prompt}])
        citations = self.extract_citations(content)
        score = await self.evaluate_experiment(content, hypothesis, "strategy")

        return ExperimentState(
            level="strategy",
            content=content,
            score=score,
            citations=citations,
            references=references
        )

    async def generate_methodology(self, parent_strategy: ExperimentState, hypothesis: str) -> ExperimentState:
        """Generate mid-level methodology from strategy"""
        literature_context = self.format_literature_context(max_chunks=3)
        references = self.create_citation_mapping(self.literature_context[:3])
        
        prompt = f"""You are an expert researcher. Based on the parent strategy and literature, generate a detailed methodology.

            HYPOTHESIS: {hypothesis}

            PARENT STRATEGY:
            {parent_strategy.content}

            {literature_context}

            Generate a MID-LEVEL METHODOLOGY that:
            1. Elaborates on ONE specific avenue from the parent strategy
            2. Defines experimental design, variables, and general procedures
            3. References supporting literature using numbered citations [1], [2], etc.
            4. Specifies data collection and analysis approaches
            5. Does NOT include specific implementation code or exact parameters

            When referencing literature, use numbered citations in format: [1], [2], [3], etc. ONLY based on the literature list above.

            Format as structured methodology description.
        """

        content = await get_llm_response([{"role": "user", "content": prompt}])
        citations = self.extract_citations(content)
        score = await self.evaluate_experiment(content, hypothesis, "methodology")

        return ExperimentState(
            level="methodology",
            content=content,
            score=score,
            citations=citations,
            references=references
        )

    async def generate_implementation(self, parent_methodology: ExperimentState, hypothesis: str) -> ExperimentState:
        """Generate concrete implementation from methodology"""
        literature_context = self.format_literature_context(max_chunks=2)
        references = self.create_citation_mapping(self.literature_context[:2])
        
        prompt = f"""You are an expert researcher. Based on the parent methodology and literature, generate a concrete implementation plan.

            HYPOTHESIS: {hypothesis}

            PARENT METHODOLOGY:
            {parent_methodology.content}

            {literature_context}

            Generate a CONCRETE IMPLEMENTATION that:
            1. Provides specific, executable experimental procedures
            2. Includes exact datasets, algorithms, and hyperparameters
            3. Specifies evaluation metrics and statistical tests
            4. References literature for technical choices using numbered citations [1], [2], etc.
            5. Must be realistically executable

            When referencing literature, use numbered citations in format: [1], [2], [3], etc. ONLY based on the literature list above.

            Format as detailed implementation plan with specific steps.
        """

        content = await get_llm_response([{"role": "user", "content": prompt}])
        citations = self.extract_citations(content)
        score = await self.evaluate_experiment(content, hypothesis, "implementation")

        return ExperimentState(
            level="implementation",
            content=content,
            score=score,
            citations=citations,
            references=references
        )

    def create_citation_mapping(self, literature_context: List[Dict]) -> Dict[int, Dict[str, str]]:
        """Create mapping from citation numbers to paper info"""
        references = {}
        for i, chunk in enumerate(literature_context, 1):
            title = chunk.get('paper_title', f'Paper {i}')
            url = chunk.get('source_url', '')
            references[i] = {'title': title, 'url': url}
        return references

    def extract_citations(self, content: str) -> list[str]:
        """Extract numbered citations from content"""
        citation_pattern = r'\[(\d+)\]'
        citations = re.findall(citation_pattern, content)
        return list(set(citations))  # Remove duplicates

    def format_references_section(self, references: Dict[int, Dict[str, str]]) -> str:
        """Format references section with paper titles and URLs"""
        if not references:
            return ""
        
        ref_text = "\n\n=== REFERENCES ===\n"
        for num in sorted(references.keys()):
            ref_info = references[num]
            title = ref_info.get('title', f'Reference {num}')
            url = ref_info.get('url', '')
            # Extract arxiv ID and create proper PDF link
            arxiv_link = ""
            if url:
                # Check if it's already a full arxiv URL
                if 'arxiv.org' in url:
                    # Extract arxiv ID from full URL (e.g., 2101.10932v3 from https://arxiv.org/abs/2101.10932v3)
                    arxiv_match = re.search(r'(\d+\.\d+v?\d*)', url)
                    if arxiv_match:
                        arxiv_id = arxiv_match.group(1)
                        arxiv_link = f"https://arxiv.org/pdf/{arxiv_id}"
                else:
                    # Check if it's just an arxiv ID (e.g., 2101.10932v3)
                    arxiv_match = re.search(r'(\d+\.\d+v?\d*)', url)
                    if arxiv_match:
                        arxiv_id = arxiv_match.group(1)
                        arxiv_link = f"https://arxiv.org/pdf/{arxiv_id}"
                    else:
                        # Use the URL as-is if it doesn't match arxiv pattern
                        arxiv_link = url
            if arxiv_link:
                ref_text += f"[{num}] {title} ({arxiv_link})\n"
            else:
                ref_text += f"[{num}] {title}\n"
        
        return ref_text


    async def evaluate_experiment(self, content: str, hypothesis: str, level: str) -> float:
        """Score experiment design with separate criteria and calculate weighted final score"""
        literature_context = self.format_literature_context(max_chunks=2)

        prompt = f"""You are an expert research evaluator. Score the following {level}-level experiment design on a 0–100 scale for each criterion.

            Criteria:
            1. Clarity & Organization – Is the design clearly stated, logically structured, and well-organized? Does it define key concepts, variables, and evaluation strategy?  
            2. Feasibility & Reproducibility – Can the experiment realistically be executed with available data, compute, and time? Are enough details provided for others to replicate it?  
            3. Novelty & Significance – Does it explore something new or provide potential for meaningful scientific insight? How original and impactful is the approach?  
            4. Soundness & Alignment – Is the design methodologically sound and does it directly test the stated hypothesis? Does it provide strong evidence for the research question?  

            Instructions:
            - Return ONLY 4 integers between 0–100, separated by commas, in this order: 
            Clarity, Feasibility, Novelty, Soundness
            
            HYPOTHESIS: {hypothesis}

            EXPERIMENT DESIGN ({level}):
            {content}

            {literature_context}
        """

        messages = [
            {"role": "system", "content": "You are an expert research evaluator. Return ONLY four numbers separated by commas."},
            {"role": "user", "content": prompt}
        ]

        try:
            response = await get_llm_response(messages, temperature=0.1)

            # Extract up to 4 integers
            numbers = re.findall(r'\d+', response)
            scores = [int(x) for x in numbers[:4]]

            if len(scores) < 4:
                print(f"Expected 4 scores but got: {response}")
                return 0.5

            # Define weights depending on node level
            # Order: Clarity, Feasibility, Novelty, Soundness
            weight_map = {
                "strategy": [0.2, 0.1, 0.4, 0.3],        # Novelty 0.4, Soundness 0.3, Clarity 0.2, Feasibility 0.1
                "methodology": [0.2, 0.3, 0.15, 0.35],    # Soundness 0.35, Feasibility 0.3, Clarity 0.2, Novelty 0.15
                "implementation": [0.2, 0.4, 0.1, 0.3]    # Feasibility 0.4, Soundness 0.3, Clarity 0.2, Novelty 0.1
            }
            weights = weight_map.get(level.lower(), [0.3, 0.3, 0.2, 0.2])

            # Weighted sum → normalize to 0–1
            weighted = sum(s * w for s, w in zip(scores, weights)) / 100.0
            return min(max(weighted, 0.0), 1.0)

        except Exception as e:
            print(f"Scoring failed: {e}")

        return 0.5  # Default score




    def find_best_leaf_node(self, tree) -> ExperimentState:
        """Find the highest-scoring leaf node (implementation level)"""
        def get_all_nodes(node):
            nodes = [node]
            for child in getattr(node, 'children', []):
                nodes.extend(get_all_nodes(child))
            return nodes
        
        all_nodes = get_all_nodes(tree.tree.root)
        leaf_nodes = [node.state for node in all_nodes 
                     if hasattr(node, 'state') and 
                     node.state.level == 'implementation' and 
                     node.state.score is not None]
        
        if not leaf_nodes:
            return None
        
        return max(leaf_nodes, key=lambda x: x.score)


# Sync wrapper functions for treequest
def sync_generate_strategy(parent_state, tree_system, hypothesis):
    import asyncio
    try:
        loop = asyncio.get_running_loop()
        # If already in an event loop, create a new one in a thread
        import concurrent.futures
        def run_in_thread():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                return new_loop.run_until_complete(tree_system.generate_strategy(hypothesis))
            finally:
                new_loop.close()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            state = executor.submit(run_in_thread).result()
    except RuntimeError:
        # No running event loop
        state = asyncio.run(tree_system.generate_strategy(hypothesis))
    return state, state.score

def sync_generate_methodology(parent_state, tree_system, hypothesis):
    import asyncio
    try:
        loop = asyncio.get_running_loop()
        import concurrent.futures
        def run_in_thread():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                return new_loop.run_until_complete(tree_system.generate_methodology(parent_state, hypothesis))
            finally:
                new_loop.close()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            state = executor.submit(run_in_thread).result()
    except RuntimeError:
        state = asyncio.run(tree_system.generate_methodology(parent_state, hypothesis))
    return state, state.score

def sync_generate_implementation(parent_state, tree_system, hypothesis):
    import asyncio
    try:
        loop = asyncio.get_running_loop()
        import concurrent.futures
        def run_in_thread():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                return new_loop.run_until_complete(tree_system.generate_implementation(parent_state, hypothesis))
            finally:
                new_loop.close()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            state = executor.submit(run_in_thread).result()
    except RuntimeError:
        state = asyncio.run(tree_system.generate_implementation(parent_state, hypothesis))
    return state, state.score

async def run_experiment_tree_search(user_input: str, num_iterations: int):
    """Main function to run the experiment tree search"""
    
    # Initialize system with literature context
    tree_system = await ExperimentTreeSystem(user_input).initialize()
    hypothesis = tree_system.research_context.get('hypotheses', [user_input])[0]
    
    print(f"🧪 Starting experiment tree search for: {hypothesis}")
    print(f"📖 Using {len(tree_system.literature_context)} literature chunks as context")
    
    # Setup sync generation functions for treequest
    def strategy_gen(parent_state):
        return sync_generate_strategy(parent_state, tree_system, hypothesis)

    def methodology_gen(parent_state):
        return sync_generate_methodology(parent_state, tree_system, hypothesis)

    def implementation_gen(parent_state):
        return sync_generate_implementation(parent_state, tree_system, hypothesis)

    generate_fns = {
        "strategy": strategy_gen,
        "methodology": methodology_gen,
        "implementation": implementation_gen
    }

    print("🌲 Building experiment design tree...(this may take a few mins)")
    
    # Monkey patch print to filter sampling messages
    original_print = print
    def filtered_print(*args, **kwargs):
        message = ' '.join(str(arg) for arg in args)
        if "Sampling:" not in message:
            original_print(*args, **kwargs)
    
    # Initialize search tree with filtered print
    import builtins
    builtins.print = filtered_print
    try:
        algo = tq.ABMCTSM()
        search_tree = algo.init_tree()
        search_tree.tree.root.state = ExperimentState(
            level="hypothesis",
            content=hypothesis,
            score=1.0
        )
    finally:
        builtins.print = original_print

    # Run tree search (sync loop) with output suppression
    for i in range(num_iterations):
        # Apply filtered print during tree search
        builtins.print = filtered_print
        try:
            search_tree = algo.step(search_tree, generate_fns)
        finally:
            builtins.print = original_print
            
        # Log best current node
        try:
            best_interim_state, _ = tq.top_k(search_tree, algo, k=1)[0]
            print(f"{i+1}/{num_iterations} iterations - best score: {best_interim_state.score:.3f} ({best_interim_state.level})")
        except Exception:
            print(f"{i+1}/{num_iterations} iterations - searching...")

    # Find best leaf node (implementation)
    best_implementation = tree_system.find_best_leaf_node(search_tree)

    if best_implementation:
        print("\n" + "="*80)
        print("🏆 BEST EXPERIMENT DESIGN (Implementation Level)")
        print("="*80)
        print(f"Score: {best_implementation.score:.3f}")
        print(f"Level: {best_implementation.level}")
        if best_implementation.citations:
            print(f"Citations: {len(best_implementation.citations)} sources")
        print("\nContent:")
        print(best_implementation.content)
        
        # Add references section
        if best_implementation.references:
            references_section = tree_system.format_references_section(best_implementation.references)
            print(references_section)
        
        print("="*80)
        return best_implementation
    else:
        print("❌ No implementation-level nodes found")
        return None
