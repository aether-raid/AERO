from __future__ import annotations
import asyncio
import re
import sys
from io import StringIO
import treequest as tq
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from aero.experiment_designer.search import build_experiment_search_workflow
from aero.experiment_designer.utils import get_llm_response, extract_research_components, stream_writer
import asyncio
import concurrent.futures
import builtins

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
class IdeaState:
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
        self.strategy_cache = {}
        self.methodology_cache = {}
        self.messages = []
        
    async def initialize(self):
        """Initialize search workflow and retrieve literature context"""
        # await stream_writer("🔍 Initializing literature search and context...", writer=state.get("writer"), stream_mode="custom")

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

    async def generate_strategy(self, hypothesis: str) -> IdeaState:
        """Generate high-level experimental strategy"""
        cache_key = hypothesis
        if cache_key in self.strategy_cache:
            return self.strategy_cache[cache_key]
        
        literature_context = self.format_literature_context(max_chunks=3)
        references = self.create_citation_mapping(self.literature_context[:3])
        
        prompt = f"""You are an expert researcher. Based on the hypothesis and research context, propose a high-level experimental strategy.

        HYPOTHESIS: {hypothesis}

        RESEARCH CONTEXT:
        - Goal: {self.research_context.get('research_goal', 'Not specified')}
        - Variables: {self.research_context.get('variables', 'Not specified')}
        - Additional info: {self.research_context.get('relevant_info', 'None')}

        Relevant literature (numbered citations):
        {literature_context}

        Generate a HIGH-LEVEL STRATEGY that:
        1. Identifies 2–3 broad experimental avenues or approaches to explore.
        2. Highlights key research questions or hypotheses each approach would address.
        3. Suggests general directions for methodology without specifying datasets, code, or parameters.
        4. Uses numbered citations [1], [2], etc., sparingly, only where literature strengthens reasoning.
        5. Includes domain knowledge and reasoning beyond what is in the literature.

        Format as clear, structured text (not JSON).
        """

        content = await get_llm_response([{"role": "user", "content": prompt}])
        citations = self.extract_citations(content)
        score = await self.evaluate_experiment(content, hypothesis, "strategy")

        state = IdeaState(
            level="strategy",
            content=content,
            score=score,
            citations=citations,
            references=references
        )
        self.strategy_cache[cache_key] = state

        return state

    async def generate_methodology(self, parent_strategy: IdeaState, hypothesis: str) -> IdeaState:
        """Generate mid-level methodology from strategy"""
        if parent_strategy is not None:
            cache_key = (parent_strategy.content, hypothesis)
        else:
            cache_key = "NO_PARENT"

        # Only use cache if not the placeholder
        if cache_key != "NO_PARENT" and cache_key in self.methodology_cache:
            return self.methodology_cache[cache_key]

        
        literature_context = self.format_literature_context(max_chunks=3)
        references = self.create_citation_mapping(self.literature_context[:3])
        
        prompt = f"""
        As an expert researcher, develop a **mid-level experimental methodology** guided primarily by the parent strategy and hypothesis. Incorporate literature only where relevant; emphasize reasoning and domain expertise over exhaustive procedural detail.

        PARENT STRATEGY:
        {parent_strategy.content if parent_strategy else "None"}

        HYPOTHESIS: {hypothesis}

        Relevant literature (numbered citations):
        {literature_context}

        Generate a **MID-LEVEL METHODOLOGY** that:
        1. Focuses on **one specific avenue** from the parent strategy.
        2. Defines the **experimental design** at a conceptual level:
        - Independent and dependent variables
        - Control or baseline conditions
        - Broad experimental procedures (no code or low-level implementation details)
        3. Suggests **data handling considerations**:
        - Types of data needed
        - Preprocessing or quality checks conceptually
        4. Recommends **evaluation metrics and success criteria** (conceptual, not exact formulas)
        5. Highlights any **robustness considerations** or potential challenges
        6. References literature **sparingly** using numbered citations [1], [2], etc.
        7. Uses reasoning **beyond the literature** to ensure a sound methodology.

        Format as **structured, human-readable text** with headings and concise subpoints. Avoid exhaustive implementation instructions.
        """

        content = await get_llm_response([{"role": "user", "content": prompt}])
        citations = self.extract_citations(content)
        score = await self.evaluate_experiment(content, hypothesis, "methodology")

        state = IdeaState(
            level="methodology",
            content=content,
            score=score,
            citations=citations,
            references=references
        )
        self.methodology_cache[cache_key] = state
        return state


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
        literature_context = self.format_literature_context(max_chunks=3)

        prompt = f"""You are an expert research evaluator. Score the following {level}-level experiment plan (strategy or methodology) on a 0–100 scale for each criterion, focusing on high-level reasoning rather than implementation details.

        Criteria:
        1. Clarity & Organization – Is the plan clearly stated, logically structured, and easy to follow? Are key ideas, experimental directions, and hypotheses communicated effectively?  
        2. Conceptual Feasibility – Are the proposed approaches or methods realistic and actionable at a conceptual level? Do they make sense for the stated hypothesis?  
        3. Novelty & Significance – Does the plan explore creative or impactful directions? How original and promising are the proposed experimental avenues?  
        4. Soundness & Alignment – Are the proposed strategies or methodologies logically coherent and aligned with the hypothesis? Do they provide a strong rationale for the research?

        Instructions:
        - Return ONLY 4 integers between 0–100, separated by commas, in this order: 
        Clarity, Conceptual Feasibility, Novelty, Soundness

        HYPOTHESIS: {hypothesis}

        EXPERIMENT PLAN ({level}):
        {content}

        Relevant literature (for reference, do not overly rely on it):
        {literature_context}
        """

        messages = [
            {"role": "system", "content": "You are an expert research evaluator. Return ONLY four numbers separated by commas."},
            {"role": "user", "content": prompt}
        ]

        response = await get_llm_response(messages, temperature=0.1)

        # Extract up to 4 integers
        numbers = re.findall(r'\d+', response)
        scores = [int(x) for x in numbers[:4]]

        # Define weights depending on node level
        # Order: Clarity, Feasibility, Novelty, Soundness
        weight_map = {
            "strategy": [0.2, 0.1, 0.4, 0.3],        # Novelty 0.4, Soundness 0.3, Clarity 0.2, Feasibility 0.1
            "methodology": [0.2, 0.3, 0.15, 0.35],    # Soundness 0.35, Feasibility 0.3, Clarity 0.2, Novelty 0.15
        }
        weights = weight_map.get(level.lower(), [0.3, 0.3, 0.2, 0.2])

        # Weighted sum → normalize to 0–1
        weighted = sum(s * w for s, w in zip(scores, weights)) / 100.0
        return min(max(weighted, 0.0), 1.0)





    def find_best_leaf_node(self, tree) -> IdeaState:
        """Find the highest-scoring leaf node (methodology level)"""
        def get_all_nodes(node):
            nodes = [node]
            for child in getattr(node, 'children', []):
                nodes.extend(get_all_nodes(child))
            return nodes
        
        all_nodes = get_all_nodes(tree.tree.root)
        leaf_nodes = [node.state for node in all_nodes 
                     if hasattr(node, 'state') and 
                     node.state.level == 'methodology' and 
                     node.state.score is not None]
        
        if not leaf_nodes:
            return None
        
        return max(leaf_nodes, key=lambda x: x.score)


# Sync wrapper functions for treequest
def sync_generate_strategy(parent_state, tree_system, hypothesis):
    try:
        # If already in an event loop, create a new one in a thread
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
    try:
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


def run_experiment_tree_search(user_input: str, num_iterations: int, writer=None, loop=None) -> Optional['IdeaState']:
    """Main function to run the experiment tree search with streaming from sync context."""

    if loop is None:
        loop = asyncio.get_event_loop()

    def sync_stream_writer(msg):
        if writer is not None:
            fut = asyncio.run_coroutine_threadsafe(
                stream_writer(msg, writer=writer, stream_mode="custom"),
                loop
            )
            # Optionally: fut.result()

    # --- Move async initialization to a helper ---
    async def async_init_and_search():
        tree_system = await ExperimentTreeSystem(user_input).initialize()
        hypothesis = tree_system.research_context.get('hypotheses', [user_input])[0]
        sync_stream_writer(f"🔎 Starting experiment tree search for: {hypothesis}")
        sync_stream_writer(f"Using {len(tree_system.literature_context)} literature chunks as context")

        def strategy_gen(parent_state):
            return sync_generate_strategy(parent_state, tree_system, hypothesis)

        def methodology_gen(parent_state):
            return sync_generate_methodology(parent_state, tree_system, hypothesis)

        generate_fns = {
            "strategy": strategy_gen,
            "methodology": methodology_gen,
        }
        sync_stream_writer("🌲 Building experiment design tree...(this may take a few mins)")

        # Monkey patch print to filter sampling messages
        original_print = print
        def filtered_print(*args, **kwargs):
            message = ' '.join(str(arg) for arg in args)
            if "Sampling:" not in message:
                original_print(*args, **kwargs)

        builtins.print = filtered_print
        try:
            algo = tq.ABMCTSA()
            search_tree = algo.init_tree()
            search_tree.tree.root.state = IdeaState(
                level="hypothesis",
                content=hypothesis,
                score=1.0
            )
        finally:
            builtins.print = original_print

        for i in range(num_iterations):
            builtins.print = filtered_print
            try:
                search_tree = algo.step(search_tree, generate_fns)
            finally:
                builtins.print = original_print

            try:
                best_interim_state, _ = tq.top_k(search_tree, algo, k=1)[0]
                sync_stream_writer(f"{i+1}/{num_iterations} iterations - best score: {best_interim_state.score:.3f} ({best_interim_state.level})")
            except Exception:
                sync_stream_writer(f"{i+1}/{num_iterations} iterations - searching...")

        best_methodology = tree_system.find_best_leaf_node(search_tree)
        if best_methodology:
            return best_methodology
        else:
            sync_stream_writer("❌ No methodology-level nodes found")
            return None

    # Run the async helper in this thread
    return asyncio.run(async_init_and_search())