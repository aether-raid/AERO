from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Any
import asyncio
import re
from langgraph.graph import StateGraph, END
from design_experiment.init_utils import get_llm_response
from design_experiment.search import build_experiment_search_workflow, search_dataset_online
import json

MAX_REFINEMENT_ROUNDS = 2

@dataclass
class ExperimentState:
    user_input: str
    summary_query: str = ""
    literature_chunks: list = field(default_factory=list)
    literature_context: str = ""
    plan_content: str = ""
    full_design_content: str = ""
    scores: dict = field(default_factory=dict)
    explanations: dict = field(default_factory=dict)
    refinement_suggestions: list = field(default_factory=list)
    refined_design_content: str = ""
    refinement_round: int = 0
    plan_state: Any = None
    full_design_state: Any = None

# --- Node: Summarize experiment design for FAISS search ---
async def summarize_node(state: ExperimentState) -> ExperimentState:
    prompt = f"""
        You are an expert research assistant. 
        Summarize the following experiment description into a concise query suitable for searching a scientific experiment database. 
        Focus on the main modalities, methods, and goals.

        EXPERIMENT DESCRIPTION:
        {state.user_input}

        Output only the search query, no explanation.
        """
    summary = await get_llm_response([{"role": "user", "content": prompt}])
    state.summary_query = summary.strip()
    return state

# --- Node: Use experiment search workflow to retrieve relevant chunks ---
async def literature_node(state: ExperimentState) -> ExperimentState:
    workflow = await build_experiment_search_workflow()
    app = workflow.compile()
    search_state = {'hypothesis': state.summary_query}
    result = await app.ainvoke(search_state)
    state.literature_chunks = result.get('results', [])[:5]
    # Format context
    context = "=== RELEVANT LITERATURE ===\n"
    for idx, chunk in enumerate(state.literature_chunks, 1):
        title = chunk.get('paper_title', 'Unknown')
        url = chunk.get('source_url', '')
        text = chunk.get('text', '')
        context += f"[{idx}] {title} ({url})\n{text}\n\n"
    state.literature_context = context
    return state

# --- Helper: Enrich datasets in plan with real links ---
def extract_canonical_dataset_name(raw_name: str) -> str:
    # Remove anything after a colon, parenthesis, or dash
    name = re.split(r'[:\(\)\-]', raw_name)[0]
    # Remove common suffixes
    name = re.sub(r'\bdataset\b', '', name, flags=re.IGNORECASE)
    return name.strip()


async def async_search_dataset_online(name):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, search_dataset_online, name, 1)

async def enrich_datasets_with_links(text):
    match = re.search(r"(1\.\s*Datasets.*?)(?=\n\d+\.)", text, re.DOTALL)
    if not match:
        return text
    datasets_text = match.group(1)
    lines = datasets_text.splitlines()
    tasks = []
    dataset_info = []
    for line in lines:
        m = re.match(r"(\s*-\s*)([^\:]+)(:?\s*)(.*)", line)
        if m:
            prefix, name, colon, desc = m.groups()
            name_clean = name.strip()
            canonical_name = extract_canonical_dataset_name(name_clean)
            if len(canonical_name) > 3:
                tasks.append(async_search_dataset_online(canonical_name))
                dataset_info.append((prefix, name_clean, colon, desc))
            else:
                dataset_info.append((prefix, name_clean, colon, desc))
                tasks.append(None)
        else:
            dataset_info.append((None, line, None, None))
            tasks.append(None)
    # Run all searches in parallel
    results = await asyncio.gather(*(t if t else asyncio.sleep(0) for t in tasks))
    enriched_lines = []
    for idx, (prefix, name_clean, colon, desc) in enumerate(dataset_info):
        if prefix is None:
            enriched_lines.append(name_clean)
            continue
        link = ""
        if results[idx]:
            link = results[idx][0]['link'] if results[idx] else ""
        if link:
            enriched_line = f"{prefix}{name_clean} ([link]({link})){colon}{desc}"
        else:
            enriched_line = f"{prefix}{name_clean}{colon}{desc}"
        enriched_lines.append(enriched_line)
    enriched_datasets_text = "\n".join(enriched_lines)
    return text.replace(datasets_text, enriched_datasets_text)

# --- Node: Plan experiment requirements ---
async def plan_node(state: ExperimentState) -> ExperimentState:
    print("📝 Generating experiment plan requirements...")
    prompt = f"""
        You are an expert researcher. Given the following experiment description and relevant literature, create a PLAN listing everything needed to execute the experiment.

        EXPERIMENT DESCRIPTION:
        {state.user_input}

        {state.literature_context}

        Instructions:
        1. List necessary datasets (names and descriptions), tools, and instruments.
        2. Identify variables to measure.
        3. Suggest experimental conditions and controls.
        4. Include evaluation metrics or success criteria.
        5. Keep output structured and human-readable.
        6. Do NOT restate the experiment description; use it as provided.
        """
    content = await get_llm_response([{"role": "user", "content": prompt}])
    # Enrich datasets section with real links (if found)
    state.plan_content = await enrich_datasets_with_links(content)
    return state

# --- Node: Generate (or refine) experiment design ---
async def design_node(state: ExperimentState) -> ExperimentState:
    # Use refined design and suggestions if this is a refinement round
    if state.refinement_round > 0:
        state.refinement_round += 1
        print("🔧 Refining experiment design...")
        prev_design = state.refined_design_content or state.full_design_content
        suggestions = "\n".join(f"- {s.strip('- ')}" for s in state.refinement_suggestions if s.strip())
        prompt = f"""
            You are an expert researcher. Refine the following experiment design by implementing the suggested improvements below. 
            Make the design clearer, more detailed, more feasible, and more novel/significant as appropriate.

            PREVIOUS DESIGN:
            {prev_design}

            SUGGESTED IMPROVEMENTS:
            {suggestions}

            Instructions:
            - Output ONLY the improved experiment design, starting directly with the required sections.
            """
        content = await get_llm_response([{"role": "user", "content": prompt}])
        state.refined_design_content = content.strip()
    else:
        state.refinement_round += 1
        print("💡 Generating initial experiment design...")
        prompt = f"""
            You are an expert researcher. Given the following context (experiment description, relevant literature, and experiment plan), generate a DETAILED, step-by-step experiment design.

            Context (for your reference only):
            EXPERIMENT DESCRIPTION:
            {state.user_input}

            {state.literature_context}

            EXPERIMENT PLAN:
            {state.plan_content}

            Instructions:
            - Output ONLY the full experiment design, starting directly with the required sections below. Do NOT restate or summarize the experiment description, literature, or plan at the top.
            - For each section below, provide detailed, step-by-step instructions and justifications.
            - Do NOT list the same dataset more than once. For each dataset, provide a valid, direct link and a citation.
            - For each dataset, only include a link if it is present in the provided literature or is a verifiable, official source (e.g., OpenNeuro, PhysioNet, official lab websites).
            - Do NOT invent or guess dataset links. If no public link is available, state "No public link available" or "Dataset available upon request".
            - For each dataset, provide its name, a brief description, and a citation from the literature context above.
            - If no suitable dataset is found in the literature, state this explicitly.
            - Use numbered citations [1], [2], etc. matching the literature context above.
            - Include all of the following sections:
                1. Datasets (with names, links, and citations)
                2. Tools & Instruments
                3. Variables to Measure
                4. Experimental Procedures (step-by-step)
                5. Experimental Conditions and Controls
                6. Evaluation Metrics and Success Criteria
                7. References (numbered, matching citations in the text)
            """
    content = await get_llm_response([{"role": "user", "content": prompt}])
    if state.refinement_round > 0:
        state.refined_design_content = content.strip()
    else:
        state.full_design_content = content.strip()
    return state

# --- Node: Score experiment design and suggest refinements ---
async def score_node(state: ExperimentState) -> ExperimentState:
    print("💯 Evaluating experiment design...")
    design = state.refined_design_content if state.refinement_round > 0 else state.full_design_content
    prompt = f"""
            You are an expert research evaluator and advisor. Review the following experiment design in detail.

            1. Score the experiment on a 0–100 scale for each criterion:
            - Feasibility & Knowledge Basis: Can it realistically be executed with available resources? Is it grounded in established scientific knowledge and principles?
            - Goal & Hypothesis Alignment: Is the outcome of the design well aligned with the research goal/hypothesis?
            - Level of Detail: Is the experiment sufficiently detailed, including datasets, tools, variables, and procedures?

            2. For each score, provide a brief explanation of why it did not receive 100.

            3. Suggest specific refinements to improve the experiment design, focusing on:
            - Adding or specifying relevant example datasets (with names if possible)
            - Clarifying or detailing any vague steps
            - Improving reproducibility or feasibility
            - Enhancing novelty or significance

            Instructions:
            - Return a structured output in JSON format with the following fields:
            {{
                "scores": {{"feasibility": int, "goal_alignment": int, "detail": int}},
                "explanations": {{"feasibility": str, "goal_alignment": str, "detail": str}},
                "refinements": [str]
            }}

            EXPERIMENT DESIGN:
            {design}
            """
    response = await get_llm_response([{"role": "user", "content": prompt}])

    # --- Clean up code block and parse JSON ---
    if isinstance(response, list) and len(response) == 1:
        response = response[0]
    response = response.strip()
    if response.startswith("```"):
        response = response.split("```")[-2] if "```" in response else response
    try:
        parsed = json.loads(response)
        state.scores = parsed.get("scores", {})
        state.explanations = parsed.get("explanations", {})
        state.refinement_suggestions = parsed.get("refinements", [])
    except Exception as e:
        state.scores = None
        state.explanations = None
        state.refinement_suggestions = [response.strip()]
    return state

# --- LangGraph workflow with cyclic refinement ---
def build_experiment_graph():
    graph = StateGraph(ExperimentState)
    graph.add_node("summarize", summarize_node)
    graph.add_node("literature", literature_node)
    graph.add_node("plan", plan_node)
    graph.add_node("design", design_node)
    graph.add_node("score", score_node)

    graph.add_edge("summarize", "literature")
    graph.add_edge("literature", "plan")
    graph.add_edge("plan", "design")
    graph.add_edge("design", "score")

    # Conditional edge: if refinement_round < MAX_REFINEMENT_ROUNDS and suggestions exist, go back to design
    async def refinement_decision(state: ExperimentState):
    # Only refine if: suggestions exist, and refinement_round < MAX_REFINEMENT_ROUNDS
    # refinement_round starts at 0 (initial), so allow up to MAX_REFINEMENT_ROUNDS total
        if (state.refinement_suggestions and
            any(s.strip() for s in state.refinement_suggestions) and
            state.refinement_round < MAX_REFINEMENT_ROUNDS):
            return "design"
        else:
            return END

    graph.add_conditional_edges("score", refinement_decision)
    graph.set_entry_point("summarize")
    return graph.compile()