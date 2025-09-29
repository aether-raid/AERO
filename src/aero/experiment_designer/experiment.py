from __future__ import annotations
import re
import json
import asyncio
from dataclasses import dataclass, field
from aero.experiment_designer.utils import get_llm_response, stream_writer
from aero.experiment_designer.search import build_experiment_search_workflow, search_dataset_online

@dataclass
class ExperimentState:
    user_input: str = ""
    experiment_input: str = ""
    plan_content: str = ""
    full_design_content: str = ""
    literature_chunks: list = field(default_factory=list)
    literature_context: str = ""
    scores: dict = field(default_factory=dict)
    refined_design_content: str = ""
    refinement_suggestions: list = field(default_factory=list)
    refinement_round: int = 0
    summary_query: str = ""
    generated_code: str = ""  
    messages: list = field(default_factory=list)  

# --- Node: Summarize experiment design for FAISS search ---
async def summarize_node(state: ExperimentState) -> ExperimentState:
    prompt = f"""
        You are an expert research assistant. 
        Summarize the following experiment description into a concise query suitable for searching a scientific experiment database. 
        Focus on the main modalities, methods, and goals.

        EXPERIMENT DESCRIPTION:
        {state.experiment_input}

        Output only the search query, no explanation.
        """
    summary = await get_llm_response([{"role": "user", "content": prompt}])
    state.summary_query = summary.strip()
    return state

# --- Node: Use experiment search workflow to retrieve relevant chunks ---
async def literature_node(state: ExperimentState, writer=None) -> ExperimentState:
    workflow = await build_experiment_search_workflow(writer=writer)
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
async def plan_node(state: ExperimentState, writer = None) -> ExperimentState:
    await stream_writer("Generating experiment plan requirements...", writer=writer, stream_mode="custom")

    prompt = f"""
        You are an expert researcher. Given the following experiment description and relevant literature, create a PLAN listing everything needed to execute the experiment.

        EXPERIMENT DESCRIPTION:
        {state.experiment_input}

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
async def design_node(state: ExperimentState, writer=None) -> ExperimentState:
    """
    Generates or refines a detailed experiment design.
    Adds [CODE_NEEDED: ...] tags for steps that can be executed in code.
    """
    # Helper: augmentation instructions to add code tags
    code_tag_instructions = """
    You are an expert researcher generating a detailed, step-by-step experiment design.

    For every major section or subsection (e.g., "Preprocessing", "Modeling", "Evaluation") that can be implemented in code, insert a single [CODE_NEEDED: <short description>] tag at the BOTTOM of that section or subsection. Do NOT add a tag for every individual step or bullet point—only one tag per self-contained, runnable section.

    - The description inside the tag must clearly state what code is needed (e.g., [CODE_NEEDED: Load EEG data from .edf files using MNE-Python]).
    - Do NOT use [CODE_NEEDED] without a description.
    - After the main design, provide a summary list of all [CODE_NEEDED] tags and their descriptions.

    Follow these rules:
    1. Only place one tag per major section or subsection that represents a logical, runnable code block.
    2. Do NOT place a tag after every bullet or micro-step.
    3. Nested tags are allowed for multi-step procedures, but each snippet must be independently executable.
    4. Include all sentences necessary for the code in the same tag.
    5. Provide a concise description for each tag.
    6. After generating the full plan with tags, return a summary list of tags and their descriptions.
    """

    # Use refined design and suggestions if this is a refinement round
    if state.refinement_round > 0:
        state.refinement_round += 1
        await stream_writer("Refining experiment design...", writer=writer, stream_mode="custom")

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
            
            {code_tag_instructions}

            """
        content = await get_llm_response([{"role": "user", "content": prompt}])
        state.refined_design_content = content.strip()
    else:
        state.refinement_round += 1
        await stream_writer("Generating initial experiment design...", writer=writer, stream_mode="custom")

        prompt = f"""
            You are an expert researcher. Given the following context (experiment description, relevant literature, and experiment plan), generate a DETAILED, step-by-step experiment design.

            Context (for your reference only):
            EXPERIMENT DESCRIPTION:
            {state.experiment_input}

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
                  
            {code_tag_instructions}

            """
    content = await get_llm_response([{"role": "user", "content": prompt}])
    if state.refinement_round > 0:
        state.refined_design_content = content.strip()
    else:
        state.full_design_content = content.strip()
    return state

# --- Node: Score experiment design and suggest refinements ---
async def score_node(state: ExperimentState, writer=None) -> ExperimentState:
    await stream_writer("Evaluating experiment design...", writer=writer, stream_mode="custom")

    design = state.refined_design_content if state.refinement_round > 0 else state.full_design_content
    prompt = f"""
            You are an expert research evaluator and advisor. Review the following experiment design in detail.

            1. Score the experiment on a 0–100 scale for each criterion:
            - Feasibility & Knowledge Basis: Can it realistically be executed with available resources? Is it grounded in established scientific knowledge and principles?
            - Goal & Hypothesis Alignment: Is the outcome of the design well aligned with the research goal/hypothesis?
            - Level of Detail: Is the experiment sufficiently detailed, including datasets, tools, variables, and procedures?

            2. Suggest specific refinements to improve the experiment design, focusing on:
            - Adding or specifying relevant example datasets (with names if possible)
            - Clarifying or detailing any vague steps
            - Improving reproducibility or feasibility
            - Enhancing novelty or significance

            Instructions:
            - Return a structured output in JSON format with the following fields:
            {{
                "scores": {{"feasibility": int, "goal_alignment": int, "detail": int}},
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
    # Remove code block markers if present
    if response.startswith("```"):
        response = re.sub(r"^```[a-zA-Z]*\n?", "", response)
        response = response.rstrip("`").strip()
    # Extract the first {...} JSON object if extra text is present
    json_match = re.search(r"\{.*\}", response, re.DOTALL)
    if json_match:
        response = json_match.group(0)
    
    try:
        parsed = json.loads(response)
        state.scores = parsed.get("scores", {})
        state.refinement_suggestions = parsed.get("refinements", [])
    except Exception as e:
        await stream_writer(("Scoring JSON parse error:", e), writer=writer, stream_mode="custom")
        state.scores = None
        state.refinement_suggestions = [response.strip()]
    if not state.scores:
        await stream_writer("No scores available.", writer=writer, stream_mode="custom")
    else:
        # Format scores as a readable string
        scores_str = "Scores:\n" + "\n".join(f"  - {k}: {v}" for k, v in state.scores.items())
        await stream_writer(scores_str, writer=writer, stream_mode="custom")

    return state

def remove_code_tags(text):
    # Remove [CODE_NEEDED: ...] and [CODE_NEEDED] ... tags (entire line)
    text = re.sub(r'\[CODE_NEEDED(?::[^\]]*)?\][^\n]*\n?', '', text)
    # Remove summary list section (header and following bullets)
    text = re.sub(
        r'(?i)Summary List of \[CODE_NEEDED\][^\n]*\n(?:\s*[\d\-\*\.]+\s*[^\n]+\n?)*', 
        '', text
    )
    # Remove any leftover "Summary List of ..." header on its own line
    text = re.sub(r'(?i)^Summary List of.*$', '', text, flags=re.MULTILINE)
    # Remove lines with only asterisks or whitespace
    text = re.sub(r'^\s*[\*\-]+\s*$', '', text, flags=re.MULTILINE)
    # Remove lines with only whitespace
    text = re.sub(r'^\s*$', '', text, flags=re.MULTILINE)
    return text.strip()
