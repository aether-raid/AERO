import asyncio
import re
from langgraph.graph import StateGraph, END
from aero.experiment_designer.experiment import (
    ExperimentState,
    summarize_node,
    literature_node,
    plan_node,
    design_node,
    score_node,
    remove_code_tags
)
from aero.experiment_designer.utils import extract_research_components, stream_writer, stream_step_name
from aero.experiment_designer.idea_tree import run_experiment_tree_search
from aero.experiment_designer.code import CodeGenState, build_codegen_graph


# --- Helper: Add arXiv links ---
def add_arxiv_links(text):
    def repl(match):
        arxiv_id = match.group(1)
        url = f"https://arxiv.org/pdf/{arxiv_id}"
        return f"([{url}]({url}))"
    return re.sub(r'\((\d{4}\.\d{5}(v\d+)?)\)', repl, text)

# --- Node: Extract research components ---
@stream_step_name
async def node_extract_components(state):
    user_input = state['user_input']
    await stream_writer("Extracting research components...", writer=state.get("writer"), stream_mode="custom")
    await asyncio.sleep(0.5)
    result = await extract_research_components(user_input)
    state['research_goal'] = result.get('research_goal', '')
    state['hypotheses'] = result.get('hypotheses', [])
    state['variables'] = result.get('variables', '')
    state['relevant_info'] = result.get('relevant_info', '')
    state['experiment_ideas'] = result.get('experiment_ideas', [])

    # Format hypotheses as a bulleted list
    hypotheses = state['hypotheses']
    if isinstance(hypotheses, list):
        hypotheses_str = "\n".join(f"- {h}" for h in hypotheses)
    else:
        hypotheses_str = str(hypotheses)

    # Format experiment ideas as a bulleted list
    experiment_ideas = state['experiment_ideas']
    if isinstance(experiment_ideas, list):
        ideas_str = ""
        for idx, idea in enumerate(experiment_ideas, 1):
            if isinstance(idea, dict):
                name = idea.get("name", f"Idea {idx}")
                details = idea.get("details", "")
                ideas_str += f"{idx}. {name}: {details}\n"
            else:
                ideas_str += f"{idx}. {idea}\n"
    else:
        ideas_str = str(experiment_ideas)

    components_str = (
        f"Research Goal: {state['research_goal']}\n"
        f"Hypotheses:\n{hypotheses_str}\n"
        f"Variables: {state['variables']}\n"
        f"Relevant Info: {state['relevant_info']}\n"
        f"Experiment Ideas:\n{ideas_str}"
    )

    await stream_writer(
        components_str,
        writer=state.get("writer"),
        stream_mode="custom"
    )
    return state

# --- Node: Tree search for experiment ideas if needed ---
@stream_step_name
async def node_tree_search(state):
    research_goal = state.get('research_goal', '')
    hypotheses = state.get('hypotheses', [])
    variables = state.get('variables', '')
    relevant_info = state.get('relevant_info', '')

    tree_search_results = []
    experiment_ideas = []

    if isinstance(hypotheses, list) and len(hypotheses) >= 2:
        await stream_writer(f"No explicit experiment ideas found. Multiple hypotheses found ({len(hypotheses)}). Running tree search for each hypothesis...", writer=state.get("writer"), stream_mode="custom")
        for idx, hypothesis in enumerate(hypotheses, 1):
            await stream_writer(f"\n--- Tree Search for Hypothesis {idx}: {hypothesis} ---", writer=state.get("writer"), stream_mode="custom")
            combined_input = f"""Research Goal: {research_goal}
                Variables: {variables}
                Relevant Info: {relevant_info}
                Hypotheses: {hypothesis}
                """
            loop = asyncio.get_running_loop()
            best_methodology = await asyncio.to_thread(
                run_experiment_tree_search,
                combined_input,
                10,
                state.get("writer"),
                loop
            )
            if best_methodology:
                tree_search_results.append({
                    "hypothesis": hypothesis,
                    "best_methodology": best_methodology.content
                })
                experiment_ideas.append(best_methodology.content)

    else:
        await stream_writer(f"No explicit experiment ideas found. Running tree search to generate experiment ideas based on hypotheses...", writer=state.get("writer"), stream_mode="custom")
        combined_input = f"""Research Goal: {research_goal}
            Variables: {variables}
            Relevant Info: {relevant_info}
            Hypotheses: {', '.join(hypotheses) if hypotheses else 'N/A'}
            """
        loop = asyncio.get_running_loop()
        best_methodology = await asyncio.to_thread(
            run_experiment_tree_search,
            combined_input,
            10,
            state.get("writer"),
            loop
        )
        if best_methodology:
            tree_search_results.append({
                "hypothesis": hypotheses[0] if hypotheses else 'N/A',
                "best_methodology": best_methodology.content
            })
            experiment_ideas.append(best_methodology.content)

    state['experiment_ideas'] = experiment_ideas
    state['tree_search_result'] = tree_search_results
    return state

# --- Node: For each experiment idea, run design, refinement, and codegen ---
async def node_design_and_codegen(state):
    research_goal = state.get('research_goal', '')
    hypotheses = state.get('hypotheses', [])
    variables = state.get('variables', '')
    relevant_info = state.get('relevant_info', '')
    experiment_ideas = state.get('experiment_ideas', [])

    all_designs = []
    await stream_writer(f"Found {len(experiment_ideas)} experiment idea(s). Generating detailed designs...", writer=state.get("writer"), stream_mode="custom")
    await asyncio.sleep(0.5)  # Allow stream message to appear before LLM calls
    for idx, exp in enumerate(experiment_ideas, 1):
        await stream_writer(f"Experiment Idea {idx}:", writer=state.get("writer"), stream_mode="custom")
        exp_desc = exp.get('description', exp) if isinstance(exp, dict) else str(exp)
        experiment_input = f"""Research Goal: {research_goal}
            Hypotheses: {', '.join(hypotheses) if hypotheses else 'N/A'}
            Variables: {variables}
            Relevant Info: {relevant_info}
            Experiment Idea: {exp_desc}
            """
        exp_state = ExperimentState(experiment_input=experiment_input.strip())
        # Run all experiment design nodes in sequence
        exp_state = await summarize_node(exp_state)

        await stream_writer(f"experiment {idx} - literature_search", writer=state.get("writer"), stream_mode="update")
        exp_state = await literature_node(exp_state, writer=state.get("writer"))

        await stream_writer(f"experiment {idx} - design_experiment", writer=state.get("writer"), stream_mode="update")
        exp_state = await plan_node(exp_state, writer=state.get("writer"))
        exp_state = await design_node(exp_state, writer=state.get("writer"))
        
        # Cyclic refinement loop
        max_refinements = 3
        for _ in range(max_refinements):
            exp_state = await score_node(exp_state, writer=state.get("writer"))
            scores = exp_state.scores or {}
            if scores and all(int(v) >= 70 for v in scores.values()):
                break
            exp_state.refinement_round += 1
            exp_state = await design_node(exp_state, writer=state.get("writer"))

        # Add arXiv links to references
        design_text = exp_state.refined_design_content or exp_state.full_design_content
        design_text = add_arxiv_links(design_text)

        # --- Run codegen workflow directly ---
        await stream_writer(f"experiment {idx} - code_generation", writer=state.get("writer"), stream_mode="update")
        code_state = CodeGenState(experiment_input=design_text)
        code_graph = build_codegen_graph(writer=state.get("writer"))
        final_code_state = await code_graph.ainvoke(code_state)
        if isinstance(final_code_state, dict):
            final_design_with_code = final_code_state.get('final_output', design_text)
        else:
            final_design_with_code = getattr(final_code_state, 'final_output', design_text)

        # Clean up any remaining code tags in the final design
        cleaned_design = remove_code_tags(design_text)
        cleaned_code = remove_code_tags(final_design_with_code)
        
        all_designs.append({
            "experiment_idea": exp_desc,
            "design": cleaned_design,
            "refinements": exp_state.refinement_suggestions,
            "code": cleaned_code
        })
    state['all_designs'] = all_designs
    return state

# --- Conditional edge: decide which path to take ---
async def decide_next_node(state):
    if state.get('experiment_ideas'):
        return 'design_and_codegen'
    else:
        return 'tree_search'

# --- Build unified LangGraph workflow ---
def experiment_designer():
    """
    Build and return the experiment design workflow as a LangGraph StateGraph.

    The workflow extracts research components, optionally performs tree search for experiment ideas,
    and generates/refines experiment designs and code.

    Returns:
        StateGraph: The configured experiment design workflow.
    """
    g = StateGraph(dict)
    g.add_node('extract_components', node_extract_components)
    g.add_node('tree_search', node_tree_search)
    g.add_node('design_and_codegen', node_design_and_codegen)
    g.add_conditional_edges('extract_components', decide_next_node)
    g.add_edge('tree_search', 'design_and_codegen')
    g.add_edge('design_and_codegen', END)
    g.set_entry_point('extract_components')
    return g

# --- Runner ---
def run_design_workflow(user_input: str):
    workflow = experiment_designer()
    state = {'user_input': user_input}
    app = workflow.compile()
    output_state = asyncio.run(app.ainvoke(state))
    # Extract the first experiment design and code from all_designs
    all_designs = output_state.get("all_designs", [])
    if all_designs:
        first_design = all_designs[0]
        design = first_design.get("design", "No design generated.")
        code = first_design.get("code", "No code generated.")
    else:
        design = "No design generated."
        code = "No code generated."
    return {
        "design": design,
        "code": code
    }


async def yield_design_workflow(user_input: str):
    """
    Async generator that yields status messages as they happen,
    and finally yields the final result as a dict.
    """
    workflow = experiment_designer()
    state = {'user_input': user_input}
    queue = asyncio.Queue()

    async def writer(msg):
        await queue.put(msg)

    state["writer"] = writer
    app = workflow.compile()

    # Run the workflow in the background
    async def run_workflow():
        await app.ainvoke(state)

    task = asyncio.create_task(run_workflow())

    # Yield status messages as they arrive
    while True:
        try:
            msg = await asyncio.wait_for(queue.get(), timeout=0.1)
            yield msg  # Stream status message in real time
        except asyncio.TimeoutError:
            if task.done():
                break

    # Yield the final result
    output_state = await app.ainvoke(state)
    all_designs = output_state.get("all_designs", [])
    if all_designs:
        first_design = all_designs[0]
        design = first_design.get("design", "No design generated.")
        code = first_design.get("code", "No code generated.")
    else:
        design = "No design generated."
        code = "No code generated."
    final_result = {
        "design": design,
        "code": code
    }
    yield final_result  # Yield the final result as the last message

async def run_experiment_designer(user_input: str, stream: bool = False):
    """
    Run the experiment design workflow for a given research plan.

    If stream=True, returns an async generator that yields status messages and the final result as they become available.
    If stream=False, returns the final result as a dictionary after the workflow completes.

    Args:
        user_input (str): The research plan or prompt.
        stream (bool): Whether to stream status messages in real time.

    Returns:
        Async generator (if stream=True) or dict (if stream=False) with experiment design and code.
    """
    if stream:
        return yield_design_workflow(user_input)
    else:
        return await run_design_workflow(user_input)