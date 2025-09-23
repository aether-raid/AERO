import asyncio
import sys
import os
import re
from langgraph.graph import StateGraph, END

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from design_experiment.experiment import (
    ExperimentState,
    summarize_node,
    literature_node,
    plan_node,
    design_node,
    score_node,
)
from design_experiment.utils import extract_research_components
from design_experiment.idea_tree import run_experiment_tree_search
from design_experiment.code import CodeGenState, build_codegen_graph

# --- Helper: Add arXiv links ---
def add_arxiv_links(text):
    def repl(match):
        arxiv_id = match.group(1)
        url = f"https://arxiv.org/pdf/{arxiv_id}"
        return f"([{url}]({url}))"
    return re.sub(r'\((\d{4}\.\d{5}(v\d+)?)\)', repl, text)

# --- Node: Extract research components ---
async def node_extract_components(state):
    user_input = state['user_input']
    print("🔎 Extracting research components...")
    result = await extract_research_components(user_input)
    state['research_goal'] = result.get('research_goal', '')
    state['hypotheses'] = result.get('hypotheses', [])
    state['variables'] = result.get('variables', '')
    state['relevant_info'] = result.get('relevant_info', '')
    state['experiment_ideas'] = result.get('experiment_ideas', [])
    return state

# --- Node: Tree search for experiment ideas if needed ---
async def node_tree_search(state):
    research_goal = state.get('research_goal', '')
    hypotheses = state.get('hypotheses', [])
    variables = state.get('variables', '')
    relevant_info = state.get('relevant_info', '')

    tree_search_results = []
    experiment_ideas = []

    if isinstance(hypotheses, list) and len(hypotheses) >= 2:
        print(f"🌳 Multiple hypotheses found ({len(hypotheses)}). Running tree search for each hypothesis...")
        for idx, hypothesis in enumerate(hypotheses, 1):
            print(f"\n--- Tree Search for Hypothesis {idx}: {hypothesis} ---")
            combined_input = f"""Research Goal: {research_goal}
                Variables: {variables}
                Relevant Info: {relevant_info}
                Hypotheses: {hypothesis}
                """
            best_methodology = await run_experiment_tree_search(combined_input, num_iterations=5)
            if best_methodology:
                print("\n--- BEST EXPERIMENT DESIGN (TREE SEARCH) ---")
                print(best_methodology.content)
                tree_search_results.append({
                    "hypothesis": hypothesis,
                    "best_methodology": best_methodology.content
                })
                experiment_ideas.append(best_methodology.content)
    else:
        print("🌳 No explicit experiment ideas found. Running tree search to generate experiment ideas based on hypotheses...")
        combined_input = f"""Research Goal: {research_goal}
            Variables: {variables}
            Relevant Info: {relevant_info}
            Hypotheses: {', '.join(hypotheses) if hypotheses else 'N/A'}
            """
        best_methodology = await run_experiment_tree_search(combined_input, num_iterations=7)
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
    print(f"🧪 Found {len(experiment_ideas)} experiment idea(s). Generating detailed designs...")
    for idx, exp in enumerate(experiment_ideas, 1):
        print(f"\n=== Experiment Idea {idx} ===")
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
        exp_state = await literature_node(exp_state)
        exp_state = await plan_node(exp_state)
        exp_state = await design_node(exp_state)
        # Cyclic refinement loop
        max_refinements = 3
        for _ in range(max_refinements):
            exp_state = await score_node(exp_state)
            scores = exp_state.scores or {}
            if scores and all(int(v) >= 70 for v in scores.values()):
                break
            exp_state.refinement_round += 1
            exp_state = await design_node(exp_state)
        # Add arXiv links to references
        design_text = exp_state.refined_design_content or exp_state.full_design_content
        design_text = add_arxiv_links(design_text)
        # --- Run codegen workflow directly (no subgraph, just call nodes) ---
        code_state = CodeGenState(experiment_input=design_text)
        code_graph = build_codegen_graph()
        final_code_state = await code_graph.ainvoke(code_state)
        if isinstance(final_code_state, dict):
            final_design_with_code = final_code_state.get('final_output', design_text)
        else:
            final_design_with_code = getattr(final_code_state, 'final_output', design_text)
        all_designs.append({
            "experiment_idea": exp_desc,
            "design": design_text,
            "refinements": exp_state.refinement_suggestions,
            "code": final_design_with_code
        })
        print("\n--- FINAL EXPERIMENT DESIGN WITH INLINE CODE ---")
        print(final_design_with_code)
        print("\n--- FINAL REFINEMENT SUGGESTIONS ---")
        if exp_state.refinement_suggestions:
            for s in exp_state.refinement_suggestions:
                print(f"- {s.strip('- ')}")
    state['all_designs'] = all_designs
    return state

# --- Conditional edge: decide which path to take ---
async def decide_next_node(state):
    if state.get('experiment_ideas'):
        return 'design_and_codegen'
    else:
        return 'tree_search'

# --- Build unified LangGraph workflow ---
def unified_design_experiment_workflow():
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
    workflow = unified_design_experiment_workflow()
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

if __name__ == "__main__":
    user_input = input("Enter your research plan: ")
    result = run_design_workflow(user_input)
    print("\n=== FINAL EXPERIMENT DESIGN ===\n")
    print(result["design"])
    print("\n=== GENERATED CODE ===\n")
    print(result["code"])
    # Save to Markdown file
    with open("experiment_designs_output1.md", "w", encoding="utf-8") as f:
        f.write("# Final Experiment Design\n\n")
        f.write(result["design"] + "\n\n")
        f.write("## Generated Code\n\n")
        f.write("```python\n")
        f.write(result["code"] + "\n")
        f.write("```\n")
    print("\nOutput saved to experiment_designs_output1.md")