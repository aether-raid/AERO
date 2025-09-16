import asyncio
from langgraph.graph import StateGraph, END
from design_experiment.experiment import ExperimentState, build_experiment_graph
from design_experiment.init_utils import extract_research_components
from design_experiment.idea_tree import run_experiment_tree_search

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

# --- Node: For each experiment idea, run experiment design workflow ---
async def node_run_experiment_design(state):
    experiment_ideas = state.get('experiment_ideas', [])
    research_goal = state.get('research_goal', '')
    hypotheses = state.get('hypotheses', [])
    variables = state.get('variables', '')
    relevant_info = state.get('relevant_info', '')

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
        initial_state = ExperimentState(user_input=experiment_input.strip())
        app = build_experiment_graph()
        final_state_dict = await app.ainvoke(initial_state)
        final_state = ExperimentState(**final_state_dict)
        all_designs.append({
            "experiment_idea": exp_desc,
            "design": final_state.refined_design_content or final_state.full_design_content,
            "refinements": final_state.refinement_suggestions
        })
        print("\n--- FINAL EXPERIMENT DESIGN ---")
        print(final_state.refined_design_content or final_state.full_design_content)
        print("\n--- FINAL REFINEMENT SUGGESTIONS ---")
        if final_state.refinement_suggestions:
            for s in final_state.refinement_suggestions:
                print(f"- {s.strip('- ')}")
    state['all_designs'] = all_designs
    return state

# --- Node: Run tree search if no experiment ideas ---
async def node_tree_search(state):
    research_goal = state.get('research_goal', '')
    hypotheses = state.get('hypotheses', [])
    variables = state.get('variables', '')
    relevant_info = state.get('relevant_info', '')

    tree_search_results = []
    all_designs = []

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
                # Now run build_experiment_graph using this methodology as the experiment idea
                experiment_input = f"""Research Goal: {research_goal}
                    Hypotheses: {hypothesis}
                    Variables: {variables}
                    Relevant Info: {relevant_info}
                    Experiment Idea: {best_methodology.content}
                    """
                initial_state = ExperimentState(user_input=experiment_input.strip())
                app = build_experiment_graph()
                final_state_dict = await app.ainvoke(initial_state)
                final_state = ExperimentState(**final_state_dict)
                all_designs.append({
                    "hypothesis": hypothesis,
                    "experiment_idea": best_methodology.content,
                    "design": final_state.refined_design_content or final_state.full_design_content,
                    "refinements": final_state.refinement_suggestions
                })
                print("\n--- FINAL EXPERIMENT DESIGN ---")
                print(final_state.refined_design_content or final_state.full_design_content)
                print("\n--- FINAL REFINEMENT SUGGESTIONS ---")
                if final_state.refinement_suggestions:
                    for s in final_state.refinement_suggestions:
                        print(f"- {s.strip('- ')}")
    else:
        print("🌳 No explicit experiment ideas found or only one hypothesis. Running tree search...")
        combined_input = f"""Research Goal: {research_goal}
            Variables: {variables}
            Relevant Info: {relevant_info}
            Hypotheses: {', '.join(hypotheses) if hypotheses else 'N/A'}
            """
        best_methodology = await run_experiment_tree_search(combined_input, num_iterations=5)
        if best_methodology:
            tree_search_results.append({
                "hypothesis": hypotheses[0] if hypotheses else 'N/A',
                "best_methodology": best_methodology.content
            })
            experiment_input = f"""Research Goal: {research_goal}
                Hypotheses: {hypotheses[0] if hypotheses else 'N/A'}
                Variables: {variables}
                Relevant Info: {relevant_info}
                Experiment Idea: {best_methodology.content}
                """
            initial_state = ExperimentState(user_input=experiment_input.strip())
            app = build_experiment_graph()
            final_state_dict = await app.ainvoke(initial_state)
            final_state = ExperimentState(**final_state_dict)
            all_designs.append({
                "hypothesis": hypotheses[0] if hypotheses else 'N/A',
                "experiment_idea": best_methodology.content,
                "design": final_state.refined_design_content or final_state.full_design_content,
                "refinements": final_state.refinement_suggestions
            })
            print("\n--- FINAL EXPERIMENT DESIGN ---")
            print(final_state.refined_design_content or final_state.full_design_content)
            print("\n--- FINAL REFINEMENT SUGGESTIONS ---")
            if final_state.refinement_suggestions:
                for s in final_state.refinement_suggestions:
                    print(f"- {s.strip('- ')}")

    state['tree_search_result'] = tree_search_results
    state['all_designs'] = all_designs
    return state

# --- Conditional edge: decide which path to take ---
async def decide_next_node(state):
    if state.get('experiment_ideas'):
        return 'run_experiment_design'
    else:
        return 'tree_search'

# --- Build LangGraph workflow ---
def design_experiment_workflow():
    g = StateGraph(dict)
    g.add_node('extract_components', node_extract_components)
    g.add_node('run_experiment_design', node_run_experiment_design)
    g.add_node('tree_search', node_tree_search)
    g.add_conditional_edges('extract_components', decide_next_node)
    g.add_edge('run_experiment_design', END)
    g.add_edge('tree_search', END)
    g.set_entry_point('extract_components')
    return g

async def main():
    print("Paste your full research plan (end with an empty line):")
    user_input = ""
    while True:
        line = input()
        if line.strip() == "":
            break
        user_input += line + "\n"

    workflow = design_experiment_workflow()
    state = {'user_input': user_input}
    app = workflow.compile()
    await app.ainvoke(state)

if __name__ == "__main__":
    asyncio.run(main())