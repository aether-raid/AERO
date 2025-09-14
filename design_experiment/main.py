"""
Experiment Design Tree Main Workflow (main.py)
==============================================

This module serves as the main entry point for the experiment design tree system. It orchestrates the workflow by:

1. Accepting user input for research plan.
2. Extracting research context and hypotheses using LLM utilities.
3. Running the experiment tree search for each hypothesis, leveraging literature retrieval and LLM-based node generation.
4. Aggregating and managing results, including best experiment designs and references.
5. (incomplete)

"""

import asyncio
from langgraph.graph import StateGraph, END
from experiment_tree import run_experiment_tree_search
from init_utils import extract_research_components

async def node_extract_components(state):
    user_input = state['user_input']
    result = await extract_research_components(user_input)
    state['research_goal'] = result.get('research_goal', '')
    state['hypotheses'] = result.get('hypotheses', [user_input])
    state['relevant_info'] = result.get('relevant_info', '')
    return state

# --- Node: Run tree workflow for each hypothesis ---
async def node_tree_search(state):
    """Run experiment tree search for each hypothesis with research context"""
    all_results = []
    
    for hyp in state['hypotheses']:
        print(f"\n🌳 Running tree search for hypothesis: {hyp}")
        
        # Create combined input that includes both hypothesis and research context
        combined_input = f"""
        Research Goal: {state.get('research_goal', 'Not specified')}
        Variables: {state.get('relevant_info', 'Not specified')}
        Additional Context: {state.get('relevant_info', 'None')}
        
        Hypothesis to test: {hyp}
        """
        
        # Run tree search with research context
        best_experiment = await run_experiment_tree_search(combined_input, num_iterations=10)
        
        all_results.append({
            'hypothesis': hyp,
            'best_experiment': best_experiment,
            'research_context': {
                'research_goal': state.get('research_goal', ''),
                'variables': state.get('relevant_info', ''),
                'additional_info': state.get('relevant_info', '')
            }
        })
    
    state['tree_results'] = all_results
    return state

def build_main_workflow():
    g = StateGraph(dict)
    g.add_node('extract_components', node_extract_components)
    g.add_node('tree_search', node_tree_search)
    g.add_edge('extract_components', 'tree_search')
    g.add_edge('tree_search', END)
    g.set_entry_point('extract_components')
    return g

async def main():
    user_input = input("Enter your research plan or hypothesis: ").strip()
    workflow = build_main_workflow()
    state = {'user_input': user_input}
    app = workflow.compile()
    results = await app.ainvoke(state)


if __name__ == "__main__":
    asyncio.run(main())
