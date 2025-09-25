"""
Experimentalist - ML Experiment Suggestion Workflow

This module provides an automated workflow for suggesting machine learning experiments
based on user requirements. The workflow analyzes research questions, searches for
relevant papers, distills methodologies, and suggests novel experimental approaches.

Main Functions:
- run_experiment_suggestion_workflow: Run the complete experiment suggestion workflow
"""

from .experiment_suggestion_nodes import run_experiment_suggestion_workflow
from typing import List, Dict, Any, Optional

# Expose the main function with a cleaner interface
async def suggest_experiments(
    prompt: str,
    experimental_results: Optional[Dict[str, Any]] = None,
    uploaded_data: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Suggest suitable ML experiments for a given research question.

    This is a convenience wrapper around run_experiment_suggestion_workflow.

    Args:
        prompt: Description of the research question or experimental goal
        experimental_results: Optional dictionary of existing experimental results
        uploaded_data: Optional list of additional data files/content.
            NOTE: Currently not processed by the workflow - reserved for future use.

    Returns:
        Dictionary containing workflow results including experiment suggestions

    Example:
        import asyncio
        from aero.experimentalist import suggest_experiments

        result = asyncio.run(suggest_experiments("How can I improve my CNN accuracy?"))
        print(result['experiment_suggestions'])
    """
    return await run_experiment_suggestion_workflow(
        user_prompt=prompt,
        experimental_results=experimental_results,
        uploaded_data=uploaded_data
    )

# Also expose the full function for advanced users
__all__ = ['suggest_experiments', 'run_experiment_suggestion_workflow']
