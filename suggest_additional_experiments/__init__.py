"""
Suggest Additional Experiments - ML Experiment Suggestion Workflow

This module provides an automated workflow for suggesting follow-up experiments
based on completed research. The workflow analyzes experimental results, searches
arXiv for relevant research papers, and provides evidence-based experiment recommendations.

Main Functions:
- run_experiment_suggestion_workflow: Run the complete experiment suggestion workflow
- run_experiment_suggestion_workflow_from_file: Run workflow with file input
"""

from .experiment_suggestion_nodes import run_experiment_suggestion_workflow, run_experiment_suggestion_workflow_from_file
from typing import List, Dict, Any, Optional

# Expose the main function with a cleaner interface
async def suggest_experiments(
    prompt: str,
    experimental_results: Optional[Dict[str, Any]] = None,
    uploaded_data: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Suggest follow-up experiments based on completed research.

    This is a convenience wrapper around run_experiment_suggestion_workflow.

    Args:
        prompt: Description of the research context and goals
        experimental_results: Optional dictionary with experimental data/results
        uploaded_data: Optional list of additional data files/content

    Returns:
        Dictionary containing workflow results including experiment suggestions

    Example:
        import asyncio
        from Suggest_additional_expirments import suggest_experiments

        result = asyncio.run(suggest_experiments(
            "I completed CNN experiments for image classification",
            experimental_results={
                "model_performance": {"accuracy": 0.87},
                "training_details": {"epochs": 50}
            }
        ))
        print(result['experiment_suggestions'])
    """
    return await run_experiment_suggestion_workflow(
        user_prompt=prompt,
        experimental_results=experimental_results,
        uploaded_data=uploaded_data
    )

# Also expose the full functions for advanced users
__all__ = ['suggest_experiments', 'run_experiment_suggestion_workflow', 'run_experiment_suggestion_workflow_from_file']