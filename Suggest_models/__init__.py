"""
Suggest Models - ML Model Suggestion Workflow

This module provides an automated workflow for suggesting machine learning models
based on user requirements. The workflow analyzes the task, searches arXiv for
relevant research papers, and provides evidence-based model recommendations.

Main Functions:
- run_model_suggestion_workflow: Run the complete model suggestion workflow
- suggest_models: Convenience wrapper function
"""

from .model_suggestion_nodes import (
    run_model_suggestion_workflow
)
from typing import List, Dict, Any, Optional, AsyncGenerator

# Expose the main function with a cleaner interface
async def suggest_models(
    prompt: str,
    uploaded_data: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Suggest suitable ML models for a given task.

    This is a convenience wrapper around run_model_suggestion_workflow.

    Args:
        prompt: Description of the ML task or problem
        uploaded_data: Optional list of additional data files/content

    Returns:
        Dictionary containing workflow results including model suggestions

    Example:
        import asyncio
        from Suggest_models import suggest_models

        result = asyncio.run(suggest_models("I need help with image classification"))
        print(result['model_suggestions']['model_suggestions'])
    """
    return await run_model_suggestion_workflow(
        user_prompt=prompt,
        uploaded_data=uploaded_data
    )

# Export all public functions
__all__ = [
    "suggest_models",
    "run_model_suggestion_workflow"
]