"""Model Researcher - ML Model Suggestion Workflow.

This module exposes convenience wrappers around the underlying workflow graph
defined in :mod:`model_suggestion_nodes`. The workflow analyzes a user-provided
machine learning task, performs literature search on arXiv, and returns
evidence-backed model recommendations.

Exports:
    - :func:`run_model_suggestion_workflow`: Full workflow entry point with
      optional streaming updates.
    - :func:`run_model_suggestion_workflow_nonstream`: Legacy helper that
      always returns the final state.
    - :func:`suggest_models`: High-level helper that can run in streamed or
      non-streamed mode.
    - :func:`stream_model_suggestions`: Async generator yielding incremental
      updates from the streaming workflow.
"""

from .model_suggestion_nodes import (
    run_model_suggestion_workflow,
    run_model_suggestion_workflow_nonstream,
)
from typing import AsyncGenerator, Dict, Any, Optional, List, Union

# Expose the main function with a cleaner interface
async def suggest_models(
    prompt: str,
    uploaded_data: Optional[List[str]] = None,
    *,
    streaming: bool = False,
) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
    """Suggest suitable ML models for a given task.

    Args:
        prompt: Description of the ML task or problem.
        uploaded_data: Optional list of additional data files/content. Reserved
            for future use.
        streaming: When ``True`` the function returns an async generator that
            yields incremental updates from the workflow. When ``False`` (the
            default) the function resolves to the final workflow state.

    Returns:
        Either the final workflow state (``streaming=False``) or an async
        generator yielding incremental updates (``streaming=True``).

    Example:
        ```python
        import asyncio
        from aero.model_researcher import suggest_models

        result = asyncio.run(suggest_models("I need help with image classification"))
        print(result["model_suggestions"]["model_suggestions"])

        async def demo_stream():
            stream = await suggest_models(
                "Stream results for GAN architectures",
                streaming=True,
            )
            async for update in stream:
                print(update.get("status"))
        asyncio.run(demo_stream())
        ```
    """
    if streaming:
        return await run_model_suggestion_workflow(
            user_prompt=prompt,
            uploaded_data=uploaded_data,
            streaming=True,
        )

    return await run_model_suggestion_workflow(
        user_prompt=prompt,
        uploaded_data=uploaded_data
    )

async def stream_model_suggestions(
    prompt: str,
    uploaded_data: Optional[List[str]] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    """Yield incremental updates from the streaming workflow.

    This helper wraps :func:`run_model_suggestion_workflow` and ensures callers
    can simply iterate with ``async for`` without handling low-level details.
    """
    stream = await run_model_suggestion_workflow(
        user_prompt=prompt,
        uploaded_data=uploaded_data,
        streaming=True,
    )

    async for update in stream:
        yield update


__all__ = [
    "suggest_models",
    "stream_model_suggestions",
    "run_model_suggestion_workflow",
    "run_model_suggestion_workflow_nonstream",
]