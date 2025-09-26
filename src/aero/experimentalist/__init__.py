"""Experimentalist (Experiment Suggestion Workflow).

This package exposes a high-level API for orchestrating the experiment suggestion
workflow defined in :mod:`aero.experimentalist.experiment_suggestion_nodes`.

The workflow can operate in **non-streaming** (single result) or **streaming**
mode. Both modes support attaching structured data via ``experimental_results``
and file-based evidence via ``file_path`` (e.g. CSV/XLSX summaries).

The primary entry point is :func:`experiment_suggestions`, which accepts a
``streaming`` flag to opt into streaming updates. Convenience helpers are
provided for the non-streaming (:func:`suggest_experiments`) and streaming
(:func:`stream_experiment_suggestions`) cases.

Example (non-streaming)::

    import asyncio
    from aero.experimentalist import suggest_experiments

    async def main():
        result = await suggest_experiments(
            prompt="Improve CNN generalisation",
            file_path="/path/to/experiments.xlsx"
        )
        print(result["experiment_suggestions"])

    asyncio.run(main())

Example (streaming updates)::

    import asyncio
    from aero.experimentalist import experiment_suggestions

    async def main():
        stream = await experiment_suggestions(
            prompt="Improve CNN generalisation",
            file_path="/path/to/experiments.xlsx",
            streaming=True,
        )

        async for update in stream:
            print("update", update)

    asyncio.run(main())
"""

from typing import Any, AsyncGenerator, Dict, List, Optional, Union, cast

from .experiment_suggestion_nodes import run_experiment_suggestion_workflow


ExperimentSuggestionStream = AsyncGenerator[Dict[str, Any], None]
ExperimentSuggestionResult = Union[Dict[str, Any], ExperimentSuggestionStream]


async def experiment_suggestions(
    prompt: str,
    experimental_results: Optional[Dict[str, Any]] = None,
    uploaded_data: Optional[List[str]] = None,
    file_path: Optional[str] = None,
    streaming: bool = False,
) -> ExperimentSuggestionResult:
    """Run the experiment suggestion workflow.

    Parameters
    ----------
    prompt:
        Description of the research question or experimental goal.
    experimental_results:
        Optional structured dictionary of existing experiment outcomes.
    uploaded_data:
        Optional list of string snippets (e.g. manual notes) to blend into prompts.
    file_path:
        Optional path to a CSV/XLS(X) file; contents are parsed and supplied to the
        workflow's uploaded-data context.
    streaming:
        When ``True`` the function returns an async generator yielding updates.
        When ``False`` (default) it returns the final workflow state.
    """

    result = await run_experiment_suggestion_workflow(
        user_prompt=prompt,
        experimental_results=experimental_results,
        uploaded_data=uploaded_data,
        file_path=file_path,
        streaming=streaming,
    )

    if streaming:
        return cast(ExperimentSuggestionStream, result)
    return cast(Dict[str, Any], result)


async def suggest_experiments_nostream(
    prompt: str,
    experimental_results: Optional[Dict[str, Any]] = None,
    uploaded_data: Optional[List[str]] = None,
    file_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Convenience wrapper returning the final state only."""

    return cast(
        Dict[str, Any],
        await experiment_suggestions(
            prompt=prompt,
            experimental_results=experimental_results,
            uploaded_data=uploaded_data,
            file_path=file_path,
            streaming=False,
        ),
    )


async def stream_experiment_suggestions(
    prompt: str,
    experimental_results: Optional[Dict[str, Any]] = None,
    uploaded_data: Optional[List[str]] = None,
    file_path: Optional[str] = None,
) -> ExperimentSuggestionStream:
    """Yield streaming updates from the workflow.

    Produces the same update payloads emitted by
    :func:`run_experiment_suggestion_workflow` when ``streaming=True``.
    """

    stream = cast(
        ExperimentSuggestionStream,
        await experiment_suggestions(
            prompt=prompt,
            experimental_results=experimental_results,
            uploaded_data=uploaded_data,
            file_path=file_path,
            streaming=True,
        ),
    )

    async for update in stream:
        yield update


__all__ = [
    "experiment_suggestions",
    "suggest_experiments_nostream",
    "stream_experiment_suggestions",
    "run_experiment_suggestion_workflow",
]
