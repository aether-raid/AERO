#!/usr/bin/env python3
"""
Example script demonstrating how to use the experiment suggestion workflow
with file input capability.
"""

import asyncio
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Suggest_additional_expirments.experiment_suggestion_nodes import run_experiment_suggestion_workflow_from_file


async def main():
    """
    Example usage of the experiment suggestion workflow with file input.
    """

    # Example 1: Use file content as both data and prompt
    print("Example 1: Using file as input data and prompt")
    print("=" * 50)

    try:
        filepath="example_experimental_results2.txt"
        result = await run_experiment_suggestion_workflow_from_file(
            file_path=filepath,
            # client, model, and arxiv_processor will be auto-initialized from environment
        )

        if result.get("experiment_suggestions"):
            print("✅ Success! Experiment suggestions generated.")
            print("Suggestions preview:")
            suggestions = str(result["experiment_suggestions"])
            print(suggestions[:500] + "..." if len(suggestions) > 500 else suggestions)
        else:
            print("❌ Failed to generate experiment suggestions")

    except Exception as e:
        print(f"❌ Error: {str(e)}")

    print("\n" + "=" * 50)
    print("Example 2: Using file as data with custom prompt")
    print("=" * 50)

    # Example 2: Use file as data but provide custom prompt
    custom_prompt = "Based on the experimental results in the attached file, suggest advanced techniques for improving medical image classification performance, focusing on domain adaptation and robustness."

    try:
        result2 = await run_experiment_suggestion_workflow_from_file(
            file_path="example_experimental_results.txt",
            user_prompt=custom_prompt
        )

        if result2.get("experiment_suggestions"):
            print("✅ Success! Custom prompt experiment suggestions generated.")
        else:
            print("❌ Failed to generate experiment suggestions with custom prompt")

    except Exception as e:
        print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    print("🚀 Running experiment suggestion workflow examples...")
    asyncio.run(main())