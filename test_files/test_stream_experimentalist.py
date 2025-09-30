#!/usr/bin/env python3
"""Minimal test for stream_experiment_suggestions function."""

import asyncio
import os
import sys
from dotenv import load_dotenv

# Add project to path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_SRC = os.path.join(ROOT_DIR, "..", "src")
if PROJECT_SRC not in sys.path:
    sys.path.insert(0, PROJECT_SRC)

async def main():
    """Minimal test of stream_experiment_suggestions."""
    load_dotenv()
    
    from aero.experimentalist import stream_experiment_suggestions
    
    # Minimal prompt
    prompt = "Improve CNN image classification performance"
    
    # Minimal experimental results
    experimental_results = {
        "model_performance": {"accuracy": 0.85},
        "observed_issues": ["overfitting", "class imbalance"]
    }
    
    # Run streaming workflow
    final_result = None
    async for update in stream_experiment_suggestions(
        prompt=prompt,
        experimental_results=experimental_results
    ):
        # Print basic status updates
        
        
        # Keep track of final result
        final_result = update
    
    # Print final experiment suggestions
    print("\n" + "="*50)
    print("FINAL EXPERIMENT SUGGESTIONS:")
    print("="*50)
    
    if final_result and "experiment_suggestions" in final_result:
        suggestions = final_result["experiment_suggestions"]
        if isinstance(suggestions, str):
            print(suggestions)
        else:
            print(f"Generated suggestions: {type(suggestions).__name__}")
    else:
        print("No experiment suggestions generated")
    
    print("="*50)

if __name__ == "__main__":
    asyncio.run(main())