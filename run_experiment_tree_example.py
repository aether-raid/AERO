"""
Simple example to run the Experiment Tree search.

Usage (PowerShell):
py    python .\run_experiment_tree_example.py "Your hypothesis here"

Notes:
- Requires OPENAI_API_KEY available via .env or env.example at project root.
- Uses the literature search workflow and treequest algorithm.
"""

import os
import sys
import asyncio
from dotenv import load_dotenv


PROJECT_DIR = os.path.dirname(__file__)

# Load environment from .env in project root, then fallback to env.example
def _load_envs_from_project_root():
    # 1) Explicitly load .env from project root if present
    dotenv_path = os.path.join(PROJECT_DIR, '.env')
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path=dotenv_path)
    else:
        # also call to load any system-level .env if present
        load_dotenv()

    # 2) Fallback: read keys from env.example in project root if still missing
    env_example = os.path.join(PROJECT_DIR, 'env.example')
    if os.path.exists(env_example):
        def ensure(key: str):
            if os.getenv(key):
                return
            try:
                with open(env_example, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith(f'{key}='):
                            value = line.split('=', 1)[1]
                            if value.startswith(("'", '"')) and value.endswith(("'", '"')):
                                value = value[1:-1]
                            if value and value not in ['your-api-key-here', 'your-base-url-here']:
                                os.environ[key] = value
                                return
            except Exception:
                pass

        for k in ("OPENAI_API_KEY", "BASE_URL", "DEFAULT_MODEL"):
            ensure(k)

_load_envs_from_project_root()

# Ensure we can import modules from the design_experiment folder
DESIGN_EXP_DIR = os.path.join(PROJECT_DIR, "design_experiment")
if DESIGN_EXP_DIR not in sys.path:
    sys.path.insert(0, DESIGN_EXP_DIR)

from experiment_tree import run_experiment_tree_search  # type: ignore


async def main():
    # Default example hypothesis; can be overridden via CLI args
    hypothesis = "Transformer-based small model distillation improves on-device ASR latency without quality loss."
    iterations = 10

    # If user provided a hypothesis on the command line, use it
    if len(sys.argv) > 1:
        hypothesis = " ".join(sys.argv[1:]).strip()

    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY is not set. Add it to .env or env.example in the project root before running.")

   

    best = await run_experiment_tree_search(hypothesis, iterations)

    if best is None:
        print("No implementation-level result produced.")
        return

    # Note: run_experiment_tree_search already prints a formatted summary.
    # Below is an extra concise recap.
    print("\n--- Summary (concise) ---")
    print(f"Best score: {best.score:.3f}")
    print(f"Level: {best.level}")
    print("First 300 chars:\n" + (best.content[:300] if best.content else ""))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user.")