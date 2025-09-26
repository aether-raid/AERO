# design_experiment/__init__.py
from aero.experiment_designer.main import run_experiment_designer
from aero.experiment_designer.main import experiment_designer 

__all__ = [
    "run_experiment_designer",  # Async entry point: takes a research plan and returns experiment design and code (along with streaming option)
    "experiment_designer"       # Returns the configured LangGraph workflow object for experiment design
]
