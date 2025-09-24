# design_experiment/__init__.py
from aero.experiment_designer.main import run_design_workflow
from aero.experiment_designer.main import design_experiment_workflow 

__all__ = [
    "run_design_workflow",  # Main entry: INPUT = research plan, OUTPUT = experiment design, executable code
    "design_experiment_workflow"  # Returns the LangGraph workflow object for advanced use
]
