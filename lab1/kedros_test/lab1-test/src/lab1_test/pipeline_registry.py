from typing import Dict
from kedro.pipeline import Pipeline
from .pipelines.assignment.pipeline import create_pipeline as assignment_pipe

def register_pipelines() -> Dict[str, Pipeline]:
    assignment = assignment_pipe()
    return {
        "assignment": assignment,
        "__default__": assignment,
    }
