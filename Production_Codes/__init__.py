"""
SageMaker inference package initializer.

This file exposes inference hooks so SageMaker can
discover model_fn, input_fn, predict_fn, and output_fn.
"""

from .inference import (
    model_fn,
    input_fn,
    predict_fn,
    output_fn,
)

__all__ = [
    "model_fn",
    "input_fn",
    "predict_fn",
    "output_fn",
]
