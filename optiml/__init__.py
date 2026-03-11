"""OptiML — train neural networks via global mathematical optimisation."""

__version__ = "0.1.0"

from .layers import (
    OptiModule,
    Linear,
    ReLU,
    Conv1D,
    Conv2D,
    AvgPool2D,
    Flatten,
)
from .model import Sequential
from .solver import SolverVariable, SolverModel, is_gurobi_wls_configured
from . import losses

__all__ = [
    "OptiModule",
    "Sequential",
    "Linear",
    "ReLU",
    "Conv1D",
    "Conv2D",
    "AvgPool2D",
    "Flatten",
    "SolverVariable",
    "SolverModel",
    "is_gurobi_wls_configured",
    "losses",
]
