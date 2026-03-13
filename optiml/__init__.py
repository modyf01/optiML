"""OptiML — train neural networks via global mathematical optimisation."""

__version__ = "1.0.13"

from .layers import (
    OptiModule,
    Linear,
    ReLU,
    PolyReLU,
    Conv1D,
    Conv2D,
    AvgPool2D,
    Flatten,
)
from .model import Sequential
from .solver import SolverVariable, SolverModel, is_gurobi_wls_configured

try:
    from .convex import ConvexReLUNet, DeepConvexReLUNet
except ImportError:
    ConvexReLUNet = None
    DeepConvexReLUNet = None

from . import losses

__all__ = [
    "OptiModule",
    "Sequential",
    "Linear",
    "ReLU",
    "PolyReLU",
    "Conv1D",
    "Conv2D",
    "AvgPool2D",
    "Flatten",
    "SolverVariable",
    "SolverModel",
    "is_gurobi_wls_configured",
    "ConvexReLUNet",
    "DeepConvexReLUNet",
    "losses",
]
