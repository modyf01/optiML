"""Loss functions expressible as Pyomo optimisation objectives."""

import numpy as np


class Loss:
    """Base class for all OptiML losses."""

    def __call__(self, y_pred, y_true, solver_model=None):
        return self.compute(y_pred, y_true, solver_model)

    def compute(self, y_pred, y_true, solver_model=None):
        raise NotImplementedError


class MSELoss(Loss):
    """Mean Squared Error: (1/n) * sum((y_pred - y_true)^2)."""

    def __init__(self, reduction='mean'):
        self.reduction = reduction

    def compute(self, predictions, targets, solver_model=None):
        total = sum(
            (pred.data.item() - target) ** 2
            for pred, target in zip(predictions, targets)
        )
        if self.reduction == 'mean':
            return total / len(predictions)
        return total


class SSELoss(Loss):
    """Sum of Squared Errors: sum((y_pred - y_true)^2).

    Equivalent to MSELoss(reduction='sum') but more explicit.
    """

    def compute(self, predictions, targets, solver_model=None):
        return sum(
            (pred.data.item() - target) ** 2
            for pred, target in zip(predictions, targets)
        )


class MAELoss(Loss):
    """Mean Absolute Error modelled with auxiliary variables.

    Introduces |e| = y_pred - y_true via:
        t >= y_pred - y_true
        t >= -(y_pred - y_true)
    then minimises sum(t).
    Requires solver_model to create auxiliary variables/constraints.
    """

    def __init__(self, reduction='mean'):
        self.reduction = reduction

    def compute(self, predictions, targets, solver_model=None):
        if solver_model is None:
            raise ValueError("MAELoss requires a solver_model to create auxiliary variables")

        total_expr = 0
        for pred, target in zip(predictions, targets):
            diff = pred.data.item() - target
            t = solver_model.create_variable(shape=(1,))
            solver_model.add_constraint(t >= diff)
            solver_model.add_constraint(t >= -diff)
            total_expr = total_expr + t.data.item()

        if self.reduction == 'mean':
            return total_expr / len(predictions)
        return total_expr


class HuberLoss(Loss):
    """Huber loss (smooth L1) modelled with auxiliary variables.

    For each sample:
        if |error| <= delta: loss = 0.5 * error^2
        else:                loss = delta * (|error| - 0.5 * delta)

    Approximated as: loss = min(0.5 * e^2, delta * |e| - 0.5 * delta^2)
    via auxiliary variables.

    For simplicity this uses a quadratic formulation that works well
    with MINLP solvers when delta is moderate.
    """

    def __init__(self, delta=1.0, reduction='mean'):
        self.delta = delta
        self.reduction = reduction

    def compute(self, predictions, targets, solver_model=None):
        if solver_model is None:
            raise ValueError("HuberLoss requires a solver_model to create auxiliary variables")

        total_expr = 0
        d = self.delta
        for pred, target in zip(predictions, targets):
            diff = pred.data.item() - target
            quad = 0.5 * diff ** 2
            t_abs = solver_model.create_variable(shape=(1,))
            solver_model.add_constraint(t_abs >= diff)
            solver_model.add_constraint(t_abs >= -diff)
            linear = d * t_abs.data.item() - 0.5 * d ** 2

            h = solver_model.create_variable(shape=(1,))
            solver_model.add_constraint(h >= quad)
            solver_model.add_constraint(h >= linear)
            total_expr = total_expr + h.data.item()

        if self.reduction == 'mean':
            return total_expr / len(predictions)
        return total_expr
