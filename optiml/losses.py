"""Loss functions expressible as Pyomo optimisation objectives.

All losses support both scalar outputs (single-target regression /
binary classification) and vector outputs (multi-target regression /
multi-class classification with one-hot targets).
"""

import numpy as np


class Loss:
    """Base class for all OptiML losses."""

    def __call__(self, y_pred, y_true, solver_model=None):
        return self.compute(y_pred, y_true, solver_model)

    def compute(self, y_pred, y_true, solver_model=None):
        raise NotImplementedError


def _squared_error_terms(predictions, targets):
    """Yield (p - t)**2 for every scalar in every sample."""
    for pred, target in zip(predictions, targets):
        p_flat = pred.data.flatten()
        t_flat = np.atleast_1d(target).flatten()
        for p_val, t_val in zip(p_flat, t_flat):
            yield (p_val - t_val) ** 2


class MSELoss(Loss):
    """Mean Squared Error: (1/n) * sum((y_pred - y_true)^2)."""

    def __init__(self, reduction='mean'):
        self.reduction = reduction

    def compute(self, predictions, targets, solver_model=None):
        total = sum(_squared_error_terms(predictions, targets))
        if self.reduction == 'mean':
            return total / len(predictions)
        return total


class SSELoss(Loss):
    """Sum of Squared Errors: sum((y_pred - y_true)^2).

    Equivalent to MSELoss(reduction='sum') but more explicit.
    """

    def compute(self, predictions, targets, solver_model=None):
        return sum(_squared_error_terms(predictions, targets))


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
            p_flat = pred.data.flatten()
            t_flat = np.atleast_1d(target).flatten()
            for p_val, t_val in zip(p_flat, t_flat):
                diff = p_val - t_val
                t = solver_model.create_variable(shape=(1,))
                solver_model.add_constraint(t >= diff)
                solver_model.add_constraint(t >= -diff)
                total_expr = total_expr + t.data.item()

        if self.reduction == 'mean':
            return total_expr / len(predictions)
        return total_expr


class HuberLoss(Loss):
    """Huber loss (smooth L1) modelled with auxiliary variables.

    For each scalar output element:
        if |error| <= delta: loss = 0.5 * error^2
        else:                loss = delta * (|error| - 0.5 * delta)

    Requires solver_model to create auxiliary variables/constraints.
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
            p_flat = pred.data.flatten()
            t_flat = np.atleast_1d(target).flatten()
            for p_val, t_val in zip(p_flat, t_flat):
                diff = p_val - t_val
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
