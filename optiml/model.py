"""High-level Sequential model that wraps layer definition, fitting, and export."""

from collections import OrderedDict

import numpy as np
import pyomo.environ as pyo

from .solver import SolverModel, SolverVariable
from .layers import OptiModule
from .losses import Loss, MSELoss


class Sequential(OptiModule):
    """An ordered container of OptiML layers with fit/export capabilities.

    Usage::

        import optiml

        model = optiml.Sequential(
            optiml.Linear(2, 2),
            optiml.ReLU(M=10),
            optiml.Linear(2, 1),
        )

        model.fit(X_train, y_train, solver='couenne')
        pytorch_model = model.export('pytorch')
    """

    def __init__(self, *layers):
        super().__init__()
        self._layers = OrderedDict()
        for i, layer in enumerate(layers):
            name = f"{layer.__class__.__name__.lower()}_{i}"
            self._layers[name] = layer
            setattr(self, name, layer)

        self._solver_model = None
        self._fitted = False

    def forward(self, x, solver_model):
        out = x
        for layer in self._layers.values():
            out = layer(out, solver_model)
        return out

    def fit(
        self,
        X,
        y,
        loss=None,
        solver='couenne',
        weight_decay=0.0,
        weight_bounds=None,
        time_limit=None,
        node_limit=None,
        tee=True,
        verbose=True,
    ):
        """Build the optimisation model from data and solve it.

        Parameters
        ----------
        X : array-like, shape (n_samples, ...)
            Training inputs (without batch dimension per sample).
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            Training targets.
        loss : Loss, optional
            Loss function to minimise. Defaults to ``MSELoss(reduction='sum')``.
        solver : str
            Name of a Pyomo-compatible MINLP solver (e.g. ``'couenne'``, ``'scip'``).
        weight_decay : float
            L2 regularisation coefficient.  Adds ``weight_decay * sum(w²)``
            to the objective.  Prevents over-fitting and encourages small
            weights suitable for quantisation on micro-controllers.
        weight_bounds : tuple[float, float] or None
            If given, bound every trainable weight to ``(lo, hi)``.
            Tightens the solver relaxation and prevents degenerate solutions.
            Also useful for fixed-point quantisation on edge devices.
        time_limit : float or None
            Maximum solver wall-clock time in seconds.  The solver returns
            the best feasible solution found within the time budget.
        tee : bool
            Whether to print solver output.
        verbose : bool
            Whether to print progress messages.

        Returns
        -------
        results : SolverResults
            Pyomo solver results object.
        """
        if loss is None:
            loss = MSELoss(reduction='sum')

        X = np.asarray(X)
        y = np.asarray(y)

        if weight_bounds is not None:
            for layer in self._layers.values():
                layer._weight_bounds = weight_bounds

        self._solver_model = SolverModel()
        sm = self._solver_model

        if verbose:
            print(f"[OptiML] Building constraints for {len(X)} samples...")

        predictions = []
        for i in range(len(X)):
            x_input = X[i]
            if not isinstance(x_input, SolverVariable):
                x_input = SolverVariable(x_input)
            y_pred = self.forward(x_input, sm)
            predictions.append(y_pred)

        if verbose:
            n_vars = sm._var_counter
            n_cons = len(sm.model.constraints)
            print(f"[OptiML] Model has {n_vars} variable groups, {n_cons} constraints.")
            print(f"[OptiML] Computing {loss.__class__.__name__}...")

        objective_expr = loss(predictions, y, sm)

        if weight_decay > 0:
            reg = 0
            for layer in self._layers.values():
                for param_sv in layer._parameters.values():
                    for val in param_sv.data.flatten():
                        reg += val ** 2
            objective_expr += weight_decay * reg
            if verbose:
                print(f"[OptiML] Added L2 regularisation (λ={weight_decay})")

        if isinstance(objective_expr, SolverVariable):
            if objective_expr.data.size == 1:
                objective_expr = objective_expr.data.item()
        sm.set_objective(objective_expr, minimize=True)

        if verbose:
            print(f"[OptiML] Solving with '{solver}'...")

        results = sm.solve(solver_name=solver, tee=tee, time_limit=time_limit,
                           node_limit=node_limit)
        self._fitted = True

        if verbose:
            try:
                obj_val = pyo.value(sm.model.obj)
                print(f"[OptiML] Solved. Objective value: {obj_val:.8f}")
            except Exception:
                print("[OptiML] Solved. (could not read objective value)")

        return results

    @property
    def objective_value(self):
        """Return the objective value after fitting."""
        if not self._fitted:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        return pyo.value(self._solver_model.model.obj)

    def export(self, backend='pytorch'):
        """Export the fitted model to a deep-learning framework.

        Parameters
        ----------
        backend : str
            Target framework. Currently only ``'pytorch'`` is supported.

        Returns
        -------
        model
            A native model in the target framework with weights loaded
            from the optimisation solution.
        """
        if not self._fitted:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        if backend == 'pytorch':
            return self._export_pytorch()
        raise ValueError(f"Unknown backend '{backend}'. Supported: 'pytorch'")

    def _export_pytorch(self):
        import torch.nn as nn

        pytorch_layers = []
        for name, layer in self._layers.items():
            pytorch_layers.append((name, layer.to_pytorch()))
        return nn.Sequential(OrderedDict(pytorch_layers))

    def __repr__(self):
        lines = [f"Sequential("]
        for name, layer in self._layers.items():
            lines.append(f"  ({name}): {layer.__class__.__name__}()")
        lines.append(")")
        return "\n".join(lines)
