import pyomo.environ as pyo
import numpy as np
import operator


class SolverVariable:
    """Array-like wrapper over Pyomo decision variables supporting arithmetic."""

    def __init__(self, data):
        self.data = np.array(data, dtype=object) if not isinstance(data, np.ndarray) else data
        self.shape = self.data.shape

    def __add__(self, other):
        other_data = other.data if isinstance(other, SolverVariable) else other
        return SolverVariable(self.data + other_data)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        other_data = other.data if isinstance(other, SolverVariable) else other
        return SolverVariable(self.data - other_data)

    def __rsub__(self, other):
        other_data = other.data if isinstance(other, SolverVariable) else other
        return SolverVariable(other_data - self.data)

    def __matmul__(self, other):
        other_data = other.data if isinstance(other, SolverVariable) else other
        return SolverVariable(np.dot(self.data, other_data))

    def __mul__(self, other):
        other_data = other.data if isinstance(other, SolverVariable) else other
        return SolverVariable(self.data * other_data)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __pow__(self, power):
        if isinstance(power, (int, float)):
            flat = self.data.flatten()
            result = np.empty(len(flat), dtype=object)
            for i in range(len(flat)):
                result[i] = flat[i] ** power
            return SolverVariable(result.reshape(self.shape))
        raise NotImplementedError("Only scalar powers are supported")

    def _apply_relational(self, other, op):
        other_data = other.data if isinstance(other, SolverVariable) else other
        flat_self = self.data.flatten()
        res_array = np.empty(len(flat_self), dtype=object)

        if isinstance(other_data, np.ndarray):
            flat_other = other_data.flatten()
            for i in range(len(flat_self)):
                res_array[i] = op(flat_self[i], flat_other[i])
        else:
            for i in range(len(flat_self)):
                res_array[i] = op(flat_self[i], other_data)

        return SolverVariable(res_array.reshape(self.shape))

    def __eq__(self, other):
        return self._apply_relational(other, operator.eq)

    def __ge__(self, other):
        return self._apply_relational(other, operator.ge)

    def __le__(self, other):
        return self._apply_relational(other, operator.le)

    def __getitem__(self, key):
        return SolverVariable(self.data[key])

    def sum(self):
        return np.sum(self.data)


class SolverModel:
    """Pyomo ConcreteModel wrapper for building optimisation problems."""

    def __init__(self):
        self.model = pyo.ConcreteModel()
        self.model.constraints = pyo.ConstraintList()
        self._var_counter = 0

    def create_variable(self, shape, boolean=False, name=None, bounds=None):
        if name is None:
            name = f"var_{self._var_counter}"
            self._var_counter += 1

        domain = pyo.Binary if boolean else pyo.Reals
        flat_size = int(np.prod(shape)) if isinstance(shape, tuple) and shape else 1

        pyo_var = pyo.Var(range(flat_size), domain=domain, bounds=bounds)
        setattr(self.model, name, pyo_var)

        arr = np.array([pyo_var[i] for i in range(flat_size)], dtype=object).reshape(shape)
        return SolverVariable(arr)

    def add_constraint(self, condition):
        cond_data = condition.data if isinstance(condition, SolverVariable) else condition
        if isinstance(cond_data, np.ndarray):
            for cond in cond_data.flatten():
                self.model.constraints.add(cond)
        else:
            self.model.constraints.add(cond_data)

    def set_objective(self, expr, minimize=True):
        sense = pyo.minimize if minimize else pyo.maximize
        self.model.obj = pyo.Objective(expr=expr, sense=sense)

    def solve(self, solver_name='couenne', tee=True, time_limit=None,
              node_limit=None):
        solver = pyo.SolverFactory(solver_name)
        if time_limit is not None:
            solver.options['bonmin.time_limit'] = time_limit
        if node_limit is not None:
            solver.options['bonmin.node_limit'] = node_limit
        results = solver.solve(self.model, tee=tee)
        return results
