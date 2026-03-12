import numpy as np
import pyomo.environ as pyo

from .solver import SolverVariable


class OptiModule:
    """Base class for all OptiML layers."""

    def __init__(self):
        self._parameters = {}
        self._weight_bounds = None

    def get_parameter(self, name, shape, solver_model):
        if name not in self._parameters:
            unique_name = f"layer_{id(self)}_{name}"
            self._parameters[name] = solver_model.create_variable(
                shape, name=unique_name, bounds=self._weight_bounds,
            )
        return self._parameters[name]

    def forward(self, x, solver_model):
        raise NotImplementedError

    def __call__(self, x, solver_model):
        return self.forward(x, solver_model)

    def _extract_weights(self, param_name):
        param_solver_var = self._parameters[param_name]
        arr = np.zeros(param_solver_var.shape)
        for idx, var in np.ndenumerate(param_solver_var.data):
            arr[idx] = pyo.value(var)
        return arr

    def to_pytorch(self):
        raise NotImplementedError(
            f"PyTorch export is not implemented for {self.__class__.__name__}"
        )


class Linear(OptiModule):
    """Fully-connected layer: y = Wx + b."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x, solver_model):
        W = self.get_parameter('weight', (self.out_features, self.in_features), solver_model)
        b = self.get_parameter('bias', (self.out_features,), solver_model)
        out = solver_model.create_variable(shape=(self.out_features,))
        solver_model.add_constraint(out == W @ x + b)
        return out

    def to_pytorch(self):
        import torch
        import torch.nn as nn
        layer = nn.Linear(self.in_features, self.out_features)
        layer.weight.data = torch.tensor(self._extract_weights('weight'), dtype=torch.float32)
        layer.bias.data = torch.tensor(self._extract_weights('bias'), dtype=torch.float32)
        return layer


class ReLU(OptiModule):
    """ReLU activation: y = max(0, x).

    When ``M`` is None (default), uses complementarity formulation
    ``y*(y-x) == 0`` — no binary variables, no big-M constant.
    Requires a solver that handles non-convex quadratic constraints
    (e.g. Gurobi with ``NonConvex=2``).

    When ``M`` is given, falls back to the classical big-M formulation
    with binary indicator variables (works with any MINLP solver).
    """

    def __init__(self, M=None):
        super().__init__()
        self.M = M

    def forward(self, x, solver_model):
        shape = x.shape if hasattr(x, 'shape') else (1,)
        y = solver_model.create_variable(shape=shape)

        solver_model.add_constraint(y >= 0)
        solver_model.add_constraint(y >= x)

        if self.M is not None:
            z = solver_model.create_variable(shape=shape, boolean=True)
            solver_model.add_constraint(y <= x + self.M * (1 - z))
            solver_model.add_constraint(y <= self.M * z)
        else:
            solver_model.add_constraint(y * (y - x) == 0)

        return y

    def to_pytorch(self):
        import torch.nn as nn
        return nn.ReLU()


class PolyReLU(OptiModule):
    """Quadratic (strictly convex) approximation of ReLU.

    f(x) = 0.0937x² + 0.5x + 0.469

    All constraints are quadratic — compatible with Gurobi ``NonConvex=2``.
    No binary variables, no big-M, no auxiliary variables.

    Best with scaled inputs (e.g. MinMaxScaler to [0, 1]).
    """

    _COEFFS = (0.0937, 0.5, 0.469)

    def forward(self, x, solver_model):
        shape = x.shape if hasattr(x, 'shape') else (1,)
        a2, a1, a0 = self._COEFFS

        y = solver_model.create_variable(shape=shape)
        solver_model.add_constraint(y == a2 * (x * x) + a1 * x + a0)

        return y

    def to_pytorch(self):
        import torch.nn as nn

        a2, a1, a0 = self._COEFFS

        class _PolyReLU(nn.Module):
            def forward(self, x):
                return a2 * x ** 2 + a1 * x + a0

        return _PolyReLU()


class Conv1D(OptiModule):
    """1-D convolution layer."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x, solver_model):
        W = self.get_parameter('weight', (self.out_channels, self.in_channels, self.kernel_size), solver_model)
        b = self.get_parameter('bias', (self.out_channels,), solver_model)
        in_channels, length = x.shape
        out_length = (length - self.kernel_size) // self.stride + 1
        out = solver_model.create_variable(shape=(self.out_channels, out_length))

        for out_pos in range(out_length):
            in_start = out_pos * self.stride
            in_end = in_start + self.kernel_size
            x_slice = x[:, in_start:in_end]
            for oc in range(self.out_channels):
                conv_sum = np.sum((W[oc, :, :] * x_slice).data) + b[oc].data
                solver_model.add_constraint(out[oc, out_pos] == conv_sum)
        return out

    def to_pytorch(self):
        import torch
        import torch.nn as nn
        layer = nn.Conv1d(self.in_channels, self.out_channels, self.kernel_size, stride=self.stride)
        layer.weight.data = torch.tensor(self._extract_weights('weight'), dtype=torch.float32)
        layer.bias.data = torch.tensor(self._extract_weights('bias'), dtype=torch.float32)
        return layer


class Conv2D(OptiModule):
    """2-D convolution layer."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)

    def forward(self, x, solver_model):
        kh, kw = self.kernel_size
        sh, sw = self.stride
        W = self.get_parameter('weight', (self.out_channels, self.in_channels, kh, kw), solver_model)
        b = self.get_parameter('bias', (self.out_channels,), solver_model)
        in_channels, in_height, in_width = x.shape
        out_height = (in_height - kh) // sh + 1
        out_width = (in_width - kw) // sw + 1
        out = solver_model.create_variable(shape=(self.out_channels, out_height, out_width))

        for out_h in range(out_height):
            for out_w in range(out_width):
                h_start = out_h * sh
                h_end = h_start + kh
                w_start = out_w * sw
                w_end = w_start + kw
                x_slice = x[:, h_start:h_end, w_start:w_end]
                for oc in range(self.out_channels):
                    conv_sum = np.sum((W[oc, :, :, :] * x_slice).data) + b[oc].data
                    solver_model.add_constraint(out[oc, out_h, out_w] == conv_sum)
        return out

    def to_pytorch(self):
        import torch
        import torch.nn as nn
        layer = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, stride=self.stride)
        layer.weight.data = torch.tensor(self._extract_weights('weight'), dtype=torch.float32)
        layer.bias.data = torch.tensor(self._extract_weights('bias'), dtype=torch.float32)
        return layer


class AvgPool2D(OptiModule):
    """2-D average pooling layer."""

    def __init__(self, kernel_size, stride=None):
        super().__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        if stride is None:
            self.stride = self.kernel_size
        else:
            self.stride = stride if isinstance(stride, tuple) else (stride, stride)

    def forward(self, x, solver_model):
        kh, kw = self.kernel_size
        sh, sw = self.stride
        channels, in_height, in_width = x.shape
        out_height = (in_height - kh) // sh + 1
        out_width = (in_width - kw) // sw + 1
        out = solver_model.create_variable(shape=(channels, out_height, out_width))
        pool_area = kh * kw

        for c in range(channels):
            for out_h in range(out_height):
                for out_w in range(out_width):
                    h_start = out_h * sh
                    h_end = h_start + kh
                    w_start = out_w * sw
                    w_end = w_start + kw
                    x_slice = x[c, h_start:h_end, w_start:w_end]
                    avg_val = np.sum(x_slice.data) / pool_area
                    solver_model.add_constraint(out[c, out_h, out_w] == avg_val)
        return out

    def to_pytorch(self):
        import torch.nn as nn
        return nn.AvgPool2d(self.kernel_size, stride=self.stride)


class Flatten(OptiModule):
    """Flatten all spatial dimensions into a single vector."""

    def __init__(self):
        super().__init__()

    def forward(self, x, solver_model):
        flat_data = x.data.flatten()
        return SolverVariable(flat_data)

    def to_pytorch(self):
        import torch.nn as nn
        return nn.Flatten(start_dim=1)
