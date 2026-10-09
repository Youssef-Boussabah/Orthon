"""Private Tensor-owner capture and scalar update protocols for optimizers."""

from dataclasses import dataclass

import orthon as ortho


@dataclass
class _Binding:
    # Strong Python owner reference; storage aliases are separate bindings.
    parameter: ortho.Tensor
    state: object = None


def _capture_bindings(params):
    parameters = list(params)
    seen = set()
    for index, parameter in enumerate(parameters):
        if not isinstance(parameter, ortho.Tensor):
            raise TypeError(f"optimizer parameter {index} must be an orthon.Tensor")
        identity = id(parameter)
        if identity in seen:
            raise ValueError(f"duplicate optimizer parameter owner at index {index}")
        seen.add(identity)
    return tuple(_Binding(parameter) for parameter in parameters)


def _scaled_add_inplace(destination, source, coefficient):
    # The native add_ alpha boundary is float-only. Semantic multiplication
    # injects the scalar in the Tensor's dtype, preserving float64 precision.
    # Keep the established float32/CUDA path and destination owner identity.
    if destination.dtype == "float64":
        destination.add_(source * coefficient)
    else:
        destination.add_(source, alpha=coefficient)
