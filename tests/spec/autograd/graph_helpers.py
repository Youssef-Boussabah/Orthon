"""Finite-difference oracles built exclusively from public Tensor operations."""

import numpy as np

import orthon as ortho


def gradient_check(function, arrays, *, dtype, device):
    arrays = [np.array(array, dtype=np.float64, copy=True) for array in arrays]

    def create(values, tracked=False):
        value = ortho.from_numpy(np.array(values, dtype=dtype, order="C"))
        if device != "cpu":
            value = value.to(device)
        return value.requires_grad_(tracked)

    leaves = [create(array, True) for array in arrays]
    output = function(*leaves)
    assert output.shape == ()
    output.backward()
    step = 1e-5 if dtype == "float64" else 1e-3
    numerical = []
    for position, array in enumerate(arrays):
        estimate = np.empty_like(array)
        for index in np.ndindex(array.shape):
            plus, minus = [a.copy() for a in arrays], [a.copy() for a in arrays]
            plus[position][index] += step
            minus[position][index] -= step
            with ortho.no_grad():
                upper = function(*[create(a) for a in plus]).item()
                lower = function(*[create(a) for a in minus]).item()
            estimate[index] = (upper - lower) / (2 * step)
        numerical.append(estimate)
    return leaves, numerical
