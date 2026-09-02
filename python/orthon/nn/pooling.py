import orthon as ortho
from .module import Module


class MaxPool2d(Module):
    """2-D max pooling layer for NCHW tensors."""

    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if stride is None:
            stride = kernel_size
        elif isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)

        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = padding

    def forward(self, x):
        kh, kw = self._kernel_size
        sh, sw = self._stride
        ph, pw = self._padding
        return ortho.maxpool2d(x, kh, kw, sh, sw, ph, pw)
