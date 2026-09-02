import math

import orthon as ortho
from .._random import uniform as _uniform
from .module import Module, Parameter


def _init_conv_weight(C_out, C_in, K_h, K_w):
    """Kaiming uniform init for [C_out, C_in, K_h, K_w] weight."""
    fan_in = C_in * K_h * K_w
    bound = math.sqrt(6.0 / fan_in)
    t = ortho.empty([C_out, C_in, K_h, K_w])
    for f in range(C_out):
        for c in range(C_in):
            for i in range(K_h):
                for j in range(K_w):
                    t.set_item([f, c, i, j], _uniform(-bound, bound))
    return t


class Conv2d(Module):
    """2-D convolution layer.

    weight: [C_out, C_in, K_h, K_w]
    bias:   [C_out]  (optional)
    forward: output[n,f,h,w] = sum_{c,i,j} weight[f,c,i,j]*input[n,c,h*sh+i-ph,w*sw+j-pw] + bias[f]
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, bias=True):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)

        self._stride       = stride
        self._padding      = padding
        self._use_bias     = bias
        self._out_channels = out_channels

        K_h, K_w = kernel_size
        self.weight = Parameter(_init_conv_weight(out_channels, in_channels, K_h, K_w))
        if bias:
            self.bias = Parameter(ortho.zeros([out_channels]))

    def forward(self, x):
        sh, sw = self._stride
        ph, pw = self._padding
        if self._use_bias:
            b = self.bias
        else:
            # Native conv2d always takes a bias and requires all three operands
            # to share dtype and device, so the synthetic zero bias follows the
            # input rather than defaulting to CPU float32. Matching x (not the
            # weight) leaves a genuine input/weight mismatch for native
            # validation to reject.
            b = ortho.zeros([self._out_channels], x.dtype).to(x.device)
        return ortho.conv2d(x, self.weight, b, sh, sw, ph, pw)
