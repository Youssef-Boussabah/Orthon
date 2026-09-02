#include <orthon/ops/cpu/matmul_ops.h>
#include <orthon/ops/cpu/fill_ops.h>
#include <orthon/ops/cpu/shape_ops.h>
#include <orthon/core/dtype.h>
#include <orthon/autograd/grad_mode.h>
#include <orthon/autograd/matmul_node.h>
#include <stdexcept>
#include <string>

#ifdef WITH_CUDA
#include <orthon/ops/cuda/matmul_kernels.h>
#endif

namespace orthon {

static void validate_matmul(const Tensor& a, const Tensor& b) {
    if (!((a.ndim() == 2 && b.ndim() == 2) ||
          (a.ndim() == 3 && b.ndim() == 3)))
        throw std::runtime_error("matmul: only 2D or batched 3D tensors supported");
    if (a.ndim() == 2 && a.shape()[1] != b.shape()[0])
        throw std::runtime_error(
            "matmul: shape mismatch: (" +
            std::to_string(a.shape()[0]) + ", " + std::to_string(a.shape()[1]) +
            ") @ (" +
            std::to_string(b.shape()[0]) + ", " + std::to_string(b.shape()[1]) + ")");
    if (a.ndim() == 3 &&
        (a.shape()[0] != b.shape()[0] || a.shape()[2] != b.shape()[1]))
        throw std::runtime_error(
            "matmul: shape mismatch for batched 3D tensors");
}

template<typename T>
static Tensor cpu_matmul(const Tensor& a, const Tensor& b) {
    const T* pa = static_cast<const T*>(a.data());
    const T* pb = static_cast<const T*>(b.data());

    std::vector<int64_t> out_shape;
    if (a.ndim() == 2) {
        out_shape = {a.shape()[0], b.shape()[1]};
    } else {
        out_shape = {a.shape()[0], a.shape()[1], b.shape()[2]};
    }
    Tensor out = empty(out_shape, a.dtype(), a.device());
    T* dst = static_cast<T*>(out.data());

    if (a.ndim() == 2) {
        const int64_t m   = a.shape()[0];
        const int64_t k   = a.shape()[1];
        const int64_t n   = b.shape()[1];
        const int64_t sa0 = a.strides()[0], sa1 = a.strides()[1];
        const int64_t sb0 = b.strides()[0], sb1 = b.strides()[1];
        for (int64_t i = 0; i < m; ++i)
            for (int64_t j = 0; j < n; ++j) {
                T acc = T{0};
                for (int64_t p = 0; p < k; ++p)
                    acc += pa[i * sa0 + p * sa1] * pb[p * sb0 + j * sb1];
                dst[i * n + j] = acc;
            }
    } else {
        const int64_t batch = a.shape()[0];
        const int64_t m     = a.shape()[1];
        const int64_t k     = a.shape()[2];
        const int64_t n     = b.shape()[2];
        const int64_t sa0 = a.strides()[0], sa1 = a.strides()[1], sa2 = a.strides()[2];
        const int64_t sb0 = b.strides()[0], sb1 = b.strides()[1], sb2 = b.strides()[2];
        for (int64_t ba = 0; ba < batch; ++ba)
            for (int64_t i = 0; i < m; ++i)
                for (int64_t j = 0; j < n; ++j) {
                    T acc = T{0};
                    for (int64_t p = 0; p < k; ++p)
                        acc += pa[ba * sa0 + i * sa1 + p * sa2] *
                               pb[ba * sb0 + p * sb1 + j * sb2];
                    dst[ba * m * n + i * n + j] = acc;
                }
    }
    return out;
}

Tensor matmul(const Tensor& a, const Tensor& b) {
    validate_matmul(a, b);

#ifdef WITH_CUDA
    if (a.device().is_cuda() || b.device().is_cuda()) {
        if (a.device() != b.device())
            throw std::runtime_error("matmul: device mismatch");
        if ((a.dtype() != DType::Float32 && a.dtype() != DType::Float16 && a.dtype() != DType::BFloat16) ||
            a.dtype() != b.dtype())
            throw std::runtime_error("matmul: CUDA path requires float32, float16, or bfloat16 (same dtype)");
        // Kernel-input materialisation, not part of the public graph -
        // MatMulNode below attaches to the ORIGINAL a/b. It goes through
        // kernel_operand(), which carries its own NoGradGuard, so every CUDA
        // op normalises views through ONE audited helper rather than a
        // per-site copy.
        Tensor a_kernel = kernel_operand(a);
        Tensor b_kernel = kernel_operand(b);
        Tensor out = a.ndim() == 2
            ? cuda_matmul_tiled(a_kernel, b_kernel)
            : cuda_batched_matmul_tiled(a_kernel, b_kernel);
        const bool req = is_grad_enabled() && (a.requires_grad() || b.requires_grad());
        out.requires_grad_(req);
        out.set_is_leaf(!req);
        if (req)
            out.set_grad_fn(std::make_shared<MatMulNode>(a, b));
        return out;
    }
#endif

    // CPU path: float32 or float64.
    if (a.dtype() != b.dtype())
        throw std::runtime_error("matmul: dtype mismatch");
    if (!a.device().is_cpu())
        throw std::runtime_error("matmul: unsupported device");

    Tensor out = [&]() -> Tensor {
        if (a.dtype() == DType::Float32) return cpu_matmul<float>(a, b);
        if (a.dtype() == DType::Float64) return cpu_matmul<double>(a, b);
        throw std::runtime_error("matmul: only float32/float64 supported on CPU");
    }();

    const bool req = is_grad_enabled() && (a.requires_grad() || b.requires_grad());
    out.requires_grad_(req);
    out.set_is_leaf(!req);
    if (req)
        out.set_grad_fn(std::make_shared<MatMulNode>(a, b));
    return out;
}

}  // namespace orthon
