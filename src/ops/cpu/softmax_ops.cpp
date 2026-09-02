#include <orthon/ops/cpu/softmax_ops.h>
#include <orthon/ops/cpu/fill_ops.h>
#include <orthon/ops/cpu/unary_ops.h>
#include <orthon/autograd/grad_mode.h>
#include <orthon/autograd/log_softmax_node.h>
#include <orthon/autograd/softmax_node.h>
#include <orthon/core/dtype.h>
#include <orthon/core/tensor_impl.h>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <limits>
#include <string>

#ifdef WITH_CUDA
#include <orthon/ops/cuda/softmax_kernels.h>
#endif

namespace orthon {

static int64_t resolve_dim(int64_t dim, int64_t ndim, const char* name) {
    const int64_t original = dim;
    if (dim < 0)
        dim += ndim;
    if (dim < 0 || dim >= ndim)
        throw std::out_of_range(
            std::string(name) + ": dim " + std::to_string(original) +
            " out of range for tensor with " + std::to_string(ndim) +
            " dimensions");
    return dim;
}

static void validate_softmax_input(const Tensor& t, const char* name) {
    if (t.ndim() == 0)
        throw std::runtime_error(std::string(name) + ": scalar tensors are not supported");
    if (t.dtype() != DType::Float32)
        throw std::runtime_error(std::string(name) + ": only float32 supported");
    if (t.numel() == 0)
        throw std::runtime_error(std::string(name) + ": empty tensor");
    if (!t.is_contiguous())
        throw std::runtime_error(std::string(name) + ": requires a contiguous tensor");
}

static void dim_factors(const Tensor& t,
                        int64_t dim,
                        int64_t& outer,
                        int64_t& reduce,
                        int64_t& inner) {
    const auto& sh = t.shape();
    outer = 1;
    for (int64_t i = 0; i < dim; ++i)
        outer *= sh[static_cast<size_t>(i)];
    reduce = sh[static_cast<size_t>(dim)];
    inner = 1;
    for (int64_t i = dim + 1; i < t.ndim(); ++i)
        inner *= sh[static_cast<size_t>(i)];
}

// Numerical forward only: never attaches a graph. `resolved` is already
// normalised and `t` already validated.
static Tensor softmax_forward_values(const Tensor& t, int64_t resolved) {
    // Fast path: the pre-existing dedicated row-wise implementations, kept for
    // the 2D last-dim shape the benchmarks and the common case use.
    if (t.ndim() == 2 && resolved == 1) {
#ifdef WITH_CUDA
        if (t.device().is_cuda())
            return cuda_softmax(t);
#endif
        if (!t.device().is_cpu())
            throw std::runtime_error("softmax: unsupported device");

        const int64_t M = t.shape()[0];
        const int64_t N = t.shape()[1];
        Tensor out = empty(t.shape(), t.dtype(), t.device());

        const float* src = static_cast<const float*>(t.data());
        float*       dst = static_cast<float*>(out.data());

        for (int64_t i = 0; i < M; ++i) {
            const float* row_src = src + i * N;
            float*       row_dst = dst + i * N;

            float rmax = *std::max_element(row_src, row_src + N);

            float rsum = 0.0f;
            for (int64_t j = 0; j < N; ++j) {
                row_dst[j] = std::exp(row_src[j] - rmax);
                rsum += row_dst[j];
            }
            for (int64_t j = 0; j < N; ++j)
                row_dst[j] /= rsum;
        }

        return out;
    }

    // General case: log_softmax already handles every rank and dimension
    // stably on both devices, so exponentiating it is both correct and
    // numerically safe - no second softmax kernel is needed.
    //
    // The guard matters: without it this would attach LogSoftmaxNode and an exp
    // ActivationNode underneath the SoftmaxNode the caller gets, exposing the
    // internal decomposition as graph structure. Public softmax presents one
    // intentional node; how the values are produced is implementation detail.
    NoGradGuard guard;
    return exp_t(log_softmax(t, resolved));
}

Tensor softmax(const Tensor& t, int64_t dim) {
    validate_softmax_input(t, "softmax");
    const int64_t resolved = resolve_dim(dim, t.ndim(), "softmax");

    Tensor out = softmax_forward_values(t, resolved);

    if (is_grad_enabled() && t.requires_grad()) {
        out.requires_grad_(true);
        out.set_is_leaf(false);
        out.set_grad_fn(std::make_shared<SoftmaxNode>(t, resolved));
    }
    return out;
}

static Tensor attach_log_softmax_grad(const Tensor& input, Tensor out, int64_t dim) {
    if (is_grad_enabled() && input.requires_grad()) {
        out.requires_grad_(true);
        out.set_is_leaf(false);
        out.set_grad_fn(std::make_shared<LogSoftmaxNode>(input, dim));
    }
    return out;
}

Tensor log_softmax(const Tensor& t, int64_t dim) {
    validate_softmax_input(t, "log_softmax");
    const int64_t resolved = resolve_dim(dim, t.ndim(), "log_softmax");

#ifdef WITH_CUDA
    if (t.device().is_cuda())
        return attach_log_softmax_grad(t, cuda_log_softmax(t, resolved), resolved);
#endif

    if (!t.device().is_cpu())
        throw std::runtime_error("log_softmax: unsupported device");

    int64_t outer, reduce, inner;
    dim_factors(t, resolved, outer, reduce, inner);
    Tensor out = empty(t.shape(), t.dtype(), t.device());

    const float* src = static_cast<const float*>(t.data());
    float* dst = static_cast<float*>(out.data());

    for (int64_t o = 0; o < outer; ++o) {
        for (int64_t in = 0; in < inner; ++in) {
            const int64_t base = o * reduce * inner + in;
            float vmax = -std::numeric_limits<float>::infinity();
            for (int64_t r = 0; r < reduce; ++r)
                vmax = std::max(vmax, src[base + r * inner]);

            double sum_exp = 0.0;
            for (int64_t r = 0; r < reduce; ++r)
                sum_exp += std::exp(static_cast<double>(src[base + r * inner] - vmax));
            const float log_denom = vmax + static_cast<float>(std::log(sum_exp));

            for (int64_t r = 0; r < reduce; ++r)
                dst[base + r * inner] = src[base + r * inner] - log_denom;
        }
    }

    return attach_log_softmax_grad(t, out, resolved);
}

Tensor log_softmax_backward(const Tensor& input,
                            const Tensor& grad_output,
                            int64_t dim) {
    validate_softmax_input(input, "log_softmax backward");
    if (grad_output.shape() != input.shape())
        throw std::runtime_error("log_softmax backward: grad_output shape mismatch");
    if (grad_output.dtype() != DType::Float32)
        throw std::runtime_error("log_softmax backward: grad_output must be float32");
    if (grad_output.device() != input.device())
        throw std::runtime_error("log_softmax backward: device mismatch");
    if (!grad_output.is_contiguous())
        throw std::runtime_error("log_softmax backward: grad_output must be contiguous");
    const int64_t resolved = resolve_dim(dim, input.ndim(), "log_softmax backward");

#ifdef WITH_CUDA
    if (input.device().is_cuda())
        return cuda_log_softmax_backward(input, grad_output, resolved);
#endif

    if (!input.device().is_cpu())
        throw std::runtime_error("log_softmax backward: unsupported device");

    int64_t outer, reduce, inner;
    dim_factors(input, resolved, outer, reduce, inner);
    Tensor grad = empty(input.shape(), DType::Float32, input.device());

    const float* src = static_cast<const float*>(input.data());
    const float* gout = static_cast<const float*>(grad_output.data());
    float* dst = static_cast<float*>(grad.data());

    for (int64_t o = 0; o < outer; ++o) {
        for (int64_t in = 0; in < inner; ++in) {
            const int64_t base = o * reduce * inner + in;
            float vmax = -std::numeric_limits<float>::infinity();
            float grad_sum = 0.0f;
            for (int64_t r = 0; r < reduce; ++r) {
                vmax = std::max(vmax, src[base + r * inner]);
                grad_sum += gout[base + r * inner];
            }

            double sum_exp = 0.0;
            for (int64_t r = 0; r < reduce; ++r)
                sum_exp += std::exp(static_cast<double>(src[base + r * inner] - vmax));

            for (int64_t r = 0; r < reduce; ++r) {
                const int64_t idx = base + r * inner;
                const float prob = static_cast<float>(
                    std::exp(static_cast<double>(src[idx] - vmax)) / sum_exp);
                dst[idx] = gout[idx] - prob * grad_sum;
            }
        }
    }

    return grad;
}

}  // namespace orthon
