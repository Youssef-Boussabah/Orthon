#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <orthon/core/tensor.h>
#include <orthon/core/tensor_impl.h>
#include <orthon/core/dtype.h>
#include <orthon/core/float_formats.h>
#include <orthon/ops/cpu/fill_ops.h>
#include <orthon/ops/cpu/binary_ops.h>
#include <orthon/ops/cpu/scalar_ops.h>
#include <orthon/ops/cpu/shape_ops.h>
#include <orthon/ops/cpu/reduce_ops.h>
#include <orthon/ops/cpu/unary_ops.h>
#include <orthon/ops/cpu/matmul_ops.h>
#include <orthon/ops/cpu/softmax_ops.h>
#include <orthon/ops/cpu/cross_entropy_ops.h>
#include <orthon/ops/cpu/nll_loss_ops.h>
#include <orthon/ops/cpu/conv2d_ops.h>
#include <orthon/ops/cpu/maxpool2d_ops.h>
#include <orthon/ops/cpu/fusion_ops.h>
#include <orthon/ops/cpu/embedding_ops.h>
#include <orthon/ops/cpu/cast_ops.h>

#include <cstring>
#include <optional>
#include <string>
#include <vector>

#ifdef WITH_CUDA
#include <orthon/ops/cuda/matmul_kernels.h>
#include <orthon/ops/cuda/conv2d_kernels.h>
#endif

namespace py = pybind11;

using orthon::DType;
using orthon::Tensor;

static DType dtype_from_string(const std::string& s) {
    if (s == "float32")  return DType::Float32;
    if (s == "float64")  return DType::Float64;
    if (s == "float16")  return DType::Float16;
    if (s == "bfloat16") return DType::BFloat16;
    if (s == "int32")    return DType::Int32;
    if (s == "int64")    return DType::Int64;
    if (s == "uint8")    return DType::UInt8;
    if (s == "int8")     return DType::Int8;
    if (s == "bool")     return DType::Bool;
    throw std::invalid_argument("unknown dtype string: " + s);
}

void bind_ops(py::module_& m) {
    // Scoped to this function, not the file: the module itself stays in the
    // global namespace, so the extension remains the boundary between the
    // native `orthon` namespace and the Python name `orthon`.
    using namespace orthon;  // NOLINT(build/namespaces)

    m.def("empty", [](std::vector<int64_t> shape, const std::string& dtype_s) {
        return empty(shape, dtype_from_string(dtype_s));
    }, py::arg("shape"), py::arg("dtype") = "float32");

    m.def("zeros", [](std::vector<int64_t> shape, const std::string& dtype_s) {
        return zeros(shape, dtype_from_string(dtype_s));
    }, py::arg("shape"), py::arg("dtype") = "float32");

    m.def("ones", [](std::vector<int64_t> shape, const std::string& dtype_s) {
        return ones(shape, dtype_from_string(dtype_s));
    }, py::arg("shape"), py::arg("dtype") = "float32");

    // `value` is bound as a DOUBLE. A Python float is a C double, and taking
    // it as a float here would re-introduce the narrowing the native signature
    // avoids: full([], 1/3, dtype="float64") would store the float32 rounding.
    m.def("full", [](std::vector<int64_t> shape, double value, const std::string& dtype_s) {
        return full(shape, value, dtype_from_string(dtype_s));
    }, py::arg("shape"), py::arg("value"), py::arg("dtype") = "float32");

    m.def("from_numpy", [](py::array array, py::object dtype_obj) {
        const char kind     = array.dtype().kind();
        const int  itemsize = static_cast<int>(array.dtype().itemsize());

        // BFloat16 special path: numpy has no native bf16 dtype, so we accept a
        // float32 array and convert each element to BF16 bits here on the host.
        if (!dtype_obj.is_none()) {
            const DType target = dtype_from_string(dtype_obj.cast<std::string>());
            if (target == DType::BFloat16) {
                if (kind != 'f' || itemsize != 4)
                    throw py::type_error(
                        "from_numpy: dtype='bfloat16' requires a float32 input array");
                if ((array.flags() & py::array::c_style) != py::array::c_style)
                    throw py::value_error("from_numpy: expected a C-contiguous array");
                py::buffer_info info = array.request();
                std::vector<int64_t> shape;
                shape.reserve(static_cast<size_t>(info.ndim));
                for (py::ssize_t dim : info.shape)
                    shape.push_back(static_cast<int64_t>(dim));
                const int64_t n = static_cast<int64_t>(info.size);
                Tensor out(std::make_shared<TensorImpl>(shape, DType::BFloat16, Device{}));
                const float*    src = static_cast<const float*>(info.ptr);
                uint16_t*       dst = static_cast<uint16_t*>(out.data());
                // The canonical ties-to-even helper, not an inline
                // `(bits + 0x8000u) >> 16` (round-half-UP). Ad-hoc rounding
                // at each call site is what would let the same float32 value
                // reach bfloat16 storage as a different bit pattern depending
                // on which entry point built it.
                for (int64_t i = 0; i < n; ++i)
                    dst[i] = f32_to_bf16_bits(src[i]);
                return out;
            }
            throw py::type_error(
                "from_numpy: explicit dtype= override only supports 'bfloat16'");
        }

        DType dtype;
        if      (kind == 'f' && itemsize == 4) dtype = DType::Float32;
        else if (kind == 'f' && itemsize == 8) dtype = DType::Float64;
        else if (kind == 'f' && itemsize == 2) dtype = DType::Float16;
        else if (kind == 'i' && itemsize == 4) dtype = DType::Int32;
        else if (kind == 'i' && itemsize == 8) dtype = DType::Int64;
        else if (kind == 'i' && itemsize == 1) dtype = DType::Int8;
        else if (kind == 'u' && itemsize == 1) dtype = DType::UInt8;
        else if (kind == 'b' && itemsize == 1) dtype = DType::Bool;
        else
            throw py::type_error(
                std::string("from_numpy: unsupported array dtype (kind='") +
                kind + "', itemsize=" + std::to_string(itemsize) + ")");

        if ((array.flags() & py::array::c_style) != py::array::c_style)
            throw py::value_error("from_numpy: expected a C-contiguous array");

        py::buffer_info info = array.request();
        std::vector<int64_t> shape;
        shape.reserve(static_cast<size_t>(info.ndim));
        for (py::ssize_t dim : info.shape)
            shape.push_back(static_cast<int64_t>(dim));

        Tensor out(std::make_shared<TensorImpl>(shape, dtype, Device{}));
        const size_t nbytes = static_cast<size_t>(info.size) * dtype_size(dtype);
        if (nbytes > 0)
            std::memcpy(out.data(), info.ptr, nbytes);
        return out;
    }, py::arg("array"), py::arg("dtype") = py::none());

    m.def("add", &add, py::arg("a"), py::arg("b"));
    m.def("mul", &mul, py::arg("a"), py::arg("b"));
    // Tensor-Tensor only, matching add/mul; the scalar forms live on the
    // Tensor operators.
    //
    // Bound through lambdas rather than &sub / &div: `div` also names the C
    // library's div(int, int), so taking its address is ambiguous. Calling it
    // with two Tensors is not, so the lambda resolves the overload by argument
    // type. `sub` is bound the same way purely for symmetry.
    m.def("sub", [](const Tensor& a, const Tensor& b) { return sub(a, b); },
          py::arg("a"), py::arg("b"));
    m.def("div", [](const Tensor& a, const Tensor& b) { return div(a, b); },
          py::arg("a"), py::arg("b"));

    // Underscore-private diagnostic. Not public API: it exists so
    // the scalar representation contract - rank 0, tensor dtype, exact Device,
    // untracked - can be pinned by a test instead of inferred from behaviour.
    m.def("_scalar_like", &scalar_like, py::arg("reference"), py::arg("value"));

    m.def("fill_", [](Tensor& t, float value) -> Tensor& {
        fill_(t, value);
        return t;
    }, py::arg("tensor"), py::arg("value"));

    m.def("reshape", [](const Tensor& t, std::vector<int64_t> new_shape) {
        return reshape(t, new_shape);
    }, py::arg("tensor"), py::arg("shape"));

    m.def("transpose", &transpose,
          py::arg("tensor"), py::arg("dim0"), py::arg("dim1"));

    m.def("contiguous", &contiguous, py::arg("tensor"));

    m.def("sum",  [](const Tensor& t, py::object dim, bool keepdim) {
        std::optional<int64_t> d = std::nullopt;
        if (!dim.is_none()) d = dim.cast<int64_t>();
        return sum(t, d, keepdim);
    }, py::arg("tensor"), py::arg("dim") = py::none(), py::arg("keepdim") = false);
    m.def("mean", [](const Tensor& t, py::object dim, bool keepdim) {
        std::optional<int64_t> d = std::nullopt;
        if (!dim.is_none()) d = dim.cast<int64_t>();
        return mean(t, d, keepdim);
    }, py::arg("tensor"), py::arg("dim") = py::none(), py::arg("keepdim") = false);
    m.def("max", [](const Tensor& t, py::object dim, bool keepdim) {
        std::optional<int64_t> d = std::nullopt;
        if (!dim.is_none()) d = dim.cast<int64_t>();
        return reduce_max(t, d, keepdim);
    }, py::arg("tensor"), py::arg("dim") = py::none(), py::arg("keepdim") = false);
    m.def("argmax", [](const Tensor& t, py::object dim, bool keepdim) {
        std::optional<int64_t> d = std::nullopt;
        if (!dim.is_none()) d = dim.cast<int64_t>();
        return argmax(t, d, keepdim);
    }, py::arg("tensor"), py::arg("dim") = py::none(), py::arg("keepdim") = false);

    m.def("neg",        [](const Tensor& t) { return neg(t); },          py::arg("tensor"));
    m.def("relu",       [](const Tensor& t) { return relu(t); },         py::arg("tensor"));
    m.def("sigmoid",    [](const Tensor& t) { return sigmoid(t); },      py::arg("tensor"));
    m.def("tanh",       [](const Tensor& t) { return tanh_t(t); },       py::arg("tensor"));
    m.def("leaky_relu", [](const Tensor& t, float negative_slope) {
        return leaky_relu(t, negative_slope);
    }, py::arg("tensor"), py::arg("negative_slope") = 0.01f);
    m.def("gelu",       [](const Tensor& t) { return gelu(t); },         py::arg("tensor"));
    m.def("exp",        [](const Tensor& t) { return exp_t(t); },        py::arg("tensor"));
    m.def("dropout",    [](const Tensor& t, float p, uint64_t seed) {
        return dropout(t, p, seed);
    }, py::arg("tensor"), py::arg("p"), py::arg("seed"));
    m.def("sqrt",       [](const Tensor& t) { return sqrt_t(t); },       py::arg("tensor"));
    m.def("reciprocal", [](const Tensor& t) { return reciprocal_t(t); }, py::arg("tensor"));
    m.def("embedding", [](const Tensor& indices,
                          const Tensor& weight,
                          py::object padding_idx) {
        std::optional<int64_t> pad = std::nullopt;
        if (!padding_idx.is_none()) pad = padding_idx.cast<int64_t>();
        return embedding(indices, weight, pad);
    }, py::arg("indices"), py::arg("weight"), py::arg("padding_idx") = py::none());

    m.def("matmul", &matmul, py::arg("a"), py::arg("b"));

    m.def("fused_linear_relu", &fused_linear_relu,
          py::arg("x"), py::arg("weight"), py::arg("bias"));

    m.def("softmax", &softmax, py::arg("tensor"), py::arg("dim") = -1);
    m.def("log_softmax", &log_softmax,
          py::arg("tensor"), py::arg("dim") = (int64_t)-1);

    m.def("conv2d", &conv2d,
          py::arg("input"), py::arg("weight"), py::arg("bias"),
          py::arg("stride_h") = (int64_t)1, py::arg("stride_w") = (int64_t)1,
          py::arg("pad_h")    = (int64_t)0, py::arg("pad_w")    = (int64_t)0);
    m.def("_conv2d_direct", &conv2d_direct,
          py::arg("input"), py::arg("weight"), py::arg("bias"),
          py::arg("stride_h") = (int64_t)1, py::arg("stride_w") = (int64_t)1,
          py::arg("pad_h")    = (int64_t)0, py::arg("pad_w")    = (int64_t)0);
    m.def("_conv2d_im2col", &conv2d_im2col,
          py::arg("input"), py::arg("weight"), py::arg("bias"),
          py::arg("stride_h") = (int64_t)1, py::arg("stride_w") = (int64_t)1,
          py::arg("pad_h")    = (int64_t)0, py::arg("pad_w")    = (int64_t)0);
    m.def("maxpool2d", &maxpool2d,
          py::arg("input"),
          py::arg("kernel_h"), py::arg("kernel_w"),
          py::arg("stride_h"), py::arg("stride_w"),
          py::arg("pad_h") = (int64_t)0, py::arg("pad_w") = (int64_t)0);

    m.def("cross_entropy",
          [](const Tensor& logits, const std::vector<int64_t>& targets) {
              return cross_entropy(logits, targets);
          },
          py::arg("logits"), py::arg("targets"));

    m.def("nll_loss",
          [](const Tensor& log_probs, py::object targets, const std::string& reduction) {
              Tensor target_tensor = [&]() {
                  if (py::isinstance<Tensor>(targets))
                      return targets.cast<Tensor>();

                  py::array_t<int64_t, py::array::c_style | py::array::forcecast> array(targets);
                  py::buffer_info info = array.request();
                  if (info.ndim != 1)
                      throw py::value_error("nll_loss: targets must be 1D");

                  Tensor out(std::make_shared<TensorImpl>(
                      std::vector<int64_t>{static_cast<int64_t>(info.shape[0])},
                      DType::Int64,
                      Device{}));
                  const size_t nbytes = static_cast<size_t>(info.size) * sizeof(int64_t);
                  if (nbytes > 0)
                      std::memcpy(out.data(), info.ptr, nbytes);
                  return out;
              }();
              if (target_tensor.device() != log_probs.device())
                  target_tensor = target_tensor.to(log_probs.device());
              return nll_loss(log_probs, target_tensor, reduction);
          },
          py::arg("log_probs"),
          py::arg("targets"),
          py::arg("reduction") = "mean");

    m.def("to", [](const Tensor& t, const Device& dev) { return t.to(dev); },
          py::arg("tensor"), py::arg("device"));
    m.def("to", [](const Tensor& t, const std::string& s) {
        return t.to(device_from_string(s));
    }, py::arg("tensor"), py::arg("device"));

    m.def("cast", [](const Tensor& t, const std::string& dtype_s) {
        return cast(t, dtype_from_string(dtype_s));
    }, py::arg("tensor"), py::arg("dtype"));

#ifdef WITH_CUDA
    // Benchmark helpers: direct access to naive and tiled CUDA matmul kernels.
    // Underscore-private like the other matmul diagnostics; both validate
    // rank/device/dtype/contiguity before launching. A is (M,K), B is (K,N).
    m.def("_matmul_naive", &cuda_matmul_naive, py::arg("a"), py::arg("b"));
    m.def("_matmul_tiled", &cuda_matmul_tiled, py::arg("a"), py::arg("b"));
    m.def("_matmul_tiled_custom", &cuda_matmul_tiled_custom,
          py::arg("a"), py::arg("b"));
    m.def("_matmul_tiled_cublas", &cuda_matmul_tiled_cublas,
          py::arg("a"), py::arg("b"));
    m.def("_matmul_batched_tiled_custom", &cuda_batched_matmul_tiled_custom,
          py::arg("a"), py::arg("b"));
    m.def("_matmul_batched_tiled_cublas", &cuda_batched_matmul_tiled_cublas,
          py::arg("a"), py::arg("b"));
    m.def("_matmul_uses_cublas", &cuda_matmul_uses_cublas,
          py::arg("a"), py::arg("b"));
    m.def("_matmul_dispatch_path", &cuda_matmul_dispatch_path,
          py::arg("a"), py::arg("b"));
    m.def("_matmul_cublas_guard_report", &cuda_matmul_cublas_guard_report,
          py::arg("a"), py::arg("b"));
    m.def("_matmul_cublas_elapsed_ms", &cuda_matmul_cublas_elapsed_ms,
          py::arg("a"), py::arg("b"), py::arg("reps") = 100);
    m.def("_matmul_cublas_profile_report", &cuda_matmul_cublas_profile_report,
          py::arg("a"), py::arg("b"), py::arg("reps") = 100);
    m.def("_conv2d_cuda_naive", &cuda_conv2d_forward_naive,
          py::arg("input"), py::arg("weight"), py::arg("bias"),
          py::arg("stride_h") = (int64_t)1, py::arg("stride_w") = (int64_t)1,
          py::arg("pad_h")    = (int64_t)0, py::arg("pad_w")    = (int64_t)0);
    m.def("_conv2d_cuda_optimized", &cuda_conv2d_forward_optimized,
          py::arg("input"), py::arg("weight"), py::arg("bias"),
          py::arg("stride_h") = (int64_t)1, py::arg("stride_w") = (int64_t)1,
          py::arg("pad_h")    = (int64_t)0, py::arg("pad_w")    = (int64_t)0);
    m.def("_conv2d_cuda_backward_input", &cuda_conv2d_backward_input,
          py::arg("input"), py::arg("weight"), py::arg("grad_output"),
          py::arg("stride_h") = (int64_t)1, py::arg("stride_w") = (int64_t)1,
          py::arg("pad_h")    = (int64_t)0, py::arg("pad_w")    = (int64_t)0);
    m.def("_conv2d_cuda_backward_input_naive", &cuda_conv2d_backward_input_naive,
          py::arg("input"), py::arg("weight"), py::arg("grad_output"),
          py::arg("stride_h") = (int64_t)1, py::arg("stride_w") = (int64_t)1,
          py::arg("pad_h")    = (int64_t)0, py::arg("pad_w")    = (int64_t)0);
    m.def("_conv2d_cuda_backward_input_optimized", &cuda_conv2d_backward_input_optimized,
          py::arg("input"), py::arg("weight"), py::arg("grad_output"),
          py::arg("stride_h") = (int64_t)1, py::arg("stride_w") = (int64_t)1,
          py::arg("pad_h")    = (int64_t)0, py::arg("pad_w")    = (int64_t)0);
    m.def("_conv2d_cuda_backward_input_optimized_v2", &cuda_conv2d_backward_input_optimized_v2,
          py::arg("input"), py::arg("weight"), py::arg("grad_output"),
          py::arg("stride_h") = (int64_t)1, py::arg("stride_w") = (int64_t)1,
          py::arg("pad_h")    = (int64_t)0, py::arg("pad_w")    = (int64_t)0);
    m.def("_conv2d_cuda_backward_weight", &cuda_conv2d_backward_weight,
          py::arg("input"), py::arg("weight"), py::arg("grad_output"),
          py::arg("stride_h") = (int64_t)1, py::arg("stride_w") = (int64_t)1,
          py::arg("pad_h")    = (int64_t)0, py::arg("pad_w")    = (int64_t)0);
    m.def("_conv2d_cuda_backward_weight_naive", &cuda_conv2d_backward_weight_naive,
          py::arg("input"), py::arg("weight"), py::arg("grad_output"),
          py::arg("stride_h") = (int64_t)1, py::arg("stride_w") = (int64_t)1,
          py::arg("pad_h")    = (int64_t)0, py::arg("pad_w")    = (int64_t)0);
    m.def("_conv2d_cuda_backward_weight_optimized", &cuda_conv2d_backward_weight_optimized,
          py::arg("input"), py::arg("weight"), py::arg("grad_output"),
          py::arg("stride_h") = (int64_t)1, py::arg("stride_w") = (int64_t)1,
          py::arg("pad_h")    = (int64_t)0, py::arg("pad_w")    = (int64_t)0);
    m.def("_conv2d_cuda_backward_weight_optimized_v2", &cuda_conv2d_backward_weight_optimized_v2,
          py::arg("input"), py::arg("weight"), py::arg("grad_output"),
          py::arg("stride_h") = (int64_t)1, py::arg("stride_w") = (int64_t)1,
          py::arg("pad_h")    = (int64_t)0, py::arg("pad_w")    = (int64_t)0);
    m.def("_conv2d_cuda_backward_bias", &cuda_conv2d_backward_bias,
          py::arg("bias"), py::arg("grad_output"));
    m.def("_conv2d_cuda_backward_bias_naive", &cuda_conv2d_backward_bias_naive,
          py::arg("bias"), py::arg("grad_output"));
    m.def("_conv2d_cuda_backward_bias_optimized", &cuda_conv2d_backward_bias_optimized,
          py::arg("bias"), py::arg("grad_output"));
    m.def("_conv2d_cuda_profile_reset", &cuda_conv2d_profile_reset);
    m.def("_conv2d_cuda_profile_summary", &cuda_conv2d_profile_summary);
#endif
}
