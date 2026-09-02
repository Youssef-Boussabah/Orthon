#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <orthon/core/tensor.h>
#include <orthon/device/device_guard.h>
#include <orthon/core/tensor_impl.h>
#include <orthon/core/dtype.h>
#include <orthon/core/float_formats.h>
#include <orthon/ops/cpu/binary_ops.h>
#include <orthon/ops/cpu/scalar_ops.h>
#include <orthon/ops/cpu/shape_ops.h>
#include <orthon/ops/cpu/fill_ops.h>
#include <orthon/ops/cpu/reduce_ops.h>
#include <orthon/ops/cpu/unary_ops.h>
#include <orthon/ops/cpu/matmul_ops.h>
#include <orthon/ops/cpu/softmax_ops.h>
#include <orthon/ops/cpu/cast_ops.h>
#include <orthon/utils/logging.h>

#include <cstring>
#include <cmath>
#include <limits>
#include <sstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef WITH_CUDA
#include <cuda_runtime_api.h>
#include <orthon/ops/cuda/cast_kernels.h>
#include <orthon/ops/cuda/unary_kernels.h>
#endif

namespace py = pybind11;

// The framework's public API lives in namespace orthon; the extension module
// deliberately does not, because `_C` is the boundary between the two. The
// file-scope helpers below therefore name what they need explicitly rather
// than importing the whole namespace here.
using orthon::Device;
using orthon::DeviceGuard;
using orthon::DType;
using orthon::Tensor;
using orthon::TensorImpl;

using orthon::bf16_bits_to_f32;
using orthon::device_to_string;
using orthon::dtype_size;
using orthon::f16_bits_to_f32;

#ifdef WITH_CUDA
using orthon::cuda_copy_cast_float32_to_;
using orthon::cuda_isfinite_all;
using orthon::cuda_unscale_to_float32_and_check;
#endif

struct PyNumel {
    int64_t value;
    int64_t operator()() const { return value; }
    bool operator==(int64_t other) const { return value == other; }
};

static std::string shape_repr(const std::vector<int64_t>& s) {
    std::string r = "[";
    for (size_t i = 0; i < s.size(); ++i) {
        if (i) r += ", ";
        r += std::to_string(s[i]);
    }
    return r + "]";
}

static std::string dtype_str(DType d) {
    switch (d) {
        case DType::Float32:  return "float32";
        case DType::Float64:  return "float64";
        case DType::Float16:  return "float16";
        case DType::BFloat16: return "bfloat16";
        case DType::Int32:    return "int32";
        case DType::Int64:    return "int64";
        case DType::UInt8:    return "uint8";
        case DType::Int8:     return "int8";
        case DType::Bool:     return "bool";
    }
    return "unknown";
}

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
    throw std::invalid_argument("unknown dtype: " + s);
}

static void copy_flat_range(Tensor& dst, int64_t dst_offset, const Tensor& src) {
    if (!dst.is_contiguous() || !src.is_contiguous())
        throw std::runtime_error("copy_flat_range: tensors must be contiguous");
    if (dst.dtype() != src.dtype())
        throw std::runtime_error("copy_flat_range: dtype mismatch");
    if (dst.device() != src.device())
        throw std::runtime_error("copy_flat_range: device mismatch");
    if (dst_offset < 0 || dst_offset + src.numel() > dst.numel())
        throw std::out_of_range("copy_flat_range: destination range out of bounds");

    const size_t esz = dtype_size(dst.dtype());
    char* dst_ptr = static_cast<char*>(dst.data()) + dst_offset * static_cast<int64_t>(esz);
    const char* src_ptr = static_cast<const char*>(src.data());
    const size_t nbytes = static_cast<size_t>(src.numel()) * esz;

    if (nbytes > 0) {
        if (dst.device().is_cuda()) {
#ifdef WITH_CUDA
            // Both pointers live on this device (equality is validated above),
            // so it is the authoritative device for the copy.
            DeviceGuard guard(dst.device());
            cudaError_t err = cudaMemcpy(dst_ptr, src_ptr, nbytes, cudaMemcpyDeviceToDevice);
            if (err != cudaSuccess)
                throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err));
#else
            throw std::runtime_error("copy_flat_range: CUDA is not available in this build");
#endif
        } else {
            std::memcpy(dst_ptr, src_ptr, nbytes);
        }
    }

    // This helper owns the destination bump for every caller that delegates
    // to it (_copy_flat_range_from, _copy_data_), so each logical copy bumps
    // exactly once. A zero-byte copy that passed validation is still a
    // successful logical mutation and bumps.
    dst.impl()->storage()->bump_version();
}

static void copy_flat_range_to(const Tensor& src, int64_t src_offset, Tensor& dst) {
    if (!src.is_contiguous() || !dst.is_contiguous())
        throw std::runtime_error("copy_flat_range_to: tensors must be contiguous");
    if (src.dtype() != dst.dtype())
        throw std::runtime_error("copy_flat_range_to: dtype mismatch");
    if (src.device() != dst.device())
        throw std::runtime_error("copy_flat_range_to: device mismatch");
    if (src_offset < 0 || src_offset + dst.numel() > src.numel())
        throw std::out_of_range("copy_flat_range_to: source range out of bounds");

    const size_t esz = dtype_size(src.dtype());
    const char* src_ptr = static_cast<const char*>(src.data()) + src_offset * static_cast<int64_t>(esz);
    char* dst_ptr = static_cast<char*>(dst.data());
    const size_t nbytes = static_cast<size_t>(dst.numel()) * esz;

    if (nbytes > 0) {
        if (src.device().is_cuda()) {
#ifdef WITH_CUDA
            // Both pointers live on this device (equality is validated above),
            // so it is the authoritative device for the copy.
            DeviceGuard guard(src.device());
            cudaError_t err = cudaMemcpy(dst_ptr, src_ptr, nbytes, cudaMemcpyDeviceToDevice);
            if (err != cudaSuccess)
                throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err));
#else
            throw std::runtime_error("copy_flat_range_to: CUDA is not available in this build");
#endif
        } else {
            std::memcpy(dst_ptr, src_ptr, nbytes);
        }
    }

    // The mutated tensor here is `dst` - the destination argument, not the
    // bound `self`. One bump per successful logical copy; zero-byte copies
    // that passed validation still bump.
    dst.impl()->storage()->bump_version();
}

// The float16/bfloat16 read-back helpers come from
// <orthon/core/float_formats.h>, so this file cannot decode a payload by a
// different rule than the one that encoded it.

static bool isfinite_all_impl(const Tensor& t) {
    const DType dt = t.dtype();
    if (dt != DType::Float32 && dt != DType::Float64 &&
        dt != DType::Float16 && dt != DType::BFloat16) {
        return true;
    }

    if (t.device().is_cuda()) {
#ifdef WITH_CUDA
        return cuda_isfinite_all(t);
#else
        throw std::runtime_error("isfinite_all: CUDA is not available in this build");
#endif
    }

    Tensor materialized = t;
    auto finite_at = [&](int64_t off) -> bool {
        if (dt == DType::Float32)
            return std::isfinite(static_cast<const float*>(materialized.data())[off]);
        if (dt == DType::Float64)
            return std::isfinite(static_cast<const double*>(materialized.data())[off]);
        if (dt == DType::Float16)
            return std::isfinite(f16_bits_to_f32(
                static_cast<const uint16_t*>(materialized.data())[off]));
        return std::isfinite(bf16_bits_to_f32(
            static_cast<const uint16_t*>(materialized.data())[off]));
    };

    const int64_t n = materialized.numel();
    if (materialized.is_contiguous()) {
        for (int64_t i = 0; i < n; ++i) {
            if (!finite_at(i)) return false;
        }
        return true;
    }

    const auto& sh = materialized.shape();
    const auto& st = materialized.strides();
    const int ndim = static_cast<int>(sh.size());
    if (ndim == 0) return finite_at(0);
    std::vector<int64_t> idx(ndim, 0);
    for (int64_t elem = 0; elem < n; ++elem) {
        int64_t off = 0;
        for (int d = 0; d < ndim; ++d) off += idx[d] * st[d];
        if (!finite_at(off)) return false;
        for (int d = ndim - 1; d >= 0; --d) {
            if (++idx[d] < sh[d]) break;
            idx[d] = 0;
        }
    }
    return true;
}

static py::tuple amp_unscale_to_float32_and_check_impl(const Tensor& grad,
                                                       float inv_scale) {
    if (!grad.device().is_cuda()) {
        throw std::runtime_error(
            "_amp_unscale_to_float32_and_check: CUDA tensor required");
    }
#ifdef WITH_CUDA
    auto result = cuda_unscale_to_float32_and_check(grad, inv_scale);
    return py::make_tuple(result.first, result.second);
#else
    throw std::runtime_error(
        "_amp_unscale_to_float32_and_check: CUDA is not available in this build");
#endif
}

static void amp_copy_master_to_model_impl(Tensor& dst, const Tensor& src) {
    if (!dst.device().is_cuda() || !src.device().is_cuda()) {
        throw std::runtime_error("_amp_copy_master_to_model_: CUDA tensors required");
    }
#ifdef WITH_CUDA
    cuda_copy_cast_float32_to_(dst, src);
    // This wrapper is the one logical in-place API around the raw cast-copy
    // kernel, so it owns the single destination bump.
    dst.impl()->storage()->bump_version();
#else
    throw std::runtime_error(
        "_amp_copy_master_to_model_: CUDA is not available in this build");
#endif
}

// Resolve a single dimension index (supports negative), with a clear error.
static int64_t resolve_idx(int64_t i, int64_t size, int dim) {
    const int64_t orig = i;
    if (i < 0) i += size;
    if (i < 0 || i >= size)
        throw std::out_of_range(
            "index " + std::to_string(orig) +
            " is out of bounds for dimension " + std::to_string(dim) +
            " with size " + std::to_string(size));
    return i;
}

// Compute element offset (in floats from data()) for a full N-dimensional index.
static int64_t multi_index_offset(const Tensor& t,
                                   const std::vector<int64_t>& indices) {
    const auto& sh = t.shape();
    const auto& st = t.strides();
    const int ndim = static_cast<int>(sh.size());
    if (static_cast<int>(indices.size()) != ndim)
        throw std::invalid_argument(
            "indexing: tensor has " + std::to_string(ndim) +
            " dimension(s) but " + std::to_string(indices.size()) +
            " index/indices were provided");
    int64_t offset = 0;
    for (int d = 0; d < ndim; ++d)
        offset += resolve_idx(indices[d], sh[d], d) * st[d];
    return offset;
}

// ---- Scalar element access across devices ----
//
// Neither accessor below may do the obvious thing:
//
//     static_cast<float*>(t.data())[offset]
//
// t.data() is a raw storage pointer. For a CUDA tensor it addresses DEVICE
// memory, which the host cannot dereference - set_item, the full-index
// __setitem__ and get_item would each terminate the process with an access
// violation (0xC0000005) on a CUDA float32 tensor. The two helpers here route
// the single element through the CUDA runtime instead.
//
// Both compute their element offset with multi_index_offset, which resolves
// negative subscripts, bounds-checks every dimension and multiplies by the
// tensor's own STRIDES - so a slice or transpose view writes/reads the element
// its logical index names, and storage_offset is already folded into data().
//
// Supported dtype stays float32, matching the existing public contract: these
// helpers make the operation safe, they do not widen it.

#ifdef WITH_CUDA
static void cuda_copy_scalar_float32(const Tensor& t,
                                     int64_t element_offset,
                                     void* host,
                                     bool to_device,
                                     const char* op) {
    // Authoritative device: the tensor's own. One element, one copy.
    DeviceGuard guard(t.device());
    float* device_ptr = static_cast<float*>(t.data()) + element_offset;
    cudaError_t err = to_device
        ? cudaMemcpy(device_ptr, host, sizeof(float), cudaMemcpyHostToDevice)
        : cudaMemcpy(host, device_ptr, sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
        throw std::runtime_error(std::string(op) + ": CUDA error: " +
                                 cudaGetErrorString(err));
}
#endif

static void write_scalar_float32(Tensor& t,
                                 const std::vector<int64_t>& indices,
                                 float value,
                                 const char* op) {
    if (t.dtype() != DType::Float32)
        throw std::runtime_error(std::string(op) + ": only float32 supported");
    const int64_t offset = multi_index_offset(t, indices);

    if (t.device().is_cuda()) {
#ifdef WITH_CUDA
        cuda_copy_scalar_float32(t, offset, &value, /*to_device=*/true, op);
#else
        throw std::runtime_error(
            std::string(op) + ": CUDA is not available in this build");
#endif
    } else if (t.device().is_cpu()) {
        static_cast<float*>(t.data())[offset] = value;
    } else {
        throw std::runtime_error(std::string(op) + ": unsupported device");
    }

    // Exactly ONE bump, and only after the write actually landed. A
    // rejected dtype, an out-of-range index and a failed cudaMemcpy all throw
    // above, so a failed scalar assignment leaves the version counter alone.
    // Views share the base Storage, so mutating through a CUDA slice or
    // transpose advances the version every alias observes.
    t.impl()->storage()->bump_version();
}

static float read_scalar_float32(const Tensor& t,
                                 const std::vector<int64_t>& indices,
                                 const char* op) {
    if (t.dtype() != DType::Float32)
        throw std::runtime_error(std::string(op) + ": only float32 supported");
    const int64_t offset = multi_index_offset(t, indices);

    if (t.device().is_cuda()) {
#ifdef WITH_CUDA
        float value = 0.0f;
        cuda_copy_scalar_float32(t, offset, &value, /*to_device=*/false, op);
        return value;
#else
        throw std::runtime_error(
            std::string(op) + ": CUDA is not available in this build");
#endif
    }
    if (!t.device().is_cpu())
        throw std::runtime_error(std::string(op) + ": unsupported device");
    return static_cast<const float*>(t.data())[offset];
}

// ---- Raw tensor byte codec ----
//
// The two halves of the checkpoint format's tensor payload. They move BYTES:
// no Python list of floats in the middle, no NumPy dtype table, no rounding.
// That is what makes a stored float16/bfloat16 payload survive a round trip
// bit-for-bit regardless of what the host conversion rules do, and what lets a
// signalling NaN keep its payload - a Tensor -> Python float -> NumPy pipeline
// quiets it to 0x7FC00001 and cannot do better.
//
// Only the logical tensor payload crosses: no TensorImpl pointer, no allocator
// bookkeeping, no Storage capacity, no autograd metadata, no version counter.

// Multiply with an explicit overflow check. Returns false if the product would
// not fit, so a malformed archive claiming an absurd shape is rejected BEFORE
// anything is allocated rather than wrapping around into a small allocation.
static bool checked_mul_u64(uint64_t a, uint64_t b, uint64_t& out) {
    if (a != 0 && b > (std::numeric_limits<uint64_t>::max)() / a) return false;
    out = a * b;
    return true;
}

static uint64_t expected_payload_bytes(const std::vector<int64_t>& shape,
                                       DType dtype,
                                       const char* op) {
    uint64_t count = 1;
    for (int64_t dim : shape) {
        if (dim < 0)
            throw py::value_error(std::string(op) + ": negative dimension " +
                                  std::to_string(dim));
        if (!checked_mul_u64(count, static_cast<uint64_t>(dim), count))
            throw py::value_error(std::string(op) +
                                  ": shape element count overflows");
    }
    uint64_t nbytes = 0;
    if (!checked_mul_u64(count, static_cast<uint64_t>(dtype_size(dtype)), nbytes))
        throw py::value_error(std::string(op) + ": payload size overflows");
    return nbytes;
}

static py::bytes tensor_to_bytes_impl(const Tensor& t) {
    // Deliberately narrow. The caller owns the snapshot policy (an explicit
    // no_grad to("cpu"), which materialises views and crosses from CUDA), so
    // this function never silently copies or transfers on someone's behalf.
    if (!t.device().is_cpu())
        throw std::runtime_error(
            "_tensor_to_bytes: tensor must be on CPU (got " +
            device_to_string(t.device()) + ")");
    if (!t.is_contiguous())
        throw std::runtime_error("_tensor_to_bytes: tensor must be contiguous");

    const uint64_t nbytes =
        expected_payload_bytes(t.shape(), t.dtype(), "_tensor_to_bytes");
    if (nbytes == 0)
        return py::bytes("", 0);   // zero-element or zero-dim-size tensor
    // data() already folds in storage_offset, so a contiguous view of a larger
    // buffer contributes exactly its own logical payload and nothing else.
    return py::bytes(static_cast<const char*>(t.data()),
                     static_cast<size_t>(nbytes));
}

static Tensor tensor_from_bytes_impl(const std::vector<int64_t>& shape,
                                     const std::string& dtype_s,
                                     const py::bytes& payload) {
    const DType dtype = dtype_from_string(dtype_s);
    const uint64_t expected =
        expected_payload_bytes(shape, dtype, "_tensor_from_bytes");

    char* buffer = nullptr;
    py::ssize_t length = 0;
    if (PYBIND11_BYTES_AS_STRING_AND_SIZE(payload.ptr(), &buffer, &length) != 0)
        throw py::value_error("_tensor_from_bytes: payload is not bytes");

    // Exact length. Not "at least": a short read would leave uninitialised
    // storage and a long one would silently drop data.
    if (static_cast<uint64_t>(length) != expected)
        throw std::runtime_error(
            "_tensor_from_bytes: expected " + std::to_string(expected) +
            " bytes for shape/dtype, got " + std::to_string(length));

    Tensor out(std::make_shared<TensorImpl>(shape, dtype, Device{}));
    if (expected > 0)
        std::memcpy(out.data(), buffer, static_cast<size_t>(expected));
    // Fresh CPU, contiguous, untracked leaf. TensorImpl defaults already give
    // requires_grad=false / is_leaf=true / no grad_fn; stated here because the
    // checkpoint loader depends on it.
    return out;
}

// ---- Python scalar operands for the Tensor arithmetic operators ----
//
// Deliberately NOT arbitrary Python object coercion. `obj` qualifies only if it
// is a real NUMBER: int, float, bool, or an object exposing the number protocol
// with a working __float__ / __index__ (numpy scalars, Decimal, Fraction). A
// str, bytes, list, dict, complex or plain object is rejected here.
//
// Rejection is reported to the CALLER as "not a scalar" rather than thrown,
// because the operator then answers NotImplemented and lets Python raise its
// own canonical message:
//
//     TypeError: unsupported operand type(s) for +: 'Tensor' and 'str'
//
// Letting pybind11 reject the argument instead would dump an overload table
// ("__add__(): incompatible function arguments. The following argument types
// are supported: ...") for every non-Tensor right-hand side.
static bool as_python_scalar(py::handle obj, double& out) {
    PyObject* p = obj.ptr();
    if (py::isinstance<Tensor>(obj)) return false;
    // A NumPy ARRAY is not a scalar, and the number protocol cannot be trusted
    // to say so: NumPy 1.25+ only DEPRECATES float(one_element_array), so
    // PyFloat_AsDouble succeeds on some NumPy versions and raises on others,
    // version-forking this contract. Decide on dimensionality instead: any
    // ndarray with ndim >= 1 (a size-1 array included) is rejected here, while
    // a 0-d ndarray carries exactly one value and stays accepted. NumPy scalar
    // objects (np.float32(...), np.int64(...)) are not ndarray instances and
    // fall through to the number-protocol path unchanged.
    if (py::isinstance<py::array>(obj)) {
        if (py::reinterpret_borrow<py::array>(obj).ndim() > 0) return false;
    }
    if (!PyNumber_Check(p))          return false;
    if (PyComplex_Check(p))          return false;   // no complex support here
    const double value = PyFloat_AsDouble(p);
    if (value == -1.0 && PyErr_Occurred()) {
        PyErr_Clear();
        return false;
    }
    out = value;
    return true;
}

static py::object py_not_implemented() {
    return py::reinterpret_borrow<py::object>(py::handle(Py_NotImplemented));
}

// Parse a Python key (int or tuple/list of ints) into a flat index vector.
static std::vector<int64_t> parse_key(py::object key) {
    if (py::isinstance<py::int_>(key))
        return {py::cast<int64_t>(key)};
    return py::cast<std::vector<int64_t>>(key);
}

void bind_tensor(py::module_& m) {
    // Scoped to this function, not the file: the registration body reaches for
    // most of the public op surface, and the module still lives in the global
    // namespace.
    using namespace orthon;  // NOLINT(build/namespaces)

    m.def("isfinite_all", &isfinite_all_impl, py::arg("tensor"));
    m.def("_amp_unscale_to_float32_and_check",
          &amp_unscale_to_float32_and_check_impl,
          py::arg("grad"),
          py::arg("inv_scale"));
    m.def("_amp_copy_master_to_model_",
          &amp_copy_master_to_model_impl,
          py::arg("destination"),
          py::arg("master"));

    // Underscore-private: the checkpoint codec is the only intended caller,
    // and the pair is not a public tensor API.
    m.def("_tensor_to_bytes", &tensor_to_bytes_impl, py::arg("tensor"),
          "_tensor_to_bytes(tensor) -> bytes\n\n"
          "Exactly numel * dtype_size(dtype) bytes of logical contiguous "
          "storage. Requires a CPU, contiguous tensor.");
    m.def("_tensor_from_bytes", &tensor_from_bytes_impl,
          py::arg("shape"), py::arg("dtype"), py::arg("payload"),
          "_tensor_from_bytes(shape, dtype, payload) -> Tensor\n\n"
          "Fresh CPU, contiguous, untracked leaf tensor built from an exact "
          "payload. The byte count must match shape/dtype exactly.");

    py::class_<PyNumel>(m, "_Numel")
        .def("__call__", &PyNumel::operator())
        .def("__eq__",   &PyNumel::operator==)
        .def("__int__",  [](const PyNumel& n) { return n.value; })
        .def("__index__",[](const PyNumel& n) { return n.value; })
        .def("__repr__", [](const PyNumel& n) { return std::to_string(n.value); });

    py::class_<Tensor> tensor_class(m, "Tensor");

    // Opt out of NumPy's binary ufunc protocol.
    //
    // The arithmetic operators below answer NotImplemented for a non-numeric
    // right-hand side so Python can raise its own canonical operand TypeError.
    // NumPy, however, accepts ANY object in its reflected operators and falls
    // back to an OBJECT-dtype array, so without this opt-out
    // `x + np.array([1., 2., 3.])` quietly yields
    // array([Tensor, Tensor, Tensor], dtype=object) instead of failing.
    //
    // Setting __array_ufunc__ to None is NumPy's documented opt-out: its
    // ufuncs return NotImplemented for any expression involving this type, so
    // the mixed expression raises TypeError from Python, in both operand
    // orders. Numeric numpy SCALARS (np.float32, np.int64, a 0-d array) are
    // unaffected: as_python_scalar converts them before NumPy is ever asked.
    tensor_class.attr("__array_ufunc__") = py::none();

    tensor_class
        // Metadata properties.
        .def_property_readonly("shape", [](const Tensor& t) {
            py::list lst = py::cast(t.shape());
            return py::tuple(lst);
        })
        .def_property_readonly("strides", [](const Tensor& t) {
            py::list lst = py::cast(t.strides());
            return py::tuple(lst);
        })
        .def_property_readonly("numel",  [](const Tensor& t) { return PyNumel{t.numel()}; })
        .def_property_readonly("ndim",   &Tensor::ndim)
        .def_property_readonly("is_contiguous", &Tensor::is_contiguous)
        .def_property_readonly("dtype", [](const Tensor& t) {
            return dtype_str(t.dtype());
        })
        .def_property_readonly("storage_offset", [](const Tensor& t) {
            return t.impl()->storage_offset();
        })
        .def_property_readonly("_impl_id", [](const Tensor& t) {
            return reinterpret_cast<uintptr_t>(t.impl().get());
        })
        // Internal diagnostic: the shared Storage's in-place mutation
        // generation. Read-only, underscore-private, exists so tests can prove
        // exactly-once bump semantics; not a public versioning API.
        .def_property_readonly("_storage_version", [](const Tensor& t) {
            return t.impl()->storage()->version();
        })
        // Autograd metadata.
        .def_property_readonly("requires_grad", &Tensor::requires_grad)
        .def("requires_grad_", [](Tensor& self, bool value) -> Tensor& {
            return self.requires_grad_(value);
        }, py::return_value_policy::reference_internal)
        .def_property_readonly("is_leaf", &Tensor::is_leaf)
        .def_property_readonly("grad", [](Tensor& self) -> py::object {
            if (!self.has_grad()) return py::none();
            return py::cast(self.grad());
        })
        .def("set_grad", &Tensor::set_grad, py::arg("grad"))
        .def("zero_grad", &Tensor::zero_grad)
        .def("isfinite_all", &isfinite_all_impl)
        .def("_register_post_accumulate_grad_hook", [](Tensor& self, py::function hook) {
            self.add_post_accumulate_grad_hook([hook]() {
                py::gil_scoped_acquire gil;
                hook();
            });
        }, py::arg("hook"))
        .def_property_readonly("_has_grad_fn", &Tensor::has_grad_fn)
        .def("backward", &Tensor::backward, py::arg("retain_graph") = false)
        // Scalar and bulk reads.
        .def("tolist", [](const Tensor& t) -> py::list {
            Tensor materialized = t;
            // For CUDA tensors, bring data to CPU first.
            if (t.device().is_cuda()) {
                orthon_log_device_fallback("Tensor.tolist",
                    "copying CUDA tensor to CPU for host materialization");
                materialized = t.to(Device{DeviceType::CPU, 0});
            }
            const int64_t n = materialized.numel();
            auto append_at = [&](py::list& out, int64_t off) {
                const DType dt = materialized.dtype();
                if (dt == DType::Float32) {
                    out.append(static_cast<const float*>(materialized.data())[off]);
                } else if (dt == DType::Float64) {
                    out.append(static_cast<const double*>(materialized.data())[off]);
                } else if (dt == DType::Int32) {
                    out.append(static_cast<const int32_t*>(materialized.data())[off]);
                } else if (dt == DType::Int64) {
                    out.append(static_cast<const int64_t*>(materialized.data())[off]);
                } else if (dt == DType::UInt8) {
                    out.append(static_cast<int>(
                        static_cast<const uint8_t*>(materialized.data())[off]));
                } else if (dt == DType::Int8) {
                    out.append(static_cast<int>(
                        static_cast<const int8_t*>(materialized.data())[off]));
                } else if (dt == DType::Bool) {
                    out.append(static_cast<bool>(
                        static_cast<const uint8_t*>(materialized.data())[off]));
                } else if (dt == DType::Float16) {
                    out.append(f16_bits_to_f32(
                        static_cast<const uint16_t*>(materialized.data())[off]));
                } else if (dt == DType::BFloat16) {
                    out.append(bf16_bits_to_f32(
                        static_cast<const uint16_t*>(materialized.data())[off]));
                } else {
                    throw std::runtime_error("tolist() unsupported dtype");
                }
            };
            py::list out;
            if (materialized.is_contiguous()) {
                for (int64_t i = 0; i < n; ++i) append_at(out, i);
                return out;
            }
            // Non-contiguous: stride-based walk in logical row-major order.
            const auto& sh = materialized.shape();
            const auto& st = materialized.strides();
            const int ndim  = static_cast<int>(sh.size());
            if (ndim == 0) { append_at(out, 0); return out; }
            std::vector<int64_t> idx(ndim, 0);
            for (int64_t elem = 0; elem < n; ++elem) {
                int64_t off = 0;
                for (int d = 0; d < ndim; ++d) off += idx[d] * st[d];
                append_at(out, off);
                for (int d = ndim - 1; d >= 0; --d) {
                    if (++idx[d] < sh[d]) break;
                    idx[d] = 0;
                }
            }
            return out;
        })
        .def("item", [](const Tensor& t) -> py::object {
            if (t.numel() != 1)
                throw std::runtime_error(
                    "item(): tensor has " + std::to_string(t.numel()) +
                    " element(s); item() requires exactly 1");
            Tensor cpu = t;
            if (t.device().is_cuda()) {
                orthon_log_device_fallback("Tensor.item",
                    "copying CUDA scalar to CPU for host materialization");
                cpu = t.to(Device{DeviceType::CPU, 0});
            }
            if (cpu.dtype() == DType::Float32)
                return py::cast(*static_cast<const float*>(cpu.data()));
            if (cpu.dtype() == DType::Float64)
                return py::cast(*static_cast<const double*>(cpu.data()));
            if (cpu.dtype() == DType::Int64)
                return py::cast(*static_cast<const int64_t*>(cpu.data()));
            if (cpu.dtype() == DType::Int32)
                return py::cast(*static_cast<const int32_t*>(cpu.data()));
            if (cpu.dtype() == DType::UInt8)
                return py::cast(static_cast<int>(*static_cast<const uint8_t*>(cpu.data())));
            if (cpu.dtype() == DType::Int8)
                return py::cast(static_cast<int>(*static_cast<const int8_t*>(cpu.data())));
            if (cpu.dtype() == DType::Bool)
                return py::cast(static_cast<bool>(*static_cast<const uint8_t*>(cpu.data())));
            if (cpu.dtype() == DType::Float16)
                return py::cast(f16_bits_to_f32(*static_cast<const uint16_t*>(cpu.data())));
            if (cpu.dtype() == DType::BFloat16)
                return py::cast(bf16_bits_to_f32(*static_cast<const uint16_t*>(cpu.data())));
            throw std::runtime_error("item(): unsupported dtype");
        })
        // get_item: multi-dimensional index → scalar float
        .def("get_item", [](const Tensor& t, std::vector<int64_t> indices) -> float {
            return read_scalar_float32(t, indices, "get_item");
        }, py::arg("indices"))
        // Mutation helpers.
        .def("fill_", [](Tensor& t, float value) -> Tensor& {
            fill_(t, value);
            return t;
        }, py::arg("value"))
        // In-place: self += alpha * other.  Does not touch autograd metadata.
        .def("add_", [](Tensor& self, const Tensor& other, float alpha) -> Tensor& {
            add_(self, other, alpha);
            return self;
        }, py::arg("other"), py::arg("alpha") = 1.0f,
           py::return_value_policy::reference_internal)
        // In-place: self *= alpha.  Does not touch autograd metadata.
        .def("mul_", [](Tensor& self, float alpha) -> Tensor& {
            mul_(self, alpha);
            return self;
        }, py::arg("alpha"), py::return_value_policy::reference_internal)
        .def("_copy_flat_range_from", [](Tensor& self, int64_t offset, const Tensor& source) -> Tensor& {
            copy_flat_range(self, offset, source);
            return self;
        }, py::arg("offset"), py::arg("source"), py::return_value_policy::reference_internal)
        .def("_copy_flat_range_to", [](const Tensor& self, int64_t offset, Tensor& destination) {
            copy_flat_range_to(self, offset, destination);
        }, py::arg("offset"), py::arg("destination"))
        // Internal exact-data copy used by Module.load_state_dict.
        //
        // Copies `source`'s payload into this tensor's existing storage. The
        // TensorImpl is not replaced, so the destination keeps its object
        // identity and every piece of bookkeeping hanging off it: autograd
        // metadata, requires_grad, is_leaf, .grad, registered hooks, optimizer
        // references and module registration. Only the stored values change.
        //
        // Deliberately narrow: exact shape, dtype, device and a contiguous
        // destination. Every canonical module parameter satisfies that, and the
        // actual transfer reuses copy_flat_range so the CUDA path stays inside
        // the one DeviceGuard-protected memcpy helper rather than growing a
        // second copy of that logic.
        .def("_copy_data_", [](Tensor& self, const Tensor& source) {
            if (self.shape() != source.shape())
                throw std::runtime_error(
                    "_copy_data_: shape mismatch (destination " +
                    shape_repr(self.shape()) + " vs source " +
                    shape_repr(source.shape()) + ")");
            if (self.dtype() != source.dtype())
                throw std::runtime_error(
                    "_copy_data_: dtype mismatch (destination " +
                    dtype_str(self.dtype()) + " vs source " +
                    dtype_str(source.dtype()) + ")");
            if (self.device() != source.device())
                throw std::runtime_error(
                    "_copy_data_: device mismatch (destination " +
                    device_to_string(self.device()) + " vs source " +
                    device_to_string(source.device()) + ")");
            if (!self.is_contiguous())
                throw std::runtime_error(
                    "_copy_data_: destination must be contiguous");
            if (!source.is_contiguous())
                throw std::runtime_error(
                    "_copy_data_: source must be contiguous");
            copy_flat_range(self, 0, source);
        }, py::arg("source"))
        // set_item: multi-dimensional index → write scalar float in-place
        .def("set_item", [](Tensor& t, std::vector<int64_t> indices, float value) {
            write_scalar_float32(t, indices, value, "set_item");
        }, py::arg("indices"), py::arg("value"))
        // Shape and view ops.
        .def("reshape", [](const Tensor& t, std::vector<int64_t> new_shape) {
            return reshape(t, new_shape);
        }, py::arg("shape"))
        .def("flatten", [](const Tensor& t, int64_t start_dim, int64_t end_dim) {
            const int64_t ndim = t.ndim();
            if (ndim == 0)
                throw std::runtime_error("flatten: scalar tensors cannot be flattened");
            if (start_dim < 0) start_dim += ndim;
            if (end_dim < 0) end_dim += ndim;
            if (start_dim < 0 || start_dim >= ndim || end_dim < 0 || end_dim >= ndim)
                throw std::out_of_range("flatten: dimension out of range");
            if (start_dim > end_dim)
                throw std::runtime_error("flatten: start_dim must be <= end_dim");

            std::vector<int64_t> new_shape;
            const auto& sh = t.shape();
            for (int64_t i = 0; i < start_dim; ++i)
                new_shape.push_back(sh[static_cast<size_t>(i)]);
            int64_t flat = 1;
            for (int64_t i = start_dim; i <= end_dim; ++i)
                flat *= sh[static_cast<size_t>(i)];
            new_shape.push_back(flat);
            for (int64_t i = end_dim + 1; i < ndim; ++i)
                new_shape.push_back(sh[static_cast<size_t>(i)]);
            return reshape(t, new_shape);
        }, py::arg("start_dim") = 0, py::arg("end_dim") = -1)
        .def("transpose", [](const Tensor& t, int64_t dim0, int64_t dim1) {
            return transpose(t, dim0, dim1);
        }, py::arg("dim0"), py::arg("dim1"))
        // Thin exposure of the already-existing native Tensor::slice. This is
        // not a redesign of Python indexing: __getitem__ keeps its integer /
        // tuple / list semantics and still rejects Python slice objects, and
        // there is deliberately no step, omitted bound, Ellipsis or advanced
        // indexing here. Bounds follow the native contract (negative dim/start/
        // end are normalised, everything else is an error) rather than Python's
        // clamping slice rules.
        .def("slice", [](const Tensor& t, int64_t dim, int64_t start, int64_t end) {
            return t.slice(dim, start, end);
        }, py::arg("dim"), py::arg("start"), py::arg("end"),
           "slice(dim, start, end) -> Tensor\n\n"
           "Storage-sharing view narrowing `dim` to [start, end). Negative dim, "
           "start and end are resolved against the tensor's own shape; the "
           "resulting range must satisfy 0 <= start <= end <= size(dim). "
           "Differentiable: with grad tracking on, the view carries a SliceNode.")
        .def("contiguous", [](const Tensor& t) {
            return contiguous(t);
        })
        // Indexing operators.
        // __getitem__: int → lower-dim view; tuple/list → chained views.
        .def("__getitem__", [](const Tensor& t, py::object key) -> Tensor {
            if (py::isinstance<py::int_>(key))
                return t[py::cast<int64_t>(key)];
            auto seq = py::cast<std::vector<int64_t>>(key);
            if (seq.empty())
                throw std::invalid_argument("__getitem__: index sequence is empty");
            Tensor cur = t;
            for (size_t i = 0; i < seq.size(); ++i) {
                if (cur.ndim() == 0)
                    throw std::out_of_range(
                        "too many indices: index " + std::to_string(i) +
                        " applied to a 0-dimensional tensor");
                cur = cur[seq[i]];
            }
            return cur;
        })
        // __setitem__: full index writes one element; partial index fills the sub-view.
        .def("__setitem__", [](Tensor& t, py::object key, float value) {
            if (t.dtype() != DType::Float32)
                throw std::runtime_error("__setitem__: only float32 supported");
            auto indices = parse_key(key);
            const int ndim = static_cast<int>(t.shape().size());
            if (static_cast<int>(indices.size()) > ndim)
                throw std::out_of_range(
                    "too many indices: tensor has " + std::to_string(ndim) +
                    " dimension(s) but " + std::to_string(indices.size()) +
                    " were provided");
            if (static_cast<int>(indices.size()) == ndim) {
                // Full index: one stride-aware scalar write, device-correct,
                // one version bump.
                write_scalar_float32(t, indices, value, "__setitem__");
            } else {
                // Partial assignment fills a whole sub-view, and the only
                // fill primitive in the runtime is the CPU fill_. A CUDA
                // tensor is rejected explicitly HERE, before any sub-view is
                // built, so nothing partial is attempted first and the error
                // names __setitem__ rather than an internal helper.
                //
                // A stride-aware CUDA fill kernel is what would lift this
                // limit. Full-index CUDA scalar assignment above works, and
                // CPU partial assignment is unaffected.
                if (t.device().is_cuda())
                    throw std::runtime_error(
                        "__setitem__: partial CUDA assignment is not supported");
                // Partial index: navigate to the sub-view and fill_ it.
                Tensor sub = t;
                for (int64_t i : indices) sub = sub[i];
                fill_(sub, value);
            }
        })
        // Device metadata.
        .def_property_readonly("device", &Tensor::device)
        // Device transfer.
        .def("to", [](const Tensor& t, const Device& dev) { return t.to(dev); },
             py::arg("device"))
        .def("to", [](const Tensor& t, const std::string& s) {
             return t.to(device_from_string(s));
        }, py::arg("device"))
        .def("cast", [](const Tensor& t, const std::string& dtype_s) {
            return cast(t, dtype_from_string(dtype_s));
        }, py::arg("dtype"))
        // Reductions.
        .def("sum",  [](const Tensor& t, py::object dim, bool keepdim) {
            std::optional<int64_t> d = std::nullopt;
            if (!dim.is_none()) d = dim.cast<int64_t>();
            return sum(t, d, keepdim);
        }, py::arg("dim") = py::none(), py::arg("keepdim") = false)
        .def("mean", [](const Tensor& t, py::object dim, bool keepdim) {
            std::optional<int64_t> d = std::nullopt;
            if (!dim.is_none()) d = dim.cast<int64_t>();
            return mean(t, d, keepdim);
        }, py::arg("dim") = py::none(), py::arg("keepdim") = false)
        .def("max", [](const Tensor& t, py::object dim, bool keepdim) {
            std::optional<int64_t> d = std::nullopt;
            if (!dim.is_none()) d = dim.cast<int64_t>();
            return reduce_max(t, d, keepdim);
        }, py::arg("dim") = py::none(), py::arg("keepdim") = false)
        .def("argmax", [](const Tensor& t, py::object dim, bool keepdim) {
            std::optional<int64_t> d = std::nullopt;
            if (!dim.is_none()) d = dim.cast<int64_t>();
            return argmax(t, d, keepdim);
        }, py::arg("dim") = py::none(), py::arg("keepdim") = false)
        // Unary ops.
        .def("neg",  [](const Tensor& t) { return neg(t); })
        .def("relu", [](const Tensor& t) { return relu(t); })
        .def("sigmoid", [](const Tensor& t) { return sigmoid(t); })
        .def("tanh", [](const Tensor& t) { return tanh_t(t); })
        .def("leaky_relu", [](const Tensor& t, float negative_slope) {
            return leaky_relu(t, negative_slope);
        }, py::arg("negative_slope") = 0.01f)
        .def("gelu", [](const Tensor& t) { return gelu(t); })
        .def("exp", [](const Tensor& t) { return exp_t(t); })
        .def("log_softmax", [](const Tensor& t, int64_t dim) {
            return log_softmax(t, dim);
        }, py::arg("dim") = (int64_t)-1)
        // Python operators.
        .def("matmul",    [](const Tensor& a, const Tensor& b) { return matmul(a, b); },
             py::arg("other"))
        .def("__neg__",   [](const Tensor& t) { return neg(t); })
        // One typed dispatch per operator rather than a pile of pybind
        // overloads.
        //
        // Tensor-Tensor is checked first. A NUMBER routes to the scalar path.
        // Anything else answers NotImplemented, which hands the decision back
        // to Python: it tries the reflected operator and, failing that, raises
        // the standard operand-type TypeError.
        .def("__add__", [](const Tensor& a, py::object other) -> py::object {
            if (py::isinstance<Tensor>(other))
                return py::cast(add(a, other.cast<Tensor>()));
            double v;
            if (as_python_scalar(other, v)) return py::cast(scalar_add(a, v));
            return py_not_implemented();
        })
        .def("__radd__", [](const Tensor& a, py::object other) -> py::object {
            // Addition commutes, so scalar + tensor is the same node.
            double v;
            if (as_python_scalar(other, v)) return py::cast(scalar_add(a, v));
            return py_not_implemented();
        })
        .def("__sub__", [](const Tensor& a, py::object other) -> py::object {
            if (py::isinstance<Tensor>(other))
                return py::cast(sub(a, other.cast<Tensor>()));
            double v;
            if (as_python_scalar(other, v)) return py::cast(scalar_sub(a, v));
            return py_not_implemented();
        })
        .def("__rsub__", [](const Tensor& a, py::object other) -> py::object {
            // NOT commutative: this is `other - a`.
            double v;
            if (as_python_scalar(other, v)) return py::cast(scalar_rsub(v, a));
            return py_not_implemented();
        })
        .def("__mul__", [](const Tensor& a, py::object other) -> py::object {
            if (py::isinstance<Tensor>(other))
                return py::cast(mul(a, other.cast<Tensor>()));
            double v;
            if (as_python_scalar(other, v)) return py::cast(scalar_mul(a, v));
            return py_not_implemented();
        })
        .def("__rmul__", [](const Tensor& a, py::object other) -> py::object {
            double v;
            if (as_python_scalar(other, v)) return py::cast(scalar_mul(a, v));
            return py_not_implemented();
        })
        .def("__truediv__", [](const Tensor& a, py::object other) -> py::object {
            if (py::isinstance<Tensor>(other))
                return py::cast(div(a, other.cast<Tensor>()));
            double v;
            if (as_python_scalar(other, v)) return py::cast(scalar_div(a, v));
            return py_not_implemented();
        })
        .def("__rtruediv__", [](const Tensor& a, py::object other) -> py::object {
            // NOT commutative: this is `other / a`, the one scalar form whose
            // backward genuinely depends on a's values.
            double v;
            if (as_python_scalar(other, v)) return py::cast(scalar_rdiv(v, a));
            return py_not_implemented();
        })
        .def("__matmul__",[](const Tensor& a, const Tensor& b) { return matmul(a, b); })
        .def("__repr__", [](const Tensor& t) {
            std::ostringstream oss;
            oss << "Tensor(shape=(";
            const auto& sh = t.shape();
            for (size_t i = 0; i < sh.size(); ++i) {
                if (i) oss << ", ";
                oss << sh[i];
            }
            if (sh.size() == 1) oss << ",";
            oss << "), dtype=" << dtype_str(t.dtype()) << ")";
            return oss.str();
        });
}
