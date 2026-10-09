#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <orthon/core/tensor.h>
#include <orthon/autograd/grad_mode.h>
#include <orthon/ops/tensor_access.h>
#include <orthon/core/tensor_impl.h>
#include <orthon/core/dtype.h>
#include <orthon/core/float_formats.h>
#include <orthon/ops/binary_ops.h>
#include <orthon/ops/scalar_ops.h>
#include <orthon/ops/shape_ops.h>
#include <orthon/ops/fill_ops.h>
#include <orthon/ops/reduce_ops.h>
#include <orthon/ops/unary_ops.h>
#include <orthon/ops/matmul_ops.h>
#include <orthon/ops/softmax_ops.h>
#include <orthon/ops/cast_ops.h>
#include <orthon/utils/logging.h>

#include <cstring>
#include <cmath>
#include <limits>
#include <sstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

// The framework's public API lives in namespace orthon; the extension module
// deliberately does not, because `_C` is the boundary between the two. The
// file-scope helpers below therefore name what they need explicitly rather
// than importing the whole namespace here.
using orthon::Device;
using orthon::DType;
using orthon::Tensor;
using orthon::TensorImpl;

using orthon::bf16_bits_to_f32;
using orthon::device_to_string;
using orthon::dtype_size;
using orthon::f16_bits_to_f32;


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
    if (dst_offset < 0 || src.numel() < 0 || dst_offset > dst.numel() ||
        src.numel() > dst.numel() - dst_offset)
        throw std::out_of_range("copy_flat_range: destination range out of bounds");

    orthon::copy_range_values(dst, src, {dst_offset, 0, src.numel()});
    // This helper owns the destination bump for every caller that delegates
    // to it (_copy_flat_range_from, _copy_data_), so each logical copy bumps
    // exactly once. A zero-byte copy that passed validation is still a
    // successful logical mutation and bumps.
    dst.impl()->buffer()->bump_version();
}

static void copy_flat_range_to(const Tensor& src, int64_t src_offset, Tensor& dst) {
    if (!src.is_contiguous() || !dst.is_contiguous())
        throw std::runtime_error("copy_flat_range_to: tensors must be contiguous");
    if (src.dtype() != dst.dtype())
        throw std::runtime_error("copy_flat_range_to: dtype mismatch");
    if (src.device() != dst.device())
        throw std::runtime_error("copy_flat_range_to: device mismatch");
    if (src_offset < 0 || dst.numel() < 0 || src_offset > src.numel() ||
        dst.numel() > src.numel() - src_offset)
        throw std::out_of_range("copy_flat_range_to: source range out of bounds");

    orthon::copy_range_values(dst, src, {0, src_offset, dst.numel()});
    // The mutated tensor here is `dst` - the destination argument, not the
    // bound `self`. One bump per successful logical copy; zero-byte copies
    // that passed validation still bump.
    dst.impl()->buffer()->bump_version();
}

// The float16/bfloat16 read-back helpers come from
// <orthon/core/float_formats.h>, so this file cannot decode a payload by a
// different rule than the one that encoded it.

static bool isfinite_all_impl(const Tensor& t) { return orthon::isfinite_all_values(t); }

static py::tuple amp_unscale_to_float32_and_check_impl(const Tensor& grad, float inv_scale) {
    auto result = orthon::amp_unscale_values(grad, inv_scale);
    return py::make_tuple(result.first, result.second);
}

static void amp_copy_master_to_model_impl(Tensor& dst, const Tensor& src) {
    orthon::amp_copy_values(dst, src);
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

// CUDA pointers cannot be dereferenced on the host: transfer one element
// through the runtime. Compute the stride-relative offset after validating
// indices; data() already includes the Tensor storage offset.


static void write_scalar_float32(Tensor& t,
                                 const std::vector<int64_t>& indices,
                                 float value,
                                 const char* op) {
    if (t.dtype() != DType::Float32)
        throw std::runtime_error(std::string(op) + ": only float32 supported");
    const int64_t offset = multi_index_offset(t, indices);

    orthon::scalar_write_value(t, offset, value, op);
    // Successful writes bump the shared Buffer once, including through views.
    // Rejected assignments and failed transfers leave its version unchanged.
    t.impl()->buffer()->bump_version();
}

static float read_scalar_float32(const Tensor& t,
                                 const std::vector<int64_t>& indices,
                                 const char* op) {
    if (t.dtype() != DType::Float32)
        throw std::runtime_error(std::string(op) + ": only float32 supported");
    const int64_t offset = multi_index_offset(t, indices);

    return orthon::scalar_read_value(t, offset, op);
}

// Checkpoint payloads preserve raw bits, including NaNs and low-precision
// values. Export only logical tensor data, excluding capacity and pointers.

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

// Accept Python numeric protocols, including NumPy scalars. Return
// NotImplemented for complex or nonnumeric operands so Python raises its
// normal arithmetic TypeError instead of a pybind overload error.
static bool as_python_scalar(py::handle obj, double& out) {
    PyObject* p = obj.ptr();
    if (py::isinstance<Tensor>(obj)) return false;
    // Array-to-float conversion varies across NumPy versions. Check ndim
    // directly: reject arrays with ndim >= 1, including size-one arrays.
    // Zero-dimensional arrays and NumPy scalar objects remain numeric operands.
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

    // Opt out of NumPy ufuncs to prevent reflected array arithmetic from
    // producing object arrays of Tensor objects. Direct numeric scalar
    // arithmetic still follows the Tensor scalar protocol.
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
        // Internal diagnostic: the shared Buffer's in-place mutation
        // generation. Read-only, underscore-private, exists so tests can prove
        // exactly-once bump semantics; not a public versioning API.
        .def_property_readonly("_storage_version", [](const Tensor& t) {
            return t.impl()->buffer()->version();
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
        .def("_register_backward_finalizer", [](Tensor& self, uintptr_t key, py::function callback) {
            self.register_backward_finalizer(key, [callback]() {
                py::gil_scoped_acquire gil;
                callback();
            });
        }, py::arg("key"), py::arg("callback"))
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
        // Copy exact shape/dtype/device contiguous values while retaining the
        // TensorImpl owner and its gradient, hooks and optimizer bindings. Module
        // state loading partitions strided destinations into contiguous subviews.
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
        // Keep the canonical representation alive while Module rebinds its
        // registered owner. The separate facade copies neither values nor history.
        .def("_model_value_reference", [](const Tensor& self) {
            if (orthon::is_grad_enabled())
                throw std::runtime_error("model value reference requires no_grad");
            return Tensor(self);
        })
        // Private Module-only value adoption. Tensor's ordinary assignment
        // adopts the canonical shared TensorImpl; no payload, allocator or
        // autograd owner is duplicated. The bound Python facade stays alive.
        // This is model state, not an unrestricted Tensor.data setter.
        .def("_rebind_model_value_", [](Tensor& self, const Tensor& value) {
            if (orthon::is_grad_enabled())
                throw std::runtime_error("model value rebind requires no_grad");
            if (self.shape() != value.shape())
                throw std::runtime_error("model value rebind: shape mismatch");
            if (!value.is_leaf() || value.has_grad_fn())
                throw std::runtime_error("model value rebind requires a graph-free leaf");
            if (self.requires_grad() != value.requires_grad())
                throw std::runtime_error("model value rebind: requires_grad mismatch");
            self = value;
        }, py::arg("value"))
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
            const int64_t flat = Layout::shape_numel(std::vector<int64_t>(
                sh.begin() + start_dim, sh.begin() + end_dim + 1));
            new_shape.push_back(flat);
            for (int64_t i = end_dim + 1; i < ndim; ++i)
                new_shape.push_back(sh[static_cast<size_t>(i)]);
            return reshape(t, new_shape);
        }, py::arg("start_dim") = 0, py::arg("end_dim") = -1)
        .def("transpose", [](const Tensor& t, int64_t dim0, int64_t dim1) {
            return transpose(t, dim0, dim1);
        }, py::arg("dim0"), py::arg("dim1"))
        // Explicit bounds follow the native slice contract: normalize negative
        // dim/start/end, then validate without Python slice clamping.
        // __getitem__ supports integer indices only.
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
                // Reject partial CUDA assignment before constructing a view. CPU supports
                // partial fill; full-index CUDA assignment uses the runtime transfer path.
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
        // Prefer Tensor dispatch, then numeric scalars. NotImplemented lets
        // Python try the reflected operator and raise its standard TypeError.
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
            // not commutative: this is `other - a`.
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
            // not commutative: this is `other / a`, the one scalar form whose
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
