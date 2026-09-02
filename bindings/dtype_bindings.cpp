#include <pybind11/pybind11.h>
#include <orthon/core/dtype.h>

namespace py = pybind11;

using orthon::DType;

void bind_dtype(py::module_& m) {
    py::enum_<DType>(m, "DType")
        .value("Float32",  DType::Float32)
        .value("Float64",  DType::Float64)
        .value("Float16",  DType::Float16)
        .value("BFloat16", DType::BFloat16)
        .value("Int32",    DType::Int32)
        .value("Int64",    DType::Int64)
        .value("UInt8",    DType::UInt8)
        .value("Int8",     DType::Int8)
        .value("Bool",     DType::Bool)
        .export_values();
}
