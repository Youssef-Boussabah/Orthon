#include <pybind11/pybind11.h>
#include <orthon/device/device.h>

#include <string>

namespace py = pybind11;

using orthon::Device;
using orthon::DeviceType;

void bind_device(py::module_& m) {
    py::enum_<DeviceType>(m, "DeviceType")
        .value("CPU",  DeviceType::CPU)
        .value("CUDA", DeviceType::CUDA)
        .export_values();

    py::class_<Device>(m, "Device")
        .def(py::init<>())
        .def(py::init<DeviceType, int>(), py::arg("type"), py::arg("index"))
        // Construct from the same spec string Tensor.to() accepts, so a device
        // STRING and a Device OBJECT are interchangeable everywhere. Without
        // it the two forms could not be reconciled in Python: Tensor.to() took
        // either, but Device.__eq__/__ne__ take only a Device, so any code
        // holding a caller-supplied device spec had to guess which form it
        // had. The checkpoint loader normalises map_location through this. The
        // parser is the native one, so "cuda" means cuda:0.
        .def(py::init([](const std::string& spec) {
            return orthon::device_from_string(spec);
        }), py::arg("spec"))
        .def("is_cpu",  &Device::is_cpu)
        .def("is_cuda", &Device::is_cuda)
        // Read-only: Device hashes by value, so mutating type/index on an
        // instance already inside a dict or set would silently change its
        // hash. From Python, Device is an immutable value object.
        .def_readonly("type",  &Device::type)
        .def_readonly("index", &Device::index)
        // py::is_operator() returns NotImplemented for a non-Device operand,
        // so `device == "cpu"` falls back to ordinary Python semantics
        // (False / True) instead of raising a pybind overload error.
        .def("__eq__",   &Device::operator==, py::is_operator())
        .def("__ne__",   &Device::operator!=, py::is_operator())
        // Value hash matching __eq__: equal devices hash equally, so Device
        // works as a dict key / set member.
        .def("__hash__", [](const Device& d) -> std::size_t {
            return (static_cast<std::size_t>(d.index) << 1) |
                   (d.is_cuda() ? 1u : 0u);
        })
        .def("__repr__", [](const Device& d) -> std::string {
            if (d.is_cpu()) return "device('cpu')";
            return "device('cuda:" + std::to_string(d.index) + "')";
        });
}
