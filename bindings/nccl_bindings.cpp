#include <pybind11/pybind11.h>
#include <orthon/distributed/nccl_process_group.h>
#include <orthon/runtime/runtime.h>

namespace py = pybind11;
using orthon::Tensor;
using orthon::distributed::NcclProcessGroup;
using orthon::distributed::Work;
using orthon::distributed::system_nccl_transport;

void bind_nccl(py::module_& m) {
    m.def("nccl_enabled", []() {
#ifdef WITH_NCCL
        return true;
#else
        return false;
#endif
    });
#ifdef WITH_NCCL
    py::class_<Work, std::shared_ptr<Work>>(m, "_NcclWork")
        .def("wait", &Work::wait)
        .def("is_completed", &Work::is_completed);
    py::class_<NcclProcessGroup, std::shared_ptr<NcclProcessGroup>>(m, "_NcclCommunicator")
        .def_property_readonly("rank", &NcclProcessGroup::rank)
        .def_property_readonly("world_size", &NcclProcessGroup::world_size)
        .def_property_readonly("device_index", [](const NcclProcessGroup& group) { return group.device().index; });
    m.def("_nccl_unique_id", []() { return py::bytes(system_nccl_transport()->unique_id()); });
    m.def("_nccl_init", [](py::bytes id, int rank, int world_size, int device_index) {
        auto transport = system_nccl_transport();
        auto context = orthon::default_runtime().cuda(device_index);
        return std::make_shared<NcclProcessGroup>(*context, transport, static_cast<std::string>(id), rank, world_size);
    }, py::arg("unique_id_bytes"), py::arg("rank"), py::arg("world_size"), py::arg("device_index"));
    m.def("_nccl_all_reduce_sum_async", [](std::shared_ptr<NcclProcessGroup> group, Tensor& tensor) {
        return group->all_reduce(tensor);
    });
    m.def("_nccl_all_reduce_sum", [](std::shared_ptr<NcclProcessGroup> group, Tensor& tensor) {
        group->all_reduce(tensor)->wait();
    });
    m.def("_nccl_broadcast_async", [](std::shared_ptr<NcclProcessGroup> group, Tensor& tensor, int src) {
        return group->broadcast(tensor, src);
    });
    m.def("_nccl_broadcast", [](std::shared_ptr<NcclProcessGroup> group, Tensor& tensor, int src) {
        group->broadcast(tensor, src)->wait();
    });
    m.def("_nccl_barrier", [](std::shared_ptr<NcclProcessGroup> group) { group->barrier()->wait(); });
    m.def("_nccl_shutdown", [](std::shared_ptr<NcclProcessGroup> group) { group->shutdown(); });
#else
    auto unavailable = []() { system_nccl_transport(); };
    m.def("_nccl_unique_id", unavailable);
    m.def("_nccl_init", [unavailable](py::bytes, int, int, int) { unavailable(); });
    m.def("_nccl_all_reduce_sum", [unavailable](py::object, Tensor&) { unavailable(); });
    m.def("_nccl_all_reduce_sum_async", [unavailable](py::object, Tensor&) { unavailable(); });
    m.def("_nccl_broadcast", [unavailable](py::object, Tensor&, int) { unavailable(); });
    m.def("_nccl_broadcast_async", [unavailable](py::object, Tensor&, int) { unavailable(); });
    m.def("_nccl_barrier", [unavailable](py::object) { unavailable(); });
    m.def("_nccl_shutdown", [unavailable](py::object) { unavailable(); });
#endif
}
