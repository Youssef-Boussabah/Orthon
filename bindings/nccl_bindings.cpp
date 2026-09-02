#include <pybind11/pybind11.h>

#include <orthon/core/dtype.h>
#include <orthon/core/storage.h>
#include <orthon/core/tensor.h>
#include <orthon/core/tensor_impl.h>

#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#if defined(WITH_CUDA) && defined(WITH_NCCL)
#include <cuda_runtime_api.h>
#include <nccl.h>
#endif

namespace py = pybind11;

using orthon::DType;
using orthon::Tensor;

#if defined(WITH_CUDA) && defined(WITH_NCCL)

#define ORTHON_CUDA_CHECK_OP(op_name, call) \
    do { \
        cudaError_t _err = (call); \
        if (_err != cudaSuccess) \
            throw std::runtime_error( \
                std::string(op_name) + ": CUDA error: " + cudaGetErrorString(_err)); \
    } while (0)

#define ORTHON_NCCL_CHECK_OP(op_name, call) \
    do { \
        ncclResult_t _err = (call); \
        if (_err != ncclSuccess) \
            throw std::runtime_error( \
                std::string(op_name) + ": NCCL error: " + ncclGetErrorString(_err)); \
    } while (0)

class NcclWork {
public:
    NcclWork(cudaStream_t stream,
             int device_index,
             Tensor tensor,
             std::shared_ptr<void> communicator)
        : stream_(stream),
          device_index_(device_index),
          tensor_(std::move(tensor)),
          communicator_(std::move(communicator)),
          completed_(false) {}

    ~NcclWork() {
        if (stream_ != nullptr) {
            cudaSetDevice(device_index_);
            if (!completed_) cudaStreamSynchronize(stream_);
            cudaStreamDestroy(stream_);
        }
    }

    NcclWork(const NcclWork&) = delete;
    NcclWork& operator=(const NcclWork&) = delete;

    bool wait() {
        if (!completed_) {
            ORTHON_CUDA_CHECK_OP("nccl work wait", cudaSetDevice(device_index_));
            ORTHON_CUDA_CHECK_OP("nccl work wait", cudaStreamSynchronize(stream_));
            completed_ = true;
        }
        return true;
    }

    bool is_completed() {
        if (completed_) return true;
        ORTHON_CUDA_CHECK_OP("nccl work is_completed", cudaSetDevice(device_index_));
        cudaError_t err = cudaStreamQuery(stream_);
        if (err == cudaSuccess) {
            completed_ = true;
            return true;
        }
        if (err == cudaErrorNotReady) return false;
        throw std::runtime_error(
            std::string("nccl work is_completed: CUDA error: ") + cudaGetErrorString(err));
    }

private:
    cudaStream_t stream_;
    int device_index_;
    Tensor tensor_;
    std::shared_ptr<void> communicator_;
    bool completed_;
};

class NcclCommunicator {
public:
    NcclCommunicator(const std::string& unique_id_bytes,
                     int rank,
                     int world_size,
                     int device_index)
        : rank_(rank),
          world_size_(world_size),
          device_index_(device_index),
          comm_(nullptr),
          closed_(false) {
        if (world_size < 1)
            throw std::runtime_error("nccl init: world_size must be >= 1");
        if (rank < 0 || rank >= world_size)
            throw std::runtime_error("nccl init: rank must be in [0, world_size)");
        if (device_index < 0)
            throw std::runtime_error("nccl init: device index must be >= 0");
        if (unique_id_bytes.size() != NCCL_UNIQUE_ID_BYTES)
            throw std::runtime_error("nccl init: invalid unique id byte length");

        ncclUniqueId id;
        std::memcpy(&id, unique_id_bytes.data(), NCCL_UNIQUE_ID_BYTES);
        set_device("nccl init");
        ORTHON_NCCL_CHECK_OP("nccl init", ncclCommInitRank(&comm_, world_size_, id, rank_));
        ORTHON_CUDA_CHECK_OP("nccl init", cudaGetLastError());
    }

    ~NcclCommunicator() {
        if (!closed_ && comm_ != nullptr) {
            ncclCommDestroy(comm_);
        }
    }

    NcclCommunicator(const NcclCommunicator&) = delete;
    NcclCommunicator& operator=(const NcclCommunicator&) = delete;

    int rank() const { return rank_; }
    int world_size() const { return world_size_; }
    int device_index() const { return device_index_; }

    void broadcast(Tensor& tensor, int src) {
        ensure_open();
        if (src < 0 || src >= world_size_)
            throw std::runtime_error("nccl broadcast: src must be in [0, world_size)");
        require_tensor(tensor, "nccl broadcast");
        set_device("nccl broadcast");
        void* data = tensor_data(tensor, "nccl broadcast");
        ORTHON_NCCL_CHECK_OP("nccl broadcast", ncclBroadcast(
            data,
            data,
            static_cast<size_t>(tensor.numel()),
            nccl_dtype(tensor.dtype(), "nccl broadcast"),
            src,
            comm_,
            0));
        synchronize("nccl broadcast");
        // Synchronous collective: the in-place mutation has completed by the
        // time synchronize returns, so one bump per successful call.
        tensor.impl()->storage()->bump_version();
    }

    std::shared_ptr<NcclWork> broadcast_async(Tensor& tensor,
                                              int src,
                                              std::shared_ptr<void> communicator) {
        ensure_open();
        if (src < 0 || src >= world_size_)
            throw std::runtime_error("nccl broadcast: src must be in [0, world_size)");
        require_tensor(tensor, "nccl broadcast");
        set_device("nccl broadcast");
        void* data = tensor_data(tensor, "nccl broadcast");
        cudaStream_t stream = nullptr;
        ORTHON_CUDA_CHECK_OP("nccl broadcast", cudaStreamCreate(&stream));
        try {
            ORTHON_NCCL_CHECK_OP("nccl broadcast", ncclBroadcast(
                data,
                data,
                static_cast<size_t>(tensor.numel()),
                nccl_dtype(tensor.dtype(), "nccl broadcast"),
                src,
                comm_,
                stream));
            ORTHON_CUDA_CHECK_OP("nccl broadcast", cudaGetLastError());
            // Asynchronous collective: the tensor is logically scheduled for
            // mutation once the enqueue succeeds, so the bump happens here
            // rather than at wait()/completion time.
            tensor.impl()->storage()->bump_version();
            return std::make_shared<NcclWork>(
                stream, device_index_, tensor, std::move(communicator));
        } catch (...) {
            cudaStreamDestroy(stream);
            throw;
        }
    }

    void all_reduce_sum(Tensor& tensor) {
        ensure_open();
        require_tensor(tensor, "nccl all_reduce");
        set_device("nccl all_reduce");
        void* data = tensor_data(tensor, "nccl all_reduce");
        ORTHON_NCCL_CHECK_OP("nccl all_reduce", ncclAllReduce(
            data,
            data,
            static_cast<size_t>(tensor.numel()),
            nccl_dtype(tensor.dtype(), "nccl all_reduce"),
            ncclSum,
            comm_,
            0));
        synchronize("nccl all_reduce");
        // Synchronous collective: mutation completed; bump once.
        tensor.impl()->storage()->bump_version();
    }

    std::shared_ptr<NcclWork> all_reduce_sum_async(Tensor& tensor,
                                                   std::shared_ptr<void> communicator) {
        ensure_open();
        require_tensor(tensor, "nccl all_reduce");
        set_device("nccl all_reduce");
        void* data = tensor_data(tensor, "nccl all_reduce");
        cudaStream_t stream = nullptr;
        ORTHON_CUDA_CHECK_OP("nccl all_reduce", cudaStreamCreate(&stream));
        try {
            ORTHON_NCCL_CHECK_OP("nccl all_reduce", ncclAllReduce(
                data,
                data,
                static_cast<size_t>(tensor.numel()),
                nccl_dtype(tensor.dtype(), "nccl all_reduce"),
                ncclSum,
                comm_,
                stream));
            ORTHON_CUDA_CHECK_OP("nccl all_reduce", cudaGetLastError());
            // Asynchronous collective: bump on successful enqueue; the
            // tensor is logically committed to mutation from this point.
            tensor.impl()->storage()->bump_version();
            return std::make_shared<NcclWork>(
                stream, device_index_, tensor, std::move(communicator));
        } catch (...) {
            cudaStreamDestroy(stream);
            throw;
        }
    }

    void barrier() {
        ensure_open();
        set_device("nccl barrier");
        float one = 1.0f;
        float* buffer = nullptr;
        ORTHON_CUDA_CHECK_OP("nccl barrier", cudaMalloc(&buffer, sizeof(float)));
        try {
            ORTHON_CUDA_CHECK_OP(
                "nccl barrier",
                cudaMemcpy(buffer, &one, sizeof(float), cudaMemcpyHostToDevice));
            ORTHON_NCCL_CHECK_OP("nccl barrier", ncclAllReduce(
                buffer,
                buffer,
                1,
                ncclFloat32,
                ncclSum,
                comm_,
                0));
            synchronize("nccl barrier");
            ORTHON_CUDA_CHECK_OP("nccl barrier", cudaFree(buffer));
        } catch (...) {
            cudaFree(buffer);
            throw;
        }
    }

    void shutdown() {
        if (closed_ || comm_ == nullptr) return;
        set_device("nccl shutdown");
        ORTHON_NCCL_CHECK_OP("nccl shutdown", ncclCommDestroy(comm_));
        comm_ = nullptr;
        closed_ = true;
    }

private:
    void ensure_open() const {
        if (closed_ || comm_ == nullptr)
            throw std::runtime_error("nccl communicator has been shut down");
    }

    void require_tensor(const Tensor& tensor, const char* op_name) const {
        if (!tensor.device().is_cuda())
            throw std::runtime_error(std::string(op_name) + ": tensor must be CUDA");
        if (tensor.device().index != device_index_)
            throw std::runtime_error(
                std::string(op_name) + ": tensor CUDA device does not match communicator device");
        if (!tensor.is_contiguous())
            throw std::runtime_error(std::string(op_name) + ": tensor must be contiguous");
        (void)nccl_dtype(tensor.dtype(), op_name);
        if (tensor.numel() < 0)
            throw std::runtime_error(std::string(op_name) + ": tensor numel must be non-negative");
    }

    ncclDataType_t nccl_dtype(DType dtype, const char* op_name) const {
        switch (dtype) {
            case DType::Float32:
                return ncclFloat32;
            case DType::Float16:
                return ncclHalf;
            case DType::BFloat16:
                return ncclBfloat16;
            default:
                throw std::runtime_error(
                    std::string(op_name) +
                    ": only float32, float16, and bfloat16 tensors are supported");
        }
    }

    void set_device(const char* op_name) const {
        ORTHON_CUDA_CHECK_OP(op_name, cudaSetDevice(device_index_));
    }

    void synchronize(const char* op_name) const {
        ORTHON_CUDA_CHECK_OP(op_name, cudaGetLastError());
        ORTHON_CUDA_CHECK_OP(op_name, cudaDeviceSynchronize());
        ORTHON_CUDA_CHECK_OP(op_name, cudaGetLastError());
    }

    void* tensor_data(const Tensor& tensor, const char* op_name) const {
        void* data = tensor.data();
        if (data == nullptr && tensor.numel() != 0)
            throw std::runtime_error(std::string(op_name) + ": tensor data pointer is null");
        if (data != nullptr) {
            cudaPointerAttributes attrs;
            ORTHON_CUDA_CHECK_OP(op_name, cudaPointerGetAttributes(&attrs, data));
            if (attrs.type != cudaMemoryTypeDevice)
                throw std::runtime_error(std::string(op_name) + ": tensor pointer is not CUDA device memory");
            if (attrs.device != device_index_)
                throw std::runtime_error(
                    std::string(op_name) + ": tensor pointer CUDA device does not match communicator device");
        }
        return data;
    }

    int rank_;
    int world_size_;
    int device_index_;
    ncclComm_t comm_;
    bool closed_;
};

static py::bytes nccl_unique_id() {
    ncclUniqueId id;
    ORTHON_NCCL_CHECK_OP("nccl unique_id", ncclGetUniqueId(&id));
    return py::bytes(reinterpret_cast<const char*>(&id), NCCL_UNIQUE_ID_BYTES);
}

#endif

void bind_nccl(py::module_& m) {
    m.def("nccl_enabled", []() -> bool {
#if defined(WITH_CUDA) && defined(WITH_NCCL)
        return true;
#else
        return false;
#endif
    });

#if defined(WITH_CUDA) && defined(WITH_NCCL)
    py::class_<NcclWork, std::shared_ptr<NcclWork>>(m, "_NcclWork")
        .def("wait", &NcclWork::wait)
        .def("is_completed", &NcclWork::is_completed);

    py::class_<NcclCommunicator, std::shared_ptr<NcclCommunicator>>(m, "_NcclCommunicator")
        .def_property_readonly("rank", &NcclCommunicator::rank)
        .def_property_readonly("world_size", &NcclCommunicator::world_size)
        .def_property_readonly("device_index", &NcclCommunicator::device_index);

    m.def("_nccl_unique_id", &nccl_unique_id);
    m.def("_nccl_init",
          [](py::bytes unique_id_bytes, int rank, int world_size, int device_index) {
              return std::make_shared<NcclCommunicator>(
                  static_cast<std::string>(unique_id_bytes),
                  rank,
                  world_size,
                  device_index);
          },
          py::arg("unique_id_bytes"),
          py::arg("rank"),
          py::arg("world_size"),
          py::arg("device_index"));
    m.def("_nccl_broadcast",
          [](std::shared_ptr<NcclCommunicator> comm, Tensor& tensor, int src) {
              comm->broadcast(tensor, src);
          },
          py::arg("comm"),
          py::arg("tensor"),
          py::arg("src"));
    m.def("_nccl_broadcast_async",
          [](std::shared_ptr<NcclCommunicator> comm, Tensor& tensor, int src) {
              return comm->broadcast_async(tensor, src, comm);
          },
          py::arg("comm"),
          py::arg("tensor"),
          py::arg("src"));
    m.def("_nccl_all_reduce_sum",
          [](std::shared_ptr<NcclCommunicator> comm, Tensor& tensor) {
              comm->all_reduce_sum(tensor);
          },
          py::arg("comm"),
          py::arg("tensor"));
    m.def("_nccl_all_reduce_sum_async",
          [](std::shared_ptr<NcclCommunicator> comm, Tensor& tensor) {
              return comm->all_reduce_sum_async(tensor, comm);
          },
          py::arg("comm"),
          py::arg("tensor"));
    m.def("_nccl_barrier",
          [](std::shared_ptr<NcclCommunicator> comm) {
              comm->barrier();
          },
          py::arg("comm"));
    m.def("_nccl_shutdown",
          [](std::shared_ptr<NcclCommunicator> comm) {
              comm->shutdown();
          },
          py::arg("comm"));
#else
    // One helper so the message cannot drift between stubs; the signatures
    // still mirror the real bindings above.
    auto unavailable = []() {
        throw std::runtime_error(
            "NCCL is not available in this build (recompile with -DWITH_CUDA=ON -DWITH_NCCL=ON)");
    };
    m.def("_nccl_unique_id", unavailable);
    m.def("_nccl_init", [unavailable](py::bytes, int, int, int) { unavailable(); });
    m.def("_nccl_broadcast", [unavailable](py::object, Tensor&, int) { unavailable(); });
    m.def("_nccl_all_reduce_sum", [unavailable](py::object, Tensor&) { unavailable(); });
    m.def("_nccl_barrier", [unavailable](py::object) { unavailable(); });
    m.def("_nccl_shutdown", [unavailable](py::object) { unavailable(); });
#endif
}
