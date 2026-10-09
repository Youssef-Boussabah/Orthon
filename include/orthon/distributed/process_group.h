#pragma once
#include <memory>
#include <orthon/core/tensor.h>
#include <orthon/distributed/work.h>
namespace orthon {
namespace distributed {
class ProcessGroup {
public:
    ProcessGroup(int rank, int world_size);
    virtual ~ProcessGroup() = default;
    int rank() const { return rank_; }
    int world_size() const { return world_size_; }
    virtual std::shared_ptr<Work> all_reduce(Tensor& tensor) = 0;
    virtual std::shared_ptr<Work> broadcast(Tensor& tensor, int src) = 0;
    virtual std::shared_ptr<Work> barrier() = 0;
    virtual void shutdown() = 0;
protected:
    void ensure_open() const;
    std::shared_ptr<GroupLifetime> lifetime_;
private:
    int rank_, world_size_;
};
} // namespace distributed
} // namespace orthon
