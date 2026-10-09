#include <orthon/distributed/process_group.h>
#include <stdexcept>
namespace orthon {
namespace distributed {
ProcessGroup::ProcessGroup(int rank, int world_size)
    : lifetime_(std::make_shared<GroupLifetime>()), rank_(rank), world_size_(world_size) {
    if (world_size < 1 || rank < 0 || rank >= world_size)
        throw std::invalid_argument("process group rank/world_size is invalid");
}
void ProcessGroup::ensure_open() const {
    if (lifetime_->closed) throw std::runtime_error("process group has been shut down");
    if (lifetime_->failed) throw std::runtime_error("process group failed");
}
} // namespace distributed
} // namespace orthon
