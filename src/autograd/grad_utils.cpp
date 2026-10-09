#include <orthon/autograd/grad_utils.h>
#include <orthon/ops/backward_ops.h>
namespace orthon {
Tensor unbroadcast(const Tensor& grad, const std::vector<int64_t>& target_shape) {
    return unbroadcast_values(grad, target_shape);
}
}
