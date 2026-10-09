#pragma once
#include <orthon/dispatch/operator.h>
namespace orthon {
Tensor relu_backward(const Tensor&, const Tensor&);
Tensor cross_entropy_backward(const Tensor&, const std::vector<int64_t>&, const Tensor&);
std::vector<Tensor> conv2d_backward(const Tensor&, const Tensor&, const Tensor&, const Tensor&, const dispatch::SpatialArgs&);
Tensor maxpool2d_backward(const Tensor&, const Tensor&, const dispatch::SpatialArgs&);
Tensor max_backward(const Tensor&, const Tensor&, const Tensor&, const ReductionPlan&);
Tensor unbroadcast_values(const Tensor&, const std::vector<int64_t>&);
} // namespace orthon
