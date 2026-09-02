#pragma once
#include <orthon/core/tensor.h>
#include <utility>
#include <vector>

namespace orthon {

// CUDA tensor-valued reductions. Preconditions (caller must verify):
// contiguous float32/float16/bfloat16 on CUDA, numel > 0.
Tensor cuda_sum(const Tensor& t,
                const std::vector<int64_t>& out_shape,
                int64_t dim,
                bool reduce_all);
Tensor cuda_mean(const Tensor& t,
                 const std::vector<int64_t>& out_shape,
                 int64_t dim,
                 bool reduce_all);

// Returns {values (input dtype), indices (Int64)}.
// reduce_all=true ignores dim and reduces the entire tensor.
std::pair<Tensor, Tensor> cuda_max_with_indices(const Tensor& t,
                                                const std::vector<int64_t>& out_shape,
                                                int64_t dim,
                                                bool reduce_all);

// Scatter grad_out values back to the argmax positions in grad_in (zeros).
void cuda_max_scatter_grad(const Tensor& grad_out,
                           const Tensor& indices,
                           Tensor& grad_in,
                           int64_t outer,
                           int64_t reduce,
                           int64_t inner);

}  // namespace orthon
