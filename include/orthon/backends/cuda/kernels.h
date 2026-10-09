#pragma once
#include <orthon/dispatch/operator.h>
namespace orthon {
namespace backend {
namespace cuda {
using namespace dispatch;
Tensor host_to_cuda(const Tensor& self, Device target);
Tensor cuda_to_host(const Tensor& self, Device target);
Tensor cuda_to_cuda(const Tensor& self, Device target);
Tensor cuda_add_adapter(const Tensor& a, const Tensor& b, const std::vector<int64_t>& shape);
Tensor cuda_mul_adapter(const Tensor& a, const Tensor& b, const std::vector<int64_t>& shape);
Tensor cuda_cast_adapter(const Tensor& src, DType dst_dtype);
Tensor cuda_matmul_adapter(const Tensor& a, const Tensor& b, const MatmulPlan& plan);
Tensor cuda_neg_adapter(const Tensor& t);
Tensor cuda_reciprocal_adapter(const Tensor& t);
Tensor cuda_relu_adapter(const Tensor& t);
Tensor cuda_sqrt_adapter(const Tensor& t);
Tensor cuda_sigmoid_adapter(const Tensor& t);
Tensor cuda_tanh_adapter(const Tensor& t);
Tensor cuda_gelu_adapter(const Tensor& t);
Tensor cuda_exp_adapter(const Tensor& t);
Tensor cuda_leaky_relu_adapter(const Tensor& t, float alpha, uint64_t);
Tensor cuda_dropout_mask_adapter(const Tensor& t, float p, uint64_t seed);
Tensor cuda_sigmoid_backward_adapter(const Tensor& input, const Tensor& grad_output, float alpha);
Tensor cuda_tanh_backward_adapter(const Tensor& input, const Tensor& grad_output, float alpha);
Tensor cuda_leaky_relu_backward_adapter(const Tensor& input, const Tensor& grad_output, float alpha);
Tensor cuda_gelu_backward_adapter(const Tensor& input, const Tensor& grad_output, float alpha);
Tensor cuda_exp_backward_adapter(const Tensor& input, const Tensor& grad_output, float alpha);
Tensor cuda_neg_backward_adapter(const Tensor& input, const Tensor& grad_output, float alpha);
Tensor cuda_sqrt_backward_adapter(const Tensor& input, const Tensor& grad_output, float alpha);
Tensor cuda_reciprocal_backward_adapter(const Tensor& input, const Tensor& grad_output, float alpha);
Tensor cuda_relu_backward_adapter(const Tensor& input, const Tensor& grad_output, float alpha);
ReductionResult cuda_sum_adapter(const Tensor& t, const ReductionPlan& args);
ReductionResult cuda_mean_adapter(const Tensor& t, const ReductionPlan& args);
ReductionResult cuda_max_adapter(const Tensor& t, const ReductionPlan& args);
Tensor cuda_softmax_adapter(const Tensor& t, int64_t);
Tensor cuda_log_softmax_adapter(const Tensor& t, int64_t dim);
Tensor cuda_log_softmax_backward_adapter(const Tensor& t, const Tensor& grad, int64_t dim);
Tensor cuda_fused_linear_relu_adapter(const Tensor& x, const Tensor& weight, const Tensor& bias);
Tensor cuda_conv2d_adapter(const Tensor& x, const Tensor& w, const Tensor& b, const SpatialArgs& a);
std::vector<Tensor> cuda_conv2d_backward_adapter(const Tensor& x, const Tensor& w, const Tensor& b, const Tensor& g, const SpatialArgs& a);
Tensor cuda_maxpool2d_adapter(const Tensor& x, const SpatialArgs& a);
Tensor cuda_maxpool2d_backward_adapter(const Tensor& x, const Tensor& g, const SpatialArgs& a);
Tensor cuda_cross_entropy_adapter(const Tensor& x, const std::vector<int64_t>& targets);
Tensor cuda_cross_entropy_backward_adapter(const Tensor& x, const std::vector<int64_t>& targets, const Tensor& grad);
Tensor cuda_nll_loss_adapter(const Tensor& x, const Tensor& targets, const std::string& reduction);
Tensor cuda_nll_loss_backward_adapter(const Tensor& x, const Tensor& targets, const Tensor& grad, const std::string& reduction);
Tensor cuda_embedding_adapter(const Tensor& indices, const Tensor& weight, std::optional<int64_t> pad);
Tensor cuda_embedding_backward_adapter(const Tensor& indices, const Tensor& grad, const std::vector<int64_t>& shape, std::optional<int64_t> pad);
void cuda_inplace_mul_adapter(Tensor& a, float alpha);
void cuda_inplace_add_adapter(Tensor& a, const Tensor& b, float alpha);
Tensor cuda_unbroadcast_adapter(const Tensor& grad, const std::vector<int64_t>& shape);
bool cuda_isfinite_all_adapter(const Tensor& t);
std::pair<Tensor, bool> cuda_unscale_adapter(const Tensor& t, float scale);
void cuda_amp_copy_adapter(Tensor& dst, const Tensor& src, const CopyArgs&);
void cuda_copy_range_adapter(Tensor& dst, const Tensor& src, const CopyArgs& a);
float cuda_scalar_read_adapter(const Tensor& t, int64_t offset, const char* op);
void cuda_scalar_write_adapter(const Tensor& t, int64_t offset, float value, const char* op);
Tensor cuda_max_backward_adapter(const Tensor& input, const Tensor& grad, const Tensor& indices, const ReductionPlan& args);
Tensor cuda_index_backward_adapter(const Tensor& grad, const ViewScatterArgs& args);
Tensor cuda_slice_backward_adapter(const Tensor& grad, const ViewScatterArgs& args);
} // namespace cuda

} // namespace backend
} // namespace orthon
