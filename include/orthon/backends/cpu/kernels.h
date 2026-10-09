#pragma once
#include <orthon/dispatch/operator.h>
#include <orthon/core/tensor_impl.h>
namespace orthon {
namespace backend {
namespace cpu {
using namespace dispatch;
Tensor cpu_add(const Tensor& a, const Tensor& b,
                                  const std::vector<int64_t>& shape);
Tensor cpu_mul(const Tensor& a, const Tensor& b,
                                  const std::vector<int64_t>& shape);
void inplace_mul_(Tensor& a, float alpha);
void inplace_add_(Tensor& a, const Tensor& b, float alpha);
Tensor cpu_cast(const Tensor& src, DType dst_dtype);
Tensor cpu_matmul(const Tensor& a, const Tensor& b, const MatmulPlan& plan);
ReductionResult cpu_sum(const Tensor& t, const ReductionPlan& args);
ReductionResult cpu_mean(const Tensor& t, const ReductionPlan& args);
ReductionResult cpu_max(const Tensor& t, const ReductionPlan& args);
Tensor cpu_neg(const Tensor& t);
Tensor cpu_reciprocal(const Tensor& t);
Tensor cpu_sqrt(const Tensor& t);
Tensor cpu_relu(const Tensor& t);
Tensor cpu_sigmoid(const Tensor& t);
Tensor cpu_tanh(const Tensor& t);
Tensor cpu_gelu(const Tensor& t);
Tensor cpu_exp(const Tensor& t);
Tensor cpu_leaky_relu(const Tensor& t, float negative_slope, uint64_t);
Tensor cpu_dropout_mask(const Tensor& t, float p, uint64_t seed);
Tensor cpu_sigmoid_backward(const Tensor& input, const Tensor& grad_output, float negative_slope);
Tensor cpu_tanh_backward(const Tensor& input, const Tensor& grad_output, float negative_slope);
Tensor cpu_leaky_relu_backward(const Tensor& input, const Tensor& grad_output, float negative_slope);
Tensor cpu_gelu_backward(const Tensor& input, const Tensor& grad_output, float negative_slope);
Tensor cpu_exp_backward(const Tensor& input, const Tensor& grad_output, float negative_slope);
Tensor cpu_neg_backward(const Tensor& input, const Tensor& grad_output, float negative_slope);
Tensor cpu_sqrt_backward(const Tensor& input, const Tensor& grad_output, float negative_slope);
Tensor cpu_reciprocal_backward(const Tensor& input, const Tensor& grad_output, float negative_slope);
Tensor cpu_relu_backward(const Tensor& input, const Tensor& grad, float);
Tensor cpu_conv2d_direct(const Tensor& input, const Tensor& weight, const Tensor& bias, const SpatialArgs& a);
Tensor cpu_conv2d_im2col(const Tensor& input, const Tensor& weight, const Tensor& bias, const SpatialArgs& a);
Tensor cpu_conv2d(const Tensor& input, const Tensor& weight, const Tensor& bias, const SpatialArgs& a);
std::vector<Tensor> cpu_conv2d_backward(const Tensor& inp, const Tensor& wgt, const Tensor& bia, const Tensor& grad, const SpatialArgs& args);
Tensor cpu_maxpool2d(const Tensor& input, const SpatialArgs& a);
Tensor cpu_maxpool2d_backward(const Tensor& input, const Tensor& grad, const SpatialArgs& a);
Tensor cpu_fused_linear_relu(const Tensor& x, const Tensor& weight, const Tensor& bias);
Tensor cpu_embedding(const Tensor& indices, const Tensor& weight, std::optional<int64_t> padding_idx);
Tensor cpu_embedding_backward(const Tensor& indices, const Tensor& grad_output, const std::vector<int64_t>& weight_shape, std::optional<int64_t> padding_idx);
Tensor cpu_softmax(const Tensor& t, int64_t);
Tensor cpu_log_softmax(const Tensor& t, int64_t resolved);
Tensor cpu_log_softmax_backward(const Tensor& input, const Tensor& grad_output, int64_t resolved);
Tensor cpu_nll_loss(const Tensor& log_probs,
                const Tensor& targets,
                const std::string& reduction);
Tensor cpu_nll_loss_backward(const Tensor& log_probs,
                         const Tensor& targets,
                         const Tensor& grad_output,
                         const std::string& reduction);
Tensor cpu_cross_entropy(const Tensor& logits, const std::vector<int64_t>& targets);
Tensor cpu_cross_entropy_backward(const Tensor& logits, const std::vector<int64_t>& targets_, const Tensor& grad_output);
void cpu_full_fill(Tensor& t, double value);
void cpu_zero_fill(Tensor& t, double);
void cpu_fill(Tensor& t, double fill_value);
Tensor cpu_to(const Tensor& self, Device);
Tensor cpu_index_backward(const Tensor& grad, const ViewScatterArgs& args);
Tensor cpu_slice_backward(const Tensor& grad, const ViewScatterArgs& args);
Tensor cpu_unbroadcast(const Tensor& grad, const std::vector<int64_t>& target_shape);
Tensor cpu_max_backward(const Tensor& input, const Tensor& grad, const Tensor& argmax_indices_, const ReductionPlan& args);
bool cpu_isfinite_all(const Tensor& t);
void cpu_amp_copy(Tensor& dst, const Tensor& src, const CopyArgs&);
void cpu_copy_range(Tensor& dst, const Tensor& src, const CopyArgs& a);
float cpu_scalar_read(const Tensor& t, int64_t offset, const char*);
void cpu_scalar_write(const Tensor& t, int64_t offset, float value, const char*);
inline Tensor empty(const std::vector<int64_t>& shape, DType dtype = DType::Float32, Device device = Device{}) { return Tensor(std::make_shared<TensorImpl>(shape, dtype, device)); }
inline Tensor zeros(const std::vector<int64_t>& shape, DType dtype = DType::Float32, Device device = Device{}) { Tensor t = empty(shape, dtype, device); cpu_zero_fill(t, 0); return t; }
inline Tensor full(const std::vector<int64_t>& shape, double value, DType dtype = DType::Float32, Device device = Device{}) { Tensor t = empty(shape, dtype, device); cpu_full_fill(t, value); return t; }
} // namespace cpu

} // namespace backend
} // namespace orthon
