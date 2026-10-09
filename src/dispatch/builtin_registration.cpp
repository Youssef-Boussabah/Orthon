#include <orthon/dispatch/dispatcher.h>
#include <orthon/backends/cpu/kernels.h>
#ifdef WITH_CUDA
#include <orthon/backends/cuda/kernels.h>
#endif

namespace orthon::dispatch {
namespace {
void register_schemas(KernelRegistry& r) {
    for (auto id : {OperatorId::Add, OperatorId::Sub, OperatorId::Mul, OperatorId::Div}) {
        const char* name = id == OperatorId::Add ? "add" : id == OperatorId::Sub ? "sub" :
                           id == OperatorId::Mul ? "mul" : "div";
        r.add_schema({id, name, Signature::Binary, 2, id == OperatorId::Sub || id == OperatorId::Div});
    }
    r.add_schema({OperatorId::Neg, "neg", Signature::Unary, 1});
    r.add_schema({OperatorId::Reciprocal, "reciprocal", Signature::Unary, 1});
    r.add_schema({OperatorId::Cast, "cast", Signature::Cast, 1});
    r.add_schema({OperatorId::To, "to", Signature::Transfer, 1});
    r.add_schema({OperatorId::Sum, "sum", Signature::Reduction, 1});
    r.add_schema({OperatorId::Mean, "mean", Signature::Reduction, 1});
    r.add_schema({OperatorId::Max, "max", Signature::Reduction, 1});
    r.add_schema({OperatorId::ArgMax, "argmax", Signature::Reduction, 1});
    r.add_schema({OperatorId::Matmul, "matmul", Signature::Matmul, 2});
    r.add_schema({OperatorId::Sqrt, "sqrt", Signature::Unary, signature_arity(Signature::Unary)});
    r.add_schema({OperatorId::Relu, "relu", Signature::Unary, signature_arity(Signature::Unary)});
    r.add_schema({OperatorId::Sigmoid, "sigmoid", Signature::Unary, signature_arity(Signature::Unary)});
    r.add_schema({OperatorId::Tanh, "tanh", Signature::Unary, signature_arity(Signature::Unary)});
    r.add_schema({OperatorId::Gelu, "gelu", Signature::Unary, signature_arity(Signature::Unary)});
    r.add_schema({OperatorId::Exp, "exp", Signature::Unary, signature_arity(Signature::Unary)});
    r.add_schema({OperatorId::LeakyRelu, "leaky_relu", Signature::ParamUnary, signature_arity(Signature::ParamUnary)});
    r.add_schema({OperatorId::DropoutMask, "dropout_mask", Signature::ParamUnary, signature_arity(Signature::ParamUnary)});
    r.add_schema({OperatorId::SigmoidBackward, "sigmoid_backward", Signature::Backward, signature_arity(Signature::Backward)});
    r.add_schema({OperatorId::TanhBackward, "tanh_backward", Signature::Backward, signature_arity(Signature::Backward)});
    r.add_schema({OperatorId::LeakyReluBackward, "leaky_relu_backward", Signature::Backward, signature_arity(Signature::Backward)});
    r.add_schema({OperatorId::GeluBackward, "gelu_backward", Signature::Backward, signature_arity(Signature::Backward)});
    r.add_schema({OperatorId::ExpBackward, "exp_backward", Signature::Backward, signature_arity(Signature::Backward)});
    r.add_schema({OperatorId::NegBackward, "neg_backward", Signature::Backward, signature_arity(Signature::Backward)});
    r.add_schema({OperatorId::SqrtBackward, "sqrt_backward", Signature::Backward, signature_arity(Signature::Backward)});
    r.add_schema({OperatorId::ReciprocalBackward, "reciprocal_backward", Signature::Backward, signature_arity(Signature::Backward)});
    r.add_schema({OperatorId::ReluBackward, "relu_backward", Signature::Backward, signature_arity(Signature::Backward)});
    r.add_schema({OperatorId::SoftmaxRows, "softmax", Signature::Dimension, signature_arity(Signature::Dimension)});
    r.add_schema({OperatorId::LogSoftmax, "log_softmax", Signature::Dimension, signature_arity(Signature::Dimension)});
    r.add_schema({OperatorId::LogSoftmaxBackward, "log_softmax backward", Signature::DimensionBackward, signature_arity(Signature::DimensionBackward)});
    r.add_schema({OperatorId::FusedLinearRelu, "fused_linear_relu", Signature::Ternary, signature_arity(Signature::Ternary)});
    r.add_schema({OperatorId::Conv2d, "conv2d", Signature::Convolution, signature_arity(Signature::Convolution)});
    r.add_schema({OperatorId::Conv2dDirect, "conv2d_direct", Signature::Convolution, signature_arity(Signature::Convolution)});
    r.add_schema({OperatorId::Conv2dIm2col, "conv2d_im2col", Signature::Convolution, signature_arity(Signature::Convolution)});
    r.add_schema({OperatorId::Conv2dBackward, "conv2d backward", Signature::ConvolutionBackward, signature_arity(Signature::ConvolutionBackward)});
    r.add_schema({OperatorId::MaxPool2d, "maxpool2d", Signature::Pool, signature_arity(Signature::Pool)});
    r.add_schema({OperatorId::MaxPool2dBackward, "maxpool2d backward", Signature::PoolBackward, signature_arity(Signature::PoolBackward)});
    r.add_schema({OperatorId::CrossEntropy, "cross_entropy", Signature::TargetsLoss, signature_arity(Signature::TargetsLoss)});
    r.add_schema({OperatorId::CrossEntropyBackward, "cross_entropy backward", Signature::TargetsBackward, signature_arity(Signature::TargetsBackward)});
    r.add_schema({OperatorId::NllLoss, "nll_loss", Signature::TensorLoss, signature_arity(Signature::TensorLoss)});
    r.add_schema({OperatorId::NllLossBackward, "nll_loss backward", Signature::TensorLossBackward, signature_arity(Signature::TensorLossBackward)});
    r.add_schema({OperatorId::Embedding, "embedding", Signature::Embedding, signature_arity(Signature::Embedding)});
    r.add_schema({OperatorId::EmbeddingBackward, "embedding backward", Signature::EmbeddingBackward, signature_arity(Signature::EmbeddingBackward)});
    r.add_schema({OperatorId::Fill, "fill_", Signature::Fill, signature_arity(Signature::Fill)});
    r.add_schema({OperatorId::FullFill, "full", Signature::Fill, signature_arity(Signature::Fill)});
    r.add_schema({OperatorId::ZeroFill, "zeros", Signature::Fill, signature_arity(Signature::Fill)});
    r.add_schema({OperatorId::MulInplace, "mul_", Signature::Scale, signature_arity(Signature::Scale)});
    r.add_schema({OperatorId::AddInplace, "add_", Signature::Inplace, signature_arity(Signature::Inplace)});
    r.add_schema({OperatorId::IndexBackward, "index_backward", Signature::ViewScatter, signature_arity(Signature::ViewScatter)});
    r.add_schema({OperatorId::SliceBackward, "slice_backward", Signature::ViewScatter, signature_arity(Signature::ViewScatter)});
    r.add_schema({OperatorId::MaxBackward, "max backward", Signature::MaxScatter, signature_arity(Signature::MaxScatter)});
    r.add_schema({OperatorId::Unbroadcast, "unbroadcast", Signature::Shape, signature_arity(Signature::Shape)});
    r.add_schema({OperatorId::Finite, "isfinite_all", Signature::Finite, signature_arity(Signature::Finite)});
    r.add_schema({OperatorId::AmpUnscale, "_amp_unscale_to_float32_and_check", Signature::Unscale, signature_arity(Signature::Unscale)});
    r.add_schema({OperatorId::AmpCopy, "_amp_copy_master_to_model_", Signature::Copy, signature_arity(Signature::Copy)});
    r.add_schema({OperatorId::CopyRange, "copy_flat_range", Signature::Copy, signature_arity(Signature::Copy)});
    r.add_schema({OperatorId::ScalarRead, "scalar_read", Signature::ScalarRead, signature_arity(Signature::ScalarRead)});
    r.add_schema({OperatorId::ScalarWrite, "scalar_write", Signature::ScalarWrite, signature_arity(Signature::ScalarWrite)});
}
void add(KernelRegistry& r, OperatorId id, Constraints c, Kernel kernel, const char* name) {
    r.expect_kernel(id, c);
    r.add_kernel({id, c, std::move(kernel), name});
}
void register_cpu_kernels(KernelRegistry& r) {
    using namespace backend::cpu;
    const auto compute = static_cast<DTypeMask>(dtype_bit(DType::Float32) | dtype_bit(DType::Float64));
    const Constraints c{DeviceType::CPU, compute};
    add(r, OperatorId::Add, c, BinaryKernel{cpu_add}, "cpu.add");
    add(r, OperatorId::Mul, c, BinaryKernel{cpu_mul}, "cpu.mul");
    add(r, OperatorId::Neg, c, UnaryKernel{cpu_neg}, "cpu.neg");
    add(r, OperatorId::Reciprocal, c, UnaryKernel{cpu_reciprocal}, "cpu.reciprocal");
    add(r, OperatorId::Matmul, c, MatmulKernel{cpu_matmul}, "cpu.matmul");
    add(r, OperatorId::Sum, c, ReductionKernel{cpu_sum}, "cpu.sum");
    add(r, OperatorId::Mean, c, ReductionKernel{cpu_mean}, "cpu.mean");
    add(r, OperatorId::Max, c, ReductionKernel{cpu_max}, "cpu.max");
    add(r, OperatorId::ArgMax, c, ReductionKernel{cpu_max}, "cpu.argmax");
    add(r, OperatorId::Cast, {DeviceType::CPU, all_dtypes}, CastKernel{cpu_cast}, "cpu.cast");
    add(r, OperatorId::To, {DeviceType::CPU, all_dtypes, all_layouts, DeviceType::CPU},
        TransferKernel{cpu_to}, "cpu.to.cpu");
    add(r, OperatorId::Sqrt, {DeviceType::CPU, compute}, UnaryKernel{cpu_sqrt}, "cpu.sqrt");
    add(r, OperatorId::Relu, {DeviceType::CPU, compute}, UnaryKernel{cpu_relu}, "cpu.relu");
    add(r, OperatorId::Sigmoid, {DeviceType::CPU, compute}, UnaryKernel{cpu_sigmoid}, "cpu.sigmoid");
    add(r, OperatorId::Tanh, {DeviceType::CPU, compute}, UnaryKernel{cpu_tanh}, "cpu.tanh");
    add(r, OperatorId::Gelu, {DeviceType::CPU, compute}, UnaryKernel{cpu_gelu}, "cpu.gelu");
    add(r, OperatorId::Exp, {DeviceType::CPU, compute}, UnaryKernel{cpu_exp}, "cpu.exp");
    add(r, OperatorId::LeakyRelu, {DeviceType::CPU, compute}, ParamUnaryKernel{cpu_leaky_relu}, "cpu.leaky_relu");
    add(r, OperatorId::DropoutMask, {DeviceType::CPU, dtype_bit(DType::Float32)}, ParamUnaryKernel{cpu_dropout_mask}, "cpu.dropout_mask");
    add(r, OperatorId::SigmoidBackward, {DeviceType::CPU, compute}, BackwardKernel{cpu_sigmoid_backward}, "cpu.sigmoid_backward");
    add(r, OperatorId::TanhBackward, {DeviceType::CPU, compute}, BackwardKernel{cpu_tanh_backward}, "cpu.tanh_backward");
    add(r, OperatorId::LeakyReluBackward, {DeviceType::CPU, compute}, BackwardKernel{cpu_leaky_relu_backward}, "cpu.leaky_relu_backward");
    add(r, OperatorId::GeluBackward, {DeviceType::CPU, compute}, BackwardKernel{cpu_gelu_backward}, "cpu.gelu_backward");
    add(r, OperatorId::ExpBackward, {DeviceType::CPU, compute}, BackwardKernel{cpu_exp_backward}, "cpu.exp_backward");
    add(r, OperatorId::NegBackward, {DeviceType::CPU, compute}, BackwardKernel{cpu_neg_backward}, "cpu.neg_backward");
    add(r, OperatorId::SqrtBackward, {DeviceType::CPU, compute}, BackwardKernel{cpu_sqrt_backward}, "cpu.sqrt_backward");
    add(r, OperatorId::ReciprocalBackward, {DeviceType::CPU, compute}, BackwardKernel{cpu_reciprocal_backward}, "cpu.reciprocal_backward");
    add(r, OperatorId::ReluBackward, {DeviceType::CPU, compute}, BackwardKernel{cpu_relu_backward}, "cpu.relu_backward");
    add(r, OperatorId::SoftmaxRows, {DeviceType::CPU, dtype_bit(DType::Float32)}, DimensionKernel{cpu_softmax}, "cpu.softmax");
    add(r, OperatorId::LogSoftmax, {DeviceType::CPU, dtype_bit(DType::Float32)}, DimensionKernel{cpu_log_softmax}, "cpu.log_softmax");
    add(r, OperatorId::LogSoftmaxBackward, {DeviceType::CPU, dtype_bit(DType::Float32)}, DimensionBackwardKernel{cpu_log_softmax_backward}, "cpu.log_softmax backward");
    add(r, OperatorId::FusedLinearRelu, {DeviceType::CPU, dtype_bit(DType::Float32)}, TernaryKernel{cpu_fused_linear_relu}, "cpu.fused_linear_relu");
    add(r, OperatorId::Conv2d, {DeviceType::CPU, dtype_bit(DType::Float32)}, ConvolutionKernel{cpu_conv2d}, "cpu.conv2d");
    add(r, OperatorId::Conv2dDirect, {DeviceType::CPU, dtype_bit(DType::Float32)}, ConvolutionKernel{cpu_conv2d_direct}, "cpu.conv2d_direct");
    add(r, OperatorId::Conv2dIm2col, {DeviceType::CPU, dtype_bit(DType::Float32)}, ConvolutionKernel{cpu_conv2d_im2col}, "cpu.conv2d_im2col");
    add(r, OperatorId::Conv2dBackward, {DeviceType::CPU, dtype_bit(DType::Float32)}, ConvolutionBackwardKernel{cpu_conv2d_backward}, "cpu.conv2d backward");
    add(r, OperatorId::MaxPool2d, {DeviceType::CPU, dtype_bit(DType::Float32)}, PoolKernel{cpu_maxpool2d}, "cpu.maxpool2d");
    add(r, OperatorId::MaxPool2dBackward, {DeviceType::CPU, dtype_bit(DType::Float32)}, PoolBackwardKernel{cpu_maxpool2d_backward}, "cpu.maxpool2d backward");
    add(r, OperatorId::CrossEntropy, {DeviceType::CPU, dtype_bit(DType::Float32)}, TargetsLossKernel{cpu_cross_entropy}, "cpu.cross_entropy");
    add(r, OperatorId::CrossEntropyBackward, {DeviceType::CPU, dtype_bit(DType::Float32)}, TargetsBackwardKernel{cpu_cross_entropy_backward}, "cpu.cross_entropy backward");
    add(r, OperatorId::NllLoss, {DeviceType::CPU, dtype_bit(DType::Float32)}, TensorLossKernel{cpu_nll_loss}, "cpu.nll_loss");
    add(r, OperatorId::NllLossBackward, {DeviceType::CPU, dtype_bit(DType::Float32)}, TensorLossBackwardKernel{cpu_nll_loss_backward}, "cpu.nll_loss backward");
    add(r, OperatorId::Embedding, {DeviceType::CPU, dtype_bit(DType::Float32)}, EmbeddingKernel{cpu_embedding}, "cpu.embedding");
    add(r, OperatorId::EmbeddingBackward, {DeviceType::CPU, dtype_bit(DType::Float32)}, EmbeddingBackwardKernel{cpu_embedding_backward}, "cpu.embedding backward");
    add(r, OperatorId::Fill, {DeviceType::CPU, dtype_bit(DType::Float32)}, FillKernel{cpu_fill}, "cpu.fill_");
    add(r, OperatorId::FullFill, {DeviceType::CPU, all_dtypes}, FillKernel{cpu_full_fill}, "cpu.full");
    add(r, OperatorId::ZeroFill, {DeviceType::CPU, all_dtypes}, FillKernel{cpu_zero_fill}, "cpu.zeros");
    add(r, OperatorId::MulInplace, {DeviceType::CPU, compute}, ScaleKernel{inplace_mul_}, "cpu.mul_");
    add(r, OperatorId::AddInplace, {DeviceType::CPU, compute}, InplaceKernel{inplace_add_}, "cpu.add_");
    add(r, OperatorId::IndexBackward, {DeviceType::CPU, all_dtypes}, ViewScatterKernel{cpu_index_backward}, "cpu.index_backward");
    add(r, OperatorId::SliceBackward, {DeviceType::CPU, all_dtypes}, ViewScatterKernel{cpu_slice_backward}, "cpu.slice_backward");
    add(r, OperatorId::MaxBackward, {DeviceType::CPU, compute}, MaxScatterKernel{cpu_max_backward}, "cpu.max backward");
    add(r, OperatorId::Unbroadcast, {DeviceType::CPU, compute}, ShapeKernel{cpu_unbroadcast}, "cpu.unbroadcast");
    add(r, OperatorId::Finite, {DeviceType::CPU, all_dtypes}, FiniteKernel{cpu_isfinite_all}, "cpu.isfinite_all");
    const auto amp_dtypes = static_cast<DTypeMask>(dtype_bit(DType::Float32) |
        dtype_bit(DType::Float16) | dtype_bit(DType::BFloat16));
    add(r, OperatorId::AmpCopy, {DeviceType::CPU, amp_dtypes}, CopyKernel{cpu_amp_copy}, "cpu._amp_copy_master_to_model_");
    add(r, OperatorId::CopyRange, {DeviceType::CPU, all_dtypes}, CopyKernel{cpu_copy_range}, "cpu.copy_flat_range");
    add(r, OperatorId::ScalarRead, {DeviceType::CPU, dtype_bit(DType::Float32)}, ScalarReadKernel{cpu_scalar_read}, "cpu.scalar_read");
    add(r, OperatorId::ScalarWrite, {DeviceType::CPU, dtype_bit(DType::Float32)}, ScalarWriteKernel{cpu_scalar_write}, "cpu.scalar_write");
}
#ifdef WITH_CUDA
void register_cuda_kernels(KernelRegistry& r) {
    using namespace backend::cuda;
    const auto compute = static_cast<DTypeMask>(dtype_bit(DType::Float32) | dtype_bit(DType::Float16) |
                                                dtype_bit(DType::BFloat16));
    const Constraints c{DeviceType::CUDA, compute};
    add(r, OperatorId::Add, c, BinaryKernel{cuda_add_adapter}, "cuda.add");
    add(r, OperatorId::Mul, c, BinaryKernel{cuda_mul_adapter}, "cuda.mul");
    add(r, OperatorId::Neg, c, UnaryKernel{cuda_neg_adapter}, "cuda.neg");
    add(r, OperatorId::Reciprocal, c, UnaryKernel{cuda_reciprocal_adapter}, "cuda.reciprocal");
    // BF16 custom kernels remain legal without BF16 cuBLAS eligibility. The
    // existing algorithm selector consumes context capabilities inside matmul.
    add(r, OperatorId::Matmul, c, MatmulKernel{cuda_matmul_adapter}, "cuda.matmul");
    add(r, OperatorId::Sum, c, ReductionKernel{cuda_sum_adapter}, "cuda.sum");
    add(r, OperatorId::Mean, c, ReductionKernel{cuda_mean_adapter}, "cuda.mean");
    add(r, OperatorId::Max, c, ReductionKernel{cuda_max_adapter}, "cuda.max");
    add(r, OperatorId::ArgMax, c, ReductionKernel{cuda_max_adapter}, "cuda.argmax");
    add(r, OperatorId::Cast, {DeviceType::CUDA, compute, all_layouts, {}, compute},
        CastKernel{cuda_cast_adapter}, "cuda.cast");
    add(r, OperatorId::To, {DeviceType::CPU, all_dtypes, all_layouts, DeviceType::CUDA},
        TransferKernel{host_to_cuda}, "cpu.to.cuda");
    // Storage-only CUDA dtypes can be copied contiguously; strided packing
    // supports only the existing compute trio. Disjoint layout constraints.
    for (auto target : {DeviceType::CPU, DeviceType::CUDA}) {
        auto kernel = target == DeviceType::CPU ? cuda_to_host : cuda_to_cuda;
        const char* name = target == DeviceType::CPU ? "cuda.to.cpu" : "cuda.to.cuda";
        add(r, OperatorId::To, {DeviceType::CUDA, all_dtypes, 5, target}, TransferKernel{kernel}, name);
        add(r, OperatorId::To, {DeviceType::CUDA, compute, 2, target}, TransferKernel{kernel}, name);
    }
    const auto host_compute = static_cast<DTypeMask>(dtype_bit(DType::Float32) | dtype_bit(DType::Float64));
    add(r, OperatorId::Sqrt, {DeviceType::CUDA, compute}, UnaryKernel{cuda_sqrt_adapter}, "cuda.sqrt");
    add(r, OperatorId::Relu, {DeviceType::CUDA, compute}, UnaryKernel{cuda_relu_adapter}, "cuda.relu");
    add(r, OperatorId::Sigmoid, {DeviceType::CUDA, compute}, UnaryKernel{cuda_sigmoid_adapter}, "cuda.sigmoid");
    add(r, OperatorId::Tanh, {DeviceType::CUDA, compute}, UnaryKernel{cuda_tanh_adapter}, "cuda.tanh");
    add(r, OperatorId::Gelu, {DeviceType::CUDA, compute}, UnaryKernel{cuda_gelu_adapter}, "cuda.gelu");
    add(r, OperatorId::Exp, {DeviceType::CUDA, compute}, UnaryKernel{cuda_exp_adapter}, "cuda.exp");
    add(r, OperatorId::LeakyRelu, {DeviceType::CUDA, compute}, ParamUnaryKernel{cuda_leaky_relu_adapter}, "cuda.leaky_relu");
    add(r, OperatorId::DropoutMask, {DeviceType::CUDA, dtype_bit(DType::Float32)}, ParamUnaryKernel{cuda_dropout_mask_adapter}, "cuda.dropout_mask");
    add(r, OperatorId::SigmoidBackward, {DeviceType::CUDA, compute}, BackwardKernel{cuda_sigmoid_backward_adapter}, "cuda.sigmoid_backward");
    add(r, OperatorId::TanhBackward, {DeviceType::CUDA, compute}, BackwardKernel{cuda_tanh_backward_adapter}, "cuda.tanh_backward");
    add(r, OperatorId::LeakyReluBackward, {DeviceType::CUDA, compute}, BackwardKernel{cuda_leaky_relu_backward_adapter}, "cuda.leaky_relu_backward");
    add(r, OperatorId::GeluBackward, {DeviceType::CUDA, compute}, BackwardKernel{cuda_gelu_backward_adapter}, "cuda.gelu_backward");
    add(r, OperatorId::ExpBackward, {DeviceType::CUDA, compute}, BackwardKernel{cuda_exp_backward_adapter}, "cuda.exp_backward");
    add(r, OperatorId::NegBackward, {DeviceType::CUDA, compute}, BackwardKernel{cuda_neg_backward_adapter}, "cuda.neg_backward");
    add(r, OperatorId::SqrtBackward, {DeviceType::CUDA, compute}, BackwardKernel{cuda_sqrt_backward_adapter}, "cuda.sqrt_backward");
    add(r, OperatorId::ReciprocalBackward, {DeviceType::CUDA, compute}, BackwardKernel{cuda_reciprocal_backward_adapter}, "cuda.reciprocal_backward");
    add(r, OperatorId::ReluBackward, {DeviceType::CUDA, compute}, BackwardKernel{cuda_relu_backward_adapter}, "cuda.relu_backward");
    add(r, OperatorId::SoftmaxRows, {DeviceType::CUDA, dtype_bit(DType::Float32)}, DimensionKernel{cuda_softmax_adapter}, "cuda.softmax");
    add(r, OperatorId::LogSoftmax, {DeviceType::CUDA, dtype_bit(DType::Float32)}, DimensionKernel{cuda_log_softmax_adapter}, "cuda.log_softmax");
    add(r, OperatorId::LogSoftmaxBackward, {DeviceType::CUDA, dtype_bit(DType::Float32)}, DimensionBackwardKernel{cuda_log_softmax_backward_adapter}, "cuda.log_softmax backward");
    add(r, OperatorId::FusedLinearRelu, {DeviceType::CUDA, dtype_bit(DType::Float32)}, TernaryKernel{cuda_fused_linear_relu_adapter}, "cuda.fused_linear_relu");
    add(r, OperatorId::Conv2d, {DeviceType::CUDA, compute}, ConvolutionKernel{cuda_conv2d_adapter}, "cuda.conv2d");
    add(r, OperatorId::Conv2dBackward, {DeviceType::CUDA, compute}, ConvolutionBackwardKernel{cuda_conv2d_backward_adapter}, "cuda.conv2d backward");
    add(r, OperatorId::MaxPool2d, {DeviceType::CUDA, dtype_bit(DType::Float32)}, PoolKernel{cuda_maxpool2d_adapter}, "cuda.maxpool2d");
    add(r, OperatorId::MaxPool2dBackward, {DeviceType::CUDA, dtype_bit(DType::Float32)}, PoolBackwardKernel{cuda_maxpool2d_backward_adapter}, "cuda.maxpool2d backward");
    add(r, OperatorId::CrossEntropy, {DeviceType::CUDA, dtype_bit(DType::Float32)}, TargetsLossKernel{cuda_cross_entropy_adapter}, "cuda.cross_entropy");
    add(r, OperatorId::CrossEntropyBackward, {DeviceType::CUDA, dtype_bit(DType::Float32)}, TargetsBackwardKernel{cuda_cross_entropy_backward_adapter}, "cuda.cross_entropy backward");
    add(r, OperatorId::NllLoss, {DeviceType::CUDA, dtype_bit(DType::Float32)}, TensorLossKernel{cuda_nll_loss_adapter}, "cuda.nll_loss");
    add(r, OperatorId::NllLossBackward, {DeviceType::CUDA, dtype_bit(DType::Float32)}, TensorLossBackwardKernel{cuda_nll_loss_backward_adapter}, "cuda.nll_loss backward");
    add(r, OperatorId::Embedding, {DeviceType::CUDA, dtype_bit(DType::Float32)}, EmbeddingKernel{cuda_embedding_adapter}, "cuda.embedding");
    add(r, OperatorId::EmbeddingBackward, {DeviceType::CUDA, dtype_bit(DType::Float32)}, EmbeddingBackwardKernel{cuda_embedding_backward_adapter}, "cuda.embedding backward");
    add(r, OperatorId::MulInplace, {DeviceType::CUDA, host_compute}, ScaleKernel{cuda_inplace_mul_adapter}, "cuda.mul_");
    add(r, OperatorId::AddInplace, {DeviceType::CUDA, host_compute}, InplaceKernel{cuda_inplace_add_adapter}, "cuda.add_");
    add(r, OperatorId::IndexBackward, {DeviceType::CUDA, all_dtypes}, ViewScatterKernel{cuda_index_backward_adapter}, "cuda.index_backward");
    add(r, OperatorId::SliceBackward, {DeviceType::CUDA, all_dtypes}, ViewScatterKernel{cuda_slice_backward_adapter}, "cuda.slice_backward");
    add(r, OperatorId::MaxBackward, {DeviceType::CUDA, compute}, MaxScatterKernel{cuda_max_backward_adapter}, "cuda.max backward");
    add(r, OperatorId::Unbroadcast, {DeviceType::CUDA, compute}, ShapeKernel{cuda_unbroadcast_adapter}, "cuda.unbroadcast");
    add(r, OperatorId::Finite, {DeviceType::CUDA, all_dtypes}, FiniteKernel{cuda_isfinite_all_adapter}, "cuda.isfinite_all");
    add(r, OperatorId::AmpUnscale, {DeviceType::CUDA, all_dtypes}, UnscaleKernel{cuda_unscale_adapter}, "cuda._amp_unscale_to_float32_and_check");
    add(r, OperatorId::AmpCopy, {DeviceType::CUDA, all_dtypes}, CopyKernel{cuda_amp_copy_adapter}, "cuda._amp_copy_master_to_model_");
    add(r, OperatorId::CopyRange, {DeviceType::CUDA, all_dtypes}, CopyKernel{cuda_copy_range_adapter}, "cuda.copy_flat_range");
    add(r, OperatorId::ScalarRead, {DeviceType::CUDA, dtype_bit(DType::Float32)}, ScalarReadKernel{cuda_scalar_read_adapter}, "cuda.scalar_read");
    add(r, OperatorId::ScalarWrite, {DeviceType::CUDA, dtype_bit(DType::Float32)}, ScalarWriteKernel{cuda_scalar_write_adapter}, "cuda.scalar_write");
}
#endif
}  // namespace
std::shared_ptr<const KernelRegistry> build_builtin_registry() {
    auto registry = std::make_shared<KernelRegistry>();
    register_schemas(*registry);
    register_cpu_kernels(*registry);
#ifdef WITH_CUDA
    register_cuda_kernels(*registry);
#endif
    registry->freeze();
    return registry;
}
}  // namespace orthon::dispatch
