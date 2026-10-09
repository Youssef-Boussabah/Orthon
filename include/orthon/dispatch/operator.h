#pragma once
#include <orthon/core/tensor.h>
#include <orthon/planning/matmul_plan.h>
#include <orthon/reduction/reduction.h>
#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace orthon {
namespace dispatch {

enum class OperatorId { Add, Sub, Mul, Div, Neg, Cast, To, Sum, Mean, Max, ArgMax,
                        Matmul, Reciprocal, Sqrt, Relu, Sigmoid, Tanh, Gelu, Exp, LeakyRelu, DropoutMask, SigmoidBackward, TanhBackward, LeakyReluBackward, GeluBackward, ExpBackward, NegBackward, SqrtBackward, ReciprocalBackward, ReluBackward, SoftmaxRows, LogSoftmax, LogSoftmaxBackward, FusedLinearRelu, Conv2d, Conv2dDirect, Conv2dIm2col, Conv2dBackward, MaxPool2d, MaxPool2dBackward, CrossEntropy, CrossEntropyBackward, NllLoss, NllLossBackward, Embedding, EmbeddingBackward, Fill, FullFill, ZeroFill, MulInplace, AddInplace, IndexBackward, SliceBackward, MaxBackward, Unbroadcast, Finite, AmpUnscale, AmpCopy, CopyRange, ScalarRead, ScalarWrite };
enum class Signature { Binary, Unary, Cast, Transfer, Reduction, Matmul, ParamUnary, Backward, Dimension, DimensionBackward, Ternary, Convolution, ConvolutionBackward, Pool, PoolBackward, TargetsLoss, TargetsBackward, TensorLoss, TensorLossBackward, Embedding, EmbeddingBackward, Fill, Scale, Inplace, ViewScatter, MaxScatter, Finite, Unscale, Copy, ScalarRead, ScalarWrite, Shape };
struct Schema {
    OperatorId id;
    std::string name;
    Signature signature;
    unsigned arity;
    bool composite = false;
};

// Logical metadata only. No Tensor identity, storage or autograd state.
enum class LayoutClass : unsigned { Contiguous = 1, Strided = 2, Empty = 4 };
constexpr unsigned all_layouts = 7;
using DTypeMask = uint16_t;
DTypeMask dtype_bit(DType dtype);
constexpr DTypeMask all_dtypes = (1u << 9) - 1;
struct DispatchRequest {
    OperatorId op;
    Device source;
    DType dtype;
    LayoutClass layout;
    std::optional<Device> target;
    std::optional<DType> target_dtype;
    int64_t rank = 0;
};
DispatchRequest unary_request(OperatorId id, const Tensor& input);
DispatchRequest binary_request(OperatorId id, const Tensor& a, const Tensor& b);
DispatchRequest cast_request(const Tensor& input, DType target);
DispatchRequest transfer_request(const Tensor& input, Device target);

// Normalized execution attributes; no graph or Python state.
struct SpatialArgs { int64_t sh, sw, ph, pw, kh = 0, kw = 0; };
struct ViewScatterArgs { std::vector<int64_t> shape; DType dtype; Device device; int64_t index = 0, dim = 0, start = 0, end = 0; };
struct CopyArgs { int64_t dst_offset = 0, src_offset = 0; int64_t count = 0; };
unsigned signature_arity(Signature signature);

template<Signature Tag, typename Fn>
struct TypedKernel {
    static constexpr Signature signature = Tag;
    Fn call;
};
using BinaryKernel = TypedKernel<Signature::Binary,
    Tensor (*)(const Tensor&, const Tensor&, const std::vector<int64_t>&)>;
using UnaryKernel = TypedKernel<Signature::Unary, Tensor (*)(const Tensor&)>;
using CastKernel = TypedKernel<Signature::Cast, Tensor (*)(const Tensor&, DType)>;
using TransferKernel = TypedKernel<Signature::Transfer, Tensor (*)(const Tensor&, Device)>;
using ReductionKernel = TypedKernel<Signature::Reduction,
    ReductionResult (*)(const Tensor&, const ReductionPlan&)>;
using MatmulKernel = TypedKernel<Signature::Matmul,
    Tensor (*)(const Tensor&, const Tensor&, const MatmulPlan&)>;
using ParamUnaryKernel = TypedKernel<Signature::ParamUnary, Tensor (*)(const Tensor&, float, uint64_t)>;
using BackwardKernel = TypedKernel<Signature::Backward, Tensor (*)(const Tensor&, const Tensor&, float)>;
using DimensionKernel = TypedKernel<Signature::Dimension, Tensor (*)(const Tensor&, int64_t)>;
using DimensionBackwardKernel = TypedKernel<Signature::DimensionBackward, Tensor (*)(const Tensor&, const Tensor&, int64_t)>;
using TernaryKernel = TypedKernel<Signature::Ternary, Tensor (*)(const Tensor&, const Tensor&, const Tensor&)>;
using ConvolutionKernel = TypedKernel<Signature::Convolution, Tensor (*)(const Tensor&, const Tensor&, const Tensor&, const SpatialArgs&)>;
using ConvolutionBackwardKernel = TypedKernel<Signature::ConvolutionBackward, std::vector<Tensor> (*)(const Tensor&, const Tensor&, const Tensor&, const Tensor&, const SpatialArgs&)>;
using PoolKernel = TypedKernel<Signature::Pool, Tensor (*)(const Tensor&, const SpatialArgs&)>;
using PoolBackwardKernel = TypedKernel<Signature::PoolBackward, Tensor (*)(const Tensor&, const Tensor&, const SpatialArgs&)>;
using TargetsLossKernel = TypedKernel<Signature::TargetsLoss, Tensor (*)(const Tensor&, const std::vector<int64_t>&)>;
using TargetsBackwardKernel = TypedKernel<Signature::TargetsBackward, Tensor (*)(const Tensor&, const std::vector<int64_t>&, const Tensor&)>;
using TensorLossKernel = TypedKernel<Signature::TensorLoss, Tensor (*)(const Tensor&, const Tensor&, const std::string&)>;
using TensorLossBackwardKernel = TypedKernel<Signature::TensorLossBackward, Tensor (*)(const Tensor&, const Tensor&, const Tensor&, const std::string&)>;
using EmbeddingKernel = TypedKernel<Signature::Embedding, Tensor (*)(const Tensor&, const Tensor&, std::optional<int64_t>)>;
using EmbeddingBackwardKernel = TypedKernel<Signature::EmbeddingBackward, Tensor (*)(const Tensor&, const Tensor&, const std::vector<int64_t>&, std::optional<int64_t>)>;
using FillKernel = TypedKernel<Signature::Fill, void (*)(Tensor&, double)>;
using ScaleKernel = TypedKernel<Signature::Scale, void (*)(Tensor&, float)>;
using InplaceKernel = TypedKernel<Signature::Inplace, void (*)(Tensor&, const Tensor&, float)>;
using ViewScatterKernel = TypedKernel<Signature::ViewScatter, Tensor (*)(const Tensor&, const ViewScatterArgs&)>;
using MaxScatterKernel = TypedKernel<Signature::MaxScatter, Tensor (*)(const Tensor&, const Tensor&, const Tensor&, const ReductionPlan&)>;
using FiniteKernel = TypedKernel<Signature::Finite, bool (*)(const Tensor&)>;
using UnscaleKernel = TypedKernel<Signature::Unscale, std::pair<Tensor, bool> (*)(const Tensor&, float)>;
using CopyKernel = TypedKernel<Signature::Copy, void (*)(Tensor&, const Tensor&, const CopyArgs&)>;
using ScalarReadKernel = TypedKernel<Signature::ScalarRead, float (*)(const Tensor&, int64_t, const char*)>;
using ScalarWriteKernel = TypedKernel<Signature::ScalarWrite, void (*)(const Tensor&, int64_t, float, const char*)>;
using ShapeKernel = TypedKernel<Signature::Shape, Tensor (*)(const Tensor&, const std::vector<int64_t>&)>;
using Kernel = std::variant<BinaryKernel, UnaryKernel, CastKernel, TransferKernel,
                            ReductionKernel, MatmulKernel, ParamUnaryKernel, BackwardKernel, DimensionKernel, DimensionBackwardKernel, TernaryKernel, ConvolutionKernel, ConvolutionBackwardKernel, PoolKernel, PoolBackwardKernel, TargetsLossKernel, TargetsBackwardKernel, TensorLossKernel, TensorLossBackwardKernel, EmbeddingKernel, EmbeddingBackwardKernel, FillKernel, ScaleKernel, InplaceKernel, ViewScatterKernel, MaxScatterKernel, FiniteKernel, UnscaleKernel, CopyKernel, ScalarReadKernel, ScalarWriteKernel, ShapeKernel>;
Signature signature_of(const Kernel& kernel);

}  // namespace dispatch
}  // namespace orthon
