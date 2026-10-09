#include <orthon/dispatch/dispatcher.h>
#include <orthon/planning/matmul_planner.h>
#include <orthon/runtime/runtime.h>
#include <orthon/ops/binary_ops.h>
#include <orthon/ops/matmul_ops.h>
#include <orthon/ops/fill_ops.h>
#include <orthon/ops/shape_ops.h>
#include <orthon/core/dtype.h>
#include <orthon/autograd/grad_mode.h>
#include <orthon/autograd/matmul_node.h>
#include <stdexcept>
#include <string>


namespace orthon {

static void validate_matmul(const Tensor& a, const Tensor& b) {
    if (!((a.ndim() == 2 && b.ndim() == 2) ||
          (a.ndim() == 3 && b.ndim() == 3)))
        throw std::runtime_error("matmul: only 2D or batched 3D tensors supported");
    if (a.ndim() == 2 && a.shape()[1] != b.shape()[0])
        throw std::runtime_error(
            "matmul: shape mismatch: (" +
            std::to_string(a.shape()[0]) + ", " + std::to_string(a.shape()[1]) +
            ") @ (" +
            std::to_string(b.shape()[0]) + ", " + std::to_string(b.shape()[1]) + ")");
    if (a.ndim() == 3 &&
        (a.shape()[0] != b.shape()[0] || a.shape()[2] != b.shape()[1]))
        throw std::runtime_error(
            "matmul: shape mismatch for batched 3D tensors");
}


static void validate_matmul_relationships(const Tensor& a, const Tensor& b) {
#ifdef WITH_CUDA
    if (a.device().is_cuda() || b.device().is_cuda()) {
        if (a.device() != b.device()) throw std::runtime_error("matmul: device mismatch");
        if (a.dtype() != b.dtype())
            throw std::runtime_error("matmul: CUDA path requires float32, float16, or bfloat16 (same dtype)");
        return;
    }
#endif
    if (a.dtype() != b.dtype()) throw std::runtime_error("matmul: dtype mismatch");
    if (a.device() != b.device()) throw std::runtime_error("matmul: device mismatch");
}
Tensor matmul(const Tensor& a, const Tensor& b) {
    validate_matmul(a, b);
    validate_matmul_relationships(a, b);
    using namespace dispatch;
    const auto& kernel = default_dispatcher().resolve<MatmulKernel>(
        binary_request(OperatorId::Matmul, a, b), default_runtime());
    const auto plan = default_matmul_planner().plan(matmul_problem(a, b), default_runtime());
    Tensor out = kernel.call(a, b, plan);
    const bool req = is_grad_enabled() && (a.requires_grad() || b.requires_grad());
    out.requires_grad_(req);
    out.set_is_leaf(!req);
    if (req) out.set_grad_fn(std::make_shared<MatMulNode>(a, b));
    return out;
}

}  // namespace orthon
