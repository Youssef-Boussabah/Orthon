#include <orthon/core/tensor.h>
#include <orthon/core/tensor_impl.h>
#include <orthon/device/device_guard.h>
#include <orthon/core/dtype.h>
#include <orthon/ops/cpu/fill_ops.h>
#include <orthon/ops/cpu/binary_ops.h>
#include <orthon/ops/cpu/shape_ops.h>
#include <orthon/autograd/node.h>
#include <orthon/autograd/grad_mode.h>
#include <orthon/autograd/index_node.h>
#include <orthon/autograd/slice_node.h>
#include <orthon/autograd/to_node.h>
#include <cstring>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifdef WITH_CUDA
#include <cuda_runtime_api.h>
#include <orthon/ops/cuda/unary_kernels.h>
#define ORTHON_CUDA_CHECK(call) \
    do { \
        cudaError_t _err = (call); \
        if (_err != cudaSuccess) \
            throw std::runtime_error( \
                std::string("CUDA error: ") + cudaGetErrorString(_err)); \
    } while (0)
#endif

namespace orthon {

Tensor::Tensor(std::shared_ptr<TensorImpl> impl) : impl_(std::move(impl)) {}

const std::vector<int64_t>& Tensor::shape()   const { return impl_->shape(); }
const std::vector<int64_t>& Tensor::strides() const { return impl_->strides(); }
int64_t Tensor::numel()         const { return impl_->numel(); }
int64_t Tensor::ndim()          const { return impl_->ndim(); }
DType   Tensor::dtype()         const { return impl_->dtype(); }
Device  Tensor::device()        const { return impl_->device(); }
void*   Tensor::data()          const { return impl_->data(); }
bool    Tensor::is_contiguous() const { return impl_->is_contiguous(); }

Tensor Tensor::operator[](int64_t i) const {
    if (impl_->ndim() == 0)
        throw std::runtime_error("operator[]: cannot index a 0-d tensor");
    const auto& sh = impl_->shape();
    const auto& st = impl_->strides();
    if (i < 0) i += sh[0];
    if (i < 0 || i >= sh[0])
        throw std::out_of_range("operator[]: index out of range");

    int64_t new_offset = impl_->storage_offset() + i * st[0];
    std::vector<int64_t> new_shape (sh.begin() + 1, sh.end());
    std::vector<int64_t> new_strides(st.begin() + 1, st.end());
    Tensor out(std::make_shared<TensorImpl>(
        impl_->storage(), std::move(new_shape), std::move(new_strides), new_offset));

    // Graph metadata is the only thing autograd adds here: `out` keeps sharing
    // this tensor's storage, shape, strides and offset either way. `i` has been
    // normalised above, so IndexNode never re-resolves a negative subscript.
    if (is_grad_enabled() && requires_grad()) {
        out.requires_grad_(true);
        out.set_is_leaf(false);
        out.set_grad_fn(std::make_shared<IndexNode>(*this, i));
    }

    return out;
}

Tensor Tensor::slice(int64_t dim, int64_t start, int64_t end) const {
    const int64_t nd = impl_->ndim();
    if (dim < 0) dim += nd;
    if (dim < 0 || dim >= nd)
        throw std::out_of_range("slice: dim out of range");

    const auto& sh = impl_->shape();
    const auto& st = impl_->strides();
    if (start < 0) start += sh[dim];
    if (end   < 0) end   += sh[dim];
    if (start < 0 || end > sh[dim] || start > end)
        throw std::out_of_range("slice: invalid start/end");

    int64_t new_offset = impl_->storage_offset() + start * st[dim];
    std::vector<int64_t> new_shape  = sh;
    std::vector<int64_t> new_strides = st;
    new_shape[dim] = end - start;
    Tensor out(std::make_shared<TensorImpl>(
        impl_->storage(), std::move(new_shape), std::move(new_strides), new_offset));

    // Graph metadata is the only thing autograd adds here: `out` keeps sharing
    // this tensor's storage, and keeps the same shape, strides and offset either
    // way. dim/start/end have all been normalised above, so SliceNode stores
    // resolved values and never re-interprets a negative argument at backward.
    if (is_grad_enabled() && requires_grad()) {
        out.requires_grad_(true);
        out.set_is_leaf(false);
        out.set_grad_fn(std::make_shared<SliceNode>(*this, dim, start, end));
    }

    return out;
}

// Linear (contiguous) copy of this tensor on the same device.
static Tensor make_contiguous_copy(const Tensor& src) {
    const int64_t n = src.numel();
    Tensor out(std::make_shared<TensorImpl>(src.shape(), src.dtype(), src.device()));
    if (src.is_contiguous()) {
        std::memcpy(out.data(), src.data(),
                    static_cast<size_t>(n) * dtype_size(src.dtype()));
    } else {
        const size_t esz = dtype_size(src.dtype());
        char*        dst = static_cast<char*>(out.data());
        const char*  ptr = static_cast<const char*>(src.data());
        const auto& sh = src.shape();
        const auto& st = src.strides();
        const int   nd = static_cast<int>(sh.size());
        std::vector<int64_t> idx(nd, 0);
        for (int64_t elem = 0; elem < n; ++elem) {
            int64_t off = 0;
            for (int d = 0; d < nd; ++d) off += idx[d] * st[d];
            std::memcpy(dst + elem * static_cast<int64_t>(esz),
                        ptr  + off  * static_cast<int64_t>(esz), esz);
            for (int d = nd - 1; d >= 0; --d) {
                if (++idx[d] < sh[d]) break;
                idx[d] = 0;
            }
        }
    }
    return out;
}

// The physical transfer half of Tensor::to: same-device moves return a
// contiguous physical copy, cross-device moves linearize + allocate + memcpy.
// Never attaches autograd metadata - the public wrapper below owns that
// decision, which keeps the transfer usable from internal no-grad paths.
static Tensor to_physical(const Tensor& self, Device target) {
    if (self.device() == target) {
#ifdef WITH_CUDA
        if (self.device().is_cuda()) {
            // Authoritative device: this tensor's own.
            DeviceGuard guard(self.device());
            return cuda_make_contiguous(self);
        }
#endif
        return make_contiguous_copy(self);
    }

    // Cross-device transfer requires CUDA support.
#ifdef WITH_CUDA
    // Linearize non-contiguous source on its current device before crossing.
    // The linearisation is CUDA work belonging to the SOURCE device; the
    // transfer below belongs to whichever side owns the device pointer.
    Tensor src_c = [&] {
        if (!self.is_contiguous() && self.device().is_cuda()) {
            DeviceGuard guard(self.device());
            return cuda_make_contiguous(self);
        }
        return self.is_contiguous() ? self : make_contiguous_copy(self);
    }();

    const size_t nbytes =
        static_cast<size_t>(src_c.numel()) * dtype_size(src_c.dtype());
    // The allocation itself is already context-correct: the memory pool guards
    // its own raw allocations against `target`.
    Tensor out(std::make_shared<TensorImpl>(src_c.shape(), src_c.dtype(), target));

    if (src_c.device().is_cpu() && target.is_cuda()) {
        // Host -> device: the only device pointer is the destination.
        DeviceGuard guard(target);
        ORTHON_CUDA_CHECK(cudaMemcpy(
            out.data(), src_c.data(), nbytes, cudaMemcpyHostToDevice));
    } else if (src_c.device().is_cuda() && target.is_cpu()) {
        // Device -> host: the only device pointer is the source.
        DeviceGuard guard(src_c.device());
        ORTHON_CUDA_CHECK(cudaMemcpy(
            out.data(), src_c.data(), nbytes, cudaMemcpyDeviceToHost));
    } else {
        // Both sides are CUDA and the indices differ (equal indices returned
        // above).
        //
        // cudaMemcpy(..., cudaMemcpyDeviceToDevice) describes a copy WITHIN one
        // device's address space; which device that is came from ambient
        // context, so the call said nothing about the pair actually involved.
        // cudaMemcpyPeer names both ordinals explicitly:
        //
        //     cudaMemcpyPeer(dst, dstDevice, src, srcDevice, count)
        //
        // The installed CUDA 13.2 runtime documents it as "Copies memory from
        // one device to memory on another device", parameterised by the two
        // device ordinals, synchronous with respect to pending and future work
        // on the current, source and destination devices, and returning
        // cudaErrorInvalidDevice / cudaErrorInvalidValue on a bad pair. It does
        // NOT document a cudaDeviceEnablePeerAccess precondition, so none is
        // requested here: naming the ordinals is the whole fix, and installing
        // permanent process-wide peer mappings as a side effect is not.
        //
        // The guard is kept so the surrounding CUDA context is still the
        // destination's and is restored on the way out, but correctness no
        // longer depends on it. A pair the hardware cannot service fails
        // through ORTHON_CUDA_CHECK with the runtime's own error, so `out` is
        // never handed back mislabelled.
        DeviceGuard guard(target);
        ORTHON_CUDA_CHECK(cudaMemcpyPeer(
            out.data(), target.index,
            src_c.data(), src_c.device().index,
            nbytes));
    }
    return out;
#else
    throw std::runtime_error(
        "to: CUDA is not available in this build (recompile with -DWITH_CUDA=ON)");
#endif
}

Tensor Tensor::to(Device target) const {
    Tensor out = to_physical(*this, target);

    // Public to() participates in autograd. The forward stays the physical
    // copy above; only graph metadata is added, and only when it means
    // something: grad mode on, the source tracked, and a FLOATING dtype
    // (integer/bool transfers never join a graph, matching the cast
    // contract). Under NoGradGuard - which includes every internal
    // materialisation that wraps itself in one, and all of Tensor::backward -
    // this attaches nothing, which is what internal snapshot/detach call
    // sites rely on EXPLICITLY rather than on a detach side effect.
    if (is_grad_enabled() && requires_grad() && dtype_is_floating(dtype())) {
        out.requires_grad_(true);
        out.set_is_leaf(false);
        out.set_grad_fn(std::make_shared<ToNode>(*this));
    }
    return out;
}

const std::shared_ptr<TensorImpl>& Tensor::impl() const { return impl_; }

bool    Tensor::requires_grad() const { return impl_->requires_grad(); }
Tensor& Tensor::requires_grad_(bool value) {
    impl_->set_requires_grad(value);
    return *this;
}
bool Tensor::is_leaf()  const { return impl_->is_leaf(); }
void Tensor::set_is_leaf(bool value) { impl_->set_is_leaf(value); }
bool Tensor::has_grad() const { return impl_->has_grad(); }

Tensor Tensor::grad() const {
    if (!impl_->has_grad())
        throw std::runtime_error("grad(): tensor has no gradient");
    return Tensor(impl_->grad_impl());
}

void Tensor::set_grad(const Tensor& grad) {
    impl_->set_grad_impl(grad.impl());
}

void Tensor::zero_grad() { impl_->zero_grad(); }

void Tensor::add_post_accumulate_grad_hook(std::function<void()> hook) {
    impl_->add_post_accumulate_grad_hook(std::move(hook));
}

bool                  Tensor::has_grad_fn() const { return impl_->has_grad_fn(); }
std::shared_ptr<Node> Tensor::grad_fn()     const { return impl_->grad_fn(); }
void                  Tensor::set_grad_fn(std::shared_ptr<Node> fn) { impl_->set_grad_fn(std::move(fn)); }
void                  Tensor::clear_grad_fn() { impl_->clear_grad_fn(); }

static void build_topo(const Tensor& t,
                        std::vector<Tensor>& topo,
                        std::unordered_set<TensorImpl*>& visited) {
    if (!visited.insert(t.impl().get()).second) return;
    if (t.has_grad_fn()) {
        for (const Tensor& inp : t.grad_fn()->inputs)
            build_topo(inp, topo, visited);
    }
    topo.push_back(t);
}

// A Node may legitimately return a strided gradient - TransposeNode hands back
// a view rather than a copy - but add() dispatches to kernels that require
// contiguous operands on CUDA. Normalising here keeps that requirement out of
// every Node and out of the raw binary kernels.
//
// contiguous() is a no-op for an already-contiguous gradient, and it attaches a
// node only when its input requires grad. That qualifier cannot fire here:
// Tensor::backward holds a NoGradGuard for the whole traversal, which makes "a
// gradient never requires grad" true by construction rather than by assumption.
static Tensor accumulate_grad(const Tensor& existing, const Tensor& incoming) {
    return add(contiguous(existing), contiguous(incoming));
}

// A leaf's .grad is public, mutable state, and the owner of that leaf is
// entitled to modify it in place: DDP rescales it after all-reduce, AMP's
// optimizer unscales it, and user code is free to do the same. So two
// DISTINCT leaves must never be handed gradient tensors that share a Storage.
//
// Nothing upstream guarantees that. A Node may legitimately return the SAME
// gradient object for two of its inputs - AddNode does, because both partial
// derivatives are the identity - unbroadcast forwards its input untouched when
// the shapes already match, and contiguous() is an identity for an
// already-contiguous tensor. No step in that chain allocates, so both leaves
// end up pointing at one buffer and `a.grad.mul_(2)` silently scales b.grad.
//
// The VALUES are right at that moment: a Node only aliases when both gradients
// are the same quantity. This is therefore an ownership defect, not an
// arithmetic one, and it belongs here - the single boundary where a gradient
// stops being engine-internal and becomes public per-leaf state - rather than
// in every Node that might one day alias.
//
// The cost is confined to the case that needs it. The first leaf to claim a
// Storage keeps exactly the tensor it was given; only a later leaf that would
// land on the same buffer pays for a private copy. A graph whose leaves each
// receive a distinct gradient - which is every module in this repository -
// allocates nothing extra.
//
// Keyed on Storage rather than TensorImpl because two TensorImpls can be views
// over one buffer; sharing that buffer is what makes an in-place edit visible
// through the other leaf, so the buffer is the thing to track.
static Tensor privatise_leaf_grad(const Tensor& candidate,
                                  std::unordered_set<const Storage*>& claimed) {
    if (claimed.insert(candidate.impl()->storage().get()).second)
        return candidate;

    // Already owned by an earlier leaf. to() onto the tensor's own device is
    // the engine's physical-copy primitive; Tensor::backward's NoGradGuard is
    // what keeps it from attaching a ToNode to a gradient.
    Tensor owned = candidate.to(candidate.device());
    claimed.insert(owned.impl()->storage().get());
    return owned;
}

void Tensor::backward(bool retain_graph) {
    if (!requires_grad())
        throw std::runtime_error("backward(): tensor does not require grad");
    if (!is_leaf() && !has_grad_fn())
        throw std::runtime_error(
            "Cannot call backward twice on a freed graph. Use retain_graph=True.");

    // Everything below runs with autograd graph construction disabled.
    //
    // Node::backward implementations legitimately build gradients out of public
    // differentiable ops applied to the very inputs they hold - MulNode does
    // mul(grad, inputs[1]), MatMulNode transposes and re-multiplies its operands
    // - and those inputs require grad by definition. Without this guard those
    // calls attach nodes, so the returned gradient carries a live graph:
    // x.grad after `z = x * y` would report requires_grad=True and a grad_fn,
    // on both CPU and CUDA.
    //
    // Making it structural here rather than per-Node is deliberate. A guard in
    // every Node is a per-site pattern: correct only as long as every future
    // Node remembers it.
    //
    // Grad mode governs graph CONSTRUCTION, not whether an existing graph may
    // run, so disabling it here does not affect what backward computes. The
    // guard is RAII, so the caller's previous setting is restored on normal
    // return and on an exception thrown from a Node or a hook.
    //
    // If higher-order autograd (create_graph) is ever added, this is the guard
    // that must become conditional on it.
    NoGradGuard backward_guard;

    // Collect nodes in post-order DFS; reverse gives output-first topo order.
    std::vector<Tensor> topo;
    std::unordered_set<TensorImpl*> visited;
    build_topo(*this, topo, visited);

    // Preflight: every Node's saved-value versions are validated BEFORE
    // the seed is created, before grad_map is touched, before any Node runs
    // and before any leaf .grad or hook fires. A tensor mutated in-place
    // between forward and backward therefore fails the whole call cleanly,
    // with no partially written gradients left behind.
    for (const Tensor& t : topo) {
        if (t.has_grad_fn())
            t.grad_fn()->validate_saved_versions();
    }

    std::unordered_map<TensorImpl*, Tensor> grad_map;
    Tensor seed = ones(shape(), dtype());
    if (device().is_cuda())
        seed = seed.to(device());
    grad_map.emplace(impl().get(), seed);

    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        const Tensor& t = *it;
        if (!t.has_grad_fn()) continue;

        auto git = grad_map.find(t.impl().get());
        if (git == grad_map.end()) continue;

        // Per-node re-check: the preflight above covers mutations made
        // before backward started, but a hook or callback can mutate storage
        // WHILE the traversal runs. Re-validate immediately before this
        // Node's formula consumes its saved values.
        t.grad_fn()->validate_saved_versions();

        auto grads  = t.grad_fn()->backward(git->second);
        const auto& inputs = t.grad_fn()->inputs;

        for (size_t i = 0; i < inputs.size() && i < grads.size(); ++i) {
            if (!inputs[i].requires_grad()) continue;
            TensorImpl* inp_key = inputs[i].impl().get();
            auto [ins_it, inserted] = grad_map.emplace(inp_key, grads[i]);
            if (!inserted)
                ins_it->second = accumulate_grad(ins_it->second, grads[i]);
        }
    }

    // Write accumulated gradients to leaf tensors, accumulating across calls.
    // claimed_grad_storages enforces the one-leaf-one-gradient-buffer
    // invariant documented on privatise_leaf_grad.
    std::unordered_set<const Storage*> claimed_grad_storages;
    for (Tensor& t : topo) {
        if (!t.is_leaf() || !t.requires_grad()) continue;
        auto it = grad_map.find(t.impl().get());
        if (it == grad_map.end()) continue;
        t.set_grad(privatise_leaf_grad(
            t.has_grad() ? accumulate_grad(t.grad(), it->second)
                         : contiguous(it->second),
            claimed_grad_storages));
        t.impl()->run_post_accumulate_grad_hooks();
    }

    // Free the graph unless the caller wants to reuse it.
    if (!retain_graph) {
        for (Tensor& t : topo) {
            if (!t.is_leaf())
                t.clear_grad_fn();
        }
    }
}

}  // namespace orthon
