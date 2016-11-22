/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/training_ops.h"
#include <algorithm>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/bounds_check.h"

namespace tensorflow {

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

namespace {
template <class T>
inline T sgn(const T x) {
  T zero(0);
  T one(1);
  return (x == zero ? zero : (x < zero ? -one : one));
}
}

namespace functor {
template <typename T>
struct ApplyGradientDescent<CPUDevice, T> {
  void operator()(const CPUDevice& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstFlat grad) {
    var.device(d) -= grad * lr();
  }
};

template <typename T>
struct ApplyAdadelta<CPUDevice, T> {
  void operator()(const CPUDevice& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::Flat accum,
                  typename TTypes<T>::Flat accum_update,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstScalar rho,
                  typename TTypes<T>::ConstScalar epsilon,
                  typename TTypes<T>::ConstFlat grad) {
    accum.device(d) =
        accum * rho() + grad.square() * (static_cast<T>(1) - rho());
    const auto update =
        (accum_update + epsilon()).sqrt() * (accum + epsilon()).rsqrt() * grad;
    accum_update.device(d) =
        accum_update * rho() + update.square() * (static_cast<T>(1) - rho());
    var.device(d) -= update * lr();
  }
};

template <typename T>
struct ApplyProximalGradientDescent<CPUDevice, T> {
  void operator()(const CPUDevice& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstScalar l1,
                  typename TTypes<T>::ConstScalar l2,
                  typename TTypes<T>::ConstFlat grad) {
    // Note that here is Fobos update, for details please refer:
    // http://papers.nips.cc/paper/3793-efficient-learning-using-forward-backward-splitting.pdf
    // TODO(xbing): merge the logic for ProximalGradientDescent and
    // ProximalAdagrad.
    auto prox_var = var;
    // compute v = w - lr * grad.
    prox_var.device(d) -= grad * lr();
    if (l1() > 0) {
      var.device(d) = prox_var.abs() - var.constant(lr() * l1());
      // compute sign(v) * max(|v| - lr * l1, 0)
      var.device(d) = prox_var.sign() * var.cwiseMax(T(0.0));
    } else {
      var.device(d) = prox_var;
    }
    if (l2() > 0) {
      // compute v / (1.0 + l2 * lr)
      var.device(d) = var / (var.constant(1.0) + var.constant(l2() * lr()));
    }
  }
};

template <typename T>
struct ApplyAdagradDA<CPUDevice, T> {
  void operator()(const CPUDevice& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::Flat gradient_accum,
                  typename TTypes<T>::Flat gradient_squared_accum,
                  typename TTypes<T>::ConstScalar lr, int64 global_step,
                  typename TTypes<T>::ConstScalar l1,
                  typename TTypes<T>::ConstScalar l2,
                  typename TTypes<T>::ConstFlat grad) {
    // Accumulate gradient, and gradient_squared
    gradient_accum.device(d) += grad;
    gradient_squared_accum.device(d) += grad.square();

    // AdagradDA update:
    // Let g to be gradient accumulator, gg to be gradient squared accumulator,
    // T be the global step, lr is the learning rate, and k the initial
    // gradient squared accumulator value.
    // w = \dfrac{sign(-g)*lr*|g - l1*T|_{+}}{l2*T*lr + \sqrt{k+gg})}
    if (l1() > 0) {
      var.device(d) =
          lr() * var.constant(-1.0) * gradient_accum.sign() *
          (gradient_accum.abs() -
           var.constant(static_cast<float>(global_step)) * var.constant(l1()))
              .cwiseMax(T(0.0)) /
          (var.constant(l2()) *
               var.constant(static_cast<float>(global_step) * lr()) +
           gradient_squared_accum.sqrt());
    } else {
      var.device(d) =
          lr() * gradient_accum * var.constant(-1.0) /
          (var.constant(l2()) *
               var.constant(static_cast<float>(global_step) * lr()) +
           gradient_squared_accum.sqrt());
    }
  }
};

template <typename T>
struct ApplyAdagrad<CPUDevice, T> {
  void operator()(const CPUDevice& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::Flat accum,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstFlat grad) {
    accum.device(d) += grad.square();
    var.device(d) -= grad * lr() * accum.rsqrt();
  }
};

template <typename T>
struct ApplyProximalAdagrad<CPUDevice, T> {
  void operator()(const CPUDevice& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::Flat accum,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstScalar l1,
                  typename TTypes<T>::ConstScalar l2,
                  typename TTypes<T>::ConstFlat grad) {
    // Fobos update per paper with Adagrad learning rate.
    accum.device(d) += grad.square();
    // Adagrad learning rate.
    auto learning_rate = accum.constant(lr()) * accum.rsqrt();
    auto prox_var = var;
    // compute v = w - lr * grad.
    prox_var.device(d) -= grad * learning_rate;
    if (l1() > 0) {
      var.device(d) = prox_var.abs() - learning_rate * prox_var.constant(l1());
      // compute sign(v) * max(|v| - lr * l1, 0)
      var.device(d) = prox_var.sign() * var.cwiseMax(T(0.0));
    } else {
      var.device(d) = prox_var;
    }
    if (l2() > 0) {
      var.device(d) =
          var / (var.constant(1.0) + var.constant(l2()) * learning_rate);
    }
  }
};

template <typename T>
struct ApplyFtrl<CPUDevice, T> {
  void operator()(const CPUDevice& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::Flat accum,
                  typename TTypes<T>::Flat linear,
                  typename TTypes<T>::ConstFlat grad,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstScalar l1,
                  typename TTypes<T>::ConstScalar l2,
                  typename TTypes<T>::ConstScalar lr_power) {
    auto new_accum = accum + grad.square();
    // special case for which lr_power=-0.5.
    if (lr_power() == static_cast<T>(-0.5)) {
      linear.device(d) += grad - (new_accum.sqrt() - accum.sqrt()) / lr() * var;
    } else {
      linear.device(d) +=
          grad -
          (new_accum.pow(-lr_power()) - accum.pow(-lr_power())) / lr() * var;
    }
    auto x = (linear.constant(l1()) * linear.sign() - linear);
    if (lr_power() == static_cast<T>(-0.5)) {
      auto y = new_accum.sqrt() / new_accum.constant(lr()) +
               linear.constant(static_cast<T>(2) * l2());
      var.device(d) = x / y;
    } else {
      auto y = new_accum.pow(-lr_power()) / new_accum.constant(lr()) +
               linear.constant(static_cast<T>(2) * l2());
      var.device(d) = x / y;
    }
    var.device(d) = (linear.abs() > linear.constant(l1()))
                        .select(var, var.constant(static_cast<T>(0)));
    accum.device(d) += grad.square();
  }
};

template <typename T>
struct ApplyMomentum<CPUDevice, T> {
  void operator()(const CPUDevice& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::Flat accum,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstFlat grad,
                  typename TTypes<T>::ConstScalar momentum, bool use_nesterov) {
    accum.device(d) = accum * momentum() + grad;
    if (use_nesterov) {
      var.device(d) -= grad * lr() + accum * momentum() * lr();
    } else {
      var.device(d) -= accum * lr();
    }
  }
};

template <typename T>
struct ApplyAdam<CPUDevice, T> {
  void operator()(const CPUDevice& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::Flat m, typename TTypes<T>::Flat v,
                  typename TTypes<T>::ConstScalar beta1_power,
                  typename TTypes<T>::ConstScalar beta2_power,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstScalar beta1,
                  typename TTypes<T>::ConstScalar beta2,
                  typename TTypes<T>::ConstScalar epsilon,
                  typename TTypes<T>::ConstFlat grad) {
    const T alpha = lr() * Eigen::numext::sqrt(T(1) - beta2_power()) /
                    (T(1) - beta1_power());
    m.device(d) += (grad - m) * (T(1) - beta1());
    v.device(d) += (grad.square() - v) * (T(1) - beta2());
    var.device(d) -= (m * alpha) / (v.sqrt() + epsilon());
  }
};

template <typename T>
struct ApplyRMSProp<CPUDevice, T> {
  void operator()(const CPUDevice& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::Flat ms, typename TTypes<T>::Flat mom,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstScalar rho,
                  typename TTypes<T>::ConstScalar momentum,
                  typename TTypes<T>::ConstScalar epsilon,
                  typename TTypes<T>::ConstFlat grad) {
    ms.device(d) += (grad.square() - ms) * (static_cast<T>(1) - rho());
    mom.device(d) =
        mom * momentum() + (grad * lr()) / ((ms + epsilon()).sqrt());
    var.device(d) -= mom;
  }
};

template <typename T>
struct ApplyCenteredRMSProp<CPUDevice, T> {
  void operator()(const CPUDevice& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::Flat mg, typename TTypes<T>::Flat ms,
                  typename TTypes<T>::Flat mom,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstScalar rho,
                  typename TTypes<T>::ConstScalar momentum,
                  typename TTypes<T>::ConstScalar epsilon,
                  typename TTypes<T>::ConstFlat grad) {
    ms.device(d) += (grad.square() - ms) * (static_cast<T>(1) - rho());
    mg.device(d) += (grad - mg) * (static_cast<T>(1) - rho());
    auto denom = (ms - mg.square()) + epsilon();
    mom.device(d) = mom * momentum() + (grad * lr()) / denom.sqrt();
    var.device(d) -= mom;
  }
};

}  // namespace functor

// MaybeLockMutexesInOrder is a helper function to acquire mutexes in address
// order to mitigate deadlock.  Returns a vector of acquired mutexes.
// Safe to pass duplicates - will only lock each distinct mutex once.
// If do_lock is false, returns immediately.
std::vector<mutex_lock> MaybeLockMutexesInOrder(
    OpKernelContext* ctx, bool do_lock, const std::vector<int>& input_ids) {
  std::vector<mutex_lock> locks;
  if (!do_lock) {
    return locks;
  }
  std::vector<mutex*> mutexes;
  std::vector<int> acquire_order;
  for (auto input : input_ids) {
    auto* mutex = ctx->input_ref_mutex(input);
    // Only lock each mutex once if duplicates exist (n^2 but n is 2 or 3).
    if (std::find(mutexes.begin(), mutexes.end(), mutex) == mutexes.end()) {
      acquire_order.push_back(input);
      mutexes.push_back(mutex);
    }
  }
  std::sort(acquire_order.begin(), acquire_order.end(),
            [&mutexes](int a, int b) { return mutexes[a] < mutexes[b]; });

  for (auto input : acquire_order) {
    locks.emplace_back(*ctx->input_ref_mutex(input));
  }
  return locks;
}

template <typename Device, typename T>
class ApplyGradientDescentOp : public OpKernel {
 public:
  explicit ApplyGradientDescentOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override {
    auto locks = MaybeLockMutexesInOrder(ctx, use_exclusive_lock_, {0});
    Tensor var = ctx->mutable_input(0, use_exclusive_lock_);

    OP_REQUIRES(
        ctx, var.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", def().input(0)));
    const Tensor& alpha = ctx->input(1);
    OP_REQUIRES(ctx, IsLegacyScalar(alpha.shape()),
                errors::InvalidArgument("alpha is not a scalar: ",
                                        alpha.shape().DebugString()));
    const Tensor& delta = ctx->input(2);
    OP_REQUIRES(
        ctx, var.shape().IsSameSize(delta.shape()),
        errors::InvalidArgument("var and delta do not have the same shape",
                                var.shape().DebugString(), " ",
                                delta.shape().DebugString()));

    const Device& device = ctx->template eigen_device<Device>();
    functor::ApplyGradientDescent<Device, T>()(
        device, var.flat<T>(), alpha.scalar<T>(), delta.flat<T>());

    ctx->forward_ref_input_to_ref_output(0, 0);
  }

 private:
  bool use_exclusive_lock_;
};

#define REGISTER_KERNELS(D, T)                                                \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("ApplyGradientDescent").Device(DEVICE_##D).TypeConstraint<T>("T"), \
      ApplyGradientDescentOp<D##Device, T>);
#define REGISTER_CPU_KERNELS(T) REGISTER_KERNELS(CPU, T);

TF_CALL_half(REGISTER_CPU_KERNELS);
TF_CALL_float(REGISTER_CPU_KERNELS);
TF_CALL_double(REGISTER_CPU_KERNELS);

#if GOOGLE_CUDA
// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC(T)                             \
  template <>                                           \
  void ApplyGradientDescent<GPUDevice, T>::operator()(  \
      const GPUDevice& d, typename TTypes<T>::Flat var, \
      typename TTypes<T>::ConstScalar alpha,            \
      typename TTypes<T>::ConstFlat delta);             \
  extern template struct ApplyGradientDescent<GPUDevice, T>;
DECLARE_GPU_SPEC(Eigen::half);
DECLARE_GPU_SPEC(float);
DECLARE_GPU_SPEC(double);
#undef DECLARE_GPU_SPEC
}  // namespace functor

REGISTER_KERNELS(GPU, Eigen::half);
REGISTER_KERNELS(GPU, float);
REGISTER_KERNELS(GPU, double);
#endif
#undef REGISTER_CPU_KERNELS
#undef REGISTER_KERNELS

template <typename Device, typename T>
class ApplyAdadeltaOp : public OpKernel {
 public:
  explicit ApplyAdadeltaOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override {
    if (use_exclusive_lock_) {
      mutex_lock l1(*ctx->input_ref_mutex(0));
      // Don't try to acquire a lock on the second ref as they share the same
      // mutex.
      //
      // mutex_lock l2(*ctx->input_ref_mutex(1));
      DoValidate(ctx);
      if (!ctx->status().ok()) return;
      DoCompute(ctx);
    } else {
      DoValidate(ctx);
      if (!ctx->status().ok()) return;
      DoCompute(ctx);
    }
    ctx->forward_ref_input_to_ref_output(0, 0);
  }

 private:
  bool use_exclusive_lock_;

  void DoValidate(OpKernelContext* ctx) {
    Tensor var = ctx->mutable_input(0, use_exclusive_lock_);
    Tensor accum = ctx->mutable_input(1, use_exclusive_lock_);
    Tensor accum_update = ctx->mutable_input(2, use_exclusive_lock_);

    OP_REQUIRES(
        ctx, var.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", def().input(0)));
    OP_REQUIRES(
        ctx, accum.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", def().input(1)));
    OP_REQUIRES(
        ctx, accum_update.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", def().input(2)));

    const Tensor& lr = ctx->input(3);
    const Tensor& rho = ctx->input(4);
    const Tensor& epsilon = ctx->input(5);
    const Tensor& grad = ctx->input(6);

    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(lr.shape()),
                errors::InvalidArgument("lr is not a scalar: ",
                                        lr.shape().DebugString()));

    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(rho.shape()),
                errors::InvalidArgument("rho is not a scalar: ",
                                        rho.shape().DebugString()));

    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(epsilon.shape()),
                errors::InvalidArgument("epsilon is not a scalar: ",
                                        epsilon.shape().DebugString()));

    OP_REQUIRES(
        ctx, var.shape().IsSameSize(accum.shape()),
        errors::InvalidArgument("var and accum do not have the same shape",
                                var.shape().DebugString(), " ",
                                accum.shape().DebugString()));
    OP_REQUIRES(
        ctx, var.shape().IsSameSize(grad.shape()),
        errors::InvalidArgument("var and grad do not have the same shape",
                                var.shape().DebugString(), " ",
                                grad.shape().DebugString()));
  }

  void DoCompute(OpKernelContext* ctx) {
    const Device& device = ctx->template eigen_device<Device>();
    Tensor var = ctx->mutable_input(0, use_exclusive_lock_);
    Tensor accum = ctx->mutable_input(1, use_exclusive_lock_);
    Tensor accum_update = ctx->mutable_input(2, use_exclusive_lock_);

    const Tensor& lr = ctx->input(3);
    const Tensor& rho = ctx->input(4);
    const Tensor& epsilon = ctx->input(5);
    const Tensor& grad = ctx->input(6);

    functor::ApplyAdadelta<Device, T>()(
        device, var.flat<T>(), accum.flat<T>(), accum_update.flat<T>(),
        lr.scalar<T>(), rho.scalar<T>(), epsilon.scalar<T>(), grad.flat<T>());
  }
};

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

#define REGISTER_KERNELS(D, T)                                         \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("ApplyAdadelta").Device(DEVICE_##D).TypeConstraint<T>("T"), \
      ApplyAdadeltaOp<D##Device, T>);
#define REGISTER_CPU_KERNELS(T) REGISTER_KERNELS(CPU, T);

TF_CALL_half(REGISTER_CPU_KERNELS);
TF_CALL_float(REGISTER_CPU_KERNELS);
TF_CALL_double(REGISTER_CPU_KERNELS);

#if GOOGLE_CUDA
// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC(T)                                                    \
  template <>                                                                  \
  void ApplyAdadelta<GPUDevice, T>::operator()(                                \
      const GPUDevice& d, typename TTypes<T>::Flat var,                        \
      typename TTypes<T>::Flat accum, typename TTypes<T>::Flat accum_update,   \
      typename TTypes<T>::ConstScalar lr, typename TTypes<T>::ConstScalar rho, \
      typename TTypes<T>::ConstScalar epsilon,                                 \
      typename TTypes<T>::ConstFlat grad);                                     \
  extern template struct ApplyAdadelta<GPUDevice, T>;
DECLARE_GPU_SPEC(Eigen::half);
DECLARE_GPU_SPEC(float);
DECLARE_GPU_SPEC(double);
#undef DECLARE_GPU_SPEC
}  // namespace functor

REGISTER_KERNELS(GPU, Eigen::half);
REGISTER_KERNELS(GPU, float);
REGISTER_KERNELS(GPU, double);
#endif
#undef REGISTER_CPU_KERNELS
#undef REGISTER_KERNELS

// Note, this op works on cpu only.
template <typename T, typename Tindex>
class SparseApplyAdadeltaOp : public OpKernel {
 public:
  explicit SparseApplyAdadeltaOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override NO_THREAD_SAFETY_ANALYSIS {
    mutex* mu_var = ctx->input_ref_mutex(0);
    // mu_accum is actually the same mutex as mu_var since currently we use a
    // global mutex.
    //
    // mutex* mu_accum = ctx->input_ref_mutex(1);
    if (use_exclusive_lock_) {
      mu_var->lock();
    }
    Tensor var = ctx->mutable_input(0, use_exclusive_lock_);
    Tensor accum_grad = ctx->mutable_input(1, use_exclusive_lock_);
    Tensor accum_update = ctx->mutable_input(2, use_exclusive_lock_);
    OP_REQUIRES(
        ctx, var.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", def().input(0)));
    OP_REQUIRES(
        ctx, accum_grad.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", def().input(1)));
    OP_REQUIRES(
        ctx, accum_update.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", def().input(2)));
    OP_REQUIRES(
        ctx, var.shape().IsSameSize(accum_grad.shape()),
        errors::InvalidArgument("var and accum_grad do not have the same shape",
                                var.shape().DebugString(), " ",
                                accum_grad.shape().DebugString()));
    OP_REQUIRES(ctx, var.shape().IsSameSize(accum_update.shape()),
                errors::InvalidArgument(
                    "var and accum_update do not have the same shape",
                    var.shape().DebugString(), " ",
                    accum_update.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVectorOrHigher(var.shape()),
                errors::InvalidArgument("var must be at least 1 dimensional"));

    const Tensor& lr = ctx->input(3);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(lr.shape()),
                errors::InvalidArgument("lr is not a scalar: ",
                                        lr.shape().DebugString()));
    const Tensor& rho = ctx->input(4);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(rho.shape()),
                errors::InvalidArgument("rho is not a scalar: ",
                                        rho.shape().DebugString()));
    const Tensor& epsilon = ctx->input(5);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(epsilon.shape()),
                errors::InvalidArgument("epsilon is not a scalar: ",
                                        epsilon.shape().DebugString()));
    const Tensor& grad = ctx->input(6);
    const Tensor& indices = ctx->input(7);
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(indices.shape()),
                errors::InvalidArgument("indices must be one-dimensional"));

    for (int d = 1; d < var.dims(); d++) {
      OP_REQUIRES(ctx, var.dim_size(d) == grad.dim_size(d),
                  errors::InvalidArgument(strings::StrCat(
                      "var and grad must match in dimension ", d)));
    }
    const Tindex N = indices.dim_size(0);
    OP_REQUIRES(
        ctx, grad.dim_size(0) == N,
        errors::InvalidArgument(
            "grad must be the same size as indices in the first dimension."));

    if (N > 0) {
      const Tindex first_dim_size = var.dim_size(0);
      // Validate all the indices are in range
      auto indices_vec = indices.vec<Tindex>();
      for (Tindex i = 0; i < N; i++) {
        const Tindex index = indices_vec(i);
        OP_REQUIRES(ctx, index >= 0 && index < first_dim_size,
                    errors::InvalidArgument(
                        strings::StrCat("Index ", index, " at offset ", i,
                                        " in indices is out of range")));
      }

      auto var_flat = var.flat_outer_dims<T>();
      auto accum_grad_flat = accum_grad.flat_outer_dims<T>();
      auto accum_update_flat = accum_update.flat_outer_dims<T>();
      auto grad_flat = grad.flat_outer_dims<T>();
      const T lr_scalar = lr.scalar<T>()();
      const T rho_scalar = rho.scalar<T>()();
      const T epsilon_scalar = epsilon.scalar<T>()();

      for (Tindex i = 0; i < N; i++) {
        const Tindex index = indices_vec(i);
        auto accum_ = accum_grad_flat.template chip<0>(index);
        auto accum_update_ = accum_update_flat.template chip<0>(index);
        auto grad_ = grad_flat.template chip<0>(i);

        accum_ = accum_ * accum_.constant(rho_scalar) +
                 grad_.square() * grad_.constant(T(1) - rho_scalar);
        const auto update =
            (accum_update_ + accum_update_.constant(epsilon_scalar)).sqrt() *
            (accum_ + accum_.constant(epsilon_scalar)).rsqrt() * grad_;
        accum_update_ =
            accum_update_ * accum_update_.constant(rho_scalar) +
            update.square() * update.constant(static_cast<T>(1) - rho_scalar);
        auto v = var_flat.template chip<0>(index);
        v -= update * update.constant(lr_scalar);
      }
    }
    if (use_exclusive_lock_) {
      mu_var->unlock();
    }

    ctx->forward_ref_input_to_ref_output(0, 0);
  }

 private:
  bool use_exclusive_lock_;
};

#define REGISTER_KERNELS(T, Tindices)                                \
  REGISTER_KERNEL_BUILDER(Name("SparseApplyAdadelta")                \
                              .Device(DEVICE_CPU)                    \
                              .TypeConstraint<T>("T")                \
                              .TypeConstraint<Tindices>("Tindices"), \
                          SparseApplyAdadeltaOp<T, Tindices>);
#define REGISTER_CPU_KERNELS(T) \
  REGISTER_KERNELS(T, int32);   \
  REGISTER_KERNELS(T, int64);

TF_CALL_half(REGISTER_CPU_KERNELS);
TF_CALL_float(REGISTER_CPU_KERNELS);
TF_CALL_double(REGISTER_CPU_KERNELS);

#undef REGISTER_CPU_KERNELS
#undef REGISTER_KERNELS

// Note, this op works on cpu only.
template <typename Device, typename T>
class ApplyProximalGradientDescentOp : public OpKernel {
 public:
  explicit ApplyProximalGradientDescentOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override {
    auto locks = MaybeLockMutexesInOrder(ctx, use_exclusive_lock_, {0});
    Tensor var = ctx->mutable_input(0, use_exclusive_lock_);

    OP_REQUIRES(
        ctx, var.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", def().input(0)));
    const Tensor& alpha = ctx->input(1);
    OP_REQUIRES(ctx, IsLegacyScalar(alpha.shape()),
                errors::InvalidArgument("alpha is not a scalar: ",
                                        alpha.shape().DebugString()));
    const Tensor& l1 = ctx->input(2);
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsScalar(l1.shape()),
        errors::InvalidArgument("l1 regularization strength is not a scalar: ",
                                l1.shape().DebugString()));
    const Tensor& l2 = ctx->input(3);
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsScalar(l2.shape()),
        errors::InvalidArgument("l2 regularization strength is not a scalar: ",
                                l2.shape().DebugString()));

    const Tensor& delta = ctx->input(4);
    OP_REQUIRES(
        ctx, var.shape().IsSameSize(delta.shape()),
        errors::InvalidArgument("var and delta do not have the same shape",
                                var.shape().DebugString(), " ",
                                delta.shape().DebugString()));

    const Device& device = ctx->template eigen_device<Device>();
    functor::ApplyProximalGradientDescent<Device, T>()(
        device, var.flat<T>(), alpha.scalar<T>(), l1.scalar<T>(),
        l2.scalar<T>(), delta.flat<T>());

    ctx->forward_ref_input_to_ref_output(0, 0);
  }

 private:
  bool use_exclusive_lock_;
};

#define REGISTER_KERNELS(D, T)                                 \
  REGISTER_KERNEL_BUILDER(Name("ApplyProximalGradientDescent") \
                              .Device(DEVICE_##D)              \
                              .TypeConstraint<T>("T"),         \
                          ApplyProximalGradientDescentOp<D##Device, T>);

REGISTER_KERNELS(CPU, float);
REGISTER_KERNELS(CPU, double);
#undef REGISTER_KERNELS

// Note, this op works on cpu only.
template <typename T, typename Tindex>
class SparseApplyProximalGradientDescentOp : public OpKernel {
 public:
  explicit SparseApplyProximalGradientDescentOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override NO_THREAD_SAFETY_ANALYSIS {
    auto locks = MaybeLockMutexesInOrder(ctx, use_exclusive_lock_, {0, 1});
    Tensor var = ctx->mutable_input(0, use_exclusive_lock_);
    OP_REQUIRES(ctx, TensorShapeUtils::IsVectorOrHigher(var.shape()),
                errors::InvalidArgument("var must be at least 1 dimensional"));

    const Tensor& lr = ctx->input(1);
    OP_REQUIRES(ctx, IsLegacyScalar(lr.shape()),
                errors::InvalidArgument("lr is not a scalar: ",
                                        lr.shape().DebugString()));
    const Tensor& l1 = ctx->input(2);
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsScalar(l1.shape()),
        errors::InvalidArgument("l1 regularization strength is not a scalar: ",
                                l1.shape().DebugString()));
    const Tensor& l2 = ctx->input(3);
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsScalar(l2.shape()),
        errors::InvalidArgument("l2 regularization strength is not a scalar: ",
                                l2.shape().DebugString()));

    const Tensor& grad = ctx->input(4);
    const Tensor& indices = ctx->input(5);
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(indices.shape()),
                errors::InvalidArgument("indices must be one-dimensional"));

    int64 inner_dim = 1;
    for (int d = 1; d < var.dims(); d++) {
      OP_REQUIRES(ctx, var.dim_size(d) == grad.dim_size(d),
                  errors::InvalidArgument(strings::StrCat(
                      "var and grad must match in dimension ", d)));
      inner_dim *= grad.dim_size(d);
    }
    const Tindex N = indices.dim_size(0);
    OP_REQUIRES(
        ctx, grad.dim_size(0) == N,
        errors::InvalidArgument(
            "grad must be the same size as indices in the first dimension."));
    OP_REQUIRES(ctx, inner_dim > 0,
                errors::InvalidArgument(
                    "Inner dimension should be greater than zero."));

    if (N > 0) {
      if (inner_dim > 1) {
        const Tindex first_dim_size = var.dim_size(0);
        auto indices_vec = indices.vec<Tindex>();
        auto var_flat = var.flat_outer_dims<T>();
        auto grad_flat = grad.flat_outer_dims<T>();
        T lr_scalar = lr.scalar<T>()();
        T l1_scalar = l1.scalar<T>()();
        T l2_scalar = l2.scalar<T>()();

        // TODO(xbing): extract the common logic for the Fobos update.
        for (Tindex i = 0; i < N; i++) {
          const Tindex index = internal::SubtleMustCopy(indices_vec(i));
          OP_REQUIRES(ctx, FastBoundsCheck(index, first_dim_size),
                      errors::InvalidArgument(
                          strings::StrCat("Index ", index, " at offset ", i,
                                          " in indices is out of range")));
          auto g = grad_flat.template chip<0>(i);
          auto v = var_flat.template chip<0>(index);
          // compute learning_rate for current step.
          auto learning_rate = v.constant(lr_scalar);
          auto prox_v = v;
          // v = w - g * learning_rate.
          prox_v -= g * learning_rate;
          if (l1_scalar > 0) {
            v = prox_v.abs() - learning_rate * prox_v.constant(l1_scalar);
            // compute sign(v) * max(|v|, 0)
            v = prox_v.sign() * v.cwiseMax(static_cast<T>(0.0));
          } else {
            v = prox_v;
          }
          if (l2_scalar > 0) {
            v /= (v.constant(1.0) + v.constant(l2_scalar) * learning_rate);
          }
        }
      } else {
        auto indices_vec = indices.vec<Tindex>();
        auto var_flat = var.flat<T>();
        auto grad_flat = grad.flat<T>();
        T lr_scalar = lr.scalar<T>()();
        T l1_scalar = l1.scalar<T>()();
        T l2_scalar = l2.scalar<T>()();
        const Tindex first_dim_size = var_flat.size();

        for (Tindex i = 0; i < N; i++) {
          const Tindex index = internal::SubtleMustCopy(indices_vec(i));
          OP_REQUIRES(ctx, FastBoundsCheck(index, first_dim_size),
                      errors::InvalidArgument(
                          strings::StrCat("Index ", index, " at offset ", i,
                                          " in indices is out of range")));
          const T& g = grad_flat(i);
          auto learning_rate = lr_scalar;
          auto prox_v = var_flat(index);
          prox_v -= learning_rate * g;
          if (l1_scalar > 0) {
            var_flat(index) = std::abs(prox_v) - learning_rate * l1_scalar;
            var_flat(index) =
                sgn(prox_v) * std::max(var_flat(index), static_cast<T>(0.0));
          } else {
            var_flat(index) = prox_v;
          }
          if (l2_scalar > 0) {
            var_flat(index) /= (1.0 + l2_scalar * learning_rate);
          }
        }
      }
    }

    ctx->forward_ref_input_to_ref_output(0, 0);
  }

 private:
  bool use_exclusive_lock_;
};

#define REGISTER_KERNELS(T, Tindices)                                \
  REGISTER_KERNEL_BUILDER(Name("SparseApplyProximalGradientDescent") \
                              .Device(DEVICE_CPU)                    \
                              .TypeConstraint<T>("T")                \
                              .TypeConstraint<Tindices>("Tindices"), \
                          SparseApplyProximalGradientDescentOp<T, Tindices>);

REGISTER_KERNELS(float, int32);
REGISTER_KERNELS(float, int64);
REGISTER_KERNELS(double, int32);
REGISTER_KERNELS(double, int64);
#undef REGISTER_KERNELS

template <typename Device, typename T>
class ApplyAdagradOp : public OpKernel {
 public:
  explicit ApplyAdagradOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override {
    auto locks = MaybeLockMutexesInOrder(ctx, use_exclusive_lock_, {0, 1});
    Tensor var = ctx->mutable_input(0, use_exclusive_lock_);
    Tensor accum = ctx->mutable_input(1, use_exclusive_lock_);
    OP_REQUIRES(
        ctx, var.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", def().input(0)));
    OP_REQUIRES(
        ctx, accum.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", def().input(1)));
    const Tensor& lr = ctx->input(2);
    OP_REQUIRES(ctx, IsLegacyScalar(lr.shape()),
                errors::InvalidArgument("lr is not a scalar: ",
                                        lr.shape().DebugString()));
    const Tensor& grad = ctx->input(3);
    OP_REQUIRES(
        ctx, var.shape().IsSameSize(accum.shape()),
        errors::InvalidArgument("var and accum do not have the same shape",
                                var.shape().DebugString(), " ",
                                accum.shape().DebugString()));
    OP_REQUIRES(
        ctx, var.shape().IsSameSize(grad.shape()),
        errors::InvalidArgument("var and grad do not have the same shape",
                                var.shape().DebugString(), " ",
                                grad.shape().DebugString()));

    const Device& device = ctx->template eigen_device<Device>();
    functor::ApplyAdagrad<Device, T>()(device, var.flat<T>(), accum.flat<T>(),
                                       lr.scalar<T>(), grad.flat<T>());

    ctx->forward_ref_input_to_ref_output(0, 0);
  }

 private:
  bool use_exclusive_lock_;
};

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

#define REGISTER_KERNELS(D, T)                                        \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("ApplyAdagrad").Device(DEVICE_##D).TypeConstraint<T>("T"), \
      ApplyAdagradOp<D##Device, T>);
#define REGISTER_CPU_KERNELS(T) REGISTER_KERNELS(CPU, T);

TF_CALL_half(REGISTER_CPU_KERNELS);
TF_CALL_float(REGISTER_CPU_KERNELS);
TF_CALL_double(REGISTER_CPU_KERNELS);

#if GOOGLE_CUDA
// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC(T)                                               \
  template <>                                                             \
  void ApplyAdagrad<GPUDevice, T>::operator()(                            \
      const GPUDevice& d, typename TTypes<T>::Flat var,                   \
      typename TTypes<T>::Flat accum, typename TTypes<T>::ConstScalar lr, \
      typename TTypes<T>::ConstFlat grad);                                \
  extern template struct ApplyAdagrad<GPUDevice, T>;
DECLARE_GPU_SPEC(Eigen::half);
DECLARE_GPU_SPEC(float);
DECLARE_GPU_SPEC(double);
#undef DECLARE_GPU_SPEC
}  // namespace functor

REGISTER_KERNELS(GPU, Eigen::half);
REGISTER_KERNELS(GPU, float);
REGISTER_KERNELS(GPU, double);
#endif
#undef REGISTER_CPU_KERNELS
#undef REGISTER_KERNELS

template <typename Device, typename T>
class ApplyProximalAdagradOp : public OpKernel {
 public:
  explicit ApplyProximalAdagradOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override {
    auto locks = MaybeLockMutexesInOrder(ctx, use_exclusive_lock_, {0, 1});
    Tensor var = ctx->mutable_input(0, use_exclusive_lock_);
    Tensor accum = ctx->mutable_input(1, use_exclusive_lock_);
    OP_REQUIRES(
        ctx, var.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", def().input(0)));
    OP_REQUIRES(
        ctx, accum.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", def().input(1)));
    OP_REQUIRES(
        ctx, var.shape().IsSameSize(accum.shape()),
        errors::InvalidArgument("var and accum do not have the same shape",
                                var.shape().DebugString(), " ",
                                accum.shape().DebugString()));
    const Tensor& lr = ctx->input(2);
    OP_REQUIRES(ctx, IsLegacyScalar(lr.shape()),
                errors::InvalidArgument("lr is not a scalar: ",
                                        lr.shape().DebugString()));
    const Tensor& l1 = ctx->input(3);
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsScalar(l1.shape()),
        errors::InvalidArgument("l1 regularization strength is not a scalar: ",
                                l1.shape().DebugString()));
    const Tensor& l2 = ctx->input(4);
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsScalar(l2.shape()),
        errors::InvalidArgument("l2 regularization strength is not a scalar: ",
                                l2.shape().DebugString()));

    const Tensor& grad = ctx->input(5);
    OP_REQUIRES(
        ctx, var.shape().IsSameSize(grad.shape()),
        errors::InvalidArgument("var and grad do not have the same shape",
                                var.shape().DebugString(), " ",
                                grad.shape().DebugString()));

    const Device& device = ctx->template eigen_device<Device>();
    functor::ApplyProximalAdagrad<Device, T>()(
        device, var.flat<T>(), accum.flat<T>(), lr.scalar<T>(), l1.scalar<T>(),
        l2.scalar<T>(), grad.flat<T>());

    ctx->forward_ref_input_to_ref_output(0, 0);
  }

 private:
  bool use_exclusive_lock_;
};

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

#define REGISTER_KERNELS(D, T)                                                \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("ApplyProximalAdagrad").Device(DEVICE_##D).TypeConstraint<T>("T"), \
      ApplyProximalAdagradOp<D##Device, T>);

REGISTER_KERNELS(CPU, float);
REGISTER_KERNELS(CPU, double);
#undef REGISTER_KERNELS

namespace {

template <typename T>
inline T FtrlCompute(const T& accum, const T& linear, const T& lr, const T& l1,
                     const T& l2, const T& lr_power) {
  T quadratic;
  if (lr_power == static_cast<T>(-0.5)) {
    quadratic = Eigen::numext::sqrt(accum) / lr + static_cast<T>(2) * l2;
  } else {
    quadratic =
        Eigen::numext::pow(accum, -lr_power) / lr + static_cast<T>(2) * l2;
  }
  if (Eigen::numext::abs(linear) > l1) {
    return (l1 * sgn(linear) - linear) / quadratic;
  } else {
    return static_cast<T>(0.0);
  }
}
}  // namespace

// Note, this op works on cpu only.
template <typename T, typename Tindex>
class SparseApplyAdagradOp : public OpKernel {
 public:
  explicit SparseApplyAdagradOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override NO_THREAD_SAFETY_ANALYSIS {
    auto locks = MaybeLockMutexesInOrder(ctx, use_exclusive_lock_, {0, 1});
    Tensor var = ctx->mutable_input(0, use_exclusive_lock_);
    Tensor accum = ctx->mutable_input(1, use_exclusive_lock_);
    OP_REQUIRES(
        ctx, var.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", def().input(0)));
    OP_REQUIRES(
        ctx, accum.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", def().input(1)));
    OP_REQUIRES(
        ctx, var.shape().IsSameSize(accum.shape()),
        errors::InvalidArgument("var and accum do not have the same shape",
                                var.shape().DebugString(), " ",
                                accum.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVectorOrHigher(var.shape()),
                errors::InvalidArgument("var must be at least 1 dimensional"));

    const Tensor& lr = ctx->input(2);
    OP_REQUIRES(ctx, IsLegacyScalar(lr.shape()),
                errors::InvalidArgument("lr is not a scalar: ",
                                        lr.shape().DebugString()));
    const Tensor& grad = ctx->input(3);
    const Tensor& indices = ctx->input(4);
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(indices.shape()),
                errors::InvalidArgument("indices must be one-dimensional"));

    int64 inner_dim = 1;
    for (int d = 1; d < var.dims(); d++) {
      OP_REQUIRES(ctx, var.dim_size(d) == grad.dim_size(d),
                  errors::InvalidArgument(strings::StrCat(
                      "var and grad must match in dimension ", d)));
      inner_dim *= grad.dim_size(d);
    }
    const Tindex N = indices.dim_size(0);
    OP_REQUIRES(
        ctx, grad.dim_size(0) == N,
        errors::InvalidArgument(
            "grad must be the same size as indices in the first dimension."));

    OP_REQUIRES(ctx, inner_dim > 0,
                errors::InvalidArgument(
                    "Inner dimension should be greater than zero."));

    if (N > 0) {
      if (inner_dim > 1) {
        const Tindex first_dim_size = var.dim_size(0);
        auto indices_vec = indices.vec<Tindex>();
        auto var_flat = var.flat_outer_dims<T>();
        auto accum_flat = accum.flat_outer_dims<T>();
        auto grad_flat = grad.flat_outer_dims<T>();
        T lr_scalar = lr.scalar<T>()();

        // Note(yonghui): It might be worth multi-threading square() and
        // rsqrt().
        for (Tindex i = 0; i < N; i++) {
          const Tindex index = internal::SubtleMustCopy(indices_vec(i));
          OP_REQUIRES(ctx, FastBoundsCheck(index, first_dim_size),
                      errors::InvalidArgument(
                          strings::StrCat("Index ", index, " at offset ", i,
                                          " in indices is out of range")));
          auto a = accum_flat.template chip<0>(index);
          auto g = grad_flat.template chip<0>(i);
          auto v = var_flat.template chip<0>(index);
          a += g.square();
          v -= g.constant(lr_scalar) * g * a.rsqrt();
        }
      } else {
        auto indices_vec = indices.vec<Tindex>();
        auto var_flat = var.flat<T>();
        auto accum_flat = accum.flat<T>();
        auto grad_flat = grad.flat<T>();
        T lr_scalar = lr.scalar<T>()();
        const Tindex first_dim_size = accum_flat.size();

        for (Tindex i = 0; i < N; i++) {
          const Tindex index = internal::SubtleMustCopy(indices_vec(i));
          OP_REQUIRES(ctx, FastBoundsCheck(index, first_dim_size),
                      errors::InvalidArgument(
                          strings::StrCat("Index ", index, " at offset ", i,
                                          " in indices is out of range")));
          T& a = accum_flat(index);
          const T& g = grad_flat(i);
          a += g * g;
          var_flat(index) -= lr_scalar * g / Eigen::numext::sqrt(a);
        }
      }
    }

    ctx->forward_ref_input_to_ref_output(0, 0);
  }

 private:
  bool use_exclusive_lock_;
};

#define REGISTER_KERNELS(T, Tindices)                                \
  REGISTER_KERNEL_BUILDER(Name("SparseApplyAdagrad")                 \
                              .Device(DEVICE_CPU)                    \
                              .TypeConstraint<T>("T")                \
                              .TypeConstraint<Tindices>("Tindices"), \
                          SparseApplyAdagradOp<T, Tindices>);
#define REGISTER_CPU_KERNELS(T) \
  REGISTER_KERNELS(T, int32);   \
  REGISTER_KERNELS(T, int64);

TF_CALL_half(REGISTER_CPU_KERNELS);
TF_CALL_float(REGISTER_CPU_KERNELS);
TF_CALL_double(REGISTER_CPU_KERNELS);

#undef REGISTER_CPU_KERNELS
#undef REGISTER_KERNELS

// Note, this op works on cpu only.
template <typename T, typename Tindex>
class SparseApplyProximalAdagradOp : public OpKernel {
 public:
  explicit SparseApplyProximalAdagradOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override NO_THREAD_SAFETY_ANALYSIS {
    auto locks = MaybeLockMutexesInOrder(ctx, use_exclusive_lock_, {0, 1});
    Tensor var = ctx->mutable_input(0, use_exclusive_lock_);
    Tensor accum = ctx->mutable_input(1, use_exclusive_lock_);
    OP_REQUIRES(
        ctx, var.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", def().input(0)));
    OP_REQUIRES(
        ctx, accum.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", def().input(1)));
    OP_REQUIRES(
        ctx, var.shape().IsSameSize(accum.shape()),
        errors::InvalidArgument("var and accum do not have the same shape",
                                var.shape().DebugString(), " ",
                                accum.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVectorOrHigher(var.shape()),
                errors::InvalidArgument("var must be at least 1 dimensional"));

    const Tensor& lr = ctx->input(2);
    OP_REQUIRES(ctx, IsLegacyScalar(lr.shape()),
                errors::InvalidArgument("lr is not a scalar: ",
                                        lr.shape().DebugString()));
    const Tensor& l1 = ctx->input(3);
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsScalar(l1.shape()),
        errors::InvalidArgument("l1 regularization strength is not a scalar: ",
                                l1.shape().DebugString()));
    const Tensor& l2 = ctx->input(4);
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsScalar(l2.shape()),
        errors::InvalidArgument("l2 regularization strength is not a scalar: ",
                                l2.shape().DebugString()));

    const Tensor& grad = ctx->input(5);
    const Tensor& indices = ctx->input(6);
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(indices.shape()),
                errors::InvalidArgument("indices must be one-dimensional"));

    int64 inner_dim = 1;
    for (int d = 1; d < var.dims(); d++) {
      OP_REQUIRES(ctx, var.dim_size(d) == grad.dim_size(d),
                  errors::InvalidArgument(strings::StrCat(
                      "var and grad must match in dimension ", d)));
      inner_dim *= grad.dim_size(d);
    }
    const Tindex N = indices.dim_size(0);
    OP_REQUIRES(
        ctx, grad.dim_size(0) == N,
        errors::InvalidArgument(
            "grad must be the same size as indices in the first dimension."));

    OP_REQUIRES(ctx, inner_dim > 0,
                errors::InvalidArgument(
                    "Inner dimension should be greater than zero."));

    if (N > 0) {
      if (inner_dim > 1) {
        const Tindex first_dim_size = var.dim_size(0);
        auto indices_vec = indices.vec<Tindex>();
        auto var_flat = var.flat_outer_dims<T>();
        auto accum_flat = accum.flat_outer_dims<T>();
        auto grad_flat = grad.flat_outer_dims<T>();
        T lr_scalar = lr.scalar<T>()();
        T l1_scalar = l1.scalar<T>()();
        T l2_scalar = l2.scalar<T>()();

        for (Tindex i = 0; i < N; i++) {
          const Tindex index = internal::SubtleMustCopy(indices_vec(i));
          OP_REQUIRES(ctx, FastBoundsCheck(index, first_dim_size),
                      errors::InvalidArgument(
                          strings::StrCat("Index ", index, " at offset ", i,
                                          " in indices is out of range")));
          auto a = accum_flat.template chip<0>(index);
          auto g = grad_flat.template chip<0>(i);
          auto v = var_flat.template chip<0>(index);
          a += g.square();
          // compute learning_rate for current step.
          auto learning_rate = a.constant(lr_scalar) * a.rsqrt();
          auto prox_v = v;
          // v = w - g * learning_rate.
          prox_v -= g * learning_rate;
          if (l1_scalar > 0) {
            v = prox_v.abs() - learning_rate * prox_v.constant(l1_scalar);
            // compute sign(v) * max(|v|, 0)
            v = prox_v.sign() * v.cwiseMax(static_cast<T>(0.0));
          } else {
            v = prox_v;
          }
          if (l2_scalar > 0) {
            v /= (v.constant(1.0) + v.constant(l2_scalar) * learning_rate);
          }
        }
      } else {
        auto indices_vec = indices.vec<Tindex>();
        auto var_flat = var.flat<T>();
        auto accum_flat = accum.flat<T>();
        auto grad_flat = grad.flat<T>();
        T lr_scalar = lr.scalar<T>()();
        T l1_scalar = l1.scalar<T>()();
        T l2_scalar = l2.scalar<T>()();
        const Tindex first_dim_size = accum_flat.size();

        for (Tindex i = 0; i < N; i++) {
          const Tindex index = internal::SubtleMustCopy(indices_vec(i));
          OP_REQUIRES(ctx, FastBoundsCheck(index, first_dim_size),
                      errors::InvalidArgument(
                          strings::StrCat("Index ", index, " at offset ", i,
                                          " in indices is out of range")));
          T& a = accum_flat(index);
          const T& g = grad_flat(i);
          a += g * g;
          auto learning_rate = lr_scalar / std::sqrt(a);
          auto prox_v = var_flat(index);
          prox_v -= learning_rate * g;
          if (l1_scalar > 0) {
            var_flat(index) = std::abs(prox_v) - learning_rate * l1_scalar;
            var_flat(index) =
                sgn(prox_v) * std::max(var_flat(index), static_cast<T>(0.0));
          } else {
            var_flat(index) = prox_v;
          }
          if (l2_scalar > 0) {
            var_flat(index) /= (1.0 + l2_scalar * learning_rate);
          }
        }
      }
    }

    ctx->forward_ref_input_to_ref_output(0, 0);
  }

 private:
  bool use_exclusive_lock_;
};

#define REGISTER_KERNELS(T, Tindices)                                \
  REGISTER_KERNEL_BUILDER(Name("SparseApplyProximalAdagrad")         \
                              .Device(DEVICE_CPU)                    \
                              .TypeConstraint<T>("T")                \
                              .TypeConstraint<Tindices>("Tindices"), \
                          SparseApplyProximalAdagradOp<T, Tindices>);

REGISTER_KERNELS(float, int32);
REGISTER_KERNELS(float, int64);
REGISTER_KERNELS(double, int32);
REGISTER_KERNELS(double, int64);
#undef REGISTER_KERNELS

template <typename Device, typename T>
class ApplyAdagradDAOp : public OpKernel {
 public:
  explicit ApplyAdagradDAOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override {
    auto locks = MaybeLockMutexesInOrder(ctx, use_exclusive_lock_, {0, 1});
    Tensor var = ctx->mutable_input(0, use_exclusive_lock_);
    Tensor gradient_accum = ctx->mutable_input(1, use_exclusive_lock_);
    Tensor gradient_squared_accum = ctx->mutable_input(2, use_exclusive_lock_);
    OP_REQUIRES(
        ctx, var.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", def().input(0)));
    OP_REQUIRES(
        ctx, gradient_accum.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", def().input(1)));
    OP_REQUIRES(
        ctx, gradient_squared_accum.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", def().input(2)));
    OP_REQUIRES(
        ctx, var.shape().IsSameSize(gradient_accum.shape()),
        errors::InvalidArgument("var and accum do not have the same shape",
                                var.shape().DebugString(), " ",
                                gradient_accum.shape().DebugString()));
    OP_REQUIRES(
        ctx, var.shape().IsSameSize(gradient_squared_accum.shape()),
        errors::InvalidArgument("var and accum do not have the same shape",
                                var.shape().DebugString(), " ",
                                gradient_squared_accum.shape().DebugString()));

    const Tensor& grad = ctx->input(3);
    OP_REQUIRES(
        ctx, var.shape().IsSameSize(grad.shape()),
        errors::InvalidArgument("var and grad do not have the same shape",
                                var.shape().DebugString(), " ",
                                grad.shape().DebugString()));

    const Tensor& lr = ctx->input(4);
    OP_REQUIRES(ctx, IsLegacyScalar(lr.shape()),
                errors::InvalidArgument("lr is not a scalar: ",
                                        lr.shape().DebugString()));
    const Tensor& l1 = ctx->input(5);
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsScalar(l1.shape()),
        errors::InvalidArgument("l1 regularization strength is not a scalar: ",
                                l1.shape().DebugString()));
    const Tensor& l2 = ctx->input(6);
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsScalar(l2.shape()),
        errors::InvalidArgument("l2 regularization strength is not a scalar: ",
                                l2.shape().DebugString()));
    const Tensor& global_step = ctx->input(7);
    OP_REQUIRES(ctx, IsLegacyScalar(global_step.shape()),
                errors::InvalidArgument("global_step is not a scalar: ",
                                        global_step.shape().DebugString()));

    const Device& device = ctx->template eigen_device<Device>();
    functor::ApplyAdagradDA<Device, T>()(
        device, var.flat<T>(), gradient_accum.flat<T>(),
        gradient_squared_accum.flat<T>(), lr.scalar<T>(),
        global_step.scalar<int64>()(), l1.scalar<T>(), l2.scalar<T>(),
        grad.flat<T>());

    ctx->forward_ref_input_to_ref_output(0, 0);
  }

 private:
  bool use_exclusive_lock_;
};

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

#define REGISTER_KERNELS(D, T)                                          \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("ApplyAdagradDA").Device(DEVICE_##D).TypeConstraint<T>("T"), \
      ApplyAdagradDAOp<D##Device, T>);

REGISTER_KERNELS(CPU, float);
REGISTER_KERNELS(CPU, double);
#undef REGISTER_KERNELS

// Note, this op works on cpu only.
template <typename T, typename Tindex>
class SparseApplyAdagradDAOp : public OpKernel {
 public:
  explicit SparseApplyAdagradDAOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override NO_THREAD_SAFETY_ANALYSIS {
    auto locks = MaybeLockMutexesInOrder(ctx, use_exclusive_lock_, {0, 1});
    Tensor var = ctx->mutable_input(0, use_exclusive_lock_);
    Tensor gradient_accum = ctx->mutable_input(1, use_exclusive_lock_);
    Tensor gradient_squared_accum = ctx->mutable_input(2, use_exclusive_lock_);
    OP_REQUIRES(
        ctx, var.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", def().input(0)));
    OP_REQUIRES(
        ctx, gradient_accum.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", def().input(1)));
    OP_REQUIRES(
        ctx, gradient_squared_accum.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", def().input(2)));
    OP_REQUIRES(
        ctx, var.shape().IsSameSize(gradient_accum.shape()),
        errors::InvalidArgument("var and accum do not have the same shape",
                                var.shape().DebugString(), " ",
                                gradient_accum.shape().DebugString()));
    OP_REQUIRES(
        ctx, var.shape().IsSameSize(gradient_squared_accum.shape()),
        errors::InvalidArgument("var and accum do not have the same shape",
                                var.shape().DebugString(), " ",
                                gradient_squared_accum.shape().DebugString()));

    OP_REQUIRES(ctx, TensorShapeUtils::IsVectorOrHigher(var.shape()),
                errors::InvalidArgument("var must be at least 1 dimensional"));

    const Tensor& grad = ctx->input(3);
    const Tensor& indices = ctx->input(4);
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(indices.shape()),
                errors::InvalidArgument("indices must be one-dimensional"));

    const Tensor& lr = ctx->input(5);
    OP_REQUIRES(ctx, IsLegacyScalar(lr.shape()),
                errors::InvalidArgument("lr is not a scalar: ",
                                        lr.shape().DebugString()));

    const Tensor& l1 = ctx->input(6);
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsScalar(l1.shape()),
        errors::InvalidArgument("l1 regularization strength is not a scalar: ",
                                l1.shape().DebugString()));

    const Tensor& l2 = ctx->input(7);
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsScalar(l2.shape()),
        errors::InvalidArgument("l2 regularization strength is not a scalar: ",
                                l2.shape().DebugString()));

    const Tensor& global_step = ctx->input(8);
    OP_REQUIRES(ctx, IsLegacyScalar(global_step.shape()),
                errors::InvalidArgument("global_step is not a scalar: ",
                                        global_step.shape().DebugString()));

    int64 inner_dim = 1;
    for (int d = 1; d < var.dims(); d++) {
      OP_REQUIRES(ctx, var.dim_size(d) == grad.dim_size(d),
                  errors::InvalidArgument(strings::StrCat(
                      "var and grad must match in dimension ", d)));
      inner_dim *= grad.dim_size(d);
    }
    const Tindex N = indices.dim_size(0);
    OP_REQUIRES(
        ctx, grad.dim_size(0) == N,
        errors::InvalidArgument(
            "grad must be the same size as indices in the first dimension."));

    OP_REQUIRES(ctx, inner_dim > 0,
                errors::InvalidArgument(
                    "Inner dimension should be greater than zero."));

    // AdagradDA update:
    // Let g to be gradient accumulator, gg to be gradient squared accumulator,
    // T be the global step, lr is the learning rate, and k the initial
    // gradient squared accumulator value.
    // w = \dfrac{sign(-g)*lr*|g - l1*T|_{+}}{l2*T*lr + \sqrt{k+gg})}
    if (N > 0) {
      if (inner_dim > 1) {
        const Tindex first_dim_size = var.dim_size(0);
        auto indices_vec = indices.vec<Tindex>();
        auto var_flat = var.flat_outer_dims<T>();
        auto gradient_accum_flat = gradient_accum.flat_outer_dims<T>();
        auto gradient_squared_accum_flat =
            gradient_squared_accum.flat_outer_dims<T>();
        auto grad_flat = grad.flat_outer_dims<T>();
        T lr_scalar = lr.scalar<T>()();
        T global_step_scalar = global_step.scalar<int64>()();
        T l1_scalar = l1.scalar<T>()();
        T l2_scalar = l2.scalar<T>()();
        const double gs_lr = global_step_scalar * lr_scalar;

        for (Tindex i = 0; i < N; i++) {
          const Tindex index = internal::SubtleMustCopy(indices_vec(i));
          OP_REQUIRES(ctx, FastBoundsCheck(index, first_dim_size),
                      errors::InvalidArgument(
                          strings::StrCat("Index ", index, " at offset ", i,
                                          " in indices is out of range")));
          auto ga = gradient_accum_flat.template chip<0>(index);
          auto da = gradient_squared_accum_flat.template chip<0>(index);
          auto g = grad_flat.template chip<0>(i);
          auto v = var_flat.template chip<0>(index);
          ga += g;
          da += g.square();
          if (l1_scalar > 0) {
            v = (ga.abs() / ga.constant(global_step_scalar)) -
                ga.constant(l1_scalar);
            v = ga.constant(-1.0) * ga.sign() *
                v.cwiseMax(static_cast<T>(0.0)) /
                (v.constant(l2_scalar) + da.sqrt() / v.constant(gs_lr));
          } else {
            v = ga.constant(-1.0) * (ga / ga.constant(global_step_scalar)) /
                (v.constant(l2_scalar) + da.sqrt() / v.constant(gs_lr));
          }
        }
      } else {
        auto indices_vec = indices.vec<Tindex>();
        auto var_flat = var.flat<T>();
        auto gradient_accum_flat = gradient_accum.flat<T>();
        auto gradient_squared_accum_flat = gradient_squared_accum.flat<T>();
        auto grad_flat = grad.flat<T>();
        const double lr_scalar = lr.scalar<T>()();
        const int64 global_step_scalar = global_step.scalar<int64>()();
        const double l1_scalar = l1.scalar<T>()();
        const double l2_scalar = l2.scalar<T>()();
        const Tindex first_dim_size = var_flat.size();
        const double gs_l1 = global_step_scalar * l1_scalar;
        const double gs_l2_lr = global_step_scalar * l2_scalar * lr_scalar;

        for (Tindex i = 0; i < N; i++) {
          const Tindex index = internal::SubtleMustCopy(indices_vec(i));
          OP_REQUIRES(ctx, FastBoundsCheck(index, first_dim_size),
                      errors::InvalidArgument(
                          strings::StrCat("Index ", index, " at offset ", i,
                                          " in indices is out of range")));
          T& ga = gradient_accum_flat(index);
          T& da = gradient_squared_accum_flat(index);
          const double g = grad_flat(i);
          ga += g;
          da += g * g;
          if (l1_scalar > 0) {
            var_flat(index) = sgn(-ga) * lr_scalar *
                              std::max((std::abs(ga) - gs_l1), 0.0) /
                              (gs_l2_lr + std::sqrt(da));
          } else {
            var_flat(index) = (-ga * lr_scalar) / (gs_l2_lr + std::sqrt(da));
          }
        }
      }
    }

    ctx->forward_ref_input_to_ref_output(0, 0);
  }

 private:
  bool use_exclusive_lock_;
};

#define REGISTER_KERNELS(T, Tindices)                                \
  REGISTER_KERNEL_BUILDER(Name("SparseApplyAdagradDA")               \
                              .Device(DEVICE_CPU)                    \
                              .TypeConstraint<T>("T")                \
                              .TypeConstraint<Tindices>("Tindices"), \
                          SparseApplyAdagradDAOp<T, Tindices>);

REGISTER_KERNELS(float, int32);
REGISTER_KERNELS(float, int64);
REGISTER_KERNELS(double, int32);
REGISTER_KERNELS(double, int64);
#undef REGISTER_KERNELS

template <typename Device, typename T>
class ApplyFtrlOp : public OpKernel {
 public:
  explicit ApplyFtrlOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override {
    auto locks = MaybeLockMutexesInOrder(ctx, use_exclusive_lock_, {0, 1, 2});

    Tensor var = ctx->mutable_input(0, use_exclusive_lock_);
    Tensor accum = ctx->mutable_input(1, use_exclusive_lock_);
    Tensor linear = ctx->mutable_input(2, use_exclusive_lock_);
    OP_REQUIRES(
        ctx, var.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", def().input(0)));
    OP_REQUIRES(
        ctx, accum.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", def().input(1)));
    OP_REQUIRES(
        ctx, linear.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", def().input(2)));

    const Tensor& grad = ctx->input(3);
    OP_REQUIRES(
        ctx, var.shape().IsSameSize(accum.shape()),
        errors::InvalidArgument("var and accum do not have the same shape",
                                var.shape().DebugString(), " ",
                                accum.shape().DebugString()));
    OP_REQUIRES(
        ctx, var.shape().IsSameSize(linear.shape()),
        errors::InvalidArgument("var and linear do not have the same shape",
                                var.shape().DebugString(), " ",
                                linear.shape().DebugString()));
    OP_REQUIRES(
        ctx, var.shape().IsSameSize(grad.shape()),
        errors::InvalidArgument("var and grad do not have the same shape",
                                var.shape().DebugString(), " ",
                                grad.shape().DebugString()));

    const Tensor& lr = ctx->input(4);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(lr.shape()) &&
                         lr.scalar<T>()() > static_cast<T>(0),
                errors::InvalidArgument("lr is not a scalar or <= 0.0: ",
                                        lr.shape().DebugString()));
    const Tensor& l1 = ctx->input(5);
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsScalar(l1.shape()),
        errors::InvalidArgument("l1 regularization strength is not a scalar: ",
                                l1.shape().DebugString()));
    const Tensor& l2 = ctx->input(6);
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsScalar(l2.shape()),
        errors::InvalidArgument("l2 regularization strength is not a scalar: ",
                                l2.shape().DebugString()));
    const Tensor& lr_power = ctx->input(7);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(lr_power.shape()),
                errors::InvalidArgument("lr power is not a scalar: ",
                                        lr_power.shape().DebugString()));

    const Device& device = ctx->template eigen_device<Device>();
    functor::ApplyFtrl<Device, T>()(device, var.flat<T>(), accum.flat<T>(),
                                    linear.flat<T>(), grad.flat<T>(),
                                    lr.scalar<T>(), l1.scalar<T>(),
                                    l2.scalar<T>(), lr_power.scalar<T>());

    ctx->forward_ref_input_to_ref_output(0, 0);
  }

 private:
  bool use_exclusive_lock_;
};

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

#define REGISTER_KERNELS(D, T)                                     \
  REGISTER_KERNEL_BUILDER(                                         \
      Name("ApplyFtrl").Device(DEVICE_##D).TypeConstraint<T>("T"), \
      ApplyFtrlOp<D##Device, T>);
#define REGISTER_CPU_KERNELS(T) REGISTER_KERNELS(CPU, T);

TF_CALL_half(REGISTER_CPU_KERNELS);
TF_CALL_float(REGISTER_CPU_KERNELS);
TF_CALL_double(REGISTER_CPU_KERNELS);

#undef REGISTER_CPU_KERNELS
#undef REGISTER_KERNELS

// Note, this op works on cpu only.
template <typename Device, typename T, typename Tindex>
class SparseApplyFtrlOp : public OpKernel {
 public:
  explicit SparseApplyFtrlOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override NO_THREAD_SAFETY_ANALYSIS {
    auto locks = MaybeLockMutexesInOrder(ctx, use_exclusive_lock_, {0, 1, 2});
    Tensor var = ctx->mutable_input(0, use_exclusive_lock_);
    Tensor accum = ctx->mutable_input(1, use_exclusive_lock_);
    Tensor linear = ctx->mutable_input(2, use_exclusive_lock_);
    OP_REQUIRES(
        ctx, var.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", def().input(0)));
    OP_REQUIRES(
        ctx, accum.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", def().input(1)));
    OP_REQUIRES(
        ctx, linear.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", def().input(2)));
    OP_REQUIRES(
        ctx, var.shape().IsSameSize(accum.shape()),
        errors::InvalidArgument("var and accum do not have the same shape",
                                var.shape().DebugString(), " ",
                                accum.shape().DebugString()));
    OP_REQUIRES(
        ctx, var.shape().IsSameSize(linear.shape()),
        errors::InvalidArgument("var and linear do not have the same shape",
                                var.shape().DebugString(), " ",
                                linear.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVectorOrHigher(var.shape()),
                errors::InvalidArgument("var must be at least 1 dimensional"));

    const Tensor& grad = ctx->input(3);
    const Tensor& indices = ctx->input(4);
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(indices.shape()),
                errors::InvalidArgument("indices must be one-dimensional"));

    const Tensor& lr = ctx->input(5);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(lr.shape()) &&
                         lr.scalar<T>()() > static_cast<T>(0),
                errors::InvalidArgument("lr is not a scalar or <= 0.0: ",
                                        lr.shape().DebugString()));

    const Tensor& l1 = ctx->input(6);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(l1.shape()),
                errors::InvalidArgument("l1 is not a scalar: ",
                                        l1.shape().DebugString()));
    const Tensor& l2 = ctx->input(7);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(l2.shape()),
                errors::InvalidArgument("l2 is not a scalar: ",
                                        l2.shape().DebugString()));
    const Tensor& lr_power = ctx->input(8);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(lr_power.shape()),
                errors::InvalidArgument("lr_power is not a scalar: ",
                                        lr_power.shape().DebugString()));

    int64 inner_dim = 1;
    for (int d = 1; d < var.dims(); d++) {
      OP_REQUIRES(ctx, var.dim_size(d) == grad.dim_size(d),
                  errors::InvalidArgument(strings::StrCat(
                      "var and grad must match in dimension ", d)));
      inner_dim *= grad.dim_size(d);
    }
    const Tindex N = indices.dim_size(0);
    OP_REQUIRES(
        ctx, grad.dim_size(0) == N,
        errors::InvalidArgument(
            "grad must be the same size as indices in the first dimension."));

    OP_REQUIRES(ctx, inner_dim > 0,
                errors::InvalidArgument(
                    "Inner dimension should be greater than zero."));

    if (N > 0) {
      if (inner_dim > 1) {
        const Tindex first_dim_size = var.dim_size(0);
        auto indices_vec = indices.vec<Tindex>();
        auto var_flat = var.flat_outer_dims<T>();
        auto accum_flat = accum.flat_outer_dims<T>();
        auto linear_flat = linear.flat_outer_dims<T>();
        auto grad_flat = grad.flat_outer_dims<T>();
        T lr_scalar = lr.scalar<T>()();
        T l1_scalar = l1.scalar<T>()();
        T l2_scalar = l2.scalar<T>()();
        T lr_power_scalar = lr_power.scalar<T>()();

        for (Tindex i = 0; i < N; i++) {
          const Tindex index = internal::SubtleMustCopy(indices_vec(i));
          OP_REQUIRES(ctx, FastBoundsCheck(index, first_dim_size),
                      errors::InvalidArgument(
                          strings::StrCat("Index ", index, " at offset ", i,
                                          " in indices is out of range")));
          auto accum = accum_flat.template chip<0>(index);
          auto linear = linear_flat.template chip<0>(index);
          auto grad = grad_flat.template chip<0>(i);
          auto var = var_flat.template chip<0>(index);

          auto new_accum = accum + grad.square();
          if (lr_power_scalar == static_cast<T>(-0.5)) {
            linear +=
                grad - (new_accum.sqrt() - accum.sqrt()) / lr_scalar * var;
          } else {
            linear += grad -
                      (new_accum.pow(-lr_power_scalar) -
                       accum.pow(-lr_power_scalar)) /
                          lr_scalar * var;
          }
          auto x = (linear.constant(l1_scalar) * linear.sign() - linear);
          if (lr_power_scalar == static_cast<T>(-0.5)) {
            auto y = new_accum.sqrt() / new_accum.constant(lr_scalar) +
                     linear.constant(static_cast<T>(2) * l2_scalar);
            var = x / y;
          } else {
            auto y = new_accum.pow(-lr_power_scalar) /
                         new_accum.constant(lr_scalar) +
                     linear.constant(static_cast<T>(2) * l2_scalar);
            var = x / y;
          }
          var = (linear.abs() > linear.constant(l1_scalar))
                    .select(var, var.constant(static_cast<T>(0)));
          accum += grad.square();
        }
      } else {
        auto indices_vec = indices.vec<Tindex>();
        auto var_flat = var.flat<T>();
        auto accum_flat = accum.flat<T>();
        auto linear_flat = linear.flat<T>();
        auto grad_flat = grad.flat<T>();
        T lr_scalar = lr.scalar<T>()();
        T l1_scalar = l1.scalar<T>()();
        T l2_scalar = l2.scalar<T>()();
        T lr_power_scalar = lr_power.scalar<T>()();
        const Tindex first_dim_size = accum_flat.size();

        for (Tindex i = 0; i < N; i++) {
          const Tindex index = internal::SubtleMustCopy(indices_vec(i));
          OP_REQUIRES(ctx, FastBoundsCheck(index, first_dim_size),
                      errors::InvalidArgument(
                          strings::StrCat("Index ", index, " at offset ", i,
                                          " in indices is out of range")));
          T& a = accum_flat(index);
          T& l = linear_flat(index);
          T& v = var_flat(index);
          const T& g = grad_flat(i);

          T updated_a = a + g * g;
          using Eigen::numext::pow;
          T sigma = pow(updated_a, -lr_power_scalar) - pow(a, -lr_power_scalar);
          sigma /= lr_scalar;
          T updated_l = l + g - sigma * v;
          v = FtrlCompute(updated_a, updated_l, lr_scalar, l1_scalar, l2_scalar,
                          lr_power_scalar);
          a = updated_a;
          l = updated_l;
        }
      }
    }

    ctx->forward_ref_input_to_ref_output(0, 0);
  }

 private:
  bool use_exclusive_lock_;
};

#define REGISTER_KERNELS(T, Tindices)                                \
  REGISTER_KERNEL_BUILDER(Name("SparseApplyFtrl")                    \
                              .Device(DEVICE_CPU)                    \
                              .TypeConstraint<T>("T")                \
                              .TypeConstraint<Tindices>("Tindices"), \
                          SparseApplyFtrlOp<CPUDevice, T, Tindices>);
#define REGISTER_CPU_KERNELS(T) \
  REGISTER_KERNELS(T, int32);   \
  REGISTER_KERNELS(T, int64);

TF_CALL_half(REGISTER_CPU_KERNELS);
TF_CALL_float(REGISTER_CPU_KERNELS);
TF_CALL_double(REGISTER_CPU_KERNELS);

#undef REGISTER_CPU_KERNELS
#undef REGISTER_KERNELS

template <typename Device, typename T>
class ApplyMomentumOp : public OpKernel {
 public:
  explicit ApplyMomentumOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_nesterov", &use_nesterov_));
  }

  void Compute(OpKernelContext* ctx) override {
    auto locks = MaybeLockMutexesInOrder(ctx, use_exclusive_lock_, {0, 1});

    Tensor var = ctx->mutable_input(0, use_exclusive_lock_);
    Tensor accum = ctx->mutable_input(1, use_exclusive_lock_);
    OP_REQUIRES(
        ctx, var.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", def().input(0)));
    OP_REQUIRES(
        ctx, accum.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", def().input(1)));
    const Tensor& lr = ctx->input(2);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(lr.shape()),
                errors::InvalidArgument("lr is not a scalar: ",
                                        lr.shape().DebugString()));
    const Tensor& grad = ctx->input(3);
    OP_REQUIRES(
        ctx, var.shape().IsSameSize(accum.shape()),
        errors::InvalidArgument("var and accum do not have the same shape",
                                var.shape().DebugString(), " ",
                                accum.shape().DebugString()));
    OP_REQUIRES(
        ctx, var.shape().IsSameSize(grad.shape()),
        errors::InvalidArgument("var and grad do not have the same shape",
                                var.shape().DebugString(), " ",
                                grad.shape().DebugString()));

    const Tensor& momentum = ctx->input(4);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(momentum.shape()),
                errors::InvalidArgument("momentum is not a scalar: ",
                                        momentum.shape().DebugString()));

    const Device& device = ctx->template eigen_device<Device>();
    functor::ApplyMomentum<Device, T>()(device, var.flat<T>(), accum.flat<T>(),
                                        lr.scalar<T>(), grad.flat<T>(),
                                        momentum.scalar<T>(), use_nesterov_);
    ctx->forward_ref_input_to_ref_output(0, 0);
  }

 private:
  bool use_exclusive_lock_;
  bool use_nesterov_;
};

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

#define REGISTER_KERNELS(D, T)                                         \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("ApplyMomentum").Device(DEVICE_##D).TypeConstraint<T>("T"), \
      ApplyMomentumOp<D##Device, T>);
#define REGISTER_CPU_KERNELS(T) REGISTER_KERNELS(CPU, T);

TF_CALL_half(REGISTER_CPU_KERNELS);
TF_CALL_float(REGISTER_CPU_KERNELS);
TF_CALL_double(REGISTER_CPU_KERNELS);

#if GOOGLE_CUDA
// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC(T)                                               \
  template <>                                                             \
  void ApplyMomentum<GPUDevice, T>::operator()(                           \
      const GPUDevice& d, typename TTypes<T>::Flat var,                   \
      typename TTypes<T>::Flat accum, typename TTypes<T>::ConstScalar lr, \
      typename TTypes<T>::ConstFlat grad,                                 \
      typename TTypes<T>::ConstScalar momentum, bool use_nesterov);       \
  extern template struct ApplyMomentum<GPUDevice, T>;
DECLARE_GPU_SPEC(Eigen::half);
DECLARE_GPU_SPEC(float);
DECLARE_GPU_SPEC(double);
#undef DECLARE_GPU_SPEC
}  // namespace functor

REGISTER_KERNELS(GPU, Eigen::half);
REGISTER_KERNELS(GPU, float);
REGISTER_KERNELS(GPU, double);
#endif
#undef REGISTER_CPU_KERNELS
#undef REGISTER_KERNELS

// Note, this op works on cpu only.
template <typename T, typename Tindex>
class SparseApplyMomentumOp : public OpKernel {
 public:
  explicit SparseApplyMomentumOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_nesterov", &use_nesterov_));
  }

  void Compute(OpKernelContext* ctx) override NO_THREAD_SAFETY_ANALYSIS {
    auto locks = MaybeLockMutexesInOrder(ctx, use_exclusive_lock_, {0, 1});

    Tensor var = ctx->mutable_input(0, use_exclusive_lock_);
    Tensor accum = ctx->mutable_input(1, use_exclusive_lock_);
    OP_REQUIRES(
        ctx, var.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", def().input(0)));
    OP_REQUIRES(
        ctx, accum.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", def().input(1)));
    OP_REQUIRES(
        ctx, var.shape().IsSameSize(accum.shape()),
        errors::InvalidArgument("var and accum do not have the same shape",
                                var.shape().DebugString(), " ",
                                accum.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVectorOrHigher(var.shape()),
                errors::InvalidArgument("var must be at least 1 dimensional"));

    const Tensor& lr = ctx->input(2);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(lr.shape()),
                errors::InvalidArgument("lr is not a scalar : ",
                                        lr.shape().DebugString()));
    const Tensor& grad = ctx->input(3);
    const Tensor& indices = ctx->input(4);
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(indices.shape()),
                errors::InvalidArgument("indices must be one-dimensional"));

    for (int d = 1; d < var.dims(); d++) {
      OP_REQUIRES(ctx, var.dim_size(d) == grad.dim_size(d),
                  errors::InvalidArgument(strings::StrCat(
                      "var and grad must match in dimension ", d)));
    }
    const Tindex N = indices.dim_size(0);
    OP_REQUIRES(
        ctx, grad.dim_size(0) == N,
        errors::InvalidArgument(
            "grad must be the same size as indices in the first dimension."));

    const Tensor& momentum = ctx->input(5);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(momentum.shape()),
                errors::InvalidArgument("momentum is not a scalar: ",
                                        momentum.shape().DebugString()));

    if (N > 0) {
      const Tindex first_dim_size = var.dim_size(0);
      auto indices_vec = indices.vec<Tindex>();
      auto var_flat = var.flat_outer_dims<T>();
      auto accum_flat = accum.flat_outer_dims<T>();
      auto grad_flat = grad.flat_outer_dims<T>();
      T lr_scalar = lr.scalar<T>()();
      T momentum_scalar = momentum.scalar<T>()();

      for (Tindex i = 0; i < N; i++) {
        const Tindex index = internal::SubtleMustCopy(indices_vec(i));
        OP_REQUIRES(ctx, FastBoundsCheck(index, first_dim_size),
                    errors::InvalidArgument(
                        strings::StrCat("Index ", index, " at offset ", i,
                                        " in indices is out of range")));
        auto a = accum_flat.template chip<0>(index);
        auto g = grad_flat.template chip<0>(i);
        auto v = var_flat.template chip<0>(index);
        a = a * a.constant(momentum_scalar) + g;
        if (use_nesterov_) {
          v -= g.constant(lr_scalar) * g +
               a.constant(lr_scalar) * a.constant(momentum_scalar) * a;
        } else {
          v -= a.constant(lr_scalar) * a;
        }
      }
    }

    ctx->forward_ref_input_to_ref_output(0, 0);
  }

 private:
  bool use_exclusive_lock_;
  bool use_nesterov_;
};

#define REGISTER_KERNELS(T, Tindices)                                \
  REGISTER_KERNEL_BUILDER(Name("SparseApplyMomentum")                \
                              .Device(DEVICE_CPU)                    \
                              .TypeConstraint<T>("T")                \
                              .TypeConstraint<Tindices>("Tindices"), \
                          SparseApplyMomentumOp<T, Tindices>);
#define REGISTER_CPU_KERNELS(T) \
  REGISTER_KERNELS(T, int32);   \
  REGISTER_KERNELS(T, int64);

TF_CALL_half(REGISTER_CPU_KERNELS);
TF_CALL_float(REGISTER_CPU_KERNELS);
TF_CALL_double(REGISTER_CPU_KERNELS);

#undef REGISTER_CPU_KERNELS
#undef REGISTER_KERNELS

template <typename Device, typename T>
class ApplyAdamOp : public OpKernel {
 public:
  explicit ApplyAdamOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override {
    auto locks = MaybeLockMutexesInOrder(ctx, use_exclusive_lock_, {0, 1, 2});

    Tensor var = ctx->mutable_input(0, use_exclusive_lock_);
    Tensor m = ctx->mutable_input(1, use_exclusive_lock_);
    Tensor v = ctx->mutable_input(2, use_exclusive_lock_);
    OP_REQUIRES(
        ctx, var.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", def().input(0)));
    OP_REQUIRES(
        ctx, m.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", def().input(1)));
    OP_REQUIRES(
        ctx, v.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", def().input(2)));

    const Tensor& beta1_power = ctx->input(3);
    const Tensor& beta2_power = ctx->input(4);
    const Tensor& lr = ctx->input(5);
    const Tensor& beta1 = ctx->input(6);
    const Tensor& beta2 = ctx->input(7);
    const Tensor& epsilon = ctx->input(8);

    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta1_power.shape()),
                errors::InvalidArgument("beta1_power is not a scalar: ",
                                        beta1_power.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta2_power.shape()),
                errors::InvalidArgument("beta2_power is not a scalar: ",
                                        beta2_power.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(lr.shape()),
                errors::InvalidArgument("lr is not a scalar : ",
                                        lr.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta1.shape()),
                errors::InvalidArgument("beta1 is not a scalar: ",
                                        beta1.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta2.shape()),
                errors::InvalidArgument("beta2 is not a scalar: ",
                                        beta2.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(epsilon.shape()),
                errors::InvalidArgument("epsilon is not a scalar: ",
                                        epsilon.shape().DebugString()));

    const Tensor& grad = ctx->input(9);
    OP_REQUIRES(ctx, var.shape().IsSameSize(m.shape()),
                errors::InvalidArgument("var and m do not have the same shape",
                                        var.shape().DebugString(), " ",
                                        m.shape().DebugString()));
    OP_REQUIRES(ctx, var.shape().IsSameSize(v.shape()),
                errors::InvalidArgument("var and v do not have the same shape",
                                        var.shape().DebugString(), " ",
                                        v.shape().DebugString()));
    OP_REQUIRES(
        ctx, var.shape().IsSameSize(grad.shape()),
        errors::InvalidArgument("var and grad do not have the same shape",
                                var.shape().DebugString(), " ",
                                grad.shape().DebugString()));

    const Device& device = ctx->template eigen_device<Device>();
    functor::ApplyAdam<Device, T>()(device, var.flat<T>(), m.flat<T>(),
                                    v.flat<T>(), beta1_power.scalar<T>(),
                                    beta2_power.scalar<T>(), lr.scalar<T>(),
                                    beta1.scalar<T>(), beta2.scalar<T>(),
                                    epsilon.scalar<T>(), grad.flat<T>());

    ctx->forward_ref_input_to_ref_output(0, 0);
  }

 private:
  bool use_exclusive_lock_;
};

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

#define REGISTER_KERNELS(D, T)                                     \
  REGISTER_KERNEL_BUILDER(                                         \
      Name("ApplyAdam").Device(DEVICE_##D).TypeConstraint<T>("T"), \
      ApplyAdamOp<D##Device, T>);
#define REGISTER_CPU_KERNELS(T) REGISTER_KERNELS(CPU, T);

TF_CALL_half(REGISTER_CPU_KERNELS);
TF_CALL_float(REGISTER_CPU_KERNELS);
TF_CALL_double(REGISTER_CPU_KERNELS);

#if GOOGLE_CUDA
// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC(T)                                   \
  template <>                                                 \
  void ApplyAdam<GPUDevice, T>::operator()(                   \
      const GPUDevice& d, typename TTypes<T>::Flat var,       \
      typename TTypes<T>::Flat m, typename TTypes<T>::Flat v, \
      typename TTypes<T>::ConstScalar beta1_power,            \
      typename TTypes<T>::ConstScalar beta2_power,            \
      typename TTypes<T>::ConstScalar lr,                     \
      typename TTypes<T>::ConstScalar beta1,                  \
      typename TTypes<T>::ConstScalar beta2,                  \
      typename TTypes<T>::ConstScalar epsilon,                \
      typename TTypes<T>::ConstFlat grad);                    \
  extern template struct ApplyAdam<GPUDevice, T>;
DECLARE_GPU_SPEC(Eigen::half);
DECLARE_GPU_SPEC(float);
DECLARE_GPU_SPEC(double);
#undef DECLARE_GPU_SPEC
}  // namespace functor

REGISTER_KERNELS(GPU, Eigen::half);
REGISTER_KERNELS(GPU, float);
REGISTER_KERNELS(GPU, double);
#endif
#undef REGISTER_CPU_KERNELS
#undef REGISTER_KERNELS

template <typename Device, typename T>
class ApplyRMSPropOp : public OpKernel {
 public:
  explicit ApplyRMSPropOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override {
    auto locks = MaybeLockMutexesInOrder(ctx, use_exclusive_lock_, {0, 1, 2});

    Tensor var = ctx->mutable_input(0, use_exclusive_lock_);
    Tensor ms = ctx->mutable_input(1, use_exclusive_lock_);
    Tensor mom = ctx->mutable_input(2, use_exclusive_lock_);

    OP_REQUIRES(
        ctx, var.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", def().input(0)));
    OP_REQUIRES(
        ctx, ms.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", def().input(1)));
    OP_REQUIRES(
        ctx, mom.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", def().input(2)));

    const Tensor& lr = ctx->input(3);
    const Tensor& rho = ctx->input(4);
    const Tensor& momentum = ctx->input(5);
    const Tensor& epsilon = ctx->input(6);
    const Tensor& grad = ctx->input(7);

    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(lr.shape()),
                errors::InvalidArgument("lr is not a scalar : ",
                                        lr.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(rho.shape()),
                errors::InvalidArgument("rho is not a scalar: ",
                                        rho.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(momentum.shape()),
                errors::InvalidArgument("momentum is not a scalar: ",
                                        momentum.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(epsilon.shape()),
                errors::InvalidArgument("epsilon is not a scalar: ",
                                        epsilon.shape().DebugString()));

    OP_REQUIRES(ctx, var.shape().IsSameSize(ms.shape()),
                errors::InvalidArgument("var and ms do not have the same shape",
                                        var.shape().DebugString(), " ",
                                        ms.shape().DebugString()));

    OP_REQUIRES(ctx, var.shape().IsSameSize(mom.shape()),
                errors::InvalidArgument(
                    "var and mom do not have the same shape",
                    var.shape().DebugString(), " ", mom.shape().DebugString()));

    OP_REQUIRES(
        ctx, var.shape().IsSameSize(grad.shape()),
        errors::InvalidArgument("var and grad do not have the same shape",
                                var.shape().DebugString(), " ",
                                grad.shape().DebugString()));

    const Device& device = ctx->template eigen_device<Device>();
    functor::ApplyRMSProp<Device, T>()(device, var.flat<T>(), ms.flat<T>(),
                                       mom.flat<T>(), lr.scalar<T>(),
                                       rho.scalar<T>(), momentum.scalar<T>(),
                                       epsilon.scalar<T>(), grad.flat<T>());

    ctx->forward_ref_input_to_ref_output(0, 0);
  }

 private:
  bool use_exclusive_lock_;
};

template <typename Device, typename T>
class ApplyCenteredRMSPropOp : public OpKernel {
 public:
  explicit ApplyCenteredRMSPropOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override {
    auto locks =
        MaybeLockMutexesInOrder(ctx, use_exclusive_lock_, {0, 1, 2, 3});

    Tensor var = ctx->mutable_input(0, use_exclusive_lock_);
    Tensor mg = ctx->mutable_input(1, use_exclusive_lock_);
    Tensor ms = ctx->mutable_input(2, use_exclusive_lock_);
    Tensor mom = ctx->mutable_input(3, use_exclusive_lock_);

    OP_REQUIRES(
        ctx, var.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", def().input(0)));
    OP_REQUIRES(
        ctx, mg.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", def().input(1)));
    OP_REQUIRES(
        ctx, ms.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", def().input(2)));
    OP_REQUIRES(
        ctx, mom.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", def().input(3)));

    const Tensor& lr = ctx->input(4);
    const Tensor& rho = ctx->input(5);
    const Tensor& momentum = ctx->input(6);
    const Tensor& epsilon = ctx->input(7);
    const Tensor& grad = ctx->input(8);

    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(lr.shape()),
                errors::InvalidArgument("lr is not a scalar : ",
                                        lr.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(rho.shape()),
                errors::InvalidArgument("rho is not a scalar: ",
                                        rho.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(momentum.shape()),
                errors::InvalidArgument("momentum is not a scalar: ",
                                        momentum.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(epsilon.shape()),
                errors::InvalidArgument("epsilon is not a scalar: ",
                                        epsilon.shape().DebugString()));

    OP_REQUIRES(ctx, var.shape().IsSameSize(mg.shape()),
                errors::InvalidArgument("var and mg do not have the same shape",
                                        var.shape().DebugString(), " ",
                                        ms.shape().DebugString()));

    OP_REQUIRES(ctx, var.shape().IsSameSize(ms.shape()),
                errors::InvalidArgument("var and ms do not have the same shape",
                                        var.shape().DebugString(), " ",
                                        ms.shape().DebugString()));

    OP_REQUIRES(ctx, var.shape().IsSameSize(mom.shape()),
                errors::InvalidArgument(
                    "var and mom do not have the same shape",
                    var.shape().DebugString(), " ", mom.shape().DebugString()));

    OP_REQUIRES(
        ctx, var.shape().IsSameSize(grad.shape()),
        errors::InvalidArgument("var and grad do not have the same shape",
                                var.shape().DebugString(), " ",
                                grad.shape().DebugString()));

    const Device& device = ctx->template eigen_device<Device>();
    functor::ApplyCenteredRMSProp<Device, T>()(
        device, var.flat<T>(), mg.flat<T>(), ms.flat<T>(), mom.flat<T>(),
        lr.scalar<T>(), rho.scalar<T>(), momentum.scalar<T>(),
        epsilon.scalar<T>(), grad.flat<T>());
    ctx->forward_ref_input_to_ref_output(0, 0);
  }

 private:
  bool use_exclusive_lock_;
};

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

#define REGISTER_KERNELS(D, T)                                                \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("ApplyRMSProp").Device(DEVICE_##D).TypeConstraint<T>("T"),         \
      ApplyRMSPropOp<D##Device, T>);                                          \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("ApplyCenteredRMSProp").Device(DEVICE_##D).TypeConstraint<T>("T"), \
      ApplyCenteredRMSPropOp<D##Device, T>);
#define REGISTER_CPU_KERNELS(T) REGISTER_KERNELS(CPU, T);

TF_CALL_half(REGISTER_CPU_KERNELS);
TF_CALL_float(REGISTER_CPU_KERNELS);
TF_CALL_double(REGISTER_CPU_KERNELS);

#if GOOGLE_CUDA
// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC(T)                                                    \
  template <>                                                                  \
  void ApplyRMSProp<GPUDevice, T>::operator()(                                 \
      const GPUDevice& d, typename TTypes<T>::Flat var,                        \
      typename TTypes<T>::Flat ms, typename TTypes<T>::Flat mom,               \
      typename TTypes<T>::ConstScalar lr, typename TTypes<T>::ConstScalar rho, \
      typename TTypes<T>::ConstScalar momentum,                                \
      typename TTypes<T>::ConstScalar epsilon,                                 \
      typename TTypes<T>::ConstFlat grad);                                     \
  extern template struct ApplyRMSProp<GPUDevice, T>;                           \
  template <>                                                                  \
  void ApplyCenteredRMSProp<GPUDevice, T>::operator()(                         \
      const GPUDevice& d, typename TTypes<T>::Flat var,                        \
      typename TTypes<T>::Flat mg, typename TTypes<T>::Flat ms,                \
      typename TTypes<T>::Flat mom, typename TTypes<T>::ConstScalar lr,        \
      typename TTypes<T>::ConstScalar rho,                                     \
      typename TTypes<T>::ConstScalar momentum,                                \
      typename TTypes<T>::ConstScalar epsilon,                                 \
      typename TTypes<T>::ConstFlat grad);                                     \
  extern template struct ApplyCenteredRMSProp<GPUDevice, T>;
DECLARE_GPU_SPEC(Eigen::half);
DECLARE_GPU_SPEC(float);
DECLARE_GPU_SPEC(double);
#undef DECLARE_GPU_SPEC
}  // namespace functor

REGISTER_KERNELS(GPU, Eigen::half);
REGISTER_KERNELS(GPU, float);
REGISTER_KERNELS(GPU, double);
#endif
#undef REGISTER_CPU_KERNELS
#undef REGISTER_KERNELS

// Note, this op works on cpu only.
template <typename T, typename Tindex>
class SparseApplyRMSPropOp : public OpKernel {
 public:
  explicit SparseApplyRMSPropOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override NO_THREAD_SAFETY_ANALYSIS {
    auto locks = MaybeLockMutexesInOrder(ctx, use_exclusive_lock_, {0, 1, 2});

    Tensor var = ctx->mutable_input(0, use_exclusive_lock_);
    Tensor ms = ctx->mutable_input(1, use_exclusive_lock_);
    Tensor mom = ctx->mutable_input(2, use_exclusive_lock_);

    OP_REQUIRES(
        ctx, var.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", def().input(0)));
    OP_REQUIRES(
        ctx, ms.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", def().input(1)));
    OP_REQUIRES(
        ctx, mom.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", def().input(2)));

    const Tensor& lr = ctx->input(3);
    const Tensor& rho = ctx->input(4);
    const Tensor& momentum = ctx->input(5);
    const Tensor& epsilon = ctx->input(6);
    const Tensor& grad = ctx->input(7);
    const Tensor& indices = ctx->input(8);

    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(lr.shape()),
                errors::InvalidArgument("lr is not a scalar: ",
                                        lr.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(rho.shape()),
                errors::InvalidArgument("rho is not a scalar: ",
                                        rho.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(momentum.shape()),
                errors::InvalidArgument("momentum is not a scalar: ",
                                        momentum.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(epsilon.shape()),
                errors::InvalidArgument("epsilon is not a scalar: ",
                                        epsilon.shape().DebugString()));

    OP_REQUIRES(ctx, var.shape().IsSameSize(ms.shape()),
                errors::InvalidArgument("var and ms do not have the same shape",
                                        var.shape().DebugString(), " ",
                                        ms.shape().DebugString()));

    OP_REQUIRES(ctx, var.shape().IsSameSize(mom.shape()),
                errors::InvalidArgument(
                    "var and mom do not have the same shape",
                    var.shape().DebugString(), " ", mom.shape().DebugString()));

    OP_REQUIRES(ctx, TensorShapeUtils::IsVectorOrHigher(var.shape()),
                errors::InvalidArgument("var must be at least 1 dimensional"));

    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(indices.shape()),
                errors::InvalidArgument("indices must be one-dimensional"));

    for (int d = 1; d < var.dims(); d++) {
      OP_REQUIRES(
          ctx, var.dim_size(d) == grad.dim_size(d),
          errors::InvalidArgument("var and grad must match in dimension ", d));
    }
    const Tindex N = indices.dim_size(0);
    OP_REQUIRES(
        ctx, grad.dim_size(0) == N,
        errors::InvalidArgument(
            "grad must be the same size as indices in the first dimension."));

    if (N > 0) {
      const Tindex first_dim_size = var.dim_size(0);
      // Validate all the indices are in range
      auto indices_vec = indices.vec<Tindex>();
      for (Tindex i = 0; i < N; i++) {
        const Tindex index = indices_vec(i);
        OP_REQUIRES(ctx, index >= 0 && index < first_dim_size,
                    errors::InvalidArgument(
                        strings::StrCat("Index ", index, " at offset ", i,
                                        " in indices is out of range")));
      }

      auto var_flat = var.flat_outer_dims<T>();
      auto ms_flat = ms.flat_outer_dims<T>();
      auto mom_flat = mom.flat_outer_dims<T>();
      auto grad_flat = grad.flat_outer_dims<T>();
      const T lr_scalar = lr.scalar<T>()();
      const T rho_scalar = rho.scalar<T>()();
      const T epsilon_scalar = epsilon.scalar<T>()();
      const T momentum_scalar = momentum.scalar<T>()();

      for (Tindex i = 0; i < N; i++) {
        const Tindex index = indices_vec(i);

        auto ms_ = ms_flat.template chip<0>(index);
        auto mom_ = mom_flat.template chip<0>(index);
        auto grad_ = grad_flat.template chip<0>(i);

        ms_ = ms_ * ms_.constant(rho_scalar) +
              grad_.square() * grad_.constant(T(1) - rho_scalar);
        mom_ = mom_ * mom_.constant(momentum_scalar) +
               (ms_ + ms_.constant(epsilon_scalar)).rsqrt() *
                   ms_.constant(lr_scalar) * grad_;

        auto v = var_flat.template chip<0>(index);
        v -= mom_;
      }
    }

    ctx->forward_ref_input_to_ref_output(0, 0);
  }

 private:
  bool use_exclusive_lock_;
};

// Note, this op works on cpu only.
template <typename T, typename Tindex>
class SparseApplyCenteredRMSPropOp : public OpKernel {
 public:
  explicit SparseApplyCenteredRMSPropOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override NO_THREAD_SAFETY_ANALYSIS {
    auto locks =
        MaybeLockMutexesInOrder(ctx, use_exclusive_lock_, {0, 1, 2, 3});

    Tensor var = ctx->mutable_input(0, use_exclusive_lock_);
    Tensor mg = ctx->mutable_input(1, use_exclusive_lock_);
    Tensor ms = ctx->mutable_input(2, use_exclusive_lock_);
    Tensor mom = ctx->mutable_input(3, use_exclusive_lock_);

    OP_REQUIRES(
        ctx, var.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", def().input(0)));
    OP_REQUIRES(
        ctx, ms.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", def().input(2)));
    OP_REQUIRES(
        ctx, mom.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", def().input(3)));

    const Tensor& lr = ctx->input(4);
    const Tensor& rho = ctx->input(5);
    const Tensor& momentum = ctx->input(6);
    const Tensor& epsilon = ctx->input(7);
    const Tensor& grad = ctx->input(8);
    const Tensor& indices = ctx->input(9);

    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(lr.shape()),
                errors::InvalidArgument("lr is not a scalar: ",
                                        lr.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(rho.shape()),
                errors::InvalidArgument("rho is not a scalar: ",
                                        rho.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(momentum.shape()),
                errors::InvalidArgument("momentum is not a scalar: ",
                                        momentum.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(epsilon.shape()),
                errors::InvalidArgument("epsilon is not a scalar: ",
                                        epsilon.shape().DebugString()));

    OP_REQUIRES(ctx, var.shape().IsSameSize(mg.shape()),
                errors::InvalidArgument("var and mg do not have the same shape",
                                        var.shape().DebugString(), " ",
                                        mg.shape().DebugString()));

    OP_REQUIRES(ctx, var.shape().IsSameSize(ms.shape()),
                errors::InvalidArgument("var and ms do not have the same shape",
                                        var.shape().DebugString(), " ",
                                        ms.shape().DebugString()));

    OP_REQUIRES(ctx, var.shape().IsSameSize(mom.shape()),
                errors::InvalidArgument(
                    "var and mom do not have the same shape",
                    var.shape().DebugString(), " ", mom.shape().DebugString()));

    OP_REQUIRES(ctx, TensorShapeUtils::IsVectorOrHigher(var.shape()),
                errors::InvalidArgument("var must be at least 1 dimensional"));

    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(indices.shape()),
                errors::InvalidArgument("indices must be one-dimensional"));

    for (int d = 1; d < var.dims(); d++) {
      OP_REQUIRES(
          ctx, var.dim_size(d) == grad.dim_size(d),
          errors::InvalidArgument("var and grad must match in dimension ", d));
    }
    const Tindex N = indices.dim_size(0);
    OP_REQUIRES(
        ctx, grad.dim_size(0) == N,
        errors::InvalidArgument(
            "grad must be the same size as indices in the first dimension."));

    if (N > 0) {
      const Tindex first_dim_size = var.dim_size(0);
      // Validate all the indices are in range
      auto indices_vec = indices.vec<Tindex>();
      for (Tindex i = 0; i < N; i++) {
        const Tindex index = indices_vec(i);
        OP_REQUIRES(ctx, index >= 0 && index < first_dim_size,
                    errors::InvalidArgument(
                        strings::StrCat("Index ", index, " at offset ", i,
                                        " in indices is out of range")));
      }

      auto var_flat = var.flat_outer_dims<T>();
      auto ms_flat = ms.flat_outer_dims<T>();
      auto mg_flat = mg.flat_outer_dims<T>();
      auto mom_flat = mom.flat_outer_dims<T>();
      auto grad_flat = grad.flat_outer_dims<T>();
      const T lr_scalar = lr.scalar<T>()();
      const T rho_scalar = rho.scalar<T>()();
      const T epsilon_scalar = epsilon.scalar<T>()();
      const T momentum_scalar = momentum.scalar<T>()();

      for (Tindex i = 0; i < N; i++) {
        const Tindex index = indices_vec(i);

        auto ms_ = ms_flat.template chip<0>(index);
        auto mom_ = mom_flat.template chip<0>(index);
        auto grad_ = grad_flat.template chip<0>(i);

        ms_ = ms_ * ms_.constant(rho_scalar) +
              grad_.square() * grad_.constant(T(1) - rho_scalar);

        auto mg_ = mg_flat.template chip<0>(index);
        mg_ = mg_ * mg_.constant(rho_scalar) +
              grad_ * grad_.constant(T(1) - rho_scalar);
        auto denom_ = ms_ + ms_.constant(epsilon_scalar) - mg_.square();
        mom_ = mom_ * mom_.constant(momentum_scalar) +
               denom_.rsqrt() * ms_.constant(lr_scalar) * grad_;
        auto v = var_flat.template chip<0>(index);
        v -= mom_;
      }
    }

    ctx->forward_ref_input_to_ref_output(0, 0);
  }

 private:
  bool use_exclusive_lock_;
};

#define REGISTER_KERNELS(T, Tindices)                                \
  REGISTER_KERNEL_BUILDER(Name("SparseApplyRMSProp")                 \
                              .Device(DEVICE_CPU)                    \
                              .TypeConstraint<T>("T")                \
                              .TypeConstraint<Tindices>("Tindices"), \
                          SparseApplyRMSPropOp<T, Tindices>);        \
  REGISTER_KERNEL_BUILDER(Name("SparseApplyCenteredRMSProp")         \
                              .Device(DEVICE_CPU)                    \
                              .TypeConstraint<T>("T")                \
                              .TypeConstraint<Tindices>("Tindices"), \
                          SparseApplyCenteredRMSPropOp<T, Tindices>);

REGISTER_KERNELS(Eigen::half, int32);
REGISTER_KERNELS(Eigen::half, int64);
REGISTER_KERNELS(float, int32);
REGISTER_KERNELS(float, int64);
REGISTER_KERNELS(double, int32);
REGISTER_KERNELS(double, int64);

#undef REGISTER_KERNELS

}  // namespace tensorflow
