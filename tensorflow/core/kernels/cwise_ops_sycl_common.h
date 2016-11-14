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

#if !TENSORFLOW_USE_SYCL
#error This file must only be included when building TensorFlow with SYCL support
#endif

#ifndef TENSORFLOW_CORE_KERNELS_CWISE_OPS_SYCL_COMMON_H_
#define TENSORFLOW_CORE_KERNELS_CWISE_OPS_SYCL_COMMON_H_

#define EIGEN_USE_SYCL

#include <SYCL/sycl.hpp>
#include "tensorflow/core/framework/register_types.h"

#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/cwise_ops.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace functor {

typedef Eigen::SyclDevice SYCLDevice;

// SYCL specific math functors
// Will be removed once implemented in Eigen

// acos
template<typename Scalar> struct scalar_acos_op_sycl {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_acos_op_sycl)
  inline const Scalar operator() (const Scalar& a) const { return cl::sycl::acos(a); }
};

template <typename T>
struct acos_sycl : base<T, scalar_acos_op_sycl<T> > {};

// asin
template<typename Scalar> struct scalar_asin_op_sycl {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_asin_op_sycl)
  inline const Scalar operator() (const Scalar& a) const { return cl::sycl::asin(a); }
};

template <typename T>
struct asin_sycl : base<T, scalar_asin_op_sycl<T> > {};

// atan
template<typename Scalar> struct scalar_atan_op_sycl {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_atan_op_sycl)
  inline const Scalar operator() (const Scalar& a) const { return cl::sycl::atan(a); }
};

template <typename T>
struct atan_sycl : base<T, scalar_atan_op_sycl<T> > {};

// ceil
template<typename Scalar> struct scalar_ceil_op_sycl {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_ceil_op_sycl)
  inline const Scalar operator() (const Scalar& a) const { return cl::sycl::ceil(a); }
};

template <typename T>
struct ceil_sycl : base<T, scalar_ceil_op_sycl<T> > {};

// cos
template<typename Scalar> struct scalar_cos_op_sycl {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_cos_op_sycl)
  inline const Scalar operator() (const Scalar& a) const { return cl::sycl::cos(a); }
};

template <typename T>
struct cos_sycl : base<T, scalar_cos_op_sycl<T> > {};

// exp
template<typename Scalar> struct scalar_exp_op_sycl {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_exp_op_sycl)
  inline const Scalar operator() (const Scalar& a) const { return cl::sycl::exp(a); }
};

template <typename T>
struct exp_sycl : base<T, scalar_exp_op_sycl<T> > {};

// floor
template<typename Scalar> struct scalar_floor_op_sycl {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_floor_op_sycl)
  inline const Scalar operator() (const Scalar& a) const { return cl::sycl::floor(a); }
};

template <typename T>
struct floor_sycl : base<T, scalar_floor_op_sycl<T> > {};

// log
template<typename Scalar> struct scalar_log_op_sycl {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_log_op_sycl)
  inline const Scalar operator() (const Scalar& a) const { return cl::sycl::log(a); }
};

template <typename T>
struct log_sycl : base<T, scalar_log_op_sycl<T> > {};

// log
template<typename Scalar> struct scalar_log1p_op_sycl {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_log1p_op_sycl)
  inline const Scalar operator() (const Scalar& a) const { return cl::sycl::log1p(a); }
};

template <typename T>
struct log1p_sycl : base<T, scalar_log1p_op_sycl<T> > {};

// sin
template<typename Scalar> struct scalar_sin_op_sycl {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_sin_op_sycl)
  inline const Scalar operator() (const Scalar& a) const { return cl::sycl::sin(a); }
};

template <typename T>
struct sin_sycl : base<T, scalar_sin_op_sycl<T> > {};


// sqrt
template<typename Scalar> struct scalar_sqrt_op_sycl {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_sqrt_op_sycl)
  inline const Scalar operator() (const Scalar& a) const { return cl::sycl::sqrt(a); }
};

template <typename T>
struct sqrt_sycl : base<T, scalar_sqrt_op_sycl<T> > {};

// isinf
template<typename Scalar> struct scalar_isinf_op_sycl {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_isinf_op_sycl)
  inline const bool operator() (const Scalar& a) const { return static_cast<bool>(cl::sycl::isinf(a)); }
};

template <typename T>
struct isinf_sycl : base<T, scalar_isinf_op_sycl<T>, bool> {};

// isnan
struct scalar_isnan_op_sycl {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_isnan_op_sycl)
  template<typename Scalar>
  inline const bool operator() (const Scalar& a) const {
#if __SYCL_DEVICE_ONLY__
    return static_cast<bool>(cl::sycl::isnan(a));
#else
    return (Eigen::numext::isnan)(a);
#endif
  }
};

template <typename T>
struct isnan_sycl : base<T, scalar_isnan_op_sycl, bool> {};

// isfinite
template<typename Scalar> struct scalar_isfinite_op_sycl {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_isfinite_op_sycl)
  inline const bool operator() (const Scalar& a) const { return static_cast<bool>(cl::sycl::isfinite(a)); }
};

template <typename T>
struct isfinite_sycl : base<T, scalar_isfinite_op_sycl<T>, bool> {};

// pow
template <typename Scalar, typename Exponent>
struct scalar_pow_op_sycl {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_pow_op_sycl)
  inline Scalar operator()(const Scalar& a, const Exponent& b) const {
  return cl::sycl::pow(a, b);
  }
};

template <typename T>
struct pow_sycl : base<T, scalar_pow_op_sycl<T, T> > {};

//rsqrt
template<typename Scalar> struct scalar_rsqrt_op_sycl {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_rsqrt_op_sycl)
  inline const Scalar operator() (const Scalar& a) const { return cl::sycl::rsqrt(a); }
};

template <typename T>
struct rsqrt_sycl : base<T, scalar_rsqrt_op_sycl<T> > {};

// tan
template<typename Scalar> struct scalar_tan_op_sycl {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_tan_op_sycl)
  inline const Scalar operator() (const Scalar& a) const { return cl::sycl::tan(a); }
};

template <typename T>
struct tan_sycl : base<T, scalar_tan_op_sycl<T> > {};


template <typename Index, int N> Eigen::array<Index, N> GenerateArrayOfOnes() {
  Eigen::array<Index, N> result;
  for (int i = 0; i < N; ++i) {
    result[i] = 1;
  }
  return result;
}

template <typename OUT, typename RHS>
void Assign(const SYCLDevice& d, OUT out, RHS rhs) {
  out.device(d) = rhs;
}

// Partial specialization of UnaryFunctor<Device=SYCLDevice, Functor>.
template <typename Functor>
struct UnaryFunctor<SYCLDevice, Functor> {
  void operator()(const SYCLDevice& d, typename Functor::tout_type out,
                  typename Functor::tin_type in) {
    To32Bit(out).device(d) = To32Bit(in).unaryExpr(typename Functor::func());
  }
};

// Partial specialization of BinaryFunctor<Device=SYCLDevice, Functor>.
template <typename Functor, int NDIMS, bool has_errors>
struct BinaryFunctor<SYCLDevice, Functor, NDIMS, has_errors> {
  void operator()(const SYCLDevice& d, typename Functor::tout_type out,
                  typename Functor::tin_type in0,
                  typename Functor::tin_type in1, bool* error) {
    To32Bit(out).device(d) = To32Bit(in0).binaryExpr(in1, typename Functor::func());
  }

  void Left(const SYCLDevice& d, typename Functor::tout_type out,
            typename Functor::tscalar_type scalar,
            typename Functor::tin_type in, bool* error) {
    typedef typename Functor::func Binary;
    constexpr int NumDims = Functor::tin_type::NumDimensions; 
    typedef typename Functor::tin_type::Scalar T;
    typedef typename Functor::tin_type::Index Index;
    Eigen::array<Index, NumDims> scalar_dim = GenerateArrayOfOnes<Index, NumDims>();
    Eigen::TensorMap<Eigen::Tensor<T, NumDims, Eigen::RowMajor>> tmp(scalar.data(), scalar_dim);
    out.device(d) = tmp.broadcast(in.dimensions()).binaryExpr(in, Binary());
  }

  void Right(const SYCLDevice& d, typename Functor::tout_type out,
             typename Functor::tin_type in,
             typename Functor::tscalar_type scalar, bool* error) {
    typedef typename Functor::func Binary;
    constexpr int NumDims = Functor::tin_type::NumDimensions;
    typedef typename Functor::tin_type::Scalar T;
    typedef typename Functor::tin_type::Index Index;
    Eigen::array<Index, NumDims> scalar_dim = GenerateArrayOfOnes<Index, NumDims>();
    Eigen::TensorMap<Eigen::Tensor<T, NumDims, Eigen::RowMajor>> tmp(scalar.data(), scalar_dim);
    out.device(d) = in.binaryExpr(tmp.broadcast(in.dimensions()), Binary());
  }

  void BCast(const SYCLDevice& d,
             typename TTypes<typename Functor::out_type, NDIMS>::Tensor out,
             typename TTypes<typename Functor::in_type, NDIMS>::ConstTensor in0,
             typename Eigen::array<Eigen::DenseIndex, NDIMS> bcast0,
             typename TTypes<typename Functor::in_type, NDIMS>::ConstTensor in1,
             typename Eigen::array<Eigen::DenseIndex, NDIMS> bcast1,
             bool* error) {
    typedef typename Functor::in_type T;
    typename Functor::func func;
    if ((NDIMS == 2) && Functor::use_bcast_optimization &&
        use_bcast_optimization<T>::value) {
      const bool bcast0_all_one = AllOne<NDIMS>(bcast0);
      const bool bcast1_all_one = AllOne<NDIMS>(bcast1);
      if (bcast0_all_one && !bcast1_all_one) {
        To32Bit(out).device(d) =
            To32Bit(in0).binaryExpr(To32Bit(in1).broadcast(bcast1), func);
        return;
      }
      if (!bcast0_all_one && bcast1_all_one) {
        To32Bit(out).device(d) =
            To32Bit(in0).broadcast(bcast0).binaryExpr(To32Bit(in1), func);
        return;
      }
    }
    To32Bit(out).device(d) = To32Bit(in0).broadcast(bcast0).binaryExpr(
        To32Bit(in1).broadcast(bcast1), func);
  }
};

// Macros to explicitly instantiate kernels on GPU for multiple types
// (T0, T1, etc.) for UnaryFunctor (e.g., functor::sqrt).
#define DEFINE_UNARY1(F, T) template struct UnaryFunctor<SYCLDevice, F<T> >
#define DEFINE_UNARY2(F, T0, T1) \
  DEFINE_UNARY1(F, T0);          \
  DEFINE_UNARY1(F, T1)
#define DEFINE_UNARY3(F, T0, T1, T2) \
  DEFINE_UNARY2(F, T0, T1);          \
  DEFINE_UNARY1(F, T2)
#define DEFINE_UNARY4(F, T0, T1, T2, T3) \
  DEFINE_UNARY2(F, T0, T1);              \
  DEFINE_UNARY2(F, T2, T3)
#define DEFINE_UNARY5(F, T0, T1, T2, T3, T4) \
  DEFINE_UNARY2(F, T0, T1);                  \
  DEFINE_UNARY3(F, T2, T3, T4)

// Macros to explicitly instantiate kernels on GPU for multiple types
// (T0, T1, etc.) for BinaryFunctor.
#define DEFINE_BINARY1(F, T)                          \
  template struct BinaryFunctor<SYCLDevice, F<T>, 1>; \
  template struct BinaryFunctor<SYCLDevice, F<T>, 2>; \
  template struct BinaryFunctor<SYCLDevice, F<T>, 3>
#define DEFINE_BINARY2(F, T0, T1) \
  DEFINE_BINARY1(F, T0);          \
  DEFINE_BINARY1(F, T1)
#define DEFINE_BINARY3(F, T0, T1, T2) \
  DEFINE_BINARY2(F, T0, T1);          \
  DEFINE_BINARY1(F, T2)
#define DEFINE_BINARY4(F, T0, T1, T2, T3) \
  DEFINE_BINARY2(F, T0, T1);              \
  DEFINE_BINARY2(F, T2, T3)
#define DEFINE_BINARY5(F, T0, T1, T2, T3, T4) \
  DEFINE_BINARY2(F, T0, T1);                  \
  DEFINE_BINARY3(F, T2, T3, T4)
#define DEFINE_BINARY6(F, T0, T1, T2, T3, T4, T5) \
  DEFINE_BINARY3(F, T0, T1, T2);                  \
  DEFINE_BINARY3(F, T3, T4, T5)
#define DEFINE_BINARY7(F, T0, T1, T2, T3, T4, T5, T6) \
  DEFINE_BINARY3(F, T0, T1, T2);                      \
  DEFINE_BINARY4(F, T3, T4, T5, T6)
#define DEFINE_BINARY8(F, T0, T1, T2, T3, T4, T5, T6, T7) \
  DEFINE_BINARY4(F, T0, T1, T2, T3);                      \
  DEFINE_BINARY4(F, T4, T5, T6, T7)
#define DEFINE_BINARY9(F, T0, T1, T2, T3, T4, T5, T6, T7, T8) \
  DEFINE_BINARY4(F, T0, T1, T2, T3);                          \
  DEFINE_BINARY5(F, T4, T5, T6, T7, T8)
#define DEFINE_BINARY10(F, T0, T1, T2, T3, T4, T5, T6, T7, T8, T9) \
  DEFINE_BINARY5(F, T0, T1, T2, T3, T4);                           \
  DEFINE_BINARY5(F, T5, T6, T7, T8, T9)

}  // end namespace functor
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_CWISE_OPS_SYCL_COMMON_H_
