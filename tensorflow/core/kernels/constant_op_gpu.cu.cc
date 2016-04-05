<<<<<<< HEAD
=======
/* Copyright 2015 Google Inc. All Rights Reserved.

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

>>>>>>> tensorflow/master
#if GOOGLE_CUDA

#define EIGEN_USE_GPU

<<<<<<< HEAD
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/platform/port.h"
=======
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/platform/types.h"
>>>>>>> tensorflow/master

namespace Eigen {
namespace internal {

template <typename T>
struct scalar_const_op {
  typedef typename packet_traits<T>::type Packet;

  const T* val;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  scalar_const_op(const scalar_const_op& x)
      : val(x.val) {}

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE scalar_const_op(const T* v) : val(v) {}

  template <typename Index>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const T operator()(Index,
                                                           Index = 0) const {
    return *val;
  }

<<<<<<< HEAD
  template <typename Index>
  EIGEN_STRONG_INLINE const Packet packetOp(Index, Index = 0) const {
    return internal::pset1<Packet>(*val);
=======
  template <typename Index, typename PacketType = Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const PacketType
      packetOp(Index, Index = 0) const {
    return internal::pset1<PacketType>(*val);
>>>>>>> tensorflow/master
  }
};

template <typename T>
struct functor_traits<scalar_const_op<T> > {
  enum {
    Cost = 1,
    PacketAccess = packet_traits<T>::Vectorizable,
    IsRepeatable = true
  };
};

}  // end namespace internal
}  // end namespace Eigen

namespace tensorflow {

namespace functor {

typedef Eigen::GpuDevice GPUDevice;

// Partial specialization FillFunctor<Device=GPUDevice, T>
template <typename T>
struct FillFunctor<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T>::Flat out,
                  typename TTypes<T>::ConstScalar in) {
    Eigen::internal::scalar_const_op<T> f(in.data());
<<<<<<< HEAD
    out.device(d) = out.nullaryExpr(f);
=======
    To32Bit(out).device(d) = To32Bit(out).nullaryExpr(f);
>>>>>>> tensorflow/master
  }
};

#define DEFINE_FILL_GPU(T) template struct FillFunctor<GPUDevice, T>
<<<<<<< HEAD
DEFINE_FILL_GPU(float);
DEFINE_FILL_GPU(double);
DEFINE_FILL_GPU(int32);
DEFINE_FILL_GPU(uint8);
DEFINE_FILL_GPU(int16);
DEFINE_FILL_GPU(int8);
DEFINE_FILL_GPU(int64);
=======
TF_CALL_REAL_NUMBER_TYPES(DEFINE_FILL_GPU);
DEFINE_FILL_GPU(bool);
DEFINE_FILL_GPU(Eigen::half);
>>>>>>> tensorflow/master
#undef DEFINE_FILL_GPU

// Partial specialization of FillFunctor<Device=GPUDevice, T>.
template <typename T>
struct SetZeroFunctor<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T>::Flat out) {
<<<<<<< HEAD
    out.device(d) = out.constant(0);
=======
    To32Bit(out).device(d) = To32Bit(out).constant(T(0));
>>>>>>> tensorflow/master
  }
};

#define DEFINE_SETZERO_GPU(T) template struct SetZeroFunctor<GPUDevice, T>
<<<<<<< HEAD
DEFINE_SETZERO_GPU(float);
=======
DEFINE_SETZERO_GPU(Eigen::half);
DEFINE_SETZERO_GPU(float);
DEFINE_SETZERO_GPU(double);
>>>>>>> tensorflow/master
#undef DEFINE_SETZERO_GPU

}  // end namespace functor
}  // end namespace tensorflow

#endif  // GOOGLE_CUDA
