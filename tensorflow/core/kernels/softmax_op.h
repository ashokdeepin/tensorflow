<<<<<<< HEAD
#ifndef TENSORFLOW_KERNELS_SOFTMAX_OP_H_
#define TENSORFLOW_KERNELS_SOFTMAX_OP_H_
// Functor definition for SoftmaxOp, must be compilable by nvcc.

#include "tensorflow/core/framework/tensor_types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {
namespace functor {

// Functor used by SoftmaxOp to do the computations.
template <typename Device, typename T>
struct SoftmaxFunctor {
  // Computes Softmax activation.
  //
  // logits: dim: batch_size, num_classes.
  // softmax: dims: batch_size, num_classes.
  void operator()(const Device& d, typename TTypes<T>::ConstMatrix logits,
                  typename TTypes<T>::Matrix softmax);
};

// Eigen code implementing SoftmaxFunctor::operator().
// This code works for both CPU and GPU and is used by the functor
// specializations for both device types.
template <typename Device, typename T>
struct SoftmaxEigenImpl {
  static void Compute(const Device& d, typename TTypes<T>::ConstMatrix logits,
                      typename TTypes<T>::Matrix softmax) {
    const int kBatchDim = 0;
    const int kClassDim = 1;

    const int batch_size = logits.dimension(kBatchDim);
    const int num_classes = logits.dimension(kClassDim);

// These arrays are used to reduce along the class dimension, and broadcast
// the resulting value to all classes.
#if !defined(EIGEN_HAS_INDEX_LIST)
    Eigen::DSizes<int, 1> along_class(kClassDim);
    Eigen::DSizes<int, 2> batch_by_one(batch_size, 1);
    Eigen::DSizes<int, 2> one_by_class(1, num_classes);
#else
    Eigen::IndexList<Eigen::type2index<kClassDim> > along_class;
    Eigen::IndexList<Eigen::type2index<1> > depth_dim;
    Eigen::IndexList<int, Eigen::type2index<1> > batch_by_one;
    batch_by_one.set(0, batch_size);
    Eigen::IndexList<Eigen::type2index<1>, int> one_by_class;
    one_by_class.set(1, num_classes);
#endif
    // NOTE(touts): If you modify this implementation please run
    // the ImageNetSoftmaxFwd benchmark in core_ops_test.cc.
    //
    // softmax = exp(logits - max(logits along classes));
    softmax.device(d) = (logits -
                         logits.maximum(along_class)
                             .eval()
                             .reshape(batch_by_one)
                             .broadcast(one_by_class)).exp();
    // softmax = softmax / sum(softmax along classes);
    softmax.device(d) = (softmax /
                         softmax.sum(along_class)
                             .eval()
                             .reshape(batch_by_one)
                             .broadcast(one_by_class));
  }
};

}  // namespace functor
}  // namespace tensorflow

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

// See docs in ../ops/nn_ops.cc.

#ifndef TENSORFLOW_KERNELS_SOFTMAX_OP_H_
#define TENSORFLOW_KERNELS_SOFTMAX_OP_H_

#define EIGEN_USE_THREADS

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/softmax_op_functor.h"

namespace tensorflow {

template <typename Device, typename T>
class SoftmaxOp : public OpKernel {
 public:
  explicit SoftmaxOp(OpKernelConstruction* context) : OpKernel(context) {
    log_ = StringPiece(name()).starts_with("Log");
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& logits_in = context->input(0);
    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(logits_in.shape()),
                errors::InvalidArgument("logits must be 2-dimensional"));
    Tensor* softmax_out = nullptr;
    OP_REQUIRES_OK(
        context, context->allocate_output(0, logits_in.shape(), &softmax_out));
    if (logits_in.NumElements()) {
      functor::SoftmaxFunctor<Device, T> functor;
      functor(context->eigen_device<Device>(), logits_in.matrix<T>(),
              softmax_out->matrix<T>(), log_);
    }
  }

 private:
  bool log_;
};

}  // namespace tensorflow

#undef EIGEN_USE_THREADS

>>>>>>> tensorflow/master
#endif  // TENSORFLOW_KERNELS_SOFTMAX_OP_H_
