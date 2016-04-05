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
// See docs in ../ops/nn_ops.cc.

#define EIGEN_USE_THREADS

<<<<<<< HEAD
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/bias_op.h"
#include "tensorflow/core/public/tensor.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
=======
#include "tensorflow/core/kernels/bias_op.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/util/tensor_format.h"

#if GOOGLE_CUDA
#include "tensorflow/core/kernels/bias_op_gpu.h"
#include "tensorflow/core/platform/stream_executor.h"
#endif  // GOOGLE_CUDA
>>>>>>> tensorflow/master

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
<<<<<<< HEAD
class BiasOp : public BinaryOp<T> {
 public:
  explicit BiasOp(OpKernelConstruction* context) : BinaryOp<T>(context) {}
=======
class BiasOp;

template <typename T>
class BiasOp<CPUDevice, T> : public BinaryOp<T> {
 public:
  typedef CPUDevice Device;
  explicit BiasOp(OpKernelConstruction* context) : BinaryOp<T>(context) {
    string data_format;
    if (context->GetAttr("data_format", &data_format).ok()) {
      OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                  errors::InvalidArgument("Invalid data format"));
    } else {
      data_format_ = FORMAT_NHWC;
    }
    OP_REQUIRES(context, data_format_ == FORMAT_NHWC,
                errors::InvalidArgument("CPU BiasOp only suuports NHWC."));
  }
>>>>>>> tensorflow/master

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& bias = context->input(1);

    OP_REQUIRES(context, TensorShapeUtils::IsMatrixOrHigher(input.shape()),
                errors::InvalidArgument("Input tensor must be at least 2D: ",
                                        input.shape().DebugString()));
    OP_REQUIRES(context, TensorShapeUtils::IsVector(bias.shape()),
                errors::InvalidArgument("Biases must be 1D: ",
                                        bias.shape().DebugString()));
    const auto last_dim = input.shape().dims() - 1;
    OP_REQUIRES(
        context, bias.shape().dim_size(0) == input.shape().dim_size(last_dim),
        errors::InvalidArgument(
            "Must provide as many biases as the last dimension "
            "of the input tensor: ",
            bias.shape().DebugString(), " vs. ", input.shape().DebugString()));

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input.shape(), &output));

    switch (input.shape().dims()) {
      case 2:
        Compute<2>(context, input, bias, output);
        break;
      case 3:
        Compute<3>(context, input, bias, output);
        break;
      case 4:
        Compute<4>(context, input, bias, output);
        break;
      case 5:
        Compute<5>(context, input, bias, output);
        break;
      default:
        OP_REQUIRES(context, false,
                    errors::InvalidArgument("Only ranks up to 5 supported: ",
                                            input.shape().DebugString()));
    }
  }

  // Add biases for an input matrix of rank Dims, by using the Bias.
  template <int Dims>
  void Compute(OpKernelContext* ctx, const Tensor& input, const Tensor& bias,
               Tensor* output) {
    functor::Bias<Device, T, Dims> functor;
    functor(ctx->eigen_device<Device>(), input.tensor<T, Dims>(), bias.vec<T>(),
            output->tensor<T, Dims>());
  }
<<<<<<< HEAD
};

#define REGISTER_KERNEL(type)                                       \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("BiasAdd").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
=======

 private:
  TensorFormat data_format_;
};

#define REGISTER_KERNEL(type)                                         \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("BiasAdd").Device(DEVICE_CPU).TypeConstraint<type>("T"),   \
      BiasOp<CPUDevice, type>);                                       \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("BiasAddV1").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
>>>>>>> tensorflow/master
      BiasOp<CPUDevice, type>);

TF_CALL_NUMBER_TYPES(REGISTER_KERNEL);
#undef REGISTER_KERNEL

<<<<<<< HEAD
#if GOOGLE_CUDA
// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC(T, Dims)                                      \
  template <>                                                          \
  void Bias<GPUDevice, T, Dims>::operator()(                           \
      const GPUDevice& d, typename TTypes<T, Dims>::ConstTensor input, \
      typename TTypes<T>::ConstVec bias,                               \
      typename TTypes<T, Dims>::Tensor output);                        \
  extern template struct Bias<GPUDevice, T, Dims>;

#define DECLARE_GPU_SPECS(T) \
  DECLARE_GPU_SPEC(T, 2);    \
  DECLARE_GPU_SPEC(T, 3);    \
  DECLARE_GPU_SPEC(T, 4);    \
  DECLARE_GPU_SPEC(T, 5);

TF_CALL_GPU_NUMBER_TYPES(DECLARE_GPU_SPECS);
}  // namespace functor

// Registration of the GPU implementations.
#define REGISTER_GPU_KERNEL(type)                                   \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("BiasAdd").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      BiasOp<GPUDevice, type>);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_KERNEL);
=======
namespace {

void GetBiasValueDims(const Tensor& value_tensor, TensorFormat data_format,
                      int32* batch, int32* height, int32* width,
                      int32* channel) {
  *batch = 1;
  *width = 1;
  *height = 1;
  *channel = 1;
  if (data_format == FORMAT_NHWC) {
    int32 channel_dim = value_tensor.dims() - 1;
    *channel = static_cast<int32>(value_tensor.dim_size(channel_dim));
    for (int32 i = 0; i < channel_dim; i++) {
      *batch *= static_cast<int32>(value_tensor.dim_size(i));
    }
  } else if (data_format == FORMAT_NCHW) {
    int32 channel_dim = value_tensor.dims() - 3;
    int32 height_dim = value_tensor.dims() - 2;
    int32 width_dim = value_tensor.dims() - 1;
    *channel = static_cast<int32>(value_tensor.dim_size(channel_dim));
    *height = static_cast<int32>(value_tensor.dim_size(height_dim));
    *width = static_cast<int32>(value_tensor.dim_size(width_dim));
    for (int32 i = 0; i < channel_dim; i++) {
      *batch *= static_cast<int32>(value_tensor.dim_size(i));
    }
  }
}

}  // namespace

template <typename Device, typename T>
class BiasGradOp;

template <typename T>
class BiasGradOp<CPUDevice, T> : public OpKernel {
 public:
  typedef CPUDevice Device;
  explicit BiasGradOp(OpKernelConstruction* context) : OpKernel(context) {
    string data_format;
    if (context->GetAttr("data_format", &data_format).ok()) {
      OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                  errors::InvalidArgument("Invalid data format"));
    } else {
      data_format_ = FORMAT_NHWC;
    }
    OP_REQUIRES(context, data_format_ == FORMAT_NHWC,
                errors::InvalidArgument("CPU BiasGradOp only suuports NHWC."));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& output_backprop = context->input(0);

    OP_REQUIRES(context,
                TensorShapeUtils::IsMatrixOrHigher(output_backprop.shape()),
                errors::InvalidArgument("Input tensor must be at least 2D: ",
                                        output_backprop.shape().DebugString()));

    OP_REQUIRES(
        context, FastBoundsCheck(output_backprop.NumElements(),
                                 std::numeric_limits<int32>::max()),
        errors::InvalidArgument("BiasGrad requires tensor size <= int32 max"));

    int32 batch, height, width, channel;
    GetBiasValueDims(output_backprop, data_format_, &batch, &height, &width,
                     &channel);
    Tensor* output = nullptr;
    TensorShape output_shape{channel};
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
    int32 total_count = static_cast<int32>(output_backprop.NumElements());
    int32 bias_size = channel;
    const T* output_backprop_data = output_backprop.template flat<T>().data();
    T* output_data = output->template flat<T>().data();
    memset(output_data, 0, sizeof(T) * bias_size);
    int32 bias_index = 0;
    for (int32 i = 0; i < total_count; i++) {
      output_data[bias_index++] += output_backprop_data[i];
      if (bias_index >= bias_size) {
        bias_index = 0;
      }
    }
  }

 private:
  TensorFormat data_format_;
};

// Registration of the GPU implementations.
#define REGISTER_KERNEL(type)                                           \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("BiasAddGrad").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      BiasGradOp<CPUDevice, type>);

TF_CALL_NUMBER_TYPES(REGISTER_KERNEL);
#undef REGISTER_KERNEL

#if GOOGLE_CUDA
template <typename T>
class BiasOp<GPUDevice, T> : public BinaryOp<T> {
 public:
  typedef GPUDevice Device;
  explicit BiasOp(OpKernelConstruction* context) : BinaryOp<T>(context) {
    string data_format;
    if (context->GetAttr("data_format", &data_format).ok()) {
      OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                  errors::InvalidArgument("Invalid data format"));
    } else {
      data_format_ = FORMAT_NHWC;
    }
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& bias = context->input(1);

    OP_REQUIRES(context, TensorShapeUtils::IsMatrixOrHigher(input.shape()),
                errors::InvalidArgument("Input tensor must be at least 2D: ",
                                        input.shape().DebugString()));
    OP_REQUIRES(context, TensorShapeUtils::IsVector(bias.shape()),
                errors::InvalidArgument("Biases must be 1D: ",
                                        bias.shape().DebugString()));
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input.shape(), &output));
    int32 batch, height, width, channel;
    GetBiasValueDims(input, data_format_, &batch, &height, &width, &channel);
    OP_REQUIRES(context, bias.shape().dim_size(0) == channel,
                errors::InvalidArgument(
                    "Must provide as many biases as the channel dimension "
                    "of the input tensor: ",
                    bias.shape().DebugString(), " vs. ", channel, " in ",
                    input.shape().DebugString()));
    BiasGPU<T>::compute(context->template eigen_device<Device>(),
                        input.flat<T>().data(), bias.flat<T>().data(),
                        output->flat<T>().data(), batch, width, height, channel,
                        data_format_);
  }

 private:
  TensorFormat data_format_;
};

// Registration of the GPU implementations.
#define REGISTER_GPU_KERNEL(type)                                     \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("BiasAdd").Device(DEVICE_GPU).TypeConstraint<type>("T"),   \
      BiasOp<GPUDevice, type>);                                       \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("BiasAddV1").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      BiasOp<GPUDevice, type>);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_KERNEL);
#undef REGISTER_GPU_KERNEL

template <typename T>
class BiasGradOp<GPUDevice, T> : public OpKernel {
 public:
  typedef GPUDevice Device;
  explicit BiasGradOp(OpKernelConstruction* context) : OpKernel(context) {
    string data_format;
    if (context->GetAttr("data_format", &data_format).ok()) {
      OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                  errors::InvalidArgument("Invalid data format"));
    } else {
      data_format_ = FORMAT_NCHW;
    }
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& output_backprop = context->input(0);

    OP_REQUIRES(context,
                TensorShapeUtils::IsMatrixOrHigher(output_backprop.shape()),
                errors::InvalidArgument("Input tensor must be at least 2D: ",
                                        output_backprop.shape().DebugString()));
    int32 batch, height, width, channel;
    GetBiasValueDims(output_backprop, data_format_, &batch, &height, &width,
                     &channel);
    Tensor* output = nullptr;
    TensorShape output_shape{channel};
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
    auto* stream = context->op_device_context()->stream();
    OP_REQUIRES(context, stream, errors::Internal("No GPU stream available."));
    perftools::gputools::DeviceMemoryBase output_ptr(
        output->flat<T>().data(), output->NumElements() * sizeof(T));
    stream->ThenMemset32(&output_ptr, 0, output->NumElements() * sizeof(T));
    BiasGradGPU<T>::compute(context->template eigen_device<Device>(),
                            output_backprop.template flat<T>().data(),
                            output->flat<T>().data(), batch, width, height,
                            channel, data_format_);
  }

 private:
  TensorFormat data_format_;
};

// Registration of the GPU implementations.
#define REGISTER_GPU_KERNEL(type)                                       \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("BiasAddGrad").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      BiasGradOp<GPUDevice, type>);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_KERNEL);
#undef REGISTER_GPU_KERNEL
>>>>>>> tensorflow/master

#endif  // GOOGLE_CUDA

}  // namespace tensorflow
