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
// See docs in ../ops/image_ops.cc

#include <memory>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
<<<<<<< HEAD
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/public/status.h"
#include "tensorflow/core/public/tensor.h"
#include "tensorflow/core/public/tensor_shape.h"
#include "tensorflow/core/lib/png/png_io.h"
=======
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/png/png_io.h"
#include "tensorflow/core/platform/logging.h"
>>>>>>> tensorflow/master

namespace tensorflow {

// Encode an image to a PNG stream
class EncodePngOp : public OpKernel {
 public:
  explicit EncodePngOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("compression", &compression_));
    OP_REQUIRES(context, -1 <= compression_ && compression_ <= 9,
                errors::InvalidArgument("compression should be in [-1,9], got ",
                                        compression_));
<<<<<<< HEAD
=======

    DataType dt = context->input_type(0);
    OP_REQUIRES(context, dt == DataType::DT_UINT8 || dt == DataType::DT_UINT16,
                errors::InvalidArgument(
                    "image must have type uint8 or uint16, got ", dt));

    if (dt == DataType::DT_UINT8) {
      desired_channel_bits_ = 8;
    } else {
      desired_channel_bits_ = 16;
    }
>>>>>>> tensorflow/master
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& image = context->input(0);
    OP_REQUIRES(context, image.dims() == 3,
                errors::InvalidArgument("image must be 3-dimensional",
<<<<<<< HEAD
                                        image.shape().ShortDebugString()));
    const int64 channels = image.dim_size(2);
    OP_REQUIRES(context, channels == 1 || channels == 3 || channels == 4,
                errors::InvalidArgument(
                    "image must have 1, 3, or 4 channels, got ", channels));
=======
                                        image.shape().DebugString()));
    const int64 channels = image.dim_size(2);
    OP_REQUIRES(context, channels >= 1 && channels <= 4,
                errors::InvalidArgument(
                    "image must have 1, 2, 3, or 4 channels, got ", channels));
>>>>>>> tensorflow/master

    // Encode image to png string
    Tensor* output = NULL;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape({}), &output));
<<<<<<< HEAD
    OP_REQUIRES(context,
                png::WriteImageToBuffer(
                    image.flat<uint8>().data(), image.dim_size(1),
                    image.dim_size(0), image.dim_size(1) * channels, channels,
                    8, compression_, &output->scalar<string>()(), nullptr),
                errors::Internal("PNG encoding failed"));
=======
    if (desired_channel_bits_ == 8) {
      OP_REQUIRES(context, png::WriteImageToBuffer(
                               image.flat<uint8>().data(), image.dim_size(1),
                               image.dim_size(0), image.dim_size(1) * channels,
                               channels, desired_channel_bits_, compression_,
                               &output->scalar<string>()(), nullptr),
                  errors::Internal("PNG encoding failed"));
    } else {
      OP_REQUIRES(
          context,
          png::WriteImageToBuffer(
              image.flat<uint16>().data(), image.dim_size(1), image.dim_size(0),
              image.dim_size(1) * channels * 2, channels, desired_channel_bits_,
              compression_, &output->scalar<string>()(), nullptr),
          errors::Internal("PNG encoding failed"));
    }
>>>>>>> tensorflow/master
  }

 private:
  int compression_;
<<<<<<< HEAD
=======
  int desired_channel_bits_;
>>>>>>> tensorflow/master
};
REGISTER_KERNEL_BUILDER(Name("EncodePng").Device(DEVICE_CPU), EncodePngOp);

}  // namespace tensorflow
