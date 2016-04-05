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
// See docs in ../ops/array_ops.cc.
#include "tensorflow/core/kernels/identity_op.h"

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
<<<<<<< HEAD
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/public/tensor.h"
=======
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
>>>>>>> tensorflow/master

namespace tensorflow {

REGISTER_KERNEL_BUILDER(Name("Identity").Device(DEVICE_CPU), IdentityOp);
// StopGradient does the same thing as Identity, but has a different
// gradient registered.
REGISTER_KERNEL_BUILDER(Name("StopGradient").Device(DEVICE_CPU), IdentityOp);
<<<<<<< HEAD
=======
// PlaceholderWithDefault does the same thing as Identity, but has a
// different shape function (and constant value function) registered.
REGISTER_KERNEL_BUILDER(Name("PlaceholderWithDefault").Device(DEVICE_CPU),
                        IdentityOp);
>>>>>>> tensorflow/master

REGISTER_KERNEL_BUILDER(Name("RefIdentity").Device(DEVICE_CPU), IdentityOp);

#define REGISTER_GPU_KERNEL(type)                                        \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("Identity").Device(DEVICE_GPU).TypeConstraint<type>("T"),     \
      IdentityOp);                                                       \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("RefIdentity").Device(DEVICE_GPU).TypeConstraint<type>("T"),  \
      IdentityOp);                                                       \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("StopGradient").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      IdentityOp)

TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_GPU_KERNEL);
<<<<<<< HEAD
REGISTER_GPU_KERNEL(bool);
=======
>>>>>>> tensorflow/master
REGISTER_GPU_KERNEL(bfloat16);

#undef REGISTER_GPU_KERNEL

<<<<<<< HEAD
// A special GPU kernel for int32.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
REGISTER_KERNEL_BUILDER(Name("Identity")
                            .Device(DEVICE_GPU)
                            .HostMemory("input")
                            .HostMemory("output")
                            .TypeConstraint<int32>("T"),
                        IdentityOp);
=======
#if GOOGLE_CUDA
// A special GPU kernel for int32 and bool.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
#define REGISTER_GPU_HOST_KERNEL(type)                    \
  REGISTER_KERNEL_BUILDER(Name("Identity")                \
                              .Device(DEVICE_GPU)         \
                              .HostMemory("input")        \
                              .HostMemory("output")       \
                              .TypeConstraint<type>("T"), \
                          IdentityOp);                    \
  REGISTER_KERNEL_BUILDER(Name("RefIdentity")             \
                              .Device(DEVICE_GPU)         \
                              .HostMemory("input")        \
                              .HostMemory("output")       \
                              .TypeConstraint<type>("T"), \
                          IdentityOp)

REGISTER_GPU_HOST_KERNEL(int32);
REGISTER_GPU_HOST_KERNEL(bool);

#undef REGISTER_GPU_HOST_KERNEL

#endif
>>>>>>> tensorflow/master

}  // namespace tensorflow
