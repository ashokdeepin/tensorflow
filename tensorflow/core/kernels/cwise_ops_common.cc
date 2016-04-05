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
#include "tensorflow/core/kernels/cwise_ops_common.h"

namespace tensorflow {

BinaryOpShared::BinaryOpShared(OpKernelConstruction* ctx, DataType out,
                               DataType in)
    : OpKernel(ctx) {
  OP_REQUIRES_OK(ctx, ctx->MatchSignature({in, in}, {out}));
}

void BinaryOpShared::SetUnimplementedError(OpKernelContext* ctx) {
  ctx->SetStatus(errors::Unimplemented(
<<<<<<< HEAD
      "Broadcast between ", ctx->input(0).shape().ShortDebugString(), " and ",
      ctx->input(1).shape().ShortDebugString(), " is not supported yet."));
}

static BCast::Vec FromShape(const TensorShape& shape) {
  BCast::Vec ret;
  for (int i = 0; i < shape.dims(); ++i) ret.push_back(shape.dim_size(i));
=======
      "Broadcast between ", ctx->input(0).shape().DebugString(), " and ",
      ctx->input(1).shape().DebugString(), " is not supported yet."));
}

static BCast::Vec FromShape(const TensorShape& shape) {
  const int N = shape.dims();
  BCast::Vec ret(N);
  for (int i = 0; i < N; ++i) {
    ret[i] = shape.dim_size(i);
  }
>>>>>>> tensorflow/master
  return ret;
}

static TensorShape ToShape(const BCast::Vec& vec) {
<<<<<<< HEAD
  TensorShape shape;
  for (auto elem : vec) shape.AddDim(elem);
=======
  TensorShape shape(vec);
>>>>>>> tensorflow/master
  return shape;
}

BinaryOpShared::BinaryOpState::BinaryOpState(OpKernelContext* ctx)
<<<<<<< HEAD
    : bcast(FromShape(ctx->input(0).shape()),
            FromShape(ctx->input(1).shape())) {
  if (!bcast.IsValid()) {
    ctx->SetStatus(errors::InvalidArgument(
        "Incompatible shapes: ", ctx->input(0).shape().ShortDebugString(),
        " vs. ", ctx->input(1).shape().ShortDebugString()));
=======
    : in0(ctx->input(0)),
      in1(ctx->input(1)),
      bcast(FromShape(in0.shape()), FromShape(in1.shape())) {
  if (!bcast.IsValid()) {
    ctx->SetStatus(errors::InvalidArgument("Incompatible shapes: ",
                                           in0.shape().DebugString(), " vs. ",
                                           in1.shape().DebugString()));
>>>>>>> tensorflow/master
    return;
  }
  OP_REQUIRES_OK(ctx,
                 ctx->allocate_output(0, ToShape(bcast.output_shape()), &out));
<<<<<<< HEAD
=======
  out_num_elements = out->NumElements();
  in0_num_elements = in0.NumElements();
  in1_num_elements = in1.NumElements();

  ndims = bcast.x_reshape().size();
>>>>>>> tensorflow/master
}

}  // namespace tensorflow
