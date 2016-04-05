<<<<<<< HEAD
// See docs in ../ops/image_ops.cc
#define EIGEN_USE_THREADS

#include <algorithm>
#include <memory>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/public/status.h"
#include "tensorflow/core/public/tensor.h"
#include "tensorflow/core/public/tensor_shape.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {
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

// See docs in ../ops/image_ops.cc
#define EIGEN_USE_THREADS

#include <math.h>
#include <algorithm>
#include <array>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/image_resizer_state.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace {

static const int64 kTableSize = (1 << 10);

const float* InitCoeffsTable() {
  // Allocate and initialize coefficients table using Bicubic
  // convolution algorithm.
  // https://en.wikipedia.org/wiki/Bicubic_interpolation
  float* coeffs_tab = new float[(kTableSize + 1) * 2];
  static const double A = -0.75;
  for (int i = 0; i <= kTableSize; ++i) {
    float x = i * 1.0 / kTableSize;
    coeffs_tab[i * 2] = ((A + 2) * x - (A + 3)) * x * x + 1;
    x += 1.0;
    coeffs_tab[i * 2 + 1] = ((A * x - 5 * A) * x + 8 * A) * x - 4 * A;
  }
  return coeffs_tab;
}

const float* GetCoeffsTable() {
  // Static so that we initialize it on first use
  static const float* coeffs_tab = InitCoeffsTable();
  return coeffs_tab;
}

inline int64 Bound(int64 val, int64 limit) {
  return std::min(limit - 1ll, std::max(0ll, val));
}

inline void GetWeightsAndIndices(float scale, int64 out_loc, int64 limit,
                                 std::array<float, 4>* weights,
                                 std::array<int64, 4>* indices) {
  const int64 in_loc = scale * out_loc;
  const float delta = scale * out_loc - in_loc;
  const int64 offset = lrintf(delta * kTableSize);
  const float* coeffs_tab = GetCoeffsTable();
  *weights = {{coeffs_tab[offset * 2 + 1], coeffs_tab[offset * 2],
               coeffs_tab[(kTableSize - offset) * 2],
               coeffs_tab[(kTableSize - offset) * 2 + 1]}};
  *indices = {{Bound(in_loc - 1, limit), Bound(in_loc, limit),
               Bound(in_loc + 1, limit), Bound(in_loc + 2, limit)}};
}

inline float Interpolate1D(const std::array<float, 4>& weights,
                           const std::array<float, 4>& values) {
  return values[0] * weights[0] + values[1] * weights[1] +
         values[2] * weights[2] + values[3] * weights[3];
}

}  // namespace
>>>>>>> tensorflow/master

typedef Eigen::ThreadPoolDevice CPUDevice;

template <typename Device, typename T>
class ResizeBicubicOp : public OpKernel {
 public:
<<<<<<< HEAD
  explicit ResizeBicubicOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    OP_REQUIRES(context, input.dims() == 4,
                errors::InvalidArgument("input must be 4-dimensional",
                                        input.shape().ShortDebugString()));
    const Tensor& shape_t = context->input(1);
    OP_REQUIRES(context, shape_t.dims() == 1,
                errors::InvalidArgument("shape_t must be 1-dimensional",
                                        shape_t.shape().ShortDebugString()));
    OP_REQUIRES(context, shape_t.NumElements() == 2,
                errors::InvalidArgument("shape_t must have two elements",
                                        shape_t.shape().ShortDebugString()));

    auto Svec = shape_t.vec<int32>();
    // Initialize shape to the batch size of the input, then add
    // the rest of the dimensions
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
                                0, TensorShape({input.dim_size(0), Svec(0),
                                                Svec(1), input.dim_size(3)}),
                                &output));
    const int64 batch_size = input.dim_size(0);
    const int64 in_height = input.dim_size(1);
    const int64 in_width = input.dim_size(2);
    const int64 channels = input.dim_size(3);
    const int64 out_height = output->dim_size(1);
    const int64 out_width = output->dim_size(2);

    typename TTypes<T, 4>::ConstTensor input_data = input.tensor<T, 4>();
    typename TTypes<float, 4>::Tensor output_data = output->tensor<float, 4>();

    const float height_scale = in_height / static_cast<float>(out_height);
    const float width_scale = in_width / static_cast<float>(out_width);

    // Initialize coefficients table using Bicubic convolution algorithm.
    // https://en.wikipedia.org/wiki/Bicubic_interpolation
    static const int64 tab_size = (1 << 10);
    static float coeffs_tab[(tab_size + 1) * 2];
    static const double A = -0.75;
    for (int i = 0; i <= tab_size; ++i) {
      float x = i * 1.0 / tab_size;
      coeffs_tab[i * 2] = ((A + 2) * x - (A + 3)) * x * x + 1;
      x += 1.0;
      coeffs_tab[i * 2 + 1] = ((A * x - 5 * A) * x + 8 * A) * x - 4 * A;
    }

    auto cal = [](float v0, float v1, float v2, float v3, float dx) {
      const int64 offset = round(dx * tab_size);
      const float a0 = coeffs_tab[offset * 2 + 1];
      const float a1 = coeffs_tab[offset * 2];
      const float a2 = coeffs_tab[(tab_size - offset) * 2];
      const float a3 = coeffs_tab[(tab_size - offset) * 2 + 1];
      return a0 * v0 + a1 * v1 + a2 * v2 + a3 * v3;
    };

    float coeff[4] = {0.0};
    for (int64 b = 0; b < batch_size; ++b) {
      for (int64 y = 0; y < out_height; ++y) {
        const int64 in_y = floor(height_scale * y);
        const float dy = height_scale * y - in_y;
        for (int64 x = 0; x < out_width; ++x) {
          const int64 in_x = floor(width_scale * x);
          const float dx = width_scale * x - in_x;
          for (int64 c = 0; c < channels; ++c) {
            for (int64 i = 0; i < 4; ++i) {
#define BOUND(val, limit) std::min(((limit)-1ll), (std::max(0ll, (val))))
              int64 bound_y = BOUND(in_y - 1 + i, in_height);
              coeff[i] =
                  cal(input_data(b, bound_y, BOUND(in_x - 1, in_width), c),
                      input_data(b, bound_y, BOUND(in_x, in_width), c),
                      input_data(b, bound_y, BOUND(in_x + 1, in_width), c),
                      input_data(b, bound_y, BOUND(in_x + 2, in_width), c), dx);
#undef BOUND
            }
            output_data(b, y, x, c) =
                cal(coeff[0], coeff[1], coeff[2], coeff[3], dy);
=======
  explicit ResizeBicubicOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("align_corners", &align_corners_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    ImageResizerState st(align_corners_);
    st.ValidateAndCreateOutput(context, input);

    if (!context->status().ok()) return;

    typename TTypes<T, 4>::ConstTensor input_data = input.tensor<T, 4>();
    typename TTypes<float, 4>::Tensor output_data =
        st.output->tensor<float, 4>();

    std::array<float, 4> coeff = {{0.0, 0.0, 0.0, 0.0}};
    for (int64 b = 0; b < st.batch_size; ++b) {
      for (int64 y = 0; y < st.out_height; ++y) {
        std::array<float, 4> y_weights;
        std::array<int64, 4> y_indices;
        GetWeightsAndIndices(st.height_scale, y, st.in_height, &y_weights,
                             &y_indices);
        for (int64 x = 0; x < st.out_width; ++x) {
          std::array<float, 4> x_weights;
          std::array<int64, 4> x_indices;
          GetWeightsAndIndices(st.width_scale, x, st.in_width, &x_weights,
                               &x_indices);
          for (int64 c = 0; c < st.channels; ++c) {
            // Use a 4x4 patch to compute the interpolated output value at
            // (b, y, x, c).
            for (int64 i = 0; i < 4; ++i) {
              const std::array<float, 4> values = {
                  {static_cast<float>(
                       input_data(b, y_indices[i], x_indices[0], c)),
                   static_cast<float>(
                       input_data(b, y_indices[i], x_indices[1], c)),
                   static_cast<float>(
                       input_data(b, y_indices[i], x_indices[2], c)),
                   static_cast<float>(
                       input_data(b, y_indices[i], x_indices[3], c))}};
              coeff[i] = Interpolate1D(x_weights, values);
            }
            output_data(b, y, x, c) = Interpolate1D(y_weights, coeff);
>>>>>>> tensorflow/master
          }
        }
      }
    }
  }
<<<<<<< HEAD
=======

 private:
  bool align_corners_;
>>>>>>> tensorflow/master
};

#define REGISTER_KERNEL(T)                            \
  REGISTER_KERNEL_BUILDER(Name("ResizeBicubic")       \
                              .Device(DEVICE_CPU)     \
                              .TypeConstraint<T>("T") \
                              .HostMemory("size"),    \
                          ResizeBicubicOp<CPUDevice, T>);

<<<<<<< HEAD
REGISTER_KERNEL(uint8);
REGISTER_KERNEL(int8);
REGISTER_KERNEL(int32);
REGISTER_KERNEL(float);
REGISTER_KERNEL(double);
=======
TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNEL);

>>>>>>> tensorflow/master
#undef REGISTER_KERNEL

}  // namespace tensorflow
