<<<<<<< HEAD
// See docs in ../ops/linalg_ops.cc.
#include <cmath>

#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/kernels/linalg_ops_common.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/port.h"
#include "tensorflow/core/public/tensor_shape.h"
#include "third_party/eigen3/Eigen/LU"
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

// See docs in ../ops/linalg_ops.cc.
#include <cmath>

#include "third_party/eigen3/Eigen/Cholesky"
#include "third_party/eigen3/Eigen/LU"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/linalg_ops_common.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"
>>>>>>> tensorflow/master

namespace tensorflow {

template <class Scalar, bool SupportsBatchOperationT>
class MatrixInverseOp
<<<<<<< HEAD
    : public LinearAlgebraOp<Scalar, SupportsBatchOperationT> {
 public:
  explicit MatrixInverseOp(OpKernelConstruction* context)
      : LinearAlgebraOp<Scalar, SupportsBatchOperationT>(context) {}
=======
    : public UnaryLinearAlgebraOp<Scalar, SupportsBatchOperationT> {
 public:
  explicit MatrixInverseOp(OpKernelConstruction* context)
      : UnaryLinearAlgebraOp<Scalar, SupportsBatchOperationT>(context) {}
>>>>>>> tensorflow/master
  ~MatrixInverseOp() override {}

  TensorShape GetOutputMatrixShape(
      const TensorShape& input_matrix_shape) override {
    return input_matrix_shape;
  }

  int64 GetCostPerUnit(const TensorShape& input_matrix_shape) override {
    const int64 rows = input_matrix_shape.dim_size(0);
    if (rows > (1LL << 20)) {
      // A big number to cap the cost in case overflow.
<<<<<<< HEAD
      return kint32max;
=======
      return kint64max;
>>>>>>> tensorflow/master
    } else {
      return rows * rows * rows;
    }
  }

<<<<<<< HEAD
  using typename LinearAlgebraOp<Scalar, SupportsBatchOperationT>::MatrixMap;
  using
      typename LinearAlgebraOp<Scalar, SupportsBatchOperationT>::ConstMatrixMap;
=======
  typedef UnaryLinearAlgebraOp<Scalar, SupportsBatchOperationT> Base;
  using Matrix = typename Base::Matrix;
  using MatrixMap = typename Base::MatrixMap;
  using ConstMatrixMap = typename Base::ConstMatrixMap;
>>>>>>> tensorflow/master

  void ComputeMatrix(OpKernelContext* context, const ConstMatrixMap& input,
                     MatrixMap* output) override {
    OP_REQUIRES(context, input.rows() == input.cols(),
                errors::InvalidArgument("Input matrix must be square."));
    if (input.rows() == 0) {
<<<<<<< HEAD
      // By definition, an empty matrix's inverse is an emptry matrix.
      return;
    }
    Eigen::FullPivLU<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic,
                                   Eigen::RowMajor>> lu_decomposition(input);
    OP_REQUIRES(context, lu_decomposition.isInvertible(),
                errors::InvalidArgument("Input is not invertible."));
    *output = lu_decomposition.inverse();
  }
=======
      // By definition, an empty matrix's inverse is an empty matrix.
      return;
    }
    Eigen::PartialPivLU<Matrix> lu_decomposition(input);
    // TODO(rmlarsen): Add check based on condition number estimation.
    // PartialPivLU cannot give strong guarantees on invertibility, but
    // we can at least guard against exact zero pivots. This can occur as
    // a result of basic user mistakes, such as providing integer valued
    // matrices that are exactly singular, or due to underflow if this
    // code is run with denormals being flushed to zero.
    const Scalar min_abs_pivot =
        lu_decomposition.matrixLU().diagonal().cwiseAbs().minCoeff();
    OP_REQUIRES(context, min_abs_pivot > Scalar(0),
                errors::InvalidArgument("Input is not invertible."));
    output->noalias() = lu_decomposition.inverse();
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(MatrixInverseOp);
>>>>>>> tensorflow/master
};

REGISTER_LINALG_OP("MatrixInverse", (MatrixInverseOp<float, false>), float);
REGISTER_LINALG_OP("MatrixInverse", (MatrixInverseOp<double, false>), double);
REGISTER_LINALG_OP("BatchMatrixInverse", (MatrixInverseOp<float, true>), float);
REGISTER_LINALG_OP("BatchMatrixInverse", (MatrixInverseOp<double, true>),
                   double);

}  // namespace tensorflow
