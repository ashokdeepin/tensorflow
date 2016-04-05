<<<<<<< HEAD
#include "tensorflow/core/framework/allocator.h"
#include <gtest/gtest.h>
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

#include <vector>
#include "tensorflow/core/framework/allocator.h"
>>>>>>> tensorflow/master
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
<<<<<<< HEAD
=======
#include "tensorflow/core/framework/tensor.h"
>>>>>>> tensorflow/master
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
<<<<<<< HEAD
#include "tensorflow/core/public/tensor.h"
=======
#include "tensorflow/core/platform/test.h"
>>>>>>> tensorflow/master

namespace tensorflow {

class AdjustContrastOpTest : public OpsTestBase {
<<<<<<< HEAD
 protected:
  void MakeOp() { RequireDefaultOps(); }
};

TEST_F(AdjustContrastOpTest, Simple_1113) {
  RequireDefaultOps();
  EXPECT_OK(NodeDefBuilder("adjust_constrast_op", "AdjustContrast")
                .Input(FakeInput(DT_FLOAT))
                .Input(FakeInput(DT_FLOAT))
                .Input(FakeInput(DT_FLOAT))
                .Input(FakeInput(DT_FLOAT))
                .Attr("T", DT_FLOAT)
                .Finalize(node_def()));
  EXPECT_OK(InitOp());
  AddInputFromArray<float>(TensorShape({1, 1, 1, 3}), {-1, 2, 3});
  AddInputFromArray<float>(TensorShape({}), {1.0});
  AddInputFromArray<float>(TensorShape({}), {0.0});
  AddInputFromArray<float>(TensorShape({}), {2.0});
  ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 1, 1, 3}));
  test::FillValues<float>(&expected, {0, 2, 2});
=======
};

TEST_F(AdjustContrastOpTest, Simple_1113) {
  TF_EXPECT_OK(NodeDefBuilder("adjust_constrast_op", "AdjustContrastv2")
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Finalize(node_def()));
  TF_EXPECT_OK(InitOp());
  AddInputFromArray<float>(TensorShape({1, 1, 1, 3}), {-1, 2, 3});
  AddInputFromArray<float>(TensorShape({}), {1.0});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 1, 1, 3}));
  test::FillValues<float>(&expected, {-1, 2, 3});
>>>>>>> tensorflow/master
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(AdjustContrastOpTest, Simple_1223) {
<<<<<<< HEAD
  RequireDefaultOps();
  EXPECT_OK(NodeDefBuilder("adjust_constrast_op", "AdjustContrast")
                .Input(FakeInput(DT_FLOAT))
                .Input(FakeInput(DT_FLOAT))
                .Input(FakeInput(DT_FLOAT))
                .Input(FakeInput(DT_FLOAT))
                .Attr("T", DT_FLOAT)
                .Finalize(node_def()));
  EXPECT_OK(InitOp());
  AddInputFromArray<float>(TensorShape({1, 2, 2, 3}),
                           {1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12});
  AddInputFromArray<float>(TensorShape({}), {0.2});
  AddInputFromArray<float>(TensorShape({}), {0.0});
  AddInputFromArray<float>(TensorShape({}), {10.0});
  ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 2, 2, 3}));
  test::FillValues<float>(
      &expected, {2.2, 6.2, 10, 2.4, 6.4, 10, 2.6, 6.6, 10, 2.8, 6.8, 10});
=======
  TF_EXPECT_OK(NodeDefBuilder("adjust_constrast_op", "AdjustContrastv2")
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Finalize(node_def()));
  TF_EXPECT_OK(InitOp());
  AddInputFromArray<float>(TensorShape({1, 2, 2, 3}),
                           {1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12});
  AddInputFromArray<float>(TensorShape({}), {0.2});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 2, 2, 3}));
  test::FillValues<float>(&expected, {2.2, 6.2, 10.2, 2.4, 6.4, 10.4, 2.6, 6.6,
                                      10.6, 2.8, 6.8, 10.8});
>>>>>>> tensorflow/master
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(AdjustContrastOpTest, Big_99x99x3) {
<<<<<<< HEAD
  EXPECT_OK(NodeDefBuilder("adjust_constrast_op", "AdjustContrast")
                .Input(FakeInput(DT_FLOAT))
                .Input(FakeInput(DT_FLOAT))
                .Input(FakeInput(DT_FLOAT))
                .Input(FakeInput(DT_FLOAT))
                .Attr("T", DT_FLOAT)
                .Finalize(node_def()));
  EXPECT_OK(InitOp());
=======
  TF_EXPECT_OK(NodeDefBuilder("adjust_constrast_op", "AdjustContrastv2")
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Finalize(node_def()));
  TF_EXPECT_OK(InitOp());
>>>>>>> tensorflow/master

  std::vector<float> values;
  for (int i = 0; i < 99 * 99 * 3; ++i) {
    values.push_back(i % 255);
  }

  AddInputFromArray<float>(TensorShape({1, 99, 99, 3}), values);
  AddInputFromArray<float>(TensorShape({}), {0.2});
<<<<<<< HEAD
  AddInputFromArray<float>(TensorShape({}), {0});
  AddInputFromArray<float>(TensorShape({}), {255});
  ASSERT_OK(RunOpKernel());
=======
  TF_ASSERT_OK(RunOpKernel());
>>>>>>> tensorflow/master
}

}  // namespace tensorflow
