<<<<<<< HEAD
"""Tests for SoftmaxOp."""
=======
# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Tests for SoftmaxOp and LogSoftmaxOp."""
>>>>>>> tensorflow/master
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

<<<<<<< HEAD
import tensorflow.python.platform
=======
import sys
>>>>>>> tensorflow/master

import numpy as np
import tensorflow as tf


class SoftmaxTest(tf.test.TestCase):

<<<<<<< HEAD
  def _npSoftmax(self, features):
=======
  def _npSoftmax(self, features, log=False):
>>>>>>> tensorflow/master
    batch_dim = 0
    class_dim = 1
    batch_size = features.shape[batch_dim]
    e = np.exp(features -
               np.reshape(np.amax(features, axis=class_dim), [batch_size, 1]))
<<<<<<< HEAD
    return e / np.reshape(np.sum(e, axis=class_dim), [batch_size, 1])

  def _testSoftmax(self, np_features, use_gpu=False):
    np_softmax = self._npSoftmax(np_features)
    with self.test_session(use_gpu=use_gpu):
      tf_softmax = tf.nn.softmax(np_features)
      out = tf_softmax.eval()
    self.assertAllClose(np_softmax, out)
    self.assertShapeEqual(np_softmax, tf_softmax)
    # Bonus check: the softmaxes should add to one in each
    # batch element.
    self.assertAllClose(np.ones(out.shape[0]),
                        np.sum(out, axis=1))

  def _testAll(self, features):
    self._testSoftmax(features, use_gpu=False)
    self._testSoftmax(features, use_gpu=True)
=======
    softmax = e / np.reshape(np.sum(e, axis=class_dim), [batch_size, 1])
    if log:
      return np.log(softmax)
    else:
      return softmax

  def _testSoftmax(self, np_features, log=False, use_gpu=False):
    np_softmax = self._npSoftmax(np_features, log=log)
    with self.test_session(use_gpu=use_gpu):
      if log:
        tf_softmax = tf.nn.log_softmax(np_features)
      else:
        tf_softmax = tf.nn.softmax(np_features)
      out = tf_softmax.eval()
    self.assertAllClose(np_softmax, out)
    self.assertShapeEqual(np_softmax, tf_softmax)
    if not log:
      # Bonus check: the softmaxes should add to one in each
      # batch element.
      self.assertAllClose(np.ones(out.shape[0]),
                          np.sum(out, axis=1))

  def _testAll(self, features):
    self._testSoftmax(features, use_gpu=False)
    self._testSoftmax(features, log=True, use_gpu=False)
    self._testSoftmax(features, use_gpu=True)
    self._testSoftmax(features, log=True, use_gpu=True)
    self._testOverflow(use_gpu=True)

>>>>>>> tensorflow/master

  def testNpSoftmax(self):
    features = [[1., 1., 1., 1.], [1., 2., 3., 4.]]
    # Batch 0: All exps are 1.  The expected result is
<<<<<<< HEAD
    # [0.25, 0.25, 0.25, 0.25]
=======
    # Softmaxes = [0.25, 0.25, 0.25, 0.25]
    # LogSoftmaxes = [-1.386294, -1.386294, -1.386294, -1.386294]
>>>>>>> tensorflow/master
    #
    # Batch 1:
    # exps = [1., 2.718, 7.389, 20.085]
    # sum = 31.192
    # Softmaxes = exps / sum = [0.0320586, 0.08714432, 0.23688282, 0.64391426]
<<<<<<< HEAD
=======
    # LogSoftmaxes = [-3.44019 , -2.44019 , -1.44019 , -0.44019]
>>>>>>> tensorflow/master
    np_sm = self._npSoftmax(np.array(features))
    self.assertAllClose(
        np.array([[0.25, 0.25, 0.25, 0.25],
                  [0.0320586, 0.08714432, 0.23688282, 0.64391426]]),
        np_sm,
        rtol=1.e-5, atol=1.e-5)
<<<<<<< HEAD
=======
    np_lsm = self._npSoftmax(np.array(features), log=True)
    self.assertAllClose(
        np.array([[-1.386294, -1.386294, -1.386294, -1.386294],
                  [-3.4401897, -2.4401897, -1.4401897, -0.4401897]]),
        np_lsm,
        rtol=1.e-5, atol=1.e-5)
>>>>>>> tensorflow/master

  def testShapeMismatch(self):
    with self.assertRaises(ValueError):
      tf.nn.softmax([0., 1., 2., 3.])
<<<<<<< HEAD
=======
    with self.assertRaises(ValueError):
      tf.nn.log_softmax([0., 1., 2., 3.])

  def _testOverflow(self, use_gpu=False):
    if use_gpu:
        type = np.float32
    else:
        type = np.float64
    max = np.finfo(type).max
    features = np.array(
        [[1., 1., 1., 1.],
         [max, 1., 2., 3.]]).astype(type)
    with self.test_session(use_gpu=use_gpu):
      tf_log_softmax = tf.nn.log_softmax(features)
      out = tf_log_softmax.eval()
    self.assertAllClose(
        np.array([[-1.386294, -1.386294, -1.386294, -1.386294],
                  [0, -max, -max, -max]]),
        out,
        rtol=1.e-5, atol=1.e-5)
>>>>>>> tensorflow/master

  def testFloat(self):
    self._testAll(
        np.array([[1., 1., 1., 1.], [1., 2., 3., 4.]]).astype(np.float32))

  def testDouble(self):
    self._testSoftmax(
        np.array([[1., 1., 1., 1.], [1., 2., 3., 4.]]).astype(np.float64),
        use_gpu=False)
<<<<<<< HEAD
=======
    self._testOverflow(use_gpu=False)


  def testEmpty(self):
    with self.test_session():
      x = tf.constant([[]], shape=[0, 3])
      self.assertEqual(0, tf.size(x).eval())
      expected_y = np.array([]).reshape(0, 3)
      np.testing.assert_array_equal(expected_y, tf.nn.softmax(x).eval())
>>>>>>> tensorflow/master


if __name__ == "__main__":
  tf.test.main()
