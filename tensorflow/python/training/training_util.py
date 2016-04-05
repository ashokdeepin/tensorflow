<<<<<<< HEAD
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

>>>>>>> tensorflow/master
"""Utility functions for training."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

from tensorflow.python.platform import gfile


def global_step(sess, global_step_tensor):
  """Small helper to get the global step.

  ```python
  # Creates a variable to hold the global_step.
  global_step_tensor = tf.Variable(10, trainable=False, name='global_step')
  # Creates a session.
  sess = tf.Session()
  # Initializes the variable.
  sess.run(global_step_tensor.initializer)
<<<<<<< HEAD
  print 'global_step:', tf.train.global_step(sess, global_step_tensor)
=======
  print('global_step: %s' % tf.train.global_step(sess, global_step_tensor))
>>>>>>> tensorflow/master

  global_step: 10
  ```

  Args:
<<<<<<< HEAD
    sess: A brain `Session` object.
=======
    sess: A TensorFlow `Session` object.
>>>>>>> tensorflow/master
    global_step_tensor:  `Tensor` or the `name` of the operation that contains
      the global step.

  Returns:
    The global step value.
  """
  return int(sess.run(global_step_tensor))


def write_graph(graph_def, logdir, name, as_text=True):
  """Writes a graph proto on disk.

<<<<<<< HEAD
  The graph is written as a binary proto unless as_text is `True`.
=======
  The graph is written as a binary proto unless `as_text` is `True`.
>>>>>>> tensorflow/master

  ```python
  v = tf.Variable(0, name='my_variable')
  sess = tf.Session()
  tf.train.write_graph(sess.graph_def, '/tmp/my-model', 'train.pbtxt')
  ```

  Args:
    graph_def: A `GraphDef` protocol buffer.
    logdir: Directory where to write the graph.
    name: Filename for the graph.
    as_text: If `True`, writes the graph as an ASCII proto.
  """
<<<<<<< HEAD
  path = os.path.join(logdir, name)
  gfile.MakeDirs(os.path.dirname(path))
  f = gfile.FastGFile(path, "w")
  if as_text:
    f.write(str(graph_def))
  else:
=======
  if not gfile.IsDirectory(logdir):
    gfile.MakeDirs(logdir)
  path = os.path.join(logdir, name)
  if as_text:
    f = gfile.FastGFile(path, "w")
    f.write(str(graph_def))
  else:
    f = gfile.FastGFile(path, "wb")
>>>>>>> tensorflow/master
    f.write(graph_def.SerializeToString())
  f.close()
