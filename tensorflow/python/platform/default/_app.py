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
"""Generic entry point script."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

from tensorflow.python.platform import flags


<<<<<<< HEAD
def run():
  f = flags.FLAGS
  f._parse_flags()
  main = sys.modules['__main__'].main
=======
def run(main=None):
  f = flags.FLAGS
  f._parse_flags()
  main = main or sys.modules['__main__'].main
>>>>>>> tensorflow/master
  sys.exit(main(sys.argv))
