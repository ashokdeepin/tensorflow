<<<<<<< HEAD
"""Switch between depending on pyglib.app or an OSS replacement."""
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

"""Generic entry point script."""
>>>>>>> tensorflow/master
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

<<<<<<< HEAD
# pylint: disable=unused-import
# pylint: disable=g-import-not-at-top
# pylint: disable=wildcard-import
import tensorflow.python.platform
from . import control_imports
if control_imports.USE_OSS and control_imports.OSS_APP:
  from tensorflow.python.platform.default._app import *
else:
  from tensorflow.python.platform.google._app import *

# Import 'flags' into this module
from tensorflow.python.platform import flags
=======
import sys

from tensorflow.python.platform import flags


def run(main=None):
  f = flags.FLAGS
  f._parse_flags()
  main = main or sys.modules['__main__'].main
  sys.exit(main(sys.argv))
>>>>>>> tensorflow/master
