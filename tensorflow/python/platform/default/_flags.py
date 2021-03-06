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
"""Implementation of the flags interface."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

<<<<<<< HEAD
import tensorflow.python.platform

=======
>>>>>>> tensorflow/master
import argparse

_global_parser = argparse.ArgumentParser()

class _FlagValues(object):

  def __init__(self):
    """Global container and accessor for flags and their values."""
    self.__dict__['__flags'] = {}
    self.__dict__['__parsed'] = False

  def _parse_flags(self):
<<<<<<< HEAD
    result = _global_parser.parse_args()
=======
    result, _ = _global_parser.parse_known_args()
>>>>>>> tensorflow/master
    for flag_name, val in vars(result).items():
      self.__dict__['__flags'][flag_name] = val
    self.__dict__['__parsed'] = True

  def __getattr__(self, name):
    """Retrieves the 'value' attribute of the flag --name."""
    if not self.__dict__['__parsed']:
      self._parse_flags()
    if name not in self.__dict__['__flags']:
      raise AttributeError(name)
    return self.__dict__['__flags'][name]

  def __setattr__(self, name, value):
    """Sets the 'value' attribute of the flag --name."""
    if not self.__dict__['__parsed']:
      self._parse_flags()
    self.__dict__['__flags'][name] = value


def _define_helper(flag_name, default_value, docstring, flagtype):
  """Registers 'flag_name' with 'default_value' and 'docstring'."""
  _global_parser.add_argument("--" + flag_name,
                              default=default_value,
                              help=docstring,
                              type=flagtype)


# Provides the global object that can be used to access flags.
FLAGS = _FlagValues()


def DEFINE_string(flag_name, default_value, docstring):
  """Defines a flag of type 'string'.

  Args:
    flag_name: The name of the flag as a string.
    default_value: The default value the flag should take as a string.
    docstring: A helpful message explaining the use of the flag.
  """
  _define_helper(flag_name, default_value, docstring, str)


def DEFINE_integer(flag_name, default_value, docstring):
  """Defines a flag of type 'int'.

  Args:
    flag_name: The name of the flag as a string.
    default_value: The default value the flag should take as an int.
    docstring: A helpful message explaining the use of the flag.
  """
  _define_helper(flag_name, default_value, docstring, int)


def DEFINE_boolean(flag_name, default_value, docstring):
  """Defines a flag of type 'boolean'.

  Args:
    flag_name: The name of the flag as a string.
    default_value: The default value the flag should take as a boolean.
    docstring: A helpful message explaining the use of the flag.
  """
<<<<<<< HEAD
  _define_helper(flag_name, default_value, docstring, bool)
=======
  # Register a custom function for 'bool' so --flag=True works.
  def str2bool(v):
    return v.lower() in ('true', 't', '1')
  _global_parser.add_argument('--' + flag_name,
                              nargs='?',
                              const=True,
                              help=docstring,
                              default=default_value,
                              type=str2bool)
>>>>>>> tensorflow/master
  _global_parser.add_argument('--no' + flag_name,
                              action='store_false',
                              dest=flag_name)


<<<<<<< HEAD
=======
# The internal google library defines the following alias, so we match
# the API for consistency.
DEFINE_bool = DEFINE_boolean  # pylint: disable=invalid-name


>>>>>>> tensorflow/master
def DEFINE_float(flag_name, default_value, docstring):
  """Defines a flag of type 'float'.

  Args:
    flag_name: The name of the flag as a string.
    default_value: The default value the flag should take as a float.
    docstring: A helpful message explaining the use of the flag.
  """
  _define_helper(flag_name, default_value, docstring, float)
