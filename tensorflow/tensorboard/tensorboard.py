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
"""Serve TensorFlow summary data to a web frontend.

This is a simple web server to proxy data from the event_loader to the web, and
serve static web files.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

<<<<<<< HEAD
import BaseHTTPServer
import functools
import os
import socket
import SocketServer

import tensorflow.python.platform
=======
import os
import socket
>>>>>>> tensorflow/master

from tensorflow.python.platform import app
from tensorflow.python.platform import flags
from tensorflow.python.platform import logging
<<<<<<< HEAD
from tensorflow.python.platform import status_bar
from tensorflow.python.summary import event_accumulator
from tensorflow.python.summary import event_multiplexer
from tensorflow.tensorboard import tensorboard_handler

flags.DEFINE_string('logdir', None, """
logdir specifies where TensorBoard will look to find TensorFlow event files
that it can display. In the simplest case, logdir is a directory containing
tfevents files. TensorBoard also supports comparing multiple TensorFlow
executions: to do this, you can use directory whose subdirectories contain
tfevents files, as in the following example:

foo/bar/logdir/
foo/bar/logdir/mnist_1/events.out.tfevents.1444088766
foo/bar/logdir/mnist_2/events.out.tfevents.1444090064

You may also pass a comma seperated list of log directories, and you can
assign names to individual log directories by putting a colon between the name
and the path, as in

tensorboard --logdir=name1:/path/to/logs/1,name2:/path/to/logs/2
""")
flags.DEFINE_boolean('debug', False, 'Whether to run the app in debug mode. '
                     'This increases log verbosity to DEBUG.')
flags.DEFINE_string('host', '0.0.0.0', 'What host to listen to. Defaults to '
                    'allowing remote access, set to 127.0.0.1 to serve only on '
                    'localhost.')
flags.DEFINE_integer('port', 6006, 'What port to serve TensorBoard on.')

FLAGS = flags.FLAGS

# How many elements to store per tag, by tag type
TENSORBOARD_SIZE_GUIDANCE = {
    event_accumulator.COMPRESSED_HISTOGRAMS: 500,
    event_accumulator.IMAGES: 4,
    event_accumulator.SCALARS: 10000,
    event_accumulator.HISTOGRAMS: 1,
}


def ParseEventFilesFlag(flag_value):
  """Parses the logdir flag into a map from paths to run group names.

  The events files flag format is a comma-separated list of path specifications.
  A path specification either looks like 'group_name:/path/to/directory' or
  '/path/to/directory'; in the latter case, the group is unnamed. Group names
  cannot start with a forward slash: /foo:bar/baz will be interpreted as a
  spec with no name and path '/foo:bar/baz'.

  Globs are not supported.

  Args:
    flag_value: A comma-separated list of run specifications.
  Returns:
    A dict mapping directory paths to names like {'/path/to/directory': 'name'}.
    Groups without an explicit name are named after their path. If flag_value
    is None, returns an empty dict, which is helpful for testing things that
    don't require any valid runs.
  """
  files = {}
  if flag_value is None:
    return files
  for specification in flag_value.split(','):
    # If the spec looks like /foo:bar/baz, then we assume it's a path with a
    # colon.
    if ':' in specification and specification[0] != '/':
      # We split at most once so run_name:/path:with/a/colon will work.
      run_name, path = specification.split(':', 1)
    else:
      run_name = None
      path = specification
    files[path] = run_name
  return files


class ThreadedHTTPServer(SocketServer.ThreadingMixIn,
                         BaseHTTPServer.HTTPServer):
  """A threaded HTTP server."""
  daemon = True


def main(unused_argv=None):
  # Change current working directory to tensorflow/'s parent directory.
  server_root = os.path.join(os.path.dirname(__file__),
                             os.pardir, os.pardir)
  os.chdir(server_root)

  if FLAGS.debug:
    logging.set_verbosity(logging.DEBUG)

  if not FLAGS.logdir:
    logging.error('A logdir must be specified. Run `tensorboard --help` for '
                  'details and examples.')
    return -1

  if FLAGS.debug:
    logging.info('Starting TensorBoard in directory %s', os.getcwd())

  path_to_run = ParseEventFilesFlag(FLAGS.logdir)
  multiplexer = event_multiplexer.AutoloadingMultiplexer(
      path_to_run=path_to_run, interval_secs=60,
      size_guidance=TENSORBOARD_SIZE_GUIDANCE)

  multiplexer.AutoUpdate(interval=30)

  factory = functools.partial(tensorboard_handler.TensorboardHandler,
                              multiplexer)
  try:
    server = ThreadedHTTPServer((FLAGS.host, FLAGS.port), factory)
  except socket.error:
    logging.error('Tried to connect to port %d, but that address is in use.',
                  FLAGS.port)
    return -2

  status_bar.SetupStatusBarInsideGoogle('TensorBoard', FLAGS.port)
  print('Starting TensorBoard on port %d' % FLAGS.port)
  print('(You can navigate to http://localhost:%d)' % FLAGS.port)
  server.serve_forever()
=======
from tensorflow.python.platform import resource_loader
from tensorflow.python.platform import status_bar
from tensorflow.python.summary import event_multiplexer
from tensorflow.tensorboard.backend import server

flags.DEFINE_string('logdir', None, """logdir specifies the directory where
TensorBoard will look to find TensorFlow event files that it can display.
TensorBoard will recursively walk the directory structure rooted at logdir,
looking for .*tfevents.* files.

You may also pass a comma separated list of log directories, and TensorBoard
will watch each directory. You can also assign names to individual log
directories by putting a colon between the name and the path, as in

tensorboard --logdir=name1:/path/to/logs/1,name2:/path/to/logs/2
""")

flags.DEFINE_boolean('debug', False, 'Whether to run the app in debug mode. '
                     'This increases log verbosity to DEBUG.')

flags.DEFINE_string('host', '0.0.0.0', 'What host to listen to. Defaults to '
                    'serving on 0.0.0.0, set to 127.0.0.1 (localhost) to'
                    'disable remote access (also quiets security warnings).')

flags.DEFINE_integer('port', 6006, 'What port to serve TensorBoard on.')

flags.DEFINE_boolean('purge_orphaned_data', True, 'Whether to purge data that '
                     'may have been orphaned due to TensorBoard restarts. '
                     'Disabling purge_orphaned_data can be used to debug data '
                     'disappearance.')

FLAGS = flags.FLAGS


def main(unused_argv=None):
  if FLAGS.debug:
    logging.set_verbosity(logging.DEBUG)
    logging.info('TensorBoard is in debug mode.')

  if not FLAGS.logdir:
    msg = ('A logdir must be specified. Run `tensorboard --help` for '
           'details and examples.')
    logging.error(msg)
    print(msg)
    return -1

  logging.info('Starting TensorBoard in directory %s', os.getcwd())
  path_to_run = server.ParseEventFilesSpec(FLAGS.logdir)
  logging.info('TensorBoard path_to_run is: %s', path_to_run)

  multiplexer = event_multiplexer.EventMultiplexer(
      size_guidance=server.TENSORBOARD_SIZE_GUIDANCE,
      purge_orphaned_data=FLAGS.purge_orphaned_data)
  server.StartMultiplexerReloadingThread(multiplexer, path_to_run)
  try:
    tb_server = server.BuildServer(multiplexer, FLAGS.host, FLAGS.port)
  except socket.error:
    if FLAGS.port == 0:
      msg = 'Unable to find any open ports.'
      logging.error(msg)
      print(msg)
      return -2
    else:
      msg = 'Tried to connect to port %d, but address is in use.' % FLAGS.port
      logging.error(msg)
      print(msg)
      return -3

  try:
    tag = resource_loader.load_resource('tensorboard/TAG').strip()
    logging.info('TensorBoard is tag: %s', tag)
  except IOError:
    logging.warning('Unable to read TensorBoard tag')
    tag = ''

  status_bar.SetupStatusBarInsideGoogle('TensorBoard %s' % tag, FLAGS.port)
  print('Starting TensorBoard %s on port %d' % (tag, FLAGS.port))
  print('(You can navigate to http://%s:%d)' % (FLAGS.host, FLAGS.port))
  tb_server.serve_forever()
>>>>>>> tensorflow/master


if __name__ == '__main__':
  app.run()
