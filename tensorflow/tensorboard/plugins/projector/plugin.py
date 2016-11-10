# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""The Embedding Projector plugin."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import imghdr
import os
import numpy as np

from google.protobuf import json_format
from google.protobuf import text_format
from tensorflow.contrib.tensorboard.plugins.projector import PROJECTOR_FILENAME
from tensorflow.contrib.tensorboard.plugins.projector.projector_config_pb2 import ProjectorConfig
from tensorflow.python.lib.io import file_io
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.pywrap_tensorflow import NewCheckpointReader
from tensorflow.python.training.saver import checkpoint_exists
from tensorflow.python.training.saver import latest_checkpoint
from tensorflow.tensorboard.plugins.base_plugin import TBPlugin

# HTTP routes.
CONFIG_ROUTE = '/info'
TENSOR_ROUTE = '/tensor'
METADATA_ROUTE = '/metadata'
RUNS_ROUTE = '/runs'
BOOKMARKS_ROUTE = '/bookmarks'
SPRITE_IMAGE_ROUTE = '/sprite_image'

_IMGHDR_TO_MIMETYPE = {
    'bmp': 'image/bmp',
    'gif': 'image/gif',
    'jpeg': 'image/jpeg',
    'png': 'image/png'
}
_DEFAULT_IMAGE_MIMETYPE = 'application/octet-stream'


def _read_tensor_file(fpath):
  with file_io.FileIO(fpath, 'r') as f:
    tensor = []
    for line in f:
      if line:
        tensor.append(map(float, line.rstrip('\n').split('\t')))
  return np.array(tensor, dtype='float32')


def _latest_checkpoints_changed(configs, run_path_pairs):
  """Returns true if the latest checkpoint has changed in any of the runs."""
  for run_name, logdir in run_path_pairs:
    if run_name not in configs:
      continue
    config = configs[run_name]
    if not config.model_checkpoint_path:
      continue

    # See if you can find a checkpoint file in the logdir.
    ckpt_path = latest_checkpoint(logdir)
    if not ckpt_path:
      # See if you can find a checkpoint in the parent of logdir.
      ckpt_path = latest_checkpoint(os.path.join('../', logdir))
      if not ckpt_path:
        continue
    if config.model_checkpoint_path != ckpt_path:
      return True
  return False


def _parse_positive_int_param(request, query_params, param_name):
  """Parses and asserts a positive (>0) integer query parameter.

  Args:
    request: The http request object.
    query_params: Dictionary of query parameters.
    param_name: Name of the parameter.

  Returns:
    None if parameter not present. -1 if parameter is not a positive integer.
  """
  param = query_params.get(param_name)
  if not param:
    return None
  try:
    param = int(param)
    if param <= 0:
      raise ValueError()
    return param
  except ValueError:
    request.respond('query parameter "%s" must be integer > 0' % param_name,
                    'text/plain', 400)
    return -1


class ProjectorPlugin(TBPlugin):
  """Embedding projector."""

  def __init__(self):
    self._handlers = None
    self.readers = {}
    self.run_paths = None
    self.logdir = None
    self._configs = None
    self.old_num_run_paths = None

  def get_plugin_handlers(self, run_paths, logdir):
    self.run_paths = run_paths
    self.logdir = logdir
    self._handlers = {
        RUNS_ROUTE: self._serve_runs,
        CONFIG_ROUTE: self._serve_config,
        TENSOR_ROUTE: self._serve_tensor,
        METADATA_ROUTE: self._serve_metadata,
        BOOKMARKS_ROUTE: self._serve_bookmarks,
        SPRITE_IMAGE_ROUTE: self._serve_sprite_image
    }
    return self._handlers

  @property
  def configs(self):
    """Returns a map of run paths to `ProjectorConfig` protos."""
    run_path_pairs = self.run_paths.items()
    # If there are no summary event files, the projector should still work,
    # treating the `logdir` as the model checkpoint directory.
    if not run_path_pairs:
      run_path_pairs.append(('.', self.logdir))
    if (self._run_paths_changed() or
        _latest_checkpoints_changed(self._configs, run_path_pairs)):
      self.readers = {}
      self._configs, self.config_fpaths = self._read_latest_config_files(
          run_path_pairs)
      self._augment_configs_with_checkpoint_info()
    return self._configs

  def _run_paths_changed(self):
    num_run_paths = len(list(self.run_paths.keys()))
    if num_run_paths != self.old_num_run_paths:
      self.old_num_run_paths = num_run_paths
      return True
    return False

  def _augment_configs_with_checkpoint_info(self):
    for run, config in self._configs.items():
      # Find the size of the embeddings that are associated with a tensor file.
      for embedding in config.embeddings:
        if embedding.tensor_path and not embedding.tensor_shape:
          tensor = _read_tensor_file(embedding.tensor_path)
          embedding.tensor_shape.extend([len(tensor), len(tensor[0])])

      reader = self._get_reader_for_run(run)
      if not reader:
        continue
      # Augment the configuration with the tensors in the checkpoint file.
      special_embedding = None
      if config.embeddings and not config.embeddings[0].tensor_name:
        special_embedding = config.embeddings[0]
        config.embeddings.remove(special_embedding)
      var_map = reader.get_variable_to_shape_map()
      for tensor_name, tensor_shape in var_map.items():
        if len(tensor_shape) != 2:
          continue
        embedding = self._get_embedding(tensor_name, config)
        if not embedding:
          embedding = config.embeddings.add()
          embedding.tensor_name = tensor_name
          if special_embedding:
            embedding.metadata_path = special_embedding.metadata_path
            embedding.bookmarks_path = special_embedding.bookmarks_path
        if not embedding.tensor_shape:
          embedding.tensor_shape.extend(tensor_shape)

    # Remove configs that do not have any valid (2D) tensors.
    runs_to_remove = []
    for run, config in self._configs.items():
      if not config.embeddings:
        runs_to_remove.append(run)
    for run in runs_to_remove:
      del self._configs[run]
      del self.config_fpaths[run]

  def _read_latest_config_files(self, run_path_pairs):
    """Reads and returns the projector config files in every run directory."""
    configs = {}
    config_fpaths = {}
    for run_name, logdir in run_path_pairs:
      config = ProjectorConfig()
      config_fpath = os.path.join(logdir, PROJECTOR_FILENAME)
      if file_io.file_exists(config_fpath):
        file_content = file_io.read_file_to_string(config_fpath).decode('utf-8')
        text_format.Merge(file_content, config)

      has_tensor_files = False
      for embedding in config.embeddings:
        if embedding.tensor_path:
          has_tensor_files = True
          break

      if not config.model_checkpoint_path:
        # See if you can find a checkpoint file in the logdir.
        ckpt_path = latest_checkpoint(logdir)
        if not ckpt_path:
          # Or in the parent of logdir.
          ckpt_path = latest_checkpoint(os.path.join('../', logdir))
          if not ckpt_path and not has_tensor_files:
            continue
        if ckpt_path:
          config.model_checkpoint_path = ckpt_path

      # Sanity check for the checkpoint file.
      if (config.model_checkpoint_path and
          not checkpoint_exists(config.model_checkpoint_path)):
        logging.warning('Checkpoint file %s not found',
                        config.model_checkpoint_path)
        continue
      configs[run_name] = config
      config_fpaths[run_name] = config_fpath
    return configs, config_fpaths

  def _get_reader_for_run(self, run):
    if run in self.readers:
      return self.readers[run]

    config = self._configs[run]
    reader = None
    if config.model_checkpoint_path:
      try:
        reader = NewCheckpointReader(config.model_checkpoint_path)
      except Exception:  # pylint: disable=broad-except
        logging.warning('Failed reading %s', config.model_checkpoint_path)
    self.readers[run] = reader
    return reader

  def _get_metadata_file_for_tensor(self, tensor_name, config):
    embedding_info = self._get_embedding(tensor_name, config)
    if embedding_info:
      return embedding_info.metadata_path
    return None

  def _get_bookmarks_file_for_tensor(self, tensor_name, config):
    embedding_info = self._get_embedding(tensor_name, config)
    if embedding_info:
      return embedding_info.bookmarks_path
    return None

  def _canonical_tensor_name(self, tensor_name):
    if ':' not in tensor_name:
      return tensor_name + ':0'
    else:
      return tensor_name

  def _get_embedding(self, tensor_name, config):
    if not config.embeddings:
      return None
    for info in config.embeddings:
      if (self._canonical_tensor_name(info.tensor_name) ==
          self._canonical_tensor_name(tensor_name)):
        return info
    return None

  def _serve_runs(self, request, query_params):
    """Returns a list of runs that have embeddings."""
    request.respond(list(self.configs.keys()), 'application/json')

  def _serve_config(self, request, query_params):
    run = query_params.get('run')
    if run is None:
      request.respond('query parameter "run" is required', 'text/plain', 400)
      return
    if run not in self.configs:
      request.respond('Unknown run: %s' % run, 'text/plain', 400)
      return

    config = self.configs[run]
    request.respond(json_format.MessageToJson(config), 'application/json')

  def _serve_metadata(self, request, query_params):
    run = query_params.get('run')
    if run is None:
      request.respond('query parameter "run" is required', 'text/plain', 400)
      return

    name = query_params.get('name')
    if name is None:
      request.respond('query parameter "name" is required', 'text/plain', 400)
      return

    num_rows = _parse_positive_int_param(request, query_params, 'num_rows')
    if num_rows == -1:
      return

    if run not in self.configs:
      request.respond('Unknown run: %s' % run, 'text/plain', 400)
      return

    config = self.configs[run]
    fpath = self._get_metadata_file_for_tensor(name, config)
    if not fpath:
      request.respond(
          'No metadata file found for tensor %s in the config file %s' %
          (name, self.config_fpaths[run]), 'text/plain', 400)
      return
    if not file_io.file_exists(fpath) or file_io.is_directory(fpath):
      request.respond('%s is not a file' % fpath, 'text/plain', 400)
      return

    num_header_rows = 0
    with file_io.FileIO(fpath, 'r') as f:
      lines = []
      # Stream reading the file with early break in case the file doesn't fit in
      # memory.
      for line in f:
        lines.append(line)
        if len(lines) == 1 and '\t' in lines[0]:
          num_header_rows = 1
        if num_rows and len(lines) >= num_rows + num_header_rows:
          break
    request.respond(''.join(lines), 'text/plain')

  def _serve_tensor(self, request, query_params):
    run = query_params.get('run')
    if run is None:
      request.respond('query parameter "run" is required', 'text/plain', 400)
      return

    name = query_params.get('name')
    if name is None:
      request.respond('query parameter "name" is required', 'text/plain', 400)
      return

    num_rows = _parse_positive_int_param(request, query_params, 'num_rows')
    if num_rows == -1:
      return

    if run not in self.configs:
      request.respond('Unknown run: %s' % run, 'text/plain', 400)
      return

    reader = self._get_reader_for_run(run)
    config = self.configs[run]

    if reader is None:
      # See if there is a tensor file in the config.
      embedding = self._get_embedding(name, config)
      if not embedding or not embedding.tensor_path:
        request.respond('Tensor %s has no tensor_path in the config' %
                        name, 'text/plain', 400)
        return
      if not file_io.file_exists(embedding.tensor_path):
        request.respond('Tensor file %s does not exist' %
                        embedding.tensor_path, 'text/plain', 400)
        return
      tensor = _read_tensor_file(embedding.tensor_path)
    else:
      if not reader.has_tensor(name):
        request.respond('Tensor %s not found in checkpoint dir %s' %
                        (name, config.model_checkpoint_path),
                        'text/plain', 400)
        return
      tensor = reader.get_tensor(name)

    if num_rows:
      tensor = tensor[:num_rows]

    if tensor.dtype != 'float32':
      tensor = tensor.astype(dtype='float32', copy=False)
    data_bytes = tensor.tobytes()
    request.respond(data_bytes, 'application/octet-stream')

  def _serve_bookmarks(self, request, query_params):
    run = query_params.get('run')
    if not run:
      request.respond('query parameter "run" is required', 'text/plain', 400)
      return

    name = query_params.get('name')
    if name is None:
      request.respond('query parameter "name" is required', 'text/plain', 400)
      return

    if run not in self.configs:
      request.respond('Unknown run: %s' % run, 'text/plain', 400)
      return

    config = self.configs[run]
    fpath = self._get_bookmarks_file_for_tensor(name, config)
    if not fpath:
      request.respond(
          'No bookmarks file found for tensor %s in the config file %s' %
          (name, self.config_fpaths[run]), 'text/plain', 400)
      return
    if not file_io.file_exists(fpath) or file_io.is_directory(fpath):
      request.respond('%s is not a file' % fpath, 'text/plain', 400)
      return

    bookmarks_json = None
    with file_io.FileIO(fpath, 'r') as f:
      bookmarks_json = f.read()
    request.respond(bookmarks_json, 'application/json')

  def _serve_sprite_image(self, request, query_params):
    run = query_params.get('run')
    if not run:
      request.respond('query parameter "run" is required', 'text/plain', 400)
      return

    name = query_params.get('name')
    if name is None:
      request.respond('query parameter "name" is required', 'text/plain', 400)
      return

    if run not in self.configs:
      request.respond('Unknown run: %s' % run, 'text/plain', 400)
      return

    config = self.configs[run]
    embedding_info = self._get_embedding(name, config)

    if not embedding_info or not embedding_info.sprite.image_path:
      request.respond(
          'No sprite image file found for tensor %s in the config file %s' %
          (name, self.config_fpaths[run]), 'text/plain', 400)
      return

    fpath = embedding_info.sprite.image_path
    if not file_io.file_exists(fpath) or file_io.is_directory(fpath):
      request.respond(
          '%s does not exist or is directory' % fpath, 'text/plain', 400)
      return
    f = file_io.FileIO(fpath, 'r')
    encoded_image_string = f.read()
    f.close()
    image_type = imghdr.what(None, encoded_image_string)
    mime_type = _IMGHDR_TO_MIMETYPE.get(image_type, _DEFAULT_IMAGE_MIMETYPE)
    request.respond(encoded_image_string, mime_type)
