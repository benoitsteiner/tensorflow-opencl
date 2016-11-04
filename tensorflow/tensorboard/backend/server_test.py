# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Integration tests for TensorBoard.

These tests start up a full-fledged TensorBoard server.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import base64
import gzip
import json
import numbers
import os
import shutil
import tempfile
import threading
import zlib

from six import BytesIO
from six.moves import http_client
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from google.protobuf import text_format
from tensorflow.contrib.tensorboard.plugins.projector.projector_config_pb2 import ProjectorConfig
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.platform import resource_loader
from tensorflow.python.summary import event_multiplexer
from tensorflow.tensorboard.backend import server
from tensorflow.tensorboard.plugins import REGISTERED_PLUGINS


class TensorboardServerTest(tf.test.TestCase):
  _only_use_meta_graph = False  # Server data contains only a GraphDef

  # Number of scalar-containing events to make.
  _SCALAR_COUNT = 99

  def setUp(self):
    temp_dir = self._GenerateTestData()
    self._multiplexer = event_multiplexer.EventMultiplexer(
        size_guidance=server.TENSORBOARD_SIZE_GUIDANCE)
    server.ReloadMultiplexer(self._multiplexer, {temp_dir: None})
    # 0 to pick an unused port.
    self._server = server.BuildServer(
        self._multiplexer, 'localhost', 0, '/foo/logdir/argument')
    self._server_thread = threading.Thread(target=self._server.serve_forever)
    self._server_thread.daemon = True
    self._server_thread.start()
    self._connection = http_client.HTTPConnection(
        'localhost', self._server.server_address[1])

  def tearDown(self):
    self._connection.close()
    self._server.shutdown()
    self._server.server_close()

  def _get(self, path, headers={}):
    """Perform a GET request for the given path."""
    self._connection.request('GET', path, None, headers)
    return self._connection.getresponse()

  def _getJson(self, path):
    """Perform a GET request and decode the result as JSON."""
    self._connection.request('GET', path)
    response = self._connection.getresponse()
    self.assertEqual(response.status, 200)
    data = response.read()
    if response.getheader('Content-Encoding') == 'gzip':
      data = gzip.GzipFile('', 'rb', 9, BytesIO(data)).read()
    return json.loads(data.decode('utf-8'))

  def testBasicStartup(self):
    """Start the server up and then shut it down immediately."""
    pass

  def testRequestMainPage(self):
    """Navigate to the main page and verify that it returns a 200."""
    response = self._get('/')
    self.assertEqual(response.status, 200)

  def testRequestNonexistentPage(self):
    """Request a page that doesn't exist; it should 404."""
    response = self._get('/asdf')
    self.assertEqual(response.status, 404)

  def testDirectoryTraversal(self):
    """Attempt a directory traversal attack."""
    response = self._get('/..' * 30 + '/etc/passwd')
    self.assertEqual(response.status, 400)

  def testLogdir(self):
    """Test the format of the data/logdir endpoint."""
    parsed_object = self._getJson('/data/logdir')
    self.assertEqual(parsed_object, {'logdir': '/foo/logdir/argument'})

  def testRuns(self):
    """Test the format of the /data/runs endpoint."""
    run_json = self._getJson('/data/runs')

    # Don't check the actual timestamp since it's time-dependent.
    self.assertTrue(isinstance(run_json['run1']['firstEventTimestamp'],
                               numbers.Number))
    del run_json['run1']['firstEventTimestamp']
    self.assertEqual(run_json, {'run1': {
        'compressedHistograms': ['histogram'],
        'scalars': ['simple_values'],
        'histograms': ['histogram'],
        'images': ['image'],
        'audio': ['audio'],
        # if only_use_meta_graph, the graph is extracted from the metagraph
        'graph': True,
        'meta_graph': self._only_use_meta_graph,
        'run_metadata': ['test run']}})

  def testApplicationPaths_getCached(self):
    """Test the format of the /data/runs endpoint."""
    for path in ('/',):  # TODO(jart): '/app.js' in open source
      connection = http_client.HTTPConnection(
          'localhost', self._server.server_address[1])
      connection.request('GET', path)
      response = connection.getresponse()
      self.assertEqual(response.status, 200, msg=path)
      self.assertEqual(response.getheader('Cache-Control'),
                       'private, max-age=3600', msg=path)
      connection.close()

  def testDataPaths_disableAllCaching(self):
    """Test the format of the /data/runs endpoint."""
    for path in ('/data/runs',
                 '/data/logdir',
                 '/data/scalars?run=run1&tag=simple_values',
                 '/data/scalars?run=run1&tag=simple_values&format=csv',
                 '/data/images?run=run1&tag=image',
                 '/data/individualImage?run=run1&tag=image&index=0',
                 '/data/audio?run=run1&tag=audio',
                 '/data/run_metadata?run=run1&tag=test%20run'):
      connection = http_client.HTTPConnection(
          'localhost', self._server.server_address[1])
      connection.request('GET', path)
      response = connection.getresponse()
      self.assertEqual(response.status, 200, msg=path)
      self.assertEqual(response.getheader('Expires'), '0', msg=path)
      response.read()
      connection.close()

  def testHistograms(self):
    """Test the format of /data/histograms."""
    self.assertEqual(
        self._getJson('/data/histograms?tag=histogram&run=run1'),
        [[0, 0, [0, 2.0, 3.0, 6.0, 5.0, [0.0, 1.0, 2.0], [1.0, 1.0, 1.0]]]])

  def testSampleScalars(self):
    """Test the sample_count parameter of /data/scalars."""
    for i in xrange(10, self._SCALAR_COUNT, 10):
      samples = self._getJson('/data/scalars?sample_count=%d' % i)
      values = samples['run1']['simple_values']
      # Verify that we got the right amount of values and that we got the
      # endpoints.
      self.assertEqual(len(values), i)
      self.assertEqual(values[0], [100, 10, 1])
      self.assertEqual(values[-1], [9900, 990, 99])

  def testSampleScalarsWithLargeSampleCount(self):
    """Test using a large sample_count."""
    samples = self._getJson('/data/scalars?sample_count=999999')
    values = samples['run1']['simple_values']
    self.assertEqual(len(values), self._SCALAR_COUNT)

  def testImages(self):
    """Test listing images and retrieving an individual image."""
    image_json = self._getJson('/data/images?tag=image&run=run1')
    image_query = image_json[0]['query']
    # We don't care about the format of the image query.
    del image_json[0]['query']
    self.assertEqual(image_json, [{
        'wall_time': 0,
        'step': 0,
        'height': 1,
        'width': 1
    }])
    response = self._get('/data/individualImage?%s' % image_query)
    self.assertEqual(response.status, 200)

  def testAudio(self):
    """Test listing audio and retrieving an individual audio clip."""
    audio_json = self._getJson('/data/audio?tag=audio&run=run1')
    audio_query = audio_json[0]['query']
    # We don't care about the format of the audio query.
    del audio_json[0]['query']
    self.assertEqual(audio_json, [{
        'wall_time': 0,
        'step': 0,
        'content_type': 'audio/wav'
    }])
    response = self._get('/data/individualAudio?%s' % audio_query)
    self.assertEqual(response.status, 200)

  def testGraph(self):
    """Test retrieving the graph definition."""
    response = self._get('/data/graph?run=run1&limit_attr_size=1024'
                         '&large_attrs_key=_very_large_attrs')
    self.assertEqual(response.status, 200)
    graph_pbtxt = response.read()
    # Parse the graph from pbtxt into a graph message.
    graph = tf.GraphDef()
    graph = text_format.Parse(graph_pbtxt, graph)
    self.assertEqual(len(graph.node), 2)
    self.assertEqual(graph.node[0].name, 'a')
    self.assertEqual(graph.node[1].name, 'b')
    # Make sure the second node has an attribute that was filtered out because
    # it was too large and was added to the "too large" attributes list.
    self.assertEqual(list(graph.node[1].attr.keys()), ['_very_large_attrs'])
    self.assertEqual(graph.node[1].attr['_very_large_attrs'].list.s,
                     [b'very_large_attr'])

  def testProjectorRunsWithEmbeddings(self):
    """Test the format of /runs endpoint in projector."""
    if 'projector' not in REGISTERED_PLUGINS:
      return

    run_json = self._getJson('/data/plugin/projector/runs')

    self.assertEqual(run_json, ['run1'])

  def testProjectorInfo(self):
    """Test the format of /info endpoint in projector."""
    if 'projector' not in REGISTERED_PLUGINS:
      return

    info_json = self._getJson('/data/plugin/projector/info?run=run1')
    self.assertItemsEqual(info_json['embeddings'], [
        {
            'tensorShape': [1, 2],
            'tensorName': 'var1'
        },
        {
            'tensorShape': [10, 10],
            'tensorName': 'var2'
        },
        {
            'tensorShape': [100, 100],
            'tensorName': 'var3'
        }
    ])

  def testProjectorTensor(self):
    """Test the format of /tensor endpoint in projector."""
    if 'projector' not in REGISTERED_PLUGINS:
      return

    tensor_tsv = (self._get('/data/plugin/projector/tensor?run=run1&name=var1')
                  .read())
    self.assertEqual(tensor_tsv, b'6.0\t6.0')

  def testAcceptGzip_compressesResponse(self):
    response = self._get('/data/graph?run=run1&limit_attr_size=1024'
                         '&large_attrs_key=_very_large_attrs',
                         {'Accept-Encoding': 'gzip'})
    self.assertEqual(response.status, 200)
    self.assertEqual(response.getheader('Content-Encoding'), 'gzip')
    pbtxt = gzip.GzipFile('', 'rb', 9, BytesIO(response.read())).read()
    graph = text_format.Parse(pbtxt, tf.GraphDef())
    self.assertEqual(len(graph.node), 2)

  def testAcceptAnyEncoding_compressesResponse(self):
    response = self._get('/data/graph?run=run1&limit_attr_size=1024'
                         '&large_attrs_key=_very_large_attrs',
                         {'Accept-Encoding': '*'})
    self.assertEqual(response.status, 200)
    self.assertEqual(response.getheader('Content-Encoding'), 'gzip')
    pbtxt = gzip.GzipFile('', 'rb', 9, BytesIO(response.read())).read()
    graph = text_format.Parse(pbtxt, tf.GraphDef())
    self.assertEqual(len(graph.node), 2)

  def testAcceptDoodleEncoding_doesNotCompressResponse(self):
    response = self._get('/data/graph?run=run1&limit_attr_size=1024'
                         '&large_attrs_key=_very_large_attrs',
                         {'Accept-Encoding': 'doodle'})
    self.assertEqual(response.status, 200)
    self.assertIsNone(response.getheader('Content-Encoding'))
    graph = text_format.Parse(response.read(), tf.GraphDef())
    self.assertEqual(len(graph.node), 2)

  def testAcceptGzip_doesNotCompressImage(self):
    response = self._get('/data/individualImage?run=run1&tag=image&index=0',
                         {'Accept-Encoding': 'gzip'})
    self.assertEqual(response.status, 200)
    self.assertEqual(response.getheader('Content-Encoding'), None)

  def testRunMetadata(self):
    """Test retrieving the run metadata information."""
    response = self._get('/data/run_metadata?run=run1&tag=test%20run')
    self.assertEqual(response.status, 200)
    run_metadata_pbtxt = response.read()
    # Parse from pbtxt into a message.
    run_metadata = tf.RunMetadata()
    text_format.Parse(run_metadata_pbtxt, run_metadata)
    self.assertEqual(len(run_metadata.step_stats.dev_stats), 1)
    self.assertEqual(run_metadata.step_stats.dev_stats[0].device, 'test device')

  def _GenerateTestData(self):
    """Generates the test data directory.

    The test data has a single run named run1 which contains:
     - a histogram
     - an image at timestamp and step 0
     - scalar events containing the value i at step 10 * i and wall time
         100 * i, for i in [1, _SCALAR_COUNT).
     - a graph definition

    Returns:
      temp_dir: The directory the test data is generated under.
    """
    temp_dir = tempfile.mkdtemp(prefix=self.get_temp_dir())
    self.addCleanup(shutil.rmtree, temp_dir)
    run1_path = os.path.join(temp_dir, 'run1')
    os.makedirs(run1_path)
    writer = tf.train.SummaryWriter(run1_path)

    histogram_value = tf.HistogramProto(min=0,
                                        max=2,
                                        num=3,
                                        sum=6,
                                        sum_squares=5,
                                        bucket_limit=[0, 1, 2],
                                        bucket=[1, 1, 1])
    # Add a simple graph event.
    graph_def = tf.GraphDef()
    node1 = graph_def.node.add()
    node1.name = 'a'
    node2 = graph_def.node.add()
    node2.name = 'b'
    node2.attr['very_large_attr'].s = b'a' * 2048  # 2 KB attribute

    meta_graph_def = meta_graph_pb2.MetaGraphDef(graph_def=graph_def)

    if self._only_use_meta_graph:
      writer.add_meta_graph(meta_graph_def)
    else:
      writer.add_graph(graph_def)

    # Add a simple run metadata event.
    run_metadata = tf.RunMetadata()
    device_stats = run_metadata.step_stats.dev_stats.add()
    device_stats.device = 'test device'
    writer.add_run_metadata(run_metadata, 'test run')

    # 1x1 transparent GIF.
    encoded_image = base64.b64decode(
        'R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7')
    image_value = tf.Summary.Image(height=1,
                                   width=1,
                                   colorspace=1,
                                   encoded_image_string=encoded_image)

    audio_value = tf.Summary.Audio(sample_rate=44100,
                                   length_frames=22050,
                                   num_channels=2,
                                   encoded_audio_string=b'',
                                   content_type='audio/wav')
    writer.add_event(tf.Event(wall_time=0,
                              step=0,
                              summary=tf.Summary(value=[
                                  tf.Summary.Value(tag='histogram',
                                                   histo=histogram_value),
                                  tf.Summary.Value(tag='image',
                                                   image=image_value),
                                  tf.Summary.Value(tag='audio',
                                                   audio=audio_value)
                              ])))

    # Write 100 simple values.
    for i in xrange(1, self._SCALAR_COUNT + 1):
      writer.add_event(tf.Event(
          # We use different values for wall time, step, and the value so we can
          # tell them apart.
          wall_time=100 * i,
          step=10 * i,
          summary=tf.Summary(value=[tf.Summary.Value(tag='simple_values',
                                                     simple_value=i)])))
    writer.flush()
    writer.close()

    if 'projector' in REGISTERED_PLUGINS:
      self._GenerateProjectorTestData(run1_path)

    return temp_dir

  def _GenerateProjectorTestData(self, run_path):
    # Write a projector config file in run1.
    config_path = os.path.join(run_path, 'projector_config.pbtxt')
    config = ProjectorConfig()
    config_pbtxt = text_format.MessageToString(config)
    with tf.gfile.GFile(config_path, 'w') as f:
      f.write(config_pbtxt)

    # Write a checkpoint with some dummy variables.
    with tf.Graph().as_default():
      sess = tf.Session()
      checkpoint_path = os.path.join(run_path, 'model')
      tf.get_variable(
          'var1', [1, 2], initializer=tf.constant_initializer(6.0))
      tf.get_variable('var2', [10, 10])
      tf.get_variable('var3', [100, 100])
      sess.run(tf.initialize_all_variables())
      saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)
      saver.save(sess, checkpoint_path)


class TensorboardServerUsingMetagraphOnlyTest(TensorboardServerTest):
  # Tests new ability to use only the MetaGraphDef
  _only_use_meta_graph = True  # Server data contains only a MetaGraphDef


class ParseEventFilesSpecTest(tf.test.TestCase):

  def testRunName(self):
    logdir_string = 'lol:/cat'
    expected = {'/cat': 'lol'}
    self.assertEqual(server.ParseEventFilesSpec(logdir_string), expected)

  def testPathWithColonThatComesAfterASlash_isNotConsideredARunName(self):
    logdir_string = '/lol:/cat'
    expected = {'/lol:/cat': None}
    self.assertEqual(server.ParseEventFilesSpec(logdir_string), expected)

  def testMultipleDirectories(self):
    logdir_string = '/a,/b'
    expected = {'/a': None, '/b': None}
    self.assertEqual(server.ParseEventFilesSpec(logdir_string), expected)

  def testNormalizesPaths(self):
    logdir_string = '/lol/.//cat/../cat'
    expected = {'/lol/cat': None}
    self.assertEqual(server.ParseEventFilesSpec(logdir_string), expected)

  def testAbsolutifies(self):
    logdir_string = 'lol/cat'
    expected = {os.path.realpath('lol/cat'): None}
    self.assertEqual(server.ParseEventFilesSpec(logdir_string), expected)

  def testRespectsGCSPath(self):
    logdir_string = 'gs://foo/path'
    expected = {'gs://foo/path': None}
    self.assertEqual(server.ParseEventFilesSpec(logdir_string), expected)

  def testDoesNotExpandUserInGCSPath(self):
    logdir_string = 'gs://~/foo/path'
    expected = {'gs://~/foo/path': None}
    self.assertEqual(server.ParseEventFilesSpec(logdir_string), expected)

  def testDoesNotNormalizeGCSPath(self):
    logdir_string = 'gs://foo/./path//..'
    expected = {'gs://foo/./path//..': None}
    self.assertEqual(server.ParseEventFilesSpec(logdir_string), expected)


class TensorBoardAssetsTest(tf.test.TestCase):

  def testTagFound(self):
    tag = resource_loader.load_resource('tensorboard/TAG')
    self.assertTrue(tag)


if __name__ == '__main__':
  tf.test.main()
