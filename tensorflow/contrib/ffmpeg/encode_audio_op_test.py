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
# =============================================================================
"""Tests for third_party.tensorflow.contrib.ffmpeg.encode_audio_op."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

from tensorflow.contrib import ffmpeg
from tensorflow.python.platform import resource_loader
from tensorflow.python.platform import test


class EncodeAudioOpTest(test.TestCase):

  def _compareWavFiles(self, original, encoded):
    """Compares the important bits of two WAV files.

    Some encoders will create a slightly different header to the WAV file.
    This compares only the important bits of the header as well as the contents.

    Args:
      original: Contents of the original .wav file.
      encoded: Contents of the new, encoded .wav file.
    """
    self.assertLess(44, len(original))
    self.assertLess(44, len(encoded))
    self.assertEqual(original[:4], encoded[:4])
    # Skip file size
    self.assertEqual(original[8:16], encoded[8:16])
    # Skip header size
    self.assertEqual(original[20:36], encoded[20:36])
    # Skip extra bits inserted by ffmpeg.
    self.assertEqual(original[original.find(b'data'):],
                     encoded[encoded.find(b'data'):])

  def testRoundTrip(self):
    """Reads a wav file, writes it, and compares them."""
    with self.test_session():
      path = os.path.join(resource_loader.get_data_files_path(),
                          'testdata/mono_10khz.wav')
      with open(path, 'rb') as f:
        original_contents = f.read()

      audio_op = ffmpeg.decode_audio(
          original_contents,
          file_format='wav',
          samples_per_second=10000,
          channel_count=1)
      encode_op = ffmpeg.encode_audio(
          audio_op, file_format='wav', samples_per_second=10000)
      encoded_contents = encode_op.eval()
      self._compareWavFiles(original_contents, encoded_contents)


if __name__ == '__main__':
  test.main()
