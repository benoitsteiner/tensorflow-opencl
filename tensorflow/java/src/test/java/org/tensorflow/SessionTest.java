/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

package org.tensorflow;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.util.ArrayList;
import java.util.Collection;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link org.tensorflow.Session}. */
@RunWith(JUnit4.class)
public class SessionTest {

  @Test
  public void run() {
    try (Graph g = new Graph();
        Session s = new Session(g)) {
      TestUtil.transpose_A_times_X(g, new int[][] {{2}, {3}});
      try (Tensor x = Tensor.create(new int[][] {{5}, {7}});
          AutoCloseableList<Tensor> outputs =
              new AutoCloseableList<Tensor>(s.runner().feed("X", x).fetch("Y").run())) {
        assertEquals(1, outputs.size());
        final int[][] expected = {{31}};
        assertEquals(expected, outputs.get(0).copyTo(new int[1][1]));
      }
    }
  }

  @Test
  public void runWithMetadata() {
    try (Graph g = new Graph();
        Session s = new Session(g)) {
      TestUtil.transpose_A_times_X(g, new int[][] {{2}, {3}});
      try (Tensor x = Tensor.create(new int[][] {{5}, {7}})) {
        Session.Run result =
            s.runner()
                .feed("X", x)
                .fetch("Y")
                .setOptions(fullTraceRunOptions())
                .runAndFetchMetadata();
        // Sanity check on outputs.
        AutoCloseableList<Tensor> outputs = new AutoCloseableList<Tensor>(result.outputs);
        assertEquals(1, outputs.size());
        final int[][] expected = {{31}};
        assertEquals(expected, outputs.get(0).copyTo(new int[1][1]));
        // Sanity check on metadatar
        // See comments in fullTraceRunOptions() for an explanation about
        // why this check is really silly. Ideally, this would be:
        /*
            RunMetadata md = RunMetadata.parseFrom(result.metadata);
            assertTrue(md.toString(), md.hasStepStats());
        */
        assertTrue(result.metadata.length > 0);
      }
    }
  }

  @Test
  public void runMultipleOutputs() {
    try (Graph g = new Graph();
        Session s = new Session(g)) {
      TestUtil.constant(g, "c1", 2718);
      TestUtil.constant(g, "c2", 31415);
      AutoCloseableList<Tensor> outputs =
          new AutoCloseableList<Tensor>(s.runner().fetch("c2").fetch("c1").run());
      assertEquals(2, outputs.size());
      assertEquals(31415, outputs.get(0).intValue());
      assertEquals(2718, outputs.get(1).intValue());
    }
  }

  @Test
  public void failOnUseAfterClose() {
    try (Graph g = new Graph()) {
      Session s = new Session(g);
      s.close();
      try {
        s.runner().run();
        fail("methods on a close()d session should fail");
      } catch (IllegalStateException e) {
        // expected exception
      }
    }
  }

  private static final class AutoCloseableList<E extends AutoCloseable> extends ArrayList<E>
      implements AutoCloseable {
    AutoCloseableList(Collection<? extends E> c) {
      super(c);
    }

    @Override
    public void close() {
      Exception toThrow = null;
      for (AutoCloseable c : this) {
        try {
          c.close();
        } catch (Exception e) {
          toThrow = e;
        }
      }
      if (toThrow != null) {
        throw new RuntimeException(toThrow);
      }
    }
  }

  private static byte[] fullTraceRunOptions() {
    // Ideally this would use the generated Java sources for protocol buffers
    // and end up with something like the snippet below. However, generating
    // the Java files for the .proto files in tensorflow/core:protos_all is
    // a bit cumbersome in bazel until the proto_library rule is setup.
    //
    // See https://github.com/bazelbuild/bazel/issues/52#issuecomment-194341866
    // https://github.com/bazelbuild/rules_go/pull/121#issuecomment-251515362
    // https://github.com/bazelbuild/rules_go/pull/121#issuecomment-251692558
    //
    // For this test, for now, the use of specific bytes sufficies.
    return new byte[] {0x08, 0x03};
    /*
    return org.tensorflow.framework.RunOptions.newBuilder()
        .setTraceLevel(RunOptions.TraceLevel.FULL_TRACE)
        .build()
        .toByteArray();
    */
  }
}
