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
"""Tests for variable store."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.platform import test


class VariableScopeTest(test.TestCase):

  def testGetVar(self):
    vs = variable_scope._get_default_variable_store()
    v = vs.get_variable("v", [1])
    v1 = vs.get_variable("v", [1])
    assert v == v1

  def testNameExists(self):
    vs = variable_scope._get_default_variable_store()
    # No check by default, so we can both create and get existing names.
    v = vs.get_variable("v", [1])
    v1 = vs.get_variable("v", [1])
    assert v == v1
    # When reuse is False, we fail when variables are already there.
    vs.get_variable("w", [1], reuse=False)  # That's ok.
    with self.assertRaises(ValueError):
      vs.get_variable("v", [1], reuse=False)  # That fails.
    # When reuse is True, we fail when variables are new.
    vs.get_variable("v", [1], reuse=True)  # That's ok.
    with self.assertRaises(ValueError):
      vs.get_variable("u", [1], reuse=True)  # That fails.

  def testNamelessStore(self):
    vs = variable_scope._get_default_variable_store()
    vs.get_variable("v1", [2])
    vs.get_variable("v2", [2])
    expected_names = ["%s:0" % name for name in ["v1", "v2"]]
    self.assertEqual(
        set(expected_names), set([v.name for v in vs._vars.values()]))

  def testVarScopeInitializer(self):
    with self.test_session() as sess:
      init = init_ops.constant_initializer(0.3)
      with variable_scope.variable_scope("tower") as tower:
        with variable_scope.variable_scope("foo", initializer=init):
          v = variable_scope.get_variable("v", [])
          sess.run(variables_lib.initialize_variables([v]))
          self.assertAllClose(v.eval(), 0.3)
        with variable_scope.variable_scope(tower, initializer=init):
          w = variable_scope.get_variable("w", [])
          sess.run(variables_lib.initialize_variables([w]))
          self.assertAllClose(w.eval(), 0.3)

  def testVarScopeDType(self):
    with self.test_session():
      with variable_scope.variable_scope("tower") as tower:
        with variable_scope.variable_scope("foo", dtype=dtypes.float16):
          v = variable_scope.get_variable("v", [])
          self.assertEqual(v.dtype, dtypes.float16_ref)
        with variable_scope.variable_scope(tower, dtype=dtypes.float16):
          w = variable_scope.get_variable("w", [])
          self.assertEqual(w.dtype, dtypes.float16_ref)

  def testInitFromNonTensorValue(self):
    with self.test_session() as sess:
      v = variable_scope.get_variable("v", initializer=4, dtype=dtypes.int32)
      sess.run(variables_lib.initialize_variables([v]))
      self.assertAllClose(v.eval(), 4)

      w = variable_scope.get_variable(
          "w", initializer=numpy.array([1, 2, 3]), dtype=dtypes.int64)
      sess.run(variables_lib.initialize_variables([w]))
      self.assertAllClose(w.eval(), [1, 2, 3])

      with self.assertRaises(TypeError):
        variable_scope.get_variable("x", initializer={})

  def testInitFromNonInitializer(self):
    with self.test_session() as sess:
      # Test various dtypes with zeros initializer as following:
      types = [
          dtypes.int8, dtypes.uint8, dtypes.int16, dtypes.uint16, dtypes.int32,
          dtypes.int64, dtypes.bool
      ]

      # Use different varibale_name to distinguish various dtypes
      for (i, dtype) in enumerate(types):
        x = variable_scope.get_variable(
            name="x%d" % i, shape=(3, 4), dtype=dtype)
        y = variable_scope.get_variable(
            name="y%d" % i,
            shape=(3, 4),
            dtype=dtype,
            initializer=init_ops.zeros_initializer(dtype=dtype))

        variables_lib.global_variables_initializer().run()
        self.assertAllEqual(x.eval(), y.eval())

  def testVarScopeCachingDevice(self):
    with self.test_session():
      caching_device = "/job:moo"
      with variable_scope.variable_scope("tower"):
        with variable_scope.variable_scope(
            "caching", caching_device=caching_device):
          v = variable_scope.get_variable("v", [])
          self.assertTrue(v.value().device.startswith(caching_device))

          with variable_scope.variable_scope("child"):
            v2 = variable_scope.get_variable("v", [])
            self.assertTrue(v2.value().device.startswith(caching_device))

          with variable_scope.variable_scope("not_cached", caching_device=""):
            v2_not_cached = variable_scope.get_variable("v", [])
            self.assertFalse(v2_not_cached.value().device.startswith(
                caching_device))

          with variable_scope.variable_scope(
              "not_cached_identity_device",
              caching_device=lambda op: op.device):
            v2_identity_device = variable_scope.get_variable("v", [])
            self.assertFalse(v2_identity_device.value().device.startswith(
                caching_device))

          with variable_scope.variable_scope("we_will_do_it_live") as vs_live:
            vs_live.set_caching_device("/job:live")
            v_live = variable_scope.get_variable("v", [])
            self.assertTrue(v_live.value().device.startswith("/job:live"))

        v_tower = variable_scope.get_variable("v", [])
        self.assertFalse(v_tower.value().device.startswith(caching_device))

  def testVarScopeRegularizer(self):
    with self.test_session() as sess:
      init = init_ops.constant_initializer(0.3)

      def regularizer1(v):
        return math_ops.reduce_mean(v) + 0.1

      def regularizer2(v):
        return math_ops.reduce_mean(v) + 0.2

      with variable_scope.variable_scope(
          "tower", regularizer=regularizer1) as tower:
        with variable_scope.variable_scope("foo", initializer=init):
          v = variable_scope.get_variable("v", [])
          sess.run(variables_lib.initialize_variables([v]))
          losses = ops.get_collection(ops.GraphKeys.REGULARIZATION_LOSSES)
          self.assertEqual(1, len(losses))
          self.assertAllClose(losses[0].eval(), 0.4)
        with variable_scope.variable_scope(tower, initializer=init) as vs:
          u = variable_scope.get_variable("u", [])
          vs.set_regularizer(regularizer2)
          w = variable_scope.get_variable("w", [])
          # Next 3 variable not regularized to test disabling regularization.
          x = variable_scope.get_variable(
              "x", [], regularizer=variable_scope.no_regularizer)
          with variable_scope.variable_scope(
              "baz", regularizer=variable_scope.no_regularizer):
            y = variable_scope.get_variable("y", [])
          vs.set_regularizer(variable_scope.no_regularizer)
          z = variable_scope.get_variable("z", [])
          # Check results.
          losses = ops.get_collection(ops.GraphKeys.REGULARIZATION_LOSSES)
          self.assertEqual(3, len(losses))
          sess.run(variables_lib.initialize_variables([u, w, x, y, z]))
          self.assertAllClose(losses[0].eval(), 0.4)
          self.assertAllClose(losses[1].eval(), 0.4)
          self.assertAllClose(losses[2].eval(), 0.5)
        with variable_scope.variable_scope("foo", reuse=True):
          v = variable_scope.get_variable("v",
                                          [])  # "v" is alredy there, reused
          losses = ops.get_collection(ops.GraphKeys.REGULARIZATION_LOSSES)
          self.assertEqual(3, len(losses))  # No new loss added.

  def testInitializeFromValue(self):
    with self.test_session() as sess:
      init = constant_op.constant(0.1)
      w = variable_scope.get_variable("v", initializer=init)
      sess.run(variables_lib.initialize_variables([w]))
      self.assertAllClose(w.eval(), 0.1)

      with self.assertRaisesRegexp(ValueError, "shape"):
        # We disallow explicit shape specification when initializer is constant.
        variable_scope.get_variable("u", [1], initializer=init)

      with variable_scope.variable_scope("foo", initializer=init):
        # Constant initializer can be passed through scopes if needed.
        v = variable_scope.get_variable("v")
        sess.run(variables_lib.initialize_variables([v]))
        self.assertAllClose(v.eval(), 0.1)

      # Check that non-float32 initializer creates a non-float32 variable.
      init = constant_op.constant(1, dtype=dtypes.int32)
      t = variable_scope.get_variable("t", initializer=init)
      self.assertEqual(t.dtype.base_dtype, dtypes.int32)

      # Raise error if `initializer` dtype and `dtype` are not identical.
      with self.assertRaisesRegexp(ValueError, "don't match"):
        variable_scope.get_variable("s", initializer=init, dtype=dtypes.float64)

  def testControlDeps(self):
    with self.test_session() as sess:
      v0 = variable_scope.get_variable(
          "v0", [1], initializer=init_ops.constant_initializer(0))
      with ops.control_dependencies([v0.value()]):
        v1 = variable_scope.get_variable(
            "v1", [1], initializer=init_ops.constant_initializer(1))
        add = v1 + v0
      # v0 should be uninitialized.
      with self.assertRaisesRegexp(errors_impl.OpError, "uninitialized"):
        sess.run(v0)
      # We should be able to initialize and run v1 without initializing
      # v0, even if the variable was created with a control dep on v0.
      sess.run(v1.initializer)
      self.assertEqual(1, sess.run(v1))
      # v0 should still be uninitialized.
      with self.assertRaisesRegexp(errors_impl.OpError, "uninitialized"):
        sess.run(v0)
      with self.assertRaisesRegexp(errors_impl.OpError, "uninitialized"):
        sess.run(add)
      # If we initialize v0 we should be able to run 'add'.
      sess.run(v0.initializer)
      sess.run(add)

  def testControlFlow(self):
    with self.test_session() as sess:
      v0 = variable_scope.get_variable(
          "v0", [], initializer=init_ops.constant_initializer(0))
      var_dict = {}

      # Call get_variable in each of the cond clauses.
      def var_in_then_clause():
        v1 = variable_scope.get_variable(
            "v1", [1], initializer=init_ops.constant_initializer(1))
        var_dict["v1"] = v1
        return v1 + v0

      def var_in_else_clause():
        v2 = variable_scope.get_variable(
            "v2", [1], initializer=init_ops.constant_initializer(2))
        var_dict["v2"] = v2
        return v2 + v0

      add = control_flow_ops.cond(
          math_ops.less(v0, 10), var_in_then_clause, var_in_else_clause)
      v1 = var_dict["v1"]
      v2 = var_dict["v2"]
      # We should be able to initialize and run v1 and v2 without initializing
      # v0, even if the variable was created with a control dep on v0.
      sess.run(v1.initializer)
      self.assertEqual([1], sess.run(v1))
      sess.run(v2.initializer)
      self.assertEqual([2], sess.run(v2))
      # v0 should still be uninitialized.
      with self.assertRaisesRegexp(errors_impl.OpError, "uninitialized"):
        sess.run(v0)
      # We should not be able to run 'add' yet.
      with self.assertRaisesRegexp(errors_impl.OpError, "uninitialized"):
        sess.run(add)
      # If we initialize v0 we should be able to run 'add'.
      sess.run(v0.initializer)
      sess.run(add)

  def testGetVariableScope(self):
    # Test the get_variable_scope() function and setting properties of result.
    with self.test_session() as sess:
      init = init_ops.constant_initializer(0.3)
      with variable_scope.variable_scope("foo"):
        new_init1 = variable_scope.get_variable_scope().initializer
        self.assertEqual(new_init1, None)
        # Check that we can set initializer like this.
        variable_scope.get_variable_scope().set_initializer(init)
        v = variable_scope.get_variable("v", [])
        sess.run(variables_lib.initialize_variables([v]))
        self.assertAllClose(v.eval(), 0.3)
        # Check that we can set reuse.
        variable_scope.get_variable_scope().reuse_variables()
        with self.assertRaises(ValueError):  # Fail, w does not exist yet.
          variable_scope.get_variable("w", [1])
      # Check that the set initializer goes away.
      new_init = variable_scope.get_variable_scope().initializer
      self.assertEqual(new_init, None)

  def testVarScope(self):
    with self.test_session():
      with variable_scope.variable_scope("tower") as tower:
        self.assertEqual(tower.name, "tower")
        with ops.name_scope("scope") as sc:
          self.assertEqual(sc, "tower/scope/")

      with variable_scope.variable_scope("foo"):
        with variable_scope.variable_scope("bar") as bar:
          self.assertEqual(bar.name, "foo/bar")
          with ops.name_scope("scope") as sc:
            self.assertEqual(sc, "foo/bar/scope/")

      with variable_scope.variable_scope("foo"):
        with variable_scope.variable_scope(tower, reuse=True) as tower_shared:
          self.assertEqual(tower_shared.name, "tower")
          with ops.name_scope("scope") as sc:
            self.assertEqual(sc, "foo_1/tower/scope/")

  def testVarScopeNameScope(self):
    with self.test_session():
      with ops.name_scope("scope1"):
        with variable_scope.variable_scope("tower") as tower:
          with ops.name_scope("scope2") as sc2:
            self.assertEqual(sc2, "scope1/tower/scope2/")
        with variable_scope.variable_scope(
            tower):  # Re-entering acts like another "tower".
          with ops.name_scope("scope2") as sc2:
            self.assertEqual(sc2, "scope1/tower_1/scope2/")
        with variable_scope.variable_scope(
            "tower"):  # Re-entering by string acts the same.
          with ops.name_scope("scope2") as sc2:
            self.assertEqual(sc2, "scope1/tower_2/scope2/")

      with ops.name_scope("scope3"):
        with variable_scope.variable_scope("tower"):
          with ops.name_scope("scope2") as sc2:
            self.assertEqual(sc2, "scope3/tower/scope2/")
        with variable_scope.variable_scope(tower):
          with ops.name_scope("scope2") as sc2:
            self.assertEqual(sc2, "scope3/tower_1/scope2/")

      root_var_scope = variable_scope.get_variable_scope()
      with ops.name_scope("scope4"):
        with variable_scope.variable_scope(root_var_scope):
          with ops.name_scope("scope2") as sc2:
            self.assertEqual(sc2, "scope4/scope2/")

  def testVarScopeOriginalNameScope(self):
    with self.test_session():
      with ops.name_scope("scope1"):
        with variable_scope.variable_scope("tower") as tower:
          self.assertEqual(tower.original_name_scope, "scope1/tower/")
          with ops.name_scope("scope2") as sc2:
            self.assertEqual(sc2, "scope1/tower/scope2/")
      with ops.name_scope("scope2"):
        with variable_scope.variable_scope(tower) as tower1:
          # Re-entering preserves original name scope.
          self.assertEqual(tower1.original_name_scope, "scope1/tower/")
          with ops.name_scope("foo") as sc2:
            self.assertEqual(sc2, "scope2/tower/foo/")
        # Test re-entering original name scope.
        with ops.name_scope(tower.original_name_scope):
          with ops.name_scope("bar") as sc3:
            self.assertEqual(sc3, "scope1/tower/bar/")
      with ops.name_scope("scope2"):
        with variable_scope.variable_scope(tower):
          with ops.name_scope(tower.original_name_scope):
            with ops.name_scope("bar") as sc3:
              self.assertEqual(sc3, "scope1/tower/bar_1/")

  def testVarScopeObjectReuse(self):
    with self.test_session():
      vs = None
      with variable_scope.variable_scope("jump", reuse=True) as scope:
        vs = scope

      with variable_scope.variable_scope(vs) as jump:
        self.assertTrue(jump.reuse)

      with variable_scope.variable_scope(vs, reuse=True) as jump_reuse:
        self.assertTrue(jump_reuse.reuse)

      with variable_scope.variable_scope(vs, reuse=False) as jump_no_reuse:
        self.assertFalse(jump_no_reuse.reuse)

      with variable_scope.variable_scope("jump", reuse=False) as scope:
        vs = scope

      with variable_scope.variable_scope(vs) as jump:
        self.assertFalse(jump.reuse)

      with variable_scope.variable_scope(vs, reuse=True) as jump_reuse:
        self.assertTrue(jump_reuse.reuse)

      with variable_scope.variable_scope(vs, reuse=False) as jump_no_reuse:
        self.assertFalse(jump_no_reuse.reuse)

  def testVarOpScope(self):
    with self.test_session():
      with ops.name_scope("scope1"):
        with variable_scope.variable_scope("tower", "default", []):
          self.assertEqual(
              variable_scope.get_variable("w", []).name, "tower/w:0")
          with ops.name_scope("scope2") as sc2:
            self.assertEqual(sc2, "scope1/tower/scope2/")
        with variable_scope.variable_scope("tower", "default", []):
          with self.assertRaises(ValueError):
            variable_scope.get_variable("w", [])
          with ops.name_scope("scope2") as sc2:
            self.assertEqual(sc2, "scope1/tower_1/scope2/")

      with ops.name_scope("scope2"):
        with variable_scope.variable_scope(None, "default", []):
          self.assertEqual(
              variable_scope.get_variable("w", []).name, "default/w:0")
          with ops.name_scope("scope2") as sc2:
            self.assertEqual(sc2, "scope2/default/scope2/")
        with variable_scope.variable_scope(None, "default", []):
          self.assertEqual(
              variable_scope.get_variable("w", []).name, "default_1/w:0")
          with ops.name_scope("scope2") as sc2:
            self.assertEqual(sc2, "scope2/default_1/scope2/")

  def testVarOpScopeUniqueNamesInterleavedSubstringScopes(self):
    with self.test_session():
      with variable_scope.variable_scope(None, "defaultScope1"):
        with variable_scope.variable_scope(None, "layer"):
          self.assertEqual(
              variable_scope.get_variable("w", []).name,
              "defaultScope1/layer/w:0")
      with variable_scope.variable_scope(None, "defaultScope1"):
        with variable_scope.variable_scope(None, "layer"):
          self.assertEqual(
              variable_scope.get_variable("w", []).name,
              "defaultScope1_1/layer/w:0")
      with variable_scope.variable_scope(None, "defaultScope"):
        with variable_scope.variable_scope(None, "layer"):
          self.assertEqual(
              variable_scope.get_variable("w", []).name,
              "defaultScope/layer/w:0")
      with variable_scope.variable_scope(None, "defaultScope1"):
        with variable_scope.variable_scope(None, "layer"):
          self.assertEqual(
              variable_scope.get_variable("w", []).name,
              "defaultScope1_2/layer/w:0")

  def testVarOpScopeUniqueNamesWithJump(self):
    with self.test_session():
      with variable_scope.variable_scope("default") as default:
        with variable_scope.variable_scope(None, "layer"):
          self.assertEqual(
              variable_scope.get_variable("w", []).name,
              "default/layer/w:0")
        with variable_scope.variable_scope(None, "layer"):
          self.assertEqual(
              variable_scope.get_variable("w", []).name,
              "default/layer_1/w:0")
        with variable_scope.variable_scope(default):
          pass
        # No matter the jump in the middle, unique numbering continues.
        with variable_scope.variable_scope(None, "layer"):
          self.assertEqual(
              variable_scope.get_variable("w", []).name,
              "default/layer_2/w:0")

  def testVarOpScopeReuse(self):
    with self.test_session():
      with variable_scope.variable_scope("outer") as outer:
        with variable_scope.variable_scope("tower", "default", []):
          self.assertEqual(
              variable_scope.get_variable("w", []).name, "outer/tower/w:0")
          with ops.name_scope("scope2") as sc2:
            self.assertEqual(sc2, "outer/tower/scope2/")
        with variable_scope.variable_scope(None, "default", []):
          self.assertEqual(
              variable_scope.get_variable("w", []).name, "outer/default/w:0")
          with ops.name_scope("scope2") as sc2:
            self.assertEqual(sc2, "outer/default/scope2/")

      with variable_scope.variable_scope(outer, reuse=True) as outer:
        with variable_scope.variable_scope("tower", "default", []):
          self.assertEqual(
              variable_scope.get_variable("w", []).name, "outer/tower/w:0")
          with ops.name_scope("scope2") as sc2:
            self.assertEqual(sc2, "outer_1/tower/scope2/")
        with variable_scope.variable_scope(None, "default", []):
          self.assertEqual(
              variable_scope.get_variable("w", []).name, "outer/default/w:0")
          with ops.name_scope("scope2") as sc2:
            self.assertEqual(sc2, "outer_1/default/scope2/")

  def testVarScopeGetVar(self):
    with self.test_session():
      with variable_scope.variable_scope("root"):
        with variable_scope.variable_scope("towerA") as tower_a:
          va = variable_scope.get_variable("v", [1])
          self.assertEqual(va.name, "root/towerA/v:0")

        with variable_scope.variable_scope(tower_a, reuse=True):
          va2 = variable_scope.get_variable("v", [1])
          self.assertEqual(va2, va)

        with variable_scope.variable_scope("towerB"):
          vb = variable_scope.get_variable("v", [1])
          self.assertEqual(vb.name, "root/towerB/v:0")

        with self.assertRaises(ValueError):
          with variable_scope.variable_scope("towerA"):
            va2 = variable_scope.get_variable("v", [1])

        with variable_scope.variable_scope("towerA", reuse=True):
          va2 = variable_scope.get_variable("v", [1])
          self.assertEqual(va2, va)

        with variable_scope.variable_scope("foo"):
          with variable_scope.variable_scope("bar"):
            v = variable_scope.get_variable("v", [1])
            self.assertEqual(v.name, "root/foo/bar/v:0")
            with variable_scope.variable_scope(tower_a, reuse=True):
              va3 = variable_scope.get_variable("v", [1])
              self.assertEqual(va, va3)

        with self.assertRaises(ValueError):
          with variable_scope.variable_scope(tower_a, reuse=True):
            with variable_scope.variable_scope("baz"):
              variable_scope.get_variable("v", [1])

        with self.assertRaises(ValueError) as exc:
          with variable_scope.variable_scope(tower_a, reuse=True):
            variable_scope.get_variable("v", [2])  # Different shape.
        self.assertEqual("shape" in str(exc.exception), True)

        with self.assertRaises(ValueError) as exc:
          with variable_scope.variable_scope(tower_a, reuse=True):
            variable_scope.get_variable("v", [1], dtype=dtypes.int32)
        self.assertEqual("dtype" in str(exc.exception), True)

  def testVarScopeOuterScope(self):
    with self.test_session():
      with variable_scope.variable_scope("outer") as outer:
        pass
      with variable_scope.variable_scope(outer):
        self.assertEqual(variable_scope.get_variable("w", []).name, "outer/w:0")
        with ops.name_scope("scope2") as sc2:
          self.assertEqual(sc2, "outer_1/scope2/")
        with variable_scope.variable_scope("default"):
          self.assertEqual(
              variable_scope.get_variable("w", []).name, "outer/default/w:0")
          with ops.name_scope("scope2") as sc2:
            self.assertEqual(sc2, "outer_1/default/scope2/")

      with variable_scope.variable_scope(outer, reuse=True):
        self.assertEqual(variable_scope.get_variable("w", []).name, "outer/w:0")
        with ops.name_scope("scope2") as sc2:
          self.assertEqual(sc2, "outer_2/scope2/")
        with variable_scope.variable_scope("default", reuse=True):
          self.assertEqual(
              variable_scope.get_variable("w", []).name, "outer/default/w:0")
          with ops.name_scope("scope2") as sc2:
            self.assertEqual(sc2, "outer_2/default/scope2/")

  def testVarScopeNestedOuterScope(self):
    with self.test_session():
      with variable_scope.variable_scope("outer") as outer:
        with variable_scope.variable_scope(outer):
          self.assertEqual(
              variable_scope.get_variable("w", []).name, "outer/w:0")
          with ops.name_scope("scope2") as sc2:
            self.assertEqual(sc2, "outer/outer/scope2/")
        with variable_scope.variable_scope("default"):
          self.assertEqual(
              variable_scope.get_variable("w", []).name, "outer/default/w:0")
          with ops.name_scope("scope2") as sc2:
            self.assertEqual(sc2, "outer/default/scope2/")

        with variable_scope.variable_scope(outer, reuse=True):
          self.assertEqual(
              variable_scope.get_variable("w", []).name, "outer/w:0")
          with ops.name_scope("scope2") as sc2:
            self.assertEqual(sc2, "outer/outer_1/scope2/")
        with variable_scope.variable_scope("default", reuse=True):
          self.assertEqual(
              variable_scope.get_variable("w", []).name, "outer/default/w:0")
          with ops.name_scope("scope2") as sc2:
            self.assertEqual(sc2, "outer/default_1/scope2/")

  def testVarOpScopeReuseParam(self):
    with self.test_session():
      with variable_scope.variable_scope("outer") as outer:
        with variable_scope.variable_scope("tower", "default", []):
          self.assertEqual(
              variable_scope.get_variable("w", []).name, "outer/tower/w:0")
          with ops.name_scope("scope2") as sc2:
            self.assertEqual(sc2, "outer/tower/scope2/")
        with variable_scope.variable_scope(None, "default", []):
          self.assertEqual(
              variable_scope.get_variable("w", []).name, "outer/default/w:0")
          with ops.name_scope("scope2") as sc2:
            self.assertEqual(sc2, "outer/default/scope2/")

      with variable_scope.variable_scope(outer) as outer:
        with variable_scope.variable_scope("tower", "default", reuse=True):
          self.assertEqual(
              variable_scope.get_variable("w", []).name, "outer/tower/w:0")
          with ops.name_scope("scope2") as sc2:
            self.assertEqual(sc2, "outer_1/tower/scope2/")
        outer.reuse_variables()
        with variable_scope.variable_scope(None, "default", []):
          self.assertEqual(
              variable_scope.get_variable("w", []).name, "outer/default/w:0")
          with ops.name_scope("scope2") as sc2:
            self.assertEqual(sc2, "outer_1/default/scope2/")

  def testVarOpScopeReuseError(self):
    with self.test_session():
      with self.assertRaises(ValueError):
        with variable_scope.variable_scope(None, "default", reuse=True):
          self.assertEqual(
              variable_scope.get_variable("w", []).name, "outer/tower/w:0")

  def testVarOpScopeOuterScope(self):
    with self.test_session():
      with variable_scope.variable_scope("outer") as outer:
        pass
      with variable_scope.variable_scope(outer, "default", []):
        self.assertEqual(variable_scope.get_variable("w", []).name, "outer/w:0")
        with ops.name_scope("scope2") as sc2:
          self.assertEqual(sc2, "outer_1/scope2/")
        with variable_scope.variable_scope(None, "default", []):
          self.assertEqual(
              variable_scope.get_variable("w", []).name, "outer/default/w:0")
          with ops.name_scope("scope2") as sc2:
            self.assertEqual(sc2, "outer_1/default/scope2/")

      with variable_scope.variable_scope(outer, "default", reuse=True):
        self.assertEqual(variable_scope.get_variable("w", []).name, "outer/w:0")
        with ops.name_scope("scope2") as sc2:
          self.assertEqual(sc2, "outer_2/scope2/")
        outer.reuse_variables()
        with variable_scope.variable_scope(None, "default", []):
          self.assertEqual(
              variable_scope.get_variable("w", []).name, "outer/default/w:0")
          with ops.name_scope("scope2") as sc2:
            self.assertEqual(sc2, "outer_2/default/scope2/")

  def testVarOpScopeNestedOuterScope(self):
    with self.test_session():
      with variable_scope.variable_scope("outer") as outer:
        with variable_scope.variable_scope(outer, "default", []):
          self.assertEqual(
              variable_scope.get_variable("w", []).name, "outer/w:0")
          with ops.name_scope("scope2") as sc2:
            self.assertEqual(sc2, "outer/outer/scope2/")
        with variable_scope.variable_scope(None, "default", []):
          self.assertEqual(
              variable_scope.get_variable("w", []).name, "outer/default/w:0")
          with ops.name_scope("scope2") as sc2:
            self.assertEqual(sc2, "outer/default/scope2/")

      with variable_scope.variable_scope(outer, "default", reuse=True):
        self.assertEqual(variable_scope.get_variable("w", []).name, "outer/w:0")
        with ops.name_scope("scope2") as sc2:
          self.assertEqual(sc2, "outer_1/scope2/")
        with variable_scope.variable_scope(None, "default", []):
          self.assertEqual(
              variable_scope.get_variable("w", []).name, "outer/default/w:0")
          with ops.name_scope("scope2") as sc2:
            self.assertEqual(sc2, "outer_1/default/scope2/")

  def testGetLocalVar(self):
    with self.test_session():
      # Check that local variable respects naming.
      with variable_scope.variable_scope("outer") as outer:
        with variable_scope.variable_scope(outer, "default", []):
          local_var = variable_scope.get_local_variable(
              "w", [], collections=["foo"])
          self.assertEqual(local_var.name, "outer/w:0")

      # Since variable is local, it should be in the local variable collection
      # but not the trainable collection.
      self.assertIn(local_var,
                    ops.get_collection(ops.GraphKeys.LOCAL_VARIABLES))
      self.assertIn(local_var, ops.get_collection("foo"))
      self.assertNotIn(local_var,
                       ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES))

      # Check that local variable respects `reuse`.
      with variable_scope.variable_scope(outer, "default", reuse=True):
        self.assertEqual(
            variable_scope.get_local_variable("w", []).name, "outer/w:0")

  def testGetVarWithDevice(self):
    g = ops.Graph()
    varname_type = []

    def device_func(op):
      if op.type in ["Variable", "VariableV2"]:
        varname_type.append((op.name, op.get_attr("dtype")))
      return "/gpu:0"

    with g.as_default():
      with ops.device(device_func):
        _ = variable_scope.get_variable("x", (100, 200))
        _ = variable_scope.get_variable(
            "y", dtype=dtypes.int64, initializer=numpy.arange(73))
    self.assertEqual(varname_type[0], ("x", dtypes.float32))
    self.assertEqual(varname_type[1], ("y", dtypes.int64))


def axis0_into1_partitioner(shape=None, **unused_kwargs):
  part = [1] * len(shape)
  return part


def axis0_into2_partitioner(shape=None, **unused_kwargs):
  part = [1] * len(shape)
  part[0] = 2
  return part


def axis0_into3_partitioner(shape=None, **unused_kwargs):
  part = [1] * len(shape)
  part[0] = 3
  return part


class VariableScopeWithPartitioningTest(test.TestCase):

  def testInitFromNonInitializer(self):
    with self.test_session() as sess:
      # Test various dtypes with zeros initializer as following:
      types = [
          dtypes.int8, dtypes.uint8, dtypes.int16, dtypes.uint16, dtypes.int32,
          dtypes.int64, dtypes.bool
      ]

      # Use different varibale_name to distinguish various dtypes
      for (i, dtype) in enumerate(types):
        x = variable_scope.get_variable(
            name="x%d" % i,
            shape=(3, 4),
            dtype=dtype,
            partitioner=axis0_into2_partitioner)
        y = variable_scope.get_variable(
            name="y%d" % i,
            shape=(6, 4),
            dtype=dtype,
            partitioner=axis0_into2_partitioner,
            initializer=init_ops.zeros_initializer(dtype=dtype))

        variables_lib.global_variables_initializer().run()
        # x and y would become var list after partition
        val_x = sess.run(list(x))
        val_y = sess.run(list(y))

        self.assertAllEqual(val_x, val_y)

  def testResultNameMatchesRequested(self):
    with variable_scope.variable_scope(
        "scope0", partitioner=axis0_into2_partitioner):
      v = variable_scope.get_variable("name0", shape=(3, 1, 1))
      self.assertEqual(v.name, "scope0/name0")
      v_concat = v.as_tensor()
      self.assertEqual(v_concat.name, "scope0/name0:0")
      variables = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
      self.assertIn("scope0/name0/part_0:0", [x.name for x in variables])
      self.assertIn("scope0/name0/part_1:0", [x.name for x in variables])
      self.assertNotIn("scope0/name0/part_2:0", [x.name for x in variables])

  def testBreaksIfPartitioningChanges(self):
    with variable_scope.variable_scope(
        "scope0", partitioner=axis0_into2_partitioner):
      variable_scope.get_variable("name0", shape=(3, 1, 1))

    with variable_scope.variable_scope(
        "scope0", partitioner=axis0_into3_partitioner, reuse=True):
      with self.assertRaisesRegexp(
          ValueError,
          "Trying to reuse partitioned variable .* but specified partitions .* "
          "and found partitions .*"):
        variable_scope.get_variable("name0", shape=(3, 1, 1))

    with variable_scope.variable_scope(
        "scope0", partitioner=axis0_into1_partitioner, reuse=True):
      with self.assertRaisesRegexp(
          ValueError,
          "Trying to reuse partitioned variable .* but specified partitions .* "
          "and found partitions .*"):
        variable_scope.get_variable("name0", shape=(3, 1, 1))

  def testReturnsExistingConcatenatedValueIfReuse(self):
    with variable_scope.variable_scope(
        "scope0", partitioner=axis0_into2_partitioner):
      v_concat = variable_scope.get_variable("name0", shape=(3, 1, 1))
      variable_scope.get_variable_scope().reuse_variables()
      v_concat_2 = variable_scope.get_variable("name0", shape=(3, 1, 1))
      self.assertEqual(v_concat, v_concat_2)

  def testAllowsReuseWithoutPartitioner(self):
    with variable_scope.variable_scope(
        "scope0", partitioner=axis0_into2_partitioner):
      v = variable_scope.get_variable("name0", shape=(3, 1, 1))
    with variable_scope.variable_scope("scope0", reuse=True):
      v_reused = variable_scope.get_variable("name0")
    self.assertEqual(v, v_reused)

  def testPropagatePartitionerOnReopening(self):
    with variable_scope.variable_scope(
        "scope0", partitioner=axis0_into2_partitioner) as vs:
      self.assertEqual(axis0_into2_partitioner, vs.partitioner)
      with variable_scope.variable_scope(vs) as vs1:
        self.assertEqual(axis0_into2_partitioner, vs1.partitioner)

  def testScalarIgnoresPartitioner(self):
    with variable_scope.variable_scope(
        "scope0", partitioner=axis0_into2_partitioner):
      v = variable_scope.get_variable("name0", shape=())
      self.assertEqual(v.name, "scope0/name0:0")
      variables = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
      self.assertIn("scope0/name0:0", [x.name for x in variables])

  def testPartitionConcatenatesAlongCorrectAxis(self):

    def _part_axis_0(**unused_kwargs):
      return (2, 1, 1)

    def _part_axis_1(**unused_kwargs):
      return (1, 2, 1)

    with variable_scope.variable_scope("root"):
      v0 = variable_scope.get_variable(
          "n0", shape=(2, 2, 2), partitioner=_part_axis_0)
      v1 = variable_scope.get_variable(
          "n1", shape=(2, 2, 2), partitioner=_part_axis_1)

    self.assertEqual(v0.get_shape(), (2, 2, 2))
    self.assertEqual(v1.get_shape(), (2, 2, 2))

    n0_0 = ops.get_default_graph().get_tensor_by_name("root/n0/part_0:0")
    n0_1 = ops.get_default_graph().get_tensor_by_name("root/n0/part_1:0")
    self.assertEqual(n0_0.get_shape(), (1, 2, 2))
    self.assertEqual(n0_1.get_shape(), (1, 2, 2))

    n1_0 = ops.get_default_graph().get_tensor_by_name("root/n1/part_0:0")
    n1_1 = ops.get_default_graph().get_tensor_by_name("root/n1/part_1:0")
    self.assertEqual(n1_0.get_shape(), (2, 1, 2))
    self.assertEqual(n1_1.get_shape(), (2, 1, 2))


class VariableScopeWithCustomGetterTest(test.TestCase):

  def testNonCallableGetterFails(self):
    with self.assertRaisesRegexp(ValueError, r"custom_getter .* not callable:"):
      with variable_scope.variable_scope("scope0", custom_getter=3):
        variable_scope.get_variable("name0")
    with self.assertRaisesRegexp(ValueError, r"custom_getter .* not callable:"):
      variable_scope.get_variable("name0", custom_getter=3)

  def testNoSideEffectsWithIdentityCustomGetter(self):
    called = [0]

    def custom_getter(getter, *args, **kwargs):
      called[0] += 1
      return getter(*args, **kwargs)

    with variable_scope.variable_scope(
        "scope", custom_getter=custom_getter) as scope:
      v = variable_scope.get_variable("v", [1])
    with variable_scope.variable_scope(scope, reuse=True):
      v2 = variable_scope.get_variable("v", [1])
    with variable_scope.variable_scope("new_scope") as new_scope:
      v3 = variable_scope.get_variable("v3", [1])
    with variable_scope.variable_scope(
        new_scope, reuse=True, custom_getter=custom_getter):
      v4 = variable_scope.get_variable("v3", [1])

    self.assertEqual(v, v2)
    self.assertEqual(v3, v4)
    self.assertEqual(3, called[0])  # skipped one in the first new_scope

  def testGetterThatCreatesTwoVariablesAndSumsThem(self):

    def custom_getter(getter, name, *args, **kwargs):
      g_0 = getter("%s/0" % name, *args, **kwargs)
      g_1 = getter("%s/1" % name, *args, **kwargs)
      with ops.name_scope("custom_getter"):
        return g_0 + g_1

    with variable_scope.variable_scope("scope", custom_getter=custom_getter):
      v = variable_scope.get_variable("v", [1, 2, 3])

    self.assertEqual([1, 2, 3], v.get_shape())
    true_vars = variables_lib.trainable_variables()
    self.assertEqual(2, len(true_vars))
    self.assertEqual("scope/v/0:0", true_vars[0].name)
    self.assertEqual("scope/v/1:0", true_vars[1].name)
    self.assertEqual("custom_getter/add:0", v.name)
    with self.test_session() as sess:
      variables_lib.global_variables_initializer().run()
      np_vars, np_v = sess.run([true_vars, v])
      self.assertAllClose(np_v, sum(np_vars))


class PartitionInfoTest(test.TestCase):

  def testConstructorChecks(self):
    # Invalid arg types.
    with self.assertRaises(TypeError):
      variable_scope._PartitionInfo(full_shape=None, var_offset=[0, 1])
    with self.assertRaises(TypeError):
      variable_scope._PartitionInfo(full_shape=[0, 1], var_offset=None)
    with self.assertRaises(TypeError):
      variable_scope._PartitionInfo(full_shape="foo", var_offset=[0, 1])
    with self.assertRaises(TypeError):
      variable_scope._PartitionInfo(full_shape=[0, 1], var_offset="foo")

    # full_shape and var_offset must have same length.
    with self.assertRaises(ValueError):
      variable_scope._PartitionInfo(full_shape=[0, 1], var_offset=[0])
    # Offset must always be less than shape.
    with self.assertRaises(ValueError):
      variable_scope._PartitionInfo(full_shape=[1, 1], var_offset=[0, 1])

  def testSingleOffset(self):
    partition_info = variable_scope._PartitionInfo(
        full_shape=[9, 3], var_offset=[4, 0])
    self.assertEqual(4, partition_info.single_offset([1, 3]))

    # Tests when the variable isn't partitioned at all.
    partition_info = variable_scope._PartitionInfo(
        full_shape=[9, 3], var_offset=[0, 0])
    self.assertEqual(0, partition_info.single_offset([9, 3]))

  def testSingleSliceDim(self):
    partition_info = variable_scope._PartitionInfo(
        full_shape=[9, 3], var_offset=[4, 0])
    # Invalid shape.
    with self.assertRaises(TypeError):
      partition_info.single_slice_dim(None)

    # Rank of shape differs from full_shape.
    with self.assertRaises(ValueError):
      partition_info.single_slice_dim([1, 2, 3])

    # Shape is too large given var_offset (4+6 > 9).
    with self.assertRaises(ValueError):
      partition_info.single_slice_dim([6, 3])

    # Multiple possible slice dim from shape.
    with self.assertRaises(ValueError):
      partition_info.single_slice_dim([1, 1])

    partition_info = variable_scope._PartitionInfo(
        full_shape=[9, 3], var_offset=[0, 0])
    self.assertEqual(1, partition_info.single_slice_dim([9, 2]))
    partition_info = variable_scope._PartitionInfo(
        full_shape=[9, 3], var_offset=[4, 0])
    self.assertEqual(0, partition_info.single_slice_dim([2, 3]))


if __name__ == "__main__":
  test.main()
