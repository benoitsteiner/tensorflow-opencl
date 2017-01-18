# TensorFlow Version Semantics

## Semantic Versioning 2.0

TensorFlow follows Semantic Versioning 2.0 ([semver](http://semver.org)) for its
public API. Each release version of TensorFlow has the form `MAJOR.MINOR.PATCH`.
Changes to the each number have the following meaning:

* **MAJOR**:  Backwards incompatible changes.  Code and data that worked with
  a previous major release will not necessarily work with a new release.
  However, in some cases existing TensorFlow data (graphs, checkpoints, and
  other protobufs) may be migratable to the newer release; see below for details
  on data compatibility.

* **MINOR**: Backwards compatible features, speed improvements, etc.  Code and
  data that worked with a previous minor release *and* which depends only the
  public API will continue to work unchanged.  For details on what is and is
  not the public API, see below.

* **PATCH**: Backwards compatible bug fixes.

## Public API

Only the public APIs of TensorFlow are backwards compatible across minor and
patch versions.  The public APIs consist of

* The documented public [Python](../api_docs/python) API, excluding `tf.contrib`.
  This includes all public functions and classes (with names not starting with
  `_`) in the tensorflow module and its submodules. Note that the code in
  the `examples/` to `tools/` directories is not reachable through the
  tensorflow Python module and is thus not covered by the compatibility
  guarantee.

  If a symbol is available through the tensorflow Python module or its
  submodules, but is not documented, then it is _not_ considered part of the
  public API.

* The [C API](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/c/c_api.h).

* The following protocol buffer files:
  [`attr_value`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/attr_value.proto),
  [`config`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/protobuf/config.proto),
  [`event`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/util/event.proto),
  [`graph`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/graph.proto),
  [`op_def`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/op_def.proto),
  [`reader_base`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/reader_base.proto),
  [`summary`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto),
  [`tensor`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/tensor.proto),
  [`tensor_shape`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/tensor_shape.proto),
  and [`types`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/types.proto).

## Other Languages

In addition to Python and C, TensorFlow also provides APIs for:

- [C++](../api_docs/cc) (exposed through header files in
[`tensorflow/cc`](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/cc).
- [Java](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/java)
([#5](https://github.com/tensorflow/tensorflow/issues/5)), and
- [Go](https://godoc.org/github.com/tensorflow/tensorflow/tensorflow/go)

However, these three are **not** covered by the versioning scheme at this time
and can be changed in backward incompatible ways between releases.

## Details That Are Not Public

The following are specifically **not** part of the public API: they are allowed
to change without notice across minor releases and even patch releases if bug
fixes require it:

* **Details of composite ops:**  Many public functions in Python expand to
  several primitive ops in the graph, and these details will be part of any
  graphs saved to disk as `GraphDef`s.  These details are allowed to change for
  minor releases. In particular, regressions tests that check for exact
  matching between graphs are likely to break across minor releases, even though
  the behavior of the graph should be unchanged and existing checkpoints will
  still work.

* **Floating point numerical details:** The specific floating point values
  computed by ops may change at any time: users should rely only on approximate
  accuracy and numerical stability, not on the specific bits computed.  Changes
  to numerical formulas in minor and patch releases should result in comparable
  or improved accuracy, with the caveat that in machine learning improved
  accuracy of specific formulas may result in worse accuracy for the overall
  system.

* **Random numbers:** The specific random numbers computed by the [random
  ops](../api_docs/python/constant_op.html#random-tensors) may change at any
  time: users should rely only on approximately correct distributions and
  statistical strength, not the specific bits computed.  However, we will make
  changes to random bits rarely and ideally never for patch releases, and all
  such intended changes will be documented.


## Compatibility for Graphs and Checkpoints

Many users of TensorFlow will be saving graphs and trained models to disk for
later evaluation or more training, often changing versions of TensorFlow in the
process.  First, following semver, any graph or checkpoint written out with one
version of TensorFlow can be loaded and evaluated with a later version of
TensorFlow with the same major release.  However, we will endeavour to preserve
backwards compatibility even across major releases when possible, so that the
serialized files are usable over long periods of time.

There are two main classes of saved TensorFlow data: graphs and checkpoints.
Graphs describe the data flow graphs of ops to be run during training and
inference, and checkpoints contain the saved tensor values of variables in a
graph.

Graphs are serialized via the `GraphDef` protocol buffer.  To facilitate (rare)
backwards incompatible changes to graphs, each `GraphDef` has an integer version
separate from the TensorFlow version.  The semantics are:

* Each version of TensorFlow supports an interval of `GraphDef` versions.  This
  interval with be constant across patch releases, and will only grow across
  minor releases.  Dropping support for a `GraphDef` version will only occur
  for a major release of TensorFlow.

* Newly created graphs use the newest `GraphDef` version.

* If a given version of TensorFlow supports the `GraphDef` version of a graph,
  it will load and evaluate with the same behavior as when it was written out
  (except for floating point numerical details and random numbers), regardless
  of the major version of TensorFlow.  In particular, all checkpoint files will
  be compatible.

* If the `GraphDef` upper bound is increased to X in a (minor) release, there
  will be at least six months before the lower bound is increased to X.

For example (numbers and versions hypothetical), TensorFlow 1.2 might support
`GraphDef` versions 4 to 7.  TensorFlow 1.3 could add `GraphDef` version 8 and
support versions 4 to 8.  At least six months later, TensorFlow 2.0.0 could drop
support for versions 4 to 7, leaving version 8 only.

Finally, when support for a `GraphDef` version is dropped, we will attempt to
provide tools for automatically converting graphs to a newer supported
`GraphDef` version.

For developer-level details about `GraphDef` versioning, including how to evolve
the versions to account for changes, see [TensorFlow Data
Versioning](data_versions.md).
