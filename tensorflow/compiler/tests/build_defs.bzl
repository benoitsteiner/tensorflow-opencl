"""Build rules for Tensorflow/XLA testing."""

load("@local_config_cuda//cuda:build_defs.bzl", "cuda_is_configured")

def all_backends():
  if cuda_is_configured():
    return ["cpu", "gpu"]
  else:
    return ["cpu"]

def tf_xla_py_test(name, srcs=[], deps=[], tags=[], data=[], main=None,
                   backends=None, **kwargs):
  """Generates py_test targets, one per XLA backend.

  This rule generates py_test() targets named name_backend, for each backend
  in all_backends(). The rule also generates a test suite with named `name` that
  tests all backends for the test.

  For example, the following rule generates test cases foo_test_cpu,
  foo_test_gpu, and a test suite name foo_test that tests both.
  tf_xla_py_test(
      name="foo_test",
      srcs="foo_test.py",
      deps=[...],
  )

  Args:
    name: Name of the target.
    srcs: Sources for the target.
    deps: Dependencies of the target.
    tags: Tags to apply to the generated targets.
    data: Data dependencies of the target.
    main: Same as py_test's main attribute.
    backends: A list of backends to test. Supported values include "cpu" and
      "gpu". If not specified, defaults to all backends.
    **kwargs: keyword arguments passed onto the generated py_test() rules.
  """
  if backends == None:
    backends = all_backends()

  test_names = []
  for backend in backends:
    test_name = "{}_{}".format(name, backend)
    backend_tags = ["tf_xla_{}".format(backend)]
    backend_args = []
    backend_deps = []
    backend_data = []
    if backend == "cpu":
      backend_args += ["--test_device=XLA_CPU",
                       "--types=DT_FLOAT,DT_DOUBLE,DT_INT32,DT_INT64,DT_BOOL"]
    elif backend == "gpu":
      backend_args += ["--test_device=XLA_GPU",
                       "--types=DT_FLOAT,DT_DOUBLE,DT_INT32,DT_INT64,DT_BOOL"]
      backend_tags += ["requires-gpu-sm35"]
    else:
      fail("Unknown backend {}".format(backend))

    native.py_test(
        name=test_name,
        srcs=srcs,
        srcs_version="PY2AND3",
        args=backend_args,
        main="{}.py".format(name) if main == None else main,
        data=data + backend_data,
        deps=deps + backend_deps,
        tags=tags + backend_tags,
        **kwargs
    )
    test_names.append(test_name)
  native.test_suite(name=name, tests=test_names)

def generate_backend_suites(backends=[]):
  """Generates per-backend test_suites that run all tests for a backend."""
  if not backends:
    backends = all_backends()
  for backend in backends:
    native.test_suite(name="%s_tests" % backend, tags=["tf_xla_%s" % backend])

