# -*- Python -*-
"""SYCL autoconfiguration.
`sycl_configure` depends on the following environment variables:

  * HOST_CXX_COMPILER:  The host C++ compiler
  * HOST_C_COMPILER:    The host C compiler
  * COMPUTECPP_TOOLKIT_PATH: The path to the ComputeCpp toolkit.
"""

_HOST_CXX_COMPILER = "HOST_CXX_COMPILER"
_HOST_C_COMPILER= "HOST_C_COMPILER"
_COMPUTECPP_TOOLKIT_PATH = "COMPUTECPP_TOOLKIT_PATH"

def _enable_sycl(repository_ctx):
  if "TF_NEED_OPENCL" in repository_ctx.os.environ:
    enable_sycl = repository_ctx.os.environ["TF_NEED_OPENCL"].strip()
    return enable_sycl == "1"
  return False
  
def auto_configure_fail(msg):
  """Output failure message when auto configuration fails."""
  red = "\033[0;31m"
  no_color = "\033[0m"
  fail("\n%sAuto-Configuration Error:%s %s\n" % (red, no_color, msg))
# END cc_configure common functions (see TODO above).

def find_c(repository_ctx):
  """Find host C compiler."""
  c_name = "gcc"
  if _HOST_C_COMPILER in repository_ctx.os.environ:
    c_name = repository_ctx.os.environ[_HOST_C_COMPILER].strip()
  if c_name.startswith("/"):
    return c_name
  c = repository_ctx.which(c_name)
  if c == None:
    fail("Cannot find C compiler, please correct your path.")
  return c

def find_cc(repository_ctx):
  """Find host C++ compiler."""
  cc_name = "g++"
  if _HOST_CXX_COMPILER in repository_ctx.os.environ:
    cc_name = repository_ctx.os.environ[_HOST_CXX_COMPILER].strip()
  if cc_name.startswith("/"):
    return cc_name
  cc = repository_ctx.which(cc_name)
  if cc == None:
    fail("Cannot find C++ compiler, please correct your path.")
  return cc

def find_computecpp_root(repository_ctx):
  """Find ComputeCpp compiler."""
  sycl_name = ""
  if _COMPUTECPP_TOOLKIT_PATH in repository_ctx.os.environ:
    sycl_name = repository_ctx.os.environ[_COMPUTECPP_TOOLKIT_PATH].strip()
  if sycl_name.startswith("/"):
    return sycl_name
  fail( "Cannot find SYCL compiler, please correct your path")

def _check_lib(repository_ctx, toolkit_path, lib):
  """Checks if lib exists under sycl_toolkit_path or fail if it doesn't.

  Args:
    repository_ctx: The repository context.
    toolkit_path: The toolkit directory containing the libraries.
    ib: The library to look for under toolkit_path.
  """
  lib_path = toolkit_path + "/" + lib
  if not repository_ctx.path(lib_path).exists:
    auto_configure_fail("Cannot find %s" % lib_path)

def _check_dir(repository_ctx, directory):
  """Checks whether the directory exists and fail if it does not.

  Args:
    repository_ctx: The repository context.
    directory: The directory to check the existence of.
  """
  if not repository_ctx.path(directory).exists:
    auto_configure_fail("Cannot find dir: %s" % directory)

def _symlink_dir(repository_ctx, src_dir, dest_dir):
  """Symlinks all the files in a directory.

  Args:
    repository_ctx: The repository context.
    src_dir: The source directory.
    dest_dir: The destination directory to create the symlinks in.
  """
  files = repository_ctx.path(src_dir).readdir()
  for src_file in files:
    repository_ctx.symlink(src_file, dest_dir + "/" + src_file.basename)

def _tpl(repository_ctx, tpl, substitutions={}, out=None):
  if not out:
    out = tpl.replace(":", "/")
  repository_ctx.template(
      out,
      Label("//third_party/sycl/%s.tpl" % tpl),
      substitutions)

def _file(repository_ctx, label):
  repository_ctx.template(
      label.replace(":", "/"),
      Label("//third_party/sycl/%s.tpl" % label),
      {})

_DUMMY_CROSSTOOL_BZL_FILE = """
def error_sycl_disabled():
  fail("ERROR: Building with --config=sycl but TensorFlow is not configured " +
       "to build with SYCL support. Please re-run ./configure and enter 'Y' " +
       "at the prompt to build with SYCL support.")

  native.genrule(
      name = "error_gen_crosstool",
      outs = ["CROSSTOOL"],
      cmd = "echo 'Should not be run.' && exit 1",
  )

  native.filegroup(
      name = "crosstool",
      srcs = [":CROSSTOOL"],
      output_licenses = ["unencumbered"],
  )
"""


_DUMMY_CROSSTOOL_BUILD_FILE = """
load("//crosstool:error_sycl_disabled.bzl", "error_sycl_disabled")

error_sycl_disabled()
"""

def _create_dummy_repository(repository_ctx):
  # Set up BUILD file for sycl/.
  _file(repository_ctx, "sycl:build_defs.bzl")
  _tpl(repository_ctx, "sycl:BUILD")
  _tpl(repository_ctx, "sycl:LICENSE.text")
  _tpl(repository_ctx, "sycl:platform.bzl")

  # Create dummy files for the SYCL toolkit since they are still required by
  # tensorflow/sycl/platform/default/build_config:sycl.
  repository_ctx.file("sycl/include/sycl.hpp", "")
  repository_ctx.file("sycl/lib/libComputeCpp.so", "")

  # If sycl_configure is not configured to build with SYCL support, and the user
  # attempts to build with --config=sycl, add a dummy build rule to intercept
  # this and fail with an actionable error message.
  repository_ctx.file("crosstool/error_sycl_disabled.bzl",
                      _DUMMY_CROSSTOOL_BZL_FILE)
  repository_ctx.file("crosstool/BUILD", _DUMMY_CROSSTOOL_BUILD_FILE)


def _sycl_autoconf_imp(repository_ctx):
  """Implementation of the sycl_autoconf rule."""
  if not _enable_sycl(repository_ctx):
    _create_dummy_repository(repository_ctx)
  else:
    # copy template files
    _file(repository_ctx, "sycl:build_defs.bzl")
    _tpl(repository_ctx, "sycl:BUILD")
    _tpl(repository_ctx, "sycl:platform.bzl")
    _file(repository_ctx, "crosstool:BUILD")
    _tpl(repository_ctx, "crosstool:computecpp",
    {
      "%{host_cxx_compiler}" : find_cc(repository_ctx),
      "%{host_c_compiler}" : find_c(repository_ctx),
    })

    computecpp_root = find_computecpp_root(repository_ctx);
    _check_dir(repository_ctx, computecpp_root)

    _tpl(repository_ctx, "crosstool:CROSSTOOL",
    {
      "%{computecpp_toolkit_path}" : computecpp_root,
    })

    # symlink libraries
    _check_lib(repository_ctx, computecpp_root+"/lib", "libComputeCpp.so" )
    _symlink_dir(repository_ctx, computecpp_root + "/lib", "sycl/lib")
    _symlink_dir(repository_ctx, computecpp_root + "/include", "sycl/include")
    _symlink_dir(repository_ctx, computecpp_root + "/bin", "sycl/bin")

sycl_configure = repository_rule(
  implementation = _sycl_autoconf_imp,
  local = True,
)
"""Detects and configures the SYCL toolchain.

Add the following to your WORKSPACE FILE:

```python
sycl_configure(name = "local_config_sycl")
```

Args:
  name: A unique name for this workspace rule.
"""
