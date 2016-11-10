/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

%include "tensorflow/python/platform/base.i"

%{

#include "tensorflow/python/client/tf_session_helper.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/public/version.h"

%}

// Required to use PyArray_* functions.
%init %{
tensorflow::ImportNumpy();
%}

// TensorFlow version and GraphDef versions
%constant const char* __version__ = TF_VERSION_STRING;
%constant int GRAPH_DEF_VERSION = TF_GRAPH_DEF_VERSION;
%constant int GRAPH_DEF_VERSION_MIN_CONSUMER = TF_GRAPH_DEF_VERSION_MIN_CONSUMER;
%constant int GRAPH_DEF_VERSION_MIN_PRODUCER = TF_GRAPH_DEF_VERSION_MIN_PRODUCER;

// Git version information
%constant const char* __git_version__ = tf_git_version();

// Compiler
%constant const char* __compiler_version__ = tf_compiler_version();

// Release the Python GIL for the duration of most methods.
%exception {
  Py_BEGIN_ALLOW_THREADS;
  $action
  Py_END_ALLOW_THREADS;
}

// The target input to TF_SetTarget() is passed as a null-terminated
// const char*.
%typemap(in) (const char* target) {
  $1 = PyBytes_AsString($input);
  if (!$1) {
    // Python has raised an error.
    SWIG_fail;
  }
}

////////////////////////////////////////////////////////////////////////////////
// BEGIN TYPEMAPS FOR tensorflow::TF_Run_wrapper()
////////////////////////////////////////////////////////////////////////////////

// The wrapper also takes a list of fetch and target names.  In Python this is
// represented as a list of strings.
%typemap(in) const tensorflow::NameVector& (
    tensorflow::NameVector temp,
    tensorflow::Safe_PyObjectPtr temp_string_list(tensorflow::make_safe(nullptr))) {
  if (!PyList_Check($input)) {
    SWIG_fail;
  }

  Py_ssize_t len = PyList_Size($input);

  temp_string_list = tensorflow::make_safe(PyList_New(len));
  if (!temp_string_list) {
    SWIG_fail;
  }

  for (Py_ssize_t i = 0; i < len; ++i) {
    PyObject* elem = PyList_GetItem($input, i);
    if (!elem) {
      SWIG_fail;
    }

    // Keep a reference to the string in case the incoming list is modified.
    PyList_SET_ITEM(temp_string_list.get(), i, elem);
    Py_INCREF(elem);

    char* fetch_name = PyBytes_AsString(elem);
    if (!fetch_name) {
      PyErr_SetString(PyExc_TypeError,
                      "a fetch or target name was not a string");
      SWIG_fail;
    }

    // TODO(mrry): Avoid copying the fetch name in, if this impacts performance.
    temp.push_back(fetch_name);
  }
  $1 = &temp;
}

// Define temporaries for the argout outputs.
%typemap(in, numinputs=0) tensorflow::PyObjectVector* out_values (
    tensorflow::PyObjectVector temp) {
  $1 = &temp;
}
%typemap(in, numinputs=0) char** out_handle (
    char* temp) {
  $1 = &temp;
}

// Build a Python list of outputs and return it.
%typemap(argout) tensorflow::PyObjectVector* out_values {
  tensorflow::Safe_PyObjectVector out_values_safe;
  for (size_t i = 0; i < $1->size(); ++i) {
    out_values_safe.emplace_back(tensorflow::make_safe($1->at(i)));
  }

  $result = PyList_New($1->size());
  if (!$result) {
    SWIG_fail;
  }

  for (size_t i = 0; i < $1->size(); ++i) {
    PyList_SET_ITEM($result, i, $1->at(i));
    out_values_safe[i].release();
  }
}

// Return the handle as a python string object.
%typemap(argout) char** out_handle {
%#if PY_MAJOR_VERSION < 3
  $result = PyString_FromStringAndSize(
%#else
  $result = PyUnicode_FromStringAndSize(
%#endif
    *$1, strlen(*$1));
  delete[] *$1;
}

////////////////////////////////////////////////////////////////////////////////
// END TYPEMAPS FOR tensorflow::TF_Run_wrapper()
////////////////////////////////////////////////////////////////////////////////

// Typemap for functions that return a TF_Buffer struct. This typemap creates a
// Python string from the TF_Buffer and returns it. The TF_Buffer.data string
// is not expected to be NULL-terminated, and TF_Buffer.length does not count
// the terminator.
%typemap(out) TF_Buffer (TF_GetOpList,TF_GetBuffer) {
  $result = PyBytes_FromStringAndSize(
      reinterpret_cast<const char*>($1.data), $1.length);
}

// Include the functions from c_api.h, except TF_Run.
%ignoreall
%unignore TF_Code;
%unignore TF_Status;
%unignore TF_Buffer;
%unignore TF_NewBuffer;
%unignore TF_NewBufferFromString;
%unignore TF_DeleteBuffer;
%unignore TF_GetBuffer;
%unignore TF_NewStatus;
%unignore TF_DeleteStatus;
%unignore TF_GetCode;
%unignore TF_Message;
%unignore TF_SessionOptions;
%rename("_TF_SetTarget") TF_SetTarget;
%rename("_TF_SetConfig") TF_SetConfig;
%rename("_TF_NewSessionOptions") TF_NewSessionOptions;
%unignore TF_DeleteSessionOptions;
%unignore TF_NewDeprecatedSession;
%unignore TF_CloseDeprecatedSession;
%unignore TF_DeleteDeprecatedSession;
%unignore TF_ExtendGraph;
%unignore TF_NewLibrary;
%unignore TF_LoadLibrary;
%unignore TF_GetOpList;
%include "tensorflow/c/c_api.h"
%ignoreall

%insert("python") %{
  def TF_NewSessionOptions(target=None, config=None):
    # NOTE: target and config are validated in the session constructor.
    opts = _TF_NewSessionOptions()
    if target is not None:
      _TF_SetTarget(opts, target)
    if config is not None:
      from tensorflow.python.framework import errors
      with errors.raise_exception_on_not_ok_status() as status:
        config_str = config.SerializeToString()
        _TF_SetConfig(opts, config_str, status)
    return opts
%}

// Include the wrapper for TF_Run from tf_session_helper.h.

// The %exception block above releases the Python GIL for the length
// of each wrapped method. We disable this behavior for TF_Run
// because it uses the Python allocator.
%noexception tensorflow::TF_Run_wrapper;
%rename(TF_Run) tensorflow::TF_Run_wrapper;
%unignore tensorflow;
%unignore TF_Run;
%unignore EqualGraphDefWrapper;

// Include the wrapper for TF_PRunSetup from tf_session_helper.h.

// The %exception block above releases the Python GIL for the length
// of each wrapped method. We disable this behavior for TF_PRunSetup
// because it uses the Python allocator.
%noexception tensorflow::TF_PRunSetup_wrapper;
%rename(TF_PRunSetup) tensorflow::TF_PRunSetup_wrapper;
%unignore tensorflow;
%unignore TF_PRunSetup;

// Include the wrapper for TF_PRun from tf_session_helper.h.

// The %exception block above releases the Python GIL for the length
// of each wrapped method. We disable this behavior for TF_PRun
// because it uses the Python allocator.
%noexception tensorflow::TF_PRun_wrapper;
%rename(TF_PRun) tensorflow::TF_PRun_wrapper;
%unignore tensorflow;
%unignore TF_PRun;

%unignore tensorflow::TF_Reset_wrapper;
%insert("python") %{
def TF_Reset(target, containers=None, config=None):
  from tensorflow.python.framework import errors
  opts = TF_NewSessionOptions(target=target, config=config)
  try:
    with errors.raise_exception_on_not_ok_status() as status:
      TF_Reset_wrapper(opts, containers, status)
  finally:
    TF_DeleteSessionOptions(opts)
%}

%include "tensorflow/python/client/tf_session_helper.h"

%unignoreall
