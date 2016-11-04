// Copyright 2016 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ============================================================================

#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

REGISTER_OP("VarHandleOp")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Attr("dtype: type")
    .Attr("shape: shape")
    .Output("resource: resource")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->Scalar());
      DataType t;
      c->GetAttr("dtype", &t);
      c->set_output_handle_dtype(0, t);
      TensorShapeProto p;
      c->GetAttr("shape", &p);
      shape_inference::ShapeHandle s;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeProto(p, &s));
      c->set_output_handle_shape(0, s);
      return Status::OK();
    })
    .Doc(R"(
Creates a handle to a Variable resource.

container: the container this variable is placed in.
shared_name: the name by which this variable is referred to.
dtype: the type of this variable. Must agree with the dtypes
  of all ops using this variable.
shape: The (possibly partially specified) shape of this variable.
)");

Status CreateAssignShapeFn(shape_inference::InferenceContext* c) {
  DataType handle_dtype = c->input_handle_dtype(0);
  DataType value_dtype;
  c->GetAttr("dtype", &value_dtype);
  if (handle_dtype != value_dtype) {
    return errors::InvalidArgument(
        "Trying to initialize handle for variable with wrong dtype. "
        "Expected ",
        handle_dtype, " got ", value_dtype);
  }
  shape_inference::ShapeHandle s = c->input_handle_shape(0);
  shape_inference::ShapeHandle value_shape = c->input(1);
  shape_inference::ShapeHandle unused;
  TF_RETURN_IF_ERROR(c->Merge(s, value_shape, &unused));
  return Status::OK();
}

REGISTER_OP("CreateVariableOp")
    .Input("resource: resource")
    .Input("value: dtype")
    .Attr("dtype: type")
    .SetShapeFn(CreateAssignShapeFn)
    .Doc(R"(
Creates a variable resource.

resource: handle to the resource in which to store the variable.
value: the value to set the new tensor to use.
dtype: the dtype of the value.
)");

REGISTER_OP("ReadVariableOp")
    .Input("resource: resource")
    .Output("value: dtype")
    .Attr("dtype: type")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      DataType handle_dtype = c->input_handle_dtype(0);
      DataType value_dtype;
      c->GetAttr("dtype", &value_dtype);
      if (handle_dtype != value_dtype) {
        return errors::InvalidArgument(
            "Trying to read variable with wrong dtype. "
            "Expected ",
            handle_dtype, " got ", value_dtype);
      }
      c->set_output(0, c->input_handle_shape(0));
      return Status::OK();
    })
    .Doc(R"(
Reads the value of a variable.

The tensor returned by this operation is immutable.

The value returned by this operation is guaranteed to be influenced by all the
writes on which this operation depends directly or indirectly, and to not be
influenced by any of the writes which depend directly or indirectly on this
operation.

resource: handle to the resource in which to store the variable.
dtype: the dtype of the value.
)");

REGISTER_OP("AssignVariableOp")
    .Input("resource: resource")
    .Input("value: dtype")
    .Attr("dtype: type")
    .SetShapeFn(CreateAssignShapeFn)
    .Doc(R"(
Assigns a new value to a variable.

Any ReadVariableOp with a control dependency on this op is guaranteed to return
this value or a subsequent newer value of the variable.

resource: handle to the resource in which to store the variable.
value: the value to set the new tensor to use.
dtype: the dtype of the value.
)");

REGISTER_OP("AssignAddVariableOp")
    .Input("resource: resource")
    .Input("value: dtype")
    .Output("new_value: dtype")
    .Attr("dtype: type")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      DataType handle_dtype = c->input_handle_dtype(0);
      DataType value_dtype;
      c->GetAttr("dtype", &value_dtype);
      if (handle_dtype != value_dtype) {
        return errors::InvalidArgument(
            "Trying to initialize handle for variable with wrong dtype. "
            "Expected ",
            handle_dtype, " got ", value_dtype);
      }
      shape_inference::ShapeHandle s = c->input_handle_shape(0);
      shape_inference::ShapeHandle value_shape = c->input(1);
      shape_inference::ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->Merge(s, value_shape, &unused));
      c->set_output(0, value_shape);
      return Status::OK();
    })
    .Doc(R"(
Adds a value to the current value of a variable.

Any ReadVariableOp which depends directly or indirectly on this assign is
guaranteed to see the incremented value or a subsequent newer one.

Outputs the incremented value, which can be used to totally order the
increments to this variable.

resource: handle to the resource in which to store the variable.
value: the value by which the variable will be incremented.
new_value: the new value of the variable.
dtype: the dtype of the value.
)");

}  // namespace tensorflow
