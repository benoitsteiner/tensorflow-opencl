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
// =============================================================================
// SampleInputs initializes candidate splits/threshold values randomly
// from incoming data for not-yet-initialized fertile nodes.
#include <ctime>
#include <unordered_map>
#include <set>

#include "tensorflow/contrib/tensor_forest/kernels/data_spec.h"
#include "tensorflow/contrib/tensor_forest/kernels/tree_utils.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/lib/random/distribution_sampler.h"
#include "tensorflow/core/lib/random/philox_random.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

using tensorforest::CheckTensorBounds;
using tensorforest::IsAllInitialized;

class SampleInputs : public OpKernel {
 public:
  explicit SampleInputs(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr(
        "split_initializations_per_input", &split_initializations_per_input_));
    OP_REQUIRES_OK(context, context->GetAttr(
        "split_sampling_random_seed", &split_sampling_random_seed_));
    // Set up the random number generator.
    if (split_sampling_random_seed_ == 0) {
      uint64 time_seed = static_cast<uint64>(std::clock());
      single_rand_ = std::unique_ptr<random::PhiloxRandom>(
          new random::PhiloxRandom(time_seed));
    } else {
      single_rand_ = std::unique_ptr<random::PhiloxRandom>(
          new random::PhiloxRandom(split_sampling_random_seed_));
    }

    rng_ = std::unique_ptr<random::SimplePhilox>(
        new random::SimplePhilox(single_rand_.get()));

    string serialized_proto;
    OP_REQUIRES_OK(context, context->GetAttr("input_spec", &serialized_proto));
    input_spec_.ParseFromString(serialized_proto);
  }

  // Returns the number of sparse values for example input_index.
  // Also returns the index where those features start in sparse_input_start
  // if any were found.
  int32 GetNumSparseFeatures(const Tensor& sparse_input_indices,
                             int32 input_index, int64* sparse_input_start) {
    // Binary search for input_index.
    // TODO(gilberth): Consider using std::lower_bound, std::upper_bound
    // for a simpler but possibly slower solution, or searching for
    // input_start and input_end simultaneously.
    const auto indices = sparse_input_indices.matrix<int64>();
    const int64 num_total = sparse_input_indices.shape().dim_size(0);
    int64 index;
    int64 low = 0;
    int64 high = num_total;

    while (true) {
      if (low == high) {
        return 0;
      }
      index = low + (high - low) / 2;
      const int64 feature_index = indices(index, 0);
      if (feature_index == input_index) {
        // found it.
        break;
      } else if (feature_index < input_index) {
        // Correct for the implicit floor in the index assignment.
        if (low == index) {
          return 0;
        }
        low = index;
      } else {
        high = index;
      }
    }

    // Scan for the start and end of the input_index range.
    int64 input_start = index;
    int64 val = indices(input_start, 0);
    while (val == input_index) {
      --input_start;
      if (input_start < 0) {
        break;
      }
      val = indices(input_start, 0);
    }
    *sparse_input_start = input_start + 1;
    int32 input_end = index;
    val = indices(input_end, 0);
    while (val == input_index) {
      ++input_end;
      if (input_end >= num_total) {
        break;
      }
      val = indices(input_end, 0);
    }
    return input_end - input_start - 1;
  }

  // increment_input implements a "++" operation for the situation when
  // you want to do something n times on an underlying iterator.
  // In an ideal world, this would be a built-in iterator adaptor.
  template <typename T>
  static void increment_input(const int n, T* it, int* count) {
    *count += 1;
    if (*count == n) {
      *count = 0;
      (*it)++;
    }
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input_data = context->input(0);
    const Tensor& sparse_input_indices = context->input(1);
    const Tensor& sparse_input_values = context->input(2);
    const Tensor& sparse_input_shape = context->input(3);
    const Tensor& input_weights = context->input(4);
    const Tensor& node_to_accumulator = context->input(5);
    const Tensor& leaves = context->input(6);
    const Tensor& split_features = context->input(7);
    const Tensor& split_thresholds = context->input(8);

    bool sparse_input = (sparse_input_indices.shape().dims() == 2);

    bool have_weights = (input_weights.shape().dim_size(0) > 0);

    if (sparse_input) {
      // TODO(gilberth): This is because we can't figure out the shape
      // of a sparse tensor at graph-build time, even if the dimension is
      // actually known.
      input_spec_.mutable_sparse(0)->set_size(
          sparse_input_shape.unaligned_flat<int64>()(1));
      OP_REQUIRES(context, sparse_input_shape.shape().dims() == 1,
                  errors::InvalidArgument(
                      "sparse_input_shape should be one-dimensional"));
      OP_REQUIRES(context,
                  sparse_input_shape.shape().dim_size(0) == 2,
                  errors::InvalidArgument(
                      "The sparse input data should be two-dimensional"));
      OP_REQUIRES(context, sparse_input_values.shape().dims() == 1,
                  errors::InvalidArgument(
                      "sparse_input_values should be one-dimensional"));
      OP_REQUIRES(context, sparse_input_indices.shape().dims() == 2,
                  errors::InvalidArgument(
                      "The sparse input data should be two-dimensional"));
      OP_REQUIRES(context,
                  sparse_input_indices.shape().dim_size(0) ==
                  sparse_input_values.shape().dim_size(0),
                  errors::InvalidArgument(
                      "sparse_input_indices and sparse_input_values should "
                      "agree on the number of non-zero values"));
      if (have_weights) {
        OP_REQUIRES(context, sparse_input_shape.unaligned_flat<int64>()(0) ==
                                 input_weights.shape().dim_size(0),
                    errors::InvalidArgument(
                        "sparse_input_values and input_weights should agree "
                        "on the number of inputs"));
      }
    }
    if (input_data.shape().dim_size(0) > 0) {
      OP_REQUIRES(context, input_data.shape().dims() == 2,
                  errors::InvalidArgument(
                  "input_data should be two-dimensional"));
      if (have_weights) {
        OP_REQUIRES(context, input_data.shape().dim_size(0) ==
                                 input_weights.shape().dim_size(0),
                    errors::InvalidArgument(
                        "input_data and input_weights should agree on the "
                        "number of inputs"));
      }
    }

    OP_REQUIRES(context, node_to_accumulator.shape().dims() == 1,
                errors::InvalidArgument(
                    "node_to_accumulator should be one-dimensional"));
    OP_REQUIRES(context, leaves.shape().dims() == 1,
                errors::InvalidArgument(
                    "leaves should be one-dimensional"));
    OP_REQUIRES(context, split_features.shape().dims() == 2,
                errors::InvalidArgument(
                    "split_features should be two-dimensional"));
    OP_REQUIRES(context, split_thresholds.shape().dims() == 2,
                errors::InvalidArgument(
                    "split_thresholds should be two-dimensional"));

    OP_REQUIRES(
        context,
        split_features.shape() == split_thresholds.shape(),
        errors::InvalidArgument(
            "split_features and split_thresholds should be the same shape."));

    // Check tensor bounds.
    if (!CheckTensorBounds(context, input_data)) return;
    if (!CheckTensorBounds(context, sparse_input_indices)) return;
    if (!CheckTensorBounds(context, sparse_input_values)) return;
    if (!CheckTensorBounds(context, sparse_input_shape)) return;
    if (!CheckTensorBounds(context, input_weights)) return;
    if (!CheckTensorBounds(context, node_to_accumulator)) return;
    if (!CheckTensorBounds(context, leaves)) return;
    if (!CheckTensorBounds(context, split_features)) return;
    if (!CheckTensorBounds(context, split_thresholds)) return;

    const auto leaves_vec = leaves.unaligned_flat<int32>();
    const auto node_map = node_to_accumulator.unaligned_flat<int32>();
    const auto features = split_features.tensor<int32, 2>();
    const auto thresholds = split_thresholds.tensor<float, 2>();
    const auto weights = input_weights.unaligned_flat<float>();

    const int32 num_data = static_cast<int32>(leaves.shape().dim_size(0));
    const int32 num_splits = static_cast<int32>(
        split_features.shape().dim_size(1));
    const int32 num_accumulators = static_cast<int32>(
        split_features.shape().dim_size(0));

    std::unordered_map<int32, std::set<int32>> accumulator_to_leaves;

    // The first pass just calculates num_output_accumulators.
    for (int32 i = 0; i < num_data; i++) {
      const int32 leaf = internal::SubtleMustCopy(leaves_vec(i));
      OP_REQUIRES(context, FastBoundsCheck(leaf, node_map.size()),
                  errors::InvalidArgument("leaf not in valid range."))
      const int32 accumulator = internal::SubtleMustCopy(node_map(leaf));

      // Check for non-fertile node or fertile node that is already
      // initialized.
      if (accumulator >= 0 &&
          !IsAllInitialized(features, accumulator, num_splits)) {
        accumulator_to_leaves[accumulator].insert(i);
      }
    }

    // Now we can allocate the outputs.
    int32 num_output_accumulators = static_cast<int32>(
        accumulator_to_leaves.size());
    VLOG(1) << "num output accumulators = " << num_output_accumulators;
    Tensor* accumulators_tensor = nullptr;
    TensorShape accumulators_shape;
    accumulators_shape.AddDim(num_output_accumulators);
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, accumulators_shape,
                                            &accumulators_tensor));
    auto accumulators_flat = accumulators_tensor->tensor<int32, 1>();

    Tensor* new_split_feature_rows_tensor = nullptr;
    TensorShape new_split_feature_rows_shape;
    new_split_feature_rows_shape.AddDim(num_output_accumulators);
    new_split_feature_rows_shape.AddDim(num_splits);
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, new_split_feature_rows_shape,
                                            &new_split_feature_rows_tensor));
    auto new_split_feature_rows_flat =
        new_split_feature_rows_tensor->tensor<int32, 2>();

    Tensor* new_split_threshold_rows_tensor = nullptr;
    TensorShape new_split_threshold_rows_shape;
    new_split_threshold_rows_shape.AddDim(num_output_accumulators);
    new_split_threshold_rows_shape.AddDim(num_splits);
    OP_REQUIRES_OK(context,
                   context->allocate_output(2, new_split_threshold_rows_shape,
                                            &new_split_threshold_rows_tensor));
    auto new_split_threshold_rows_flat =
        new_split_threshold_rows_tensor->tensor<float, 2>();

    // The second pass fills out the outputs.
    int output_slot = 0;
    for (const auto& active : accumulator_to_leaves) {
      const int32 accumulator = active.first;
      OP_REQUIRES(context, FastBoundsCheck(accumulator, num_accumulators),
                  errors::InvalidArgument("accumulator not in valid range."))
      const std::set<int32> inputs_for_accumulator = active.second;
      VLOG(1) << "Accumulator " << accumulator
                  << " gets new output slot " << output_slot;
      accumulators_flat(output_slot) = accumulator;

      // scatter_update updates entire rows, so we first copy the existing
      // rows into the output tensors, and then write over the values we
      // want to change.
      for (int split = 0; split < num_splits; split++) {
        new_split_feature_rows_flat(output_slot, split) =
            features(accumulator, split);
        new_split_threshold_rows_flat(output_slot, split) =
            thresholds(accumulator, split);
      }

      auto it = inputs_for_accumulator.begin();
      int input_used_count = 0;
      for (int split = 0;
           split < num_splits && it != inputs_for_accumulator.end(); split++) {
        if (new_split_feature_rows_flat(output_slot, split) < 0) {
          if (have_weights) {
            // If we have weights, we probabilistically reject inputs with
            // low weight.  Which means we might have to look at a bunch of
            // inputs -- maybe even all of them -- to fill this slot.
            while (it != inputs_for_accumulator.end()) {
              float w = weights(*it);
              if (rng_->RandFloat() <= w) {
                break;
              }
              increment_input(split_initializations_per_input_, &it,
                              &input_used_count);
            }
            if (it == inputs_for_accumulator.end()) {
              break;
            }
          }
          int32 index;
          float val;
          int64 sparse_input_start;
          int32 num_total_features = input_spec_.dense_features_size();
          if (sparse_input) {
            num_total_features += GetNumSparseFeatures(
                sparse_input_indices, *it, &sparse_input_start);
          }
          if (num_total_features == 0) {
            LOG(WARNING) << "num total features is zero.";
            break;
          }
          const int32 rand_feature = rng_->Uniform(num_total_features);
          if (rand_feature < input_spec_.dense_features_size()) {
            const auto inputs = input_data.tensor<float, 2>();
            index = rand_feature;
            val = inputs(*it, rand_feature);
          } else {
            const auto indices = sparse_input_indices.matrix<int64>();
            const auto values = sparse_input_values.vec<float>();
            const int32 sparse_index = sparse_input_start + rand_feature -
                                       input_spec_.dense_features_size();
            index =
                indices(sparse_index, 1) + input_spec_.dense_features_size();
            val = values(sparse_index);
          }
          CHECK(index >= 0)
              << "sample inputs chose negative feature: " << index;
          increment_input(split_initializations_per_input_, &it,
                          &input_used_count);

          VLOG(1) << "Over-writing @ " << output_slot << "," << split;
          new_split_feature_rows_flat(output_slot, split) = index;
          new_split_threshold_rows_flat(output_slot, split) = val;
        }
      }
      ++output_slot;
    }
  }

 private:
  int32 split_initializations_per_input_;
  int32 split_sampling_random_seed_;
  std::unique_ptr<random::PhiloxRandom> single_rand_;
  std::unique_ptr<random::SimplePhilox> rng_;
  tensorforest::TensorForestDataSpec input_spec_;
};

REGISTER_KERNEL_BUILDER(Name("SampleInputs").Device(DEVICE_CPU), SampleInputs);

}  // namespace tensorflow
