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

// See docs in ../ops/sdca_ops.cc.

#define EIGEN_USE_THREADS

#include <stddef.h>
#include <atomic>
#include <cmath>
#include <functional>
#include <limits>
#include <string>
#include <unordered_set>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/hinge-loss.h"
#include "tensorflow/core/kernels/logistic-loss.h"
#include "tensorflow/core/kernels/smooth-hinge-loss.h"
#include "tensorflow/core/kernels/squared-loss.h"
#include "tensorflow/core/lib/core/coding.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/lib/random/distribution_sampler.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/fingerprint.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/guarded_philox_random.h"
#include "tensorflow/core/util/sparse/group_iterator.h"
#include "tensorflow/core/util/sparse/sparse_tensor.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

namespace {

using UnalignedFloatVector = TTypes<const float>::UnalignedConstVec;
using UnalignedInt64Vector = TTypes<const int64>::UnalignedConstVec;

// Statistics computed with input (ModelWeights, Example).
struct ExampleStatistics {
  // Logits for each class.
  // For binary case, this should be a vector of length 1; while for multiclass
  // case, this vector has the same length as the number of classes, where each
  // value corresponds to one class.
  // Using InlinedVector is to avoid heap allocation for small number of
  // classes, and 3 is chosen to minimize memory usage for the multiclass case.
  gtl::InlinedVector<double, 3> wx;

  // Multiplication using the previous weights.
  gtl::InlinedVector<double, 3> prev_wx;

  // Sum of squared feature values occurring in the example divided by
  // L2 * sum(example_weights).
  double normalized_squared_norm = 0;

  // Num_weight_vectors equals to the number of classification classes in the
  // multiclass case; while for binary case, it is 1.
  ExampleStatistics(const int num_weight_vectors)
      : wx(num_weight_vectors, 0.0), prev_wx(num_weight_vectors, 0.0) {}
};

class Regularizations {
 public:
  Regularizations(){};

  // Initialize() must be called immediately after construction.
  Status Initialize(OpKernelConstruction* const context) {
    TF_RETURN_IF_ERROR(context->GetAttr("l1", &symmetric_l1_));
    TF_RETURN_IF_ERROR(context->GetAttr("l2", &symmetric_l2_));
    shrinkage_ = symmetric_l1_ / symmetric_l2_;
    return Status::OK();
  }

  // Proximal SDCA shrinking for L1 regularization.
  double Shrink(const double weight) const {
    const double shrinked = std::max(std::abs(weight) - shrinkage_, 0.0);
    if (shrinked > 0.0) {
      return std::copysign(shrinked, weight);
    }
    return 0.0;
  }

  // Vectorized float variant of the above.
  Eigen::Tensor<float, 1, Eigen::RowMajor> EigenShrinkVector(
      const Eigen::Tensor<float, 1, Eigen::RowMajor> weights) const {
    // Proximal step on the weights which is sign(w)*|w - shrinkage|+.
    return weights.sign() * ((weights.abs() - weights.constant(shrinkage_))
                                 .cwiseMax(weights.constant(0.0)));
  }

  // Matrix float variant of the above.
  Eigen::Tensor<float, 2, Eigen::RowMajor> EigenShrinkMatrix(
      const Eigen::Tensor<float, 2, Eigen::RowMajor> weights) const {
    // Proximal step on the weights which is sign(w)*|w - shrinkage|+.
    return weights.sign() * ((weights.abs() - weights.constant(shrinkage_))
                                 .cwiseMax(weights.constant(0.0)));
  }

  float symmetric_l2() const { return symmetric_l2_; }

 private:
  float symmetric_l1_ = 0;
  float symmetric_l2_ = 0;

  // L1 divided by L2, pre-computed for use during weight shrinking.
  double shrinkage_ = 0;

  TF_DISALLOW_COPY_AND_ASSIGN(Regularizations);
};

class ModelWeights;

// Struct describing a single example.
class Example {
 public:
  // Compute matrix vector product between weights (a matrix) and features
  // (a vector). This method also computes the normalized example norm used
  // in SDCA update.
  // For multiclass case, num_weight_vectors equals to the number of classes;
  // while for binary case, it is 1.
  const ExampleStatistics ComputeWxAndWeightedExampleNorm(
      const int num_loss_partitions, const ModelWeights& model_weights,
      const Regularizations& regularization,
      const int num_weight_vectors) const;

  float example_label() const { return example_label_; }

  float example_weight() const { return example_weight_; }

  double squared_norm() const { return squared_norm_; }

  // Sparse features associated with the example.
  // Indices and Values are the associated feature index, and values. Values
  // can be optionally absent, in which we case we implicitly assume a value of
  // 1.0f.
  struct SparseFeatures {
    std::unique_ptr<UnalignedInt64Vector> indices;
    std::unique_ptr<UnalignedFloatVector> values;  // nullptr encodes optional.
  };

  // A dense vector which is a row-slice of the underlying matrix.
  struct DenseVector {
    // Returns a row slice from the matrix.
    Eigen::TensorMap<Eigen::Tensor<const float, 1, Eigen::RowMajor>> Row()
        const {
      return Eigen::TensorMap<Eigen::Tensor<const float, 1, Eigen::RowMajor>>(
          data_matrix.data() + row_index * data_matrix.dimension(1),
          data_matrix.dimension(1));
    }

    // Returns a row slice as a 1 * F matrix, where F is the number of features.
    Eigen::TensorMap<Eigen::Tensor<const float, 2, Eigen::RowMajor>>
    RowAsMatrix() const {
      return Eigen::TensorMap<Eigen::Tensor<const float, 2, Eigen::RowMajor>>(
          data_matrix.data() + row_index * data_matrix.dimension(1), 1,
          data_matrix.dimension(1));
    }

    const TTypes<float>::ConstMatrix data_matrix;
    const int64 row_index;
  };

 private:
  std::vector<SparseFeatures> sparse_features_;
  std::vector<std::unique_ptr<DenseVector>> dense_vectors_;

  float example_label_ = 0;
  float example_weight_ = 0;
  double squared_norm_ = 0;  // sum squared norm of the features.

  // Examples fills Example in a multi-threaded way.
  friend class Examples;

  // ModelWeights use each example for model update w += \alpha * x_{i};
  friend class ModelWeights;
};

// Weights related to features. For example, say you have two sets of sparse
// features i.e. age bracket and country, then FeatureWeightsDenseStorage hold
// the parameters for it. We keep track of the original weight passed in and the
// delta weight which the optimizer learns in each call to the optimizer.
class FeatureWeightsDenseStorage {
 public:
  FeatureWeightsDenseStorage(const TTypes<const float>::Matrix nominals,
                             TTypes<float>::Matrix deltas)
      : nominals_(nominals), deltas_(deltas) {}

  // Check if a feature index is with-in the bounds.
  bool IndexValid(const int64 index) const {
    return index >= 0 && index < deltas_.dimension(1);
  }

  // Nominals here are the original weight matrix.
  TTypes<const float>::Matrix nominals() const { return nominals_; }

  // Delta weights durining mini-batch updates.
  TTypes<float>::Matrix deltas() const { return deltas_; }

  // Updates delta weights based on active dense features in the example and
  // the corresponding dual residual.
  void UpdateDenseDeltaWeights(
      const Eigen::ThreadPoolDevice& device,
      const Example::DenseVector& dense_vector,
      const std::vector<double>& normalized_bounded_dual_delta) {
    const size_t num_weight_vectors = normalized_bounded_dual_delta.size();
    if (num_weight_vectors == 1) {
      deltas_.device(device) =
          deltas_ +
          dense_vector.RowAsMatrix() *
              deltas_.constant(normalized_bounded_dual_delta[0]);
    } else {
      // Transform the dual vector into a column matrix.
      const Eigen::TensorMap<Eigen::Tensor<const double, 2, Eigen::RowMajor>>
          dual_matrix(normalized_bounded_dual_delta.data(), num_weight_vectors,
                      1);
      const Eigen::array<Eigen::IndexPair<int>, 1> product_dims = {
          Eigen::IndexPair<int>(1, 0)};
      // This essentially computes delta_w += delta_vector / \lamdba * N.
      deltas_.device(device) =
          (deltas_.cast<double>() +
           dual_matrix.contract(dense_vector.RowAsMatrix().cast<double>(),
                                product_dims))
              .cast<float>();
    }
  }

 private:
  // The nominal value of the weight for a feature (indexed by its id).
  const TTypes<const float>::Matrix nominals_;
  // The accumulated delta weight for a feature (indexed by its id).
  TTypes<float>::Matrix deltas_;
};

// Similar to FeatureWeightsDenseStorage, but the underlying weights are stored
// in an unordered map.
class FeatureWeightsSparseStorage {
 public:
  FeatureWeightsSparseStorage(const TTypes<const int64>::Vec indices,
                              const TTypes<const float>::Matrix nominals,
                              TTypes<float>::Matrix deltas)
      : nominals_(nominals), deltas_(deltas) {
    // Create a map from sparse index to the dense index of the underlying
    // storage.
    for (int64 j = 0; j < indices.size(); ++j) {
      indices_to_id_[indices(j)] = j;
    }
  }

  // Check if a feature index exists.
  bool IndexValid(const int64 index) const {
    return indices_to_id_.find(index) != indices_to_id_.end();
  }

  // Nominal value at a particular feature index and class label.
  float nominals(const int class_id, const int64 index) const {
    auto it = indices_to_id_.find(index);
    return nominals_(class_id, it->second);
  }

  // Delta weights durining mini-batch updates.
  float deltas(const int class_id, const int64 index) const {
    auto it = indices_to_id_.find(index);
    return deltas_(class_id, it->second);
  }

  // Updates delta weights based on active sparse features in the example and
  // the corresponding dual residual.
  void UpdateSparseDeltaWeights(
      const Eigen::ThreadPoolDevice& device,
      const Example::SparseFeatures& sparse_features,
      const std::vector<double>& normalized_bounded_dual_delta) {
    for (int64 k = 0; k < sparse_features.indices->size(); ++k) {
      const double feature_value = sparse_features.values == nullptr
                                       ? 1.0
                                       : (*sparse_features.values)(k);
      auto it = indices_to_id_.find((*sparse_features.indices)(k));
      for (size_t l = 0; l < normalized_bounded_dual_delta.size(); ++l) {
        deltas_(l, it->second) +=
            feature_value * normalized_bounded_dual_delta[l];
      }
    }
  }

 private:
  // The nominal value of the weight for a feature (indexed by its id).
  const TTypes<const float>::Matrix nominals_;
  // The accumulated delta weight for a feature (indexed by its id).
  TTypes<float>::Matrix deltas_;
  // Map from feature index to an index to the dense vector.
  std::unordered_map<int64, int64> indices_to_id_;
};

// Weights in the model, wraps both current weights, and the delta weights
// for both sparse and dense features.
class ModelWeights {
 public:
  ModelWeights() {}

  bool SparseIndexValid(const int col, const int64 index) const {
    return sparse_weights_[col].IndexValid(index);
  }

  bool DenseIndexValid(const int col, const int64 index) const {
    return dense_weights_[col].IndexValid(index);
  }

  // Go through all the features present in the example, and update the
  // weights based on the dual delta.
  void UpdateDeltaWeights(
      const Eigen::ThreadPoolDevice& device, const Example& example,
      const std::vector<double>& normalized_bounded_dual_delta) {
    // Sparse weights.
    for (size_t j = 0; j < sparse_weights_.size(); ++j) {
      sparse_weights_[j].UpdateSparseDeltaWeights(
          device, example.sparse_features_[j], normalized_bounded_dual_delta);
    }

    // Dense weights.
    for (size_t j = 0; j < dense_weights_.size(); ++j) {
      dense_weights_[j].UpdateDenseDeltaWeights(
          device, *example.dense_vectors_[j], normalized_bounded_dual_delta);
    }
  }

  Status Initialize(OpKernelContext* const context) {
    OpInputList sparse_indices_inputs;
    TF_RETURN_IF_ERROR(
        context->input_list("sparse_indices", &sparse_indices_inputs));
    OpInputList sparse_weights_inputs;
    TF_RETURN_IF_ERROR(
        context->input_list("sparse_weights", &sparse_weights_inputs));
    OpInputList dense_weights_inputs;
    TF_RETURN_IF_ERROR(
        context->input_list("dense_weights", &dense_weights_inputs));

    OpOutputList sparse_weights_outputs;
    TF_RETURN_IF_ERROR(context->output_list("out_delta_sparse_weights",
                                            &sparse_weights_outputs));

    OpOutputList dense_weights_outputs;
    TF_RETURN_IF_ERROR(context->output_list("out_delta_dense_weights",
                                            &dense_weights_outputs));

    for (int i = 0; i < sparse_weights_inputs.size(); ++i) {
      Tensor* delta_t;
      sparse_weights_outputs.allocate(i, sparse_weights_inputs[i].shape(),
                                      &delta_t);
      // Convert the input vector to a row matrix in internal representation.
      auto deltas = delta_t->shaped<float, 2>({1, delta_t->NumElements()});
      deltas.setZero();
      sparse_weights_.emplace_back(FeatureWeightsSparseStorage{
          sparse_indices_inputs[i].flat<int64>(),
          sparse_weights_inputs[i].shaped<float, 2>(
              {1, sparse_weights_inputs[i].NumElements()}),
          deltas});
    }

    // Reads in the weights, and allocates and initializes the delta weights.
    const auto intialize_weights = [&](
        const OpInputList& weight_inputs, OpOutputList* const weight_outputs,
        std::vector<FeatureWeightsDenseStorage>* const feature_weights) {
      for (int i = 0; i < weight_inputs.size(); ++i) {
        Tensor* delta_t;
        weight_outputs->allocate(i, weight_inputs[i].shape(), &delta_t);
        // Convert the input vector to a row matrix in internal representation.
        auto deltas = delta_t->shaped<float, 2>({1, delta_t->NumElements()});
        deltas.setZero();
        feature_weights->emplace_back(
            FeatureWeightsDenseStorage{weight_inputs[i].shaped<float, 2>(
                                           {1, weight_inputs[i].NumElements()}),
                                       deltas});
      }
    };

    intialize_weights(dense_weights_inputs, &dense_weights_outputs,
                      &dense_weights_);

    return Status::OK();
  }

  const std::vector<FeatureWeightsSparseStorage>& sparse_weights() const {
    return sparse_weights_;
  }

  const std::vector<FeatureWeightsDenseStorage>& dense_weights() const {
    return dense_weights_;
  }

 private:
  std::vector<FeatureWeightsSparseStorage> sparse_weights_;
  std::vector<FeatureWeightsDenseStorage> dense_weights_;

  TF_DISALLOW_COPY_AND_ASSIGN(ModelWeights);
};

// Computes the example statistics for given example, and model. Defined here
// as we need definition of ModelWeights and Regularizations.
const ExampleStatistics Example::ComputeWxAndWeightedExampleNorm(
    const int num_loss_partitions, const ModelWeights& model_weights,
    const Regularizations& regularization, const int num_weight_vectors) const {
  ExampleStatistics result(num_weight_vectors);

  result.normalized_squared_norm =
      squared_norm_ / regularization.symmetric_l2();

  // Compute w \dot x and prev_w \dot x.
  // This is for sparse features contribution to the logit.
  for (size_t j = 0; j < sparse_features_.size(); ++j) {
    const Example::SparseFeatures& sparse_features = sparse_features_[j];
    const FeatureWeightsSparseStorage& sparse_weights =
        model_weights.sparse_weights()[j];

    for (int64 k = 0; k < sparse_features.indices->size(); ++k) {
      const int64 feature_index = (*sparse_features.indices)(k);
      const double feature_value = sparse_features.values == nullptr
                                       ? 1.0
                                       : (*sparse_features.values)(k);
      for (int l = 0; l < num_weight_vectors; ++l) {
        const float sparse_weight = sparse_weights.nominals(l, feature_index);
        const double feature_weight =
            sparse_weight +
            sparse_weights.deltas(l, feature_index) * num_loss_partitions;
        result.prev_wx[l] +=
            feature_value * regularization.Shrink(sparse_weight);
        result.wx[l] += feature_value * regularization.Shrink(feature_weight);
      }
    }
  }

  // Compute w \dot x and prev_w \dot x.
  // This is for dense features contribution to the logit.
  for (size_t j = 0; j < dense_vectors_.size(); ++j) {
    const Example::DenseVector& dense_vector = *dense_vectors_[j];
    const FeatureWeightsDenseStorage& dense_weights =
        model_weights.dense_weights()[j];

    const Eigen::Tensor<float, 2, Eigen::RowMajor> feature_weights =
        dense_weights.nominals() +
        dense_weights.deltas() *
            dense_weights.deltas().constant(num_loss_partitions);
    if (num_weight_vectors == 1) {
      const Eigen::Tensor<float, 0, Eigen::RowMajor> prev_prediction =
          (dense_vector.Row() *
           regularization.EigenShrinkVector(
               Eigen::TensorMap<Eigen::Tensor<const float, 1, Eigen::RowMajor>>(
                   dense_weights.nominals().data(),
                   dense_weights.nominals().dimension(1))))
              .sum();
      const Eigen::Tensor<float, 0, Eigen::RowMajor> prediction =
          (dense_vector.Row() *
           regularization.EigenShrinkVector(
               Eigen::TensorMap<Eigen::Tensor<const float, 1, Eigen::RowMajor>>(
                   feature_weights.data(), feature_weights.dimension(1))))
              .sum();
      result.prev_wx[0] += prev_prediction();
      result.wx[0] += prediction();
    } else {
      const Eigen::array<Eigen::IndexPair<int>, 1> product_dims = {
          Eigen::IndexPair<int>(1, 1)};
      const Eigen::Tensor<float, 2, Eigen::RowMajor> prev_prediction =
          regularization.EigenShrinkMatrix(dense_weights.nominals())
              .contract(dense_vector.RowAsMatrix(), product_dims);
      const Eigen::Tensor<float, 2, Eigen::RowMajor> prediction =
          regularization.EigenShrinkMatrix(feature_weights)
              .contract(dense_vector.RowAsMatrix(), product_dims);
      // The result of "tensor contraction" (multiplication)  in the code
      // above is of dimension num_weight_vectors * 1.
      for (int l = 0; l < num_weight_vectors; ++l) {
        result.prev_wx[l] += prev_prediction(l, 0);
        result.wx[l] += prediction(l, 0);
      }
    }
  }

  return result;
}

// Examples contains all the training examples that SDCA uses for a mini-batch.
class Examples {
 public:
  Examples() {}

  // Returns the Example at |example_index|.
  const Example& example(const int example_index) const {
    return examples_.at(example_index);
  }

  int sampled_index(const int id, const bool adaptative) const {
    if (adaptative) return sampled_index_[id];
    return id;
  }

  // Adaptive SDCA in the current implementation only works for
  // binary classification, where the input argument for num_weight_vectors
  // is 1.
  Status SampleAdaptativeProbabilities(
      const int num_loss_partitions, const Regularizations& regularization,
      const ModelWeights& model_weights,
      const TTypes<float>::Matrix example_state_data,
      const std::unique_ptr<DualLossUpdater>& loss_updater,
      const int num_weight_vectors = 1) {
    if (num_weight_vectors != 1) {
      return errors::InvalidArgument(
          "Adaptive SDCA only works with binary SDCA, "
          "where num_weight_vectors should be 1.");
    }
    // Compute the probabilities
    for (int example_id = 0; example_id < num_examples(); ++example_id) {
      const Example& example = examples_[example_id];
      const double example_weight = example.example_weight();
      float label = example.example_label();
      const Status conversion_status = loss_updater->ConvertLabel(&label);
      const ExampleStatistics example_statistics =
          example.ComputeWxAndWeightedExampleNorm(num_loss_partitions,
                                                  model_weights, regularization,
                                                  num_weight_vectors);
      const double kappa = example_state_data(example_id, 0) +
                           loss_updater->PrimalLossDerivative(
                               example_statistics.wx[0], label, example_weight);
      probabilities_[example_id] =
          example_weight * sqrt(examples_[example_id].squared_norm_ +
                                regularization.symmetric_l2() *
                                    loss_updater->SmoothnessConstant()) *
          std::abs(kappa);
    }

    // Sample the index
    random::DistributionSampler sampler(probabilities_);
    GuardedPhiloxRandom generator;
    generator.Init(0, 0);
    auto local_gen = generator.ReserveSamples32(num_examples());
    random::SimplePhilox random(&local_gen);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1);

    // We use a decay of 10: the probability of an example is divided by 10
    // once that example is picked. A good approximation of that is to only
    // keep a picked example with probability (1 / 10) ^ k where k is the
    // number of times we already picked that example. We add a num_retries
    // to avoid taking too long to sample. We then fill the sampled_index with
    // unseen examples sorted by probabilities.
    int id = 0;
    int num_retries = 0;
    while (id < num_examples() && num_retries < num_examples()) {
      int picked_id = sampler.Sample(&random);
      if (dis(gen) > std::pow(0.1, sampled_count_[picked_id])) {
        num_retries++;
        continue;
      }
      sampled_count_[picked_id]++;
      sampled_index_[id++] = picked_id;
    }

    std::vector<std::pair<int, float>> examples_not_seen;
    examples_not_seen.reserve(num_examples());
    for (int i = 0; i < num_examples(); ++i) {
      if (sampled_count_[i] == 0)
        examples_not_seen.emplace_back(sampled_index_[i], probabilities_[i]);
    }
    std::sort(
        examples_not_seen.begin(), examples_not_seen.end(),
        [](const std::pair<int, float>& lhs, const std::pair<int, float>& rhs) {
          return lhs.second > rhs.second;
        });
    for (int i = id; i < num_examples(); ++i) {
      sampled_count_[i] = examples_not_seen[i - id].first;
    }
    return Status::OK();
  }

  int num_examples() const { return examples_.size(); }

  int num_features() const { return num_features_; }

  // Initialize() must be called immediately after construction.
  // TODO(sibyl-Aix6ihai): Refactor/shorten this function.
  Status Initialize(OpKernelContext* const context, const ModelWeights& weights,
                    int num_sparse_features,
                    int num_sparse_features_with_values,
                    int num_dense_features);

 private:
  // Reads the input tensors, and builds the internal representation for sparse
  // features per example. This function modifies the |examples| passed in
  // to build the sparse representations.
  static Status CreateSparseFeatureRepresentation(
      const DeviceBase::CpuWorkerThreads& worker_threads, int num_examples,
      int num_sparse_features, const ModelWeights& weights,
      const OpInputList& sparse_example_indices_inputs,
      const OpInputList& sparse_feature_indices_inputs,
      const OpInputList& sparse_feature_values_inputs,
      std::vector<Example>* const examples);

  // Reads the input tensors, and builds the internal representation for dense
  // features per example. This function modifies the |examples| passed in
  // to build the sparse representations.
  static Status CreateDenseFeatureRepresentation(
      const DeviceBase::CpuWorkerThreads& worker_threads, int num_examples,
      int num_dense_features, const ModelWeights& weights,
      const OpInputList& dense_features_inputs,
      std::vector<Example>* const examples);

  // Computes squared example norm per example i.e |x|^2. This function modifies
  // the |examples| passed in and adds the squared norm per example.
  static void ComputeSquaredNormPerExample(
      const DeviceBase::CpuWorkerThreads& worker_threads, int num_examples,
      int num_sparse_features, int num_dense_features,
      std::vector<Example>* const examples);

  // All examples in the batch.
  std::vector<Example> examples_;

  // Adaptative sampling variables
  std::vector<float> probabilities_;
  std::vector<int> sampled_index_;
  std::vector<int> sampled_count_;

  int num_features_ = 0;

  TF_DISALLOW_COPY_AND_ASSIGN(Examples);
};

Status Examples::Initialize(OpKernelContext* const context,
                            const ModelWeights& weights,
                            const int num_sparse_features,
                            const int num_sparse_features_with_values,
                            const int num_dense_features) {
  num_features_ = num_sparse_features + num_dense_features;

  OpInputList sparse_example_indices_inputs;
  TF_RETURN_IF_ERROR(context->input_list("sparse_example_indices",
                                         &sparse_example_indices_inputs));
  OpInputList sparse_feature_indices_inputs;
  TF_RETURN_IF_ERROR(context->input_list("sparse_feature_indices",
                                         &sparse_feature_indices_inputs));
  OpInputList sparse_feature_values_inputs;
  if (num_sparse_features_with_values > 0) {
    TF_RETURN_IF_ERROR(context->input_list("sparse_feature_values",
                                           &sparse_feature_values_inputs));
  }

  const Tensor* example_weights_t;
  TF_RETURN_IF_ERROR(context->input("example_weights", &example_weights_t));
  auto example_weights = example_weights_t->flat<float>();

  if (example_weights.size() >= std::numeric_limits<int>::max()) {
    return errors::InvalidArgument(strings::Printf(
        "Too many examples in a mini-batch: %zu > %d", example_weights.size(),
        std::numeric_limits<int>::max()));
  }

  // The static_cast here is safe since num_examples can be at max an int.
  const int num_examples = static_cast<int>(example_weights.size());
  const Tensor* example_labels_t;
  TF_RETURN_IF_ERROR(context->input("example_labels", &example_labels_t));
  auto example_labels = example_labels_t->flat<float>();

  OpInputList dense_features_inputs;
  TF_RETURN_IF_ERROR(
      context->input_list("dense_features", &dense_features_inputs));

  examples_.clear();
  examples_.resize(num_examples);
  probabilities_.resize(num_examples);
  sampled_index_.resize(num_examples);
  sampled_count_.resize(num_examples);
  for (int example_id = 0; example_id < num_examples; ++example_id) {
    Example* const example = &examples_[example_id];
    example->sparse_features_.resize(num_sparse_features);
    example->dense_vectors_.resize(num_dense_features);
    example->example_weight_ = example_weights(example_id);
    example->example_label_ = example_labels(example_id);
  }
  const DeviceBase::CpuWorkerThreads& worker_threads =
      *context->device()->tensorflow_cpu_worker_threads();
  TF_RETURN_IF_ERROR(CreateSparseFeatureRepresentation(
      worker_threads, num_examples, num_sparse_features, weights,
      sparse_example_indices_inputs, sparse_feature_indices_inputs,
      sparse_feature_values_inputs, &examples_));
  TF_RETURN_IF_ERROR(CreateDenseFeatureRepresentation(
      worker_threads, num_examples, num_dense_features, weights,
      dense_features_inputs, &examples_));
  ComputeSquaredNormPerExample(worker_threads, num_examples,
                               num_sparse_features, num_dense_features,
                               &examples_);
  return Status::OK();
}

Status Examples::CreateSparseFeatureRepresentation(
    const DeviceBase::CpuWorkerThreads& worker_threads, const int num_examples,
    const int num_sparse_features, const ModelWeights& weights,
    const OpInputList& sparse_example_indices_inputs,
    const OpInputList& sparse_feature_indices_inputs,
    const OpInputList& sparse_feature_values_inputs,
    std::vector<Example>* const examples) {
  mutex mu;
  Status result GUARDED_BY(mu);
  auto parse_partition = [&](const int64 begin, const int64 end) {
    // The static_cast here is safe since begin and end can be at most
    // num_examples which is an int.
    for (int i = static_cast<int>(begin); i < end; ++i) {
      auto example_indices =
          sparse_example_indices_inputs[i].template flat<int64>();
      auto feature_indices =
          sparse_feature_indices_inputs[i].template flat<int64>();

      // Parse features for each example. Features for a particular example
      // are at the offsets (start_id, end_id]
      int start_id = -1;
      int end_id = 0;
      for (int example_id = 0; example_id < num_examples; ++example_id) {
        start_id = end_id;
        while (end_id < example_indices.size() &&
               example_indices(end_id) == example_id) {
          ++end_id;
        }
        Example::SparseFeatures* const sparse_features =
            &(*examples)[example_id].sparse_features_[i];
        if (start_id < example_indices.size() &&
            example_indices(start_id) == example_id) {
          sparse_features->indices.reset(new UnalignedInt64Vector(
              &(feature_indices(start_id)), end_id - start_id));
          if (sparse_feature_values_inputs.size() > i) {
            auto feature_weights =
                sparse_feature_values_inputs[i].flat<float>();
            sparse_features->values.reset(new UnalignedFloatVector(
                &(feature_weights(start_id)), end_id - start_id));
          }
          // If features are non empty.
          if (end_id - start_id > 0) {
            // TODO(sibyl-Aix6ihai): Write this efficiently using vectorized
            // operations from eigen.
            for (int64 k = 0; k < sparse_features->indices->size(); ++k) {
              const int64 feature_index = (*sparse_features->indices)(k);
              if (!weights.SparseIndexValid(i, feature_index)) {
                mutex_lock l(mu);
                result = errors::InvalidArgument(
                    "Found sparse feature indices out of valid range: ",
                    (*sparse_features->indices)(k));
                return;
              }
            }
          }
        } else {
          // Add a Tensor that has size 0.
          sparse_features->indices.reset(
              new UnalignedInt64Vector(&(feature_indices(0)), 0));
          // If values exist for this feature group.
          if (sparse_feature_values_inputs.size() > i) {
            auto feature_weights =
                sparse_feature_values_inputs[i].flat<float>();
            sparse_features->values.reset(
                new UnalignedFloatVector(&(feature_weights(0)), 0));
          }
        }
      }
    }
  };
  // For each column, the cost of parsing it is O(num_examples). We use
  // num_examples here, as empirically Shard() creates the right amount of
  // threads based on the problem size.
  // TODO(sibyl-Aix6ihai): Tune this as a function of dataset size.
  const int64 kCostPerUnit = num_examples;
  Shard(worker_threads.num_threads, worker_threads.workers, num_sparse_features,
        kCostPerUnit, parse_partition);
  return result;
}

Status Examples::CreateDenseFeatureRepresentation(
    const DeviceBase::CpuWorkerThreads& worker_threads, const int num_examples,
    const int num_dense_features, const ModelWeights& weights,
    const OpInputList& dense_features_inputs,
    std::vector<Example>* const examples) {
  mutex mu;
  Status result GUARDED_BY(mu);
  auto parse_partition = [&](const int64 begin, const int64 end) {
    // The static_cast here is safe since begin and end can be at most
    // num_examples which is an int.
    for (int i = static_cast<int>(begin); i < end; ++i) {
      auto dense_features = dense_features_inputs[i].template matrix<float>();
      for (int example_id = 0; example_id < num_examples; ++example_id) {
        (*examples)[example_id].dense_vectors_[i].reset(
            new Example::DenseVector{dense_features, example_id});
      }
      if (!weights.DenseIndexValid(i, dense_features.dimension(1) - 1)) {
        mutex_lock l(mu);
        result = errors::InvalidArgument(
            "More dense features than we have parameters for: ",
            dense_features.dimension(1));
        return;
      }
    }
  };
  // TODO(sibyl-Aix6ihai): Tune this as a function of dataset size.
  const int64 kCostPerUnit = num_examples;
  Shard(worker_threads.num_threads, worker_threads.workers, num_dense_features,
        kCostPerUnit, parse_partition);
  return result;
}

void Examples::ComputeSquaredNormPerExample(
    const DeviceBase::CpuWorkerThreads& worker_threads, const int num_examples,
    const int num_sparse_features, const int num_dense_features,
    std::vector<Example>* const examples) {
  // Compute norm of examples.
  auto compute_example_norm = [&](const int64 begin, const int64 end) {
    // The static_cast here is safe since begin and end can be at most
    // num_examples which is an int.
    for (int example_id = static_cast<int>(begin); example_id < end;
         ++example_id) {
      double squared_norm = 0;
      Example* const example = &(*examples)[example_id];
      for (int j = 0; j < num_sparse_features; ++j) {
        const Example::SparseFeatures& sparse_features =
            example->sparse_features_[j];
        if (sparse_features.values) {
          const Eigen::Tensor<float, 0, Eigen::RowMajor> sn =
              sparse_features.values->square().sum();
          squared_norm += sn();
        } else {
          squared_norm += sparse_features.indices->size();
        }
      }
      for (int j = 0; j < num_dense_features; ++j) {
        const Eigen::Tensor<float, 0, Eigen::RowMajor> sn =
            example->dense_vectors_[j]->Row().square().sum();
        squared_norm += sn();
      }
      example->squared_norm_ = squared_norm;
    }
  };
  // TODO(sibyl-Aix6ihai): Compute the cost optimally.
  const int64 kCostPerUnit = num_dense_features + num_sparse_features;
  Shard(worker_threads.num_threads, worker_threads.workers, num_examples,
        kCostPerUnit, compute_example_norm);
}

struct ComputeOptions {
  ComputeOptions(OpKernelConstruction* const context) {
    string loss_type;
    OP_REQUIRES_OK(context, context->GetAttr("loss_type", &loss_type));
    if (loss_type == "logistic_loss") {
      loss_updater.reset(new LogisticLossUpdater);
    } else if (loss_type == "squared_loss") {
      loss_updater.reset(new SquaredLossUpdater);
    } else if (loss_type == "hinge_loss") {
      loss_updater.reset(new HingeLossUpdater);
    } else if (loss_type == "smooth_hinge_loss") {
      loss_updater.reset(new SmoothHingeLossUpdater);
    } else {
      OP_REQUIRES(context, false, errors::InvalidArgument(
                                      "Unsupported loss type: ", loss_type));
    }
    OP_REQUIRES_OK(context, context->GetAttr("adaptative", &adaptative));
    OP_REQUIRES_OK(
        context, context->GetAttr("num_sparse_features", &num_sparse_features));
    OP_REQUIRES_OK(context, context->GetAttr("num_sparse_features_with_values",
                                             &num_sparse_features_with_values));
    OP_REQUIRES_OK(context,
                   context->GetAttr("num_dense_features", &num_dense_features));
    OP_REQUIRES(
        context, num_sparse_features + num_dense_features > 0,
        errors::InvalidArgument("Requires at least one feature to train."));

    OP_REQUIRES(context, static_cast<int64>(num_sparse_features) +
                                 static_cast<int64>(num_dense_features) <=
                             std::numeric_limits<int>::max(),
                errors::InvalidArgument(
                    strings::Printf("Too many feature groups: %lld > %d",
                                    static_cast<int64>(num_sparse_features) +
                                        static_cast<int64>(num_dense_features),
                                    std::numeric_limits<int>::max())));
    OP_REQUIRES_OK(
        context, context->GetAttr("num_loss_partitions", &num_loss_partitions));
    OP_REQUIRES_OK(context, context->GetAttr("num_inner_iterations",
                                             &num_inner_iterations));
    OP_REQUIRES_OK(context, regularizations.Initialize(context));
  }

  std::unique_ptr<DualLossUpdater> loss_updater;
  int num_sparse_features = 0;
  int num_sparse_features_with_values = 0;
  int num_dense_features = 0;
  int num_inner_iterations = 0;
  int num_loss_partitions = 0;
  bool adaptative = false;
  Regularizations regularizations;
};

// TODO(shengx): The helper classes/methods are changed to support multiclass
// SDCA, which lead to changes within this function. Need to revisit the
// convergence once the multiclass SDCA is in.
void DoCompute(const ComputeOptions& options, OpKernelContext* const context) {
  ModelWeights model_weights;
  OP_REQUIRES_OK(context, model_weights.Initialize(context));

  Examples examples;
  OP_REQUIRES_OK(
      context,
      examples.Initialize(context, model_weights, options.num_sparse_features,
                          options.num_sparse_features_with_values,
                          options.num_dense_features));

  const Tensor* example_state_data_t;
  OP_REQUIRES_OK(context,
                 context->input("example_state_data", &example_state_data_t));
  TensorShape expected_example_state_shape({examples.num_examples(), 4});
  OP_REQUIRES(context,
              example_state_data_t->shape() == expected_example_state_shape,
              errors::InvalidArgument(
                  "Expected shape ", expected_example_state_shape.DebugString(),
                  " for example_state_data, got ",
                  example_state_data_t->shape().DebugString()));

  Tensor mutable_example_state_data_t(*example_state_data_t);
  auto example_state_data = mutable_example_state_data_t.matrix<float>();
  context->set_output("out_example_state_data", mutable_example_state_data_t);

  if (options.adaptative) {
    OP_REQUIRES_OK(
        context, examples.SampleAdaptativeProbabilities(
                     options.num_loss_partitions, options.regularizations,
                     model_weights, example_state_data, options.loss_updater));
  }

  mutex mu;
  Status train_step_status GUARDED_BY(mu);
  std::atomic<std::int64_t> atomic_index(-1);
  auto train_step = [&](const int64 begin, const int64 end) {
    // The static_cast here is safe since begin and end can be at most
    // num_examples which is an int.
    for (int id = static_cast<int>(begin); id < end; ++id) {
      const int64 example_index =
          examples.sampled_index(++atomic_index, options.adaptative);
      const Example& example = examples.example(example_index);
      const float dual = example_state_data(example_index, 0);
      const float example_weight = example.example_weight();
      float example_label = example.example_label();
      const Status conversion_status =
          options.loss_updater->ConvertLabel(&example_label);
      if (!conversion_status.ok()) {
        mutex_lock l(mu);
        train_step_status = conversion_status;
        // Return from this worker thread - the calling thread is
        // responsible for checking context status and returning on error.
        return;
      }

      // Compute wx, example norm weighted by regularization, dual loss,
      // primal loss.
      // For binary SDCA, num_weight_vectors should be one.
      const ExampleStatistics example_statistics =
          example.ComputeWxAndWeightedExampleNorm(
              options.num_loss_partitions, model_weights,
              options.regularizations, 1 /* num_weight_vectors */);

      const double new_dual = options.loss_updater->ComputeUpdatedDual(
          options.num_loss_partitions, example_label, example_weight, dual,
          example_statistics.wx[0], example_statistics.normalized_squared_norm);

      // Compute new weights.
      const double normalized_bounded_dual_delta =
          (new_dual - dual) * example_weight /
          options.regularizations.symmetric_l2();
      model_weights.UpdateDeltaWeights(
          context->eigen_cpu_device(), example,
          std::vector<double>{normalized_bounded_dual_delta});

      // Update example data.
      example_state_data(example_index, 0) = new_dual;
      example_state_data(example_index, 1) =
          options.loss_updater->ComputePrimalLoss(
              example_statistics.prev_wx[0], example_label, example_weight);
      example_state_data(example_index, 2) =
          options.loss_updater->ComputeDualLoss(dual, example_label,
                                                example_weight);
      example_state_data(example_index, 3) = example_weight;
    }
  };
  // TODO(sibyl-Aix6ihai): Tune this properly based on sparsity of the data,
  // number of cpus, and cost per example.
  const int64 kCostPerUnit = examples.num_features();
  const DeviceBase::CpuWorkerThreads& worker_threads =
      *context->device()->tensorflow_cpu_worker_threads();

  Shard(worker_threads.num_threads, worker_threads.workers,
        examples.num_examples(), kCostPerUnit, train_step);
  OP_REQUIRES_OK(context, train_step_status);
}

}  // namespace

class SdcaOptimizer : public OpKernel {
 public:
  explicit SdcaOptimizer(OpKernelConstruction* const context)
      : OpKernel(context), options_(context) {}

  void Compute(OpKernelContext* const context) override {
    DoCompute(options_, context);
  }

 private:
  // TODO(sibyl-Aix6ihai): We could use the type-constraint on loss_type, and
  // template the entire class to avoid the virtual table lookup penalty in
  // the inner loop.
  ComputeOptions options_;
};
REGISTER_KERNEL_BUILDER(Name("SdcaOptimizer").Device(DEVICE_CPU),
                        SdcaOptimizer);

class SdcaShrinkL1 : public OpKernel {
 public:
  explicit SdcaShrinkL1(OpKernelConstruction* const context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, regularizations_.Initialize(context));
  }

  void Compute(OpKernelContext* const context) override {
    OpMutableInputList weights_inputs;
    OP_REQUIRES_OK(context,
                   context->mutable_input_list("weights", &weights_inputs));

    auto do_work = [&](const int64 begin, const int64 end) {
      for (int i = begin; i < end; ++i) {
        auto prox_w = weights_inputs.at(i, /*lock_held=*/true).flat<float>();
        prox_w.device(context->eigen_cpu_device()) =
            regularizations_.EigenShrinkVector(prox_w);
      }
    };

    if (weights_inputs.size() > 0) {
      int64 num_weights = 0;
      for (int i = 0; i < weights_inputs.size(); ++i) {
        num_weights += weights_inputs.at(i, /*lock_held=*/true).NumElements();
      }
      // TODO(sibyl-Aix6ihai): Tune this value.
      const int64 kCostPerUnit = (num_weights * 50) / weights_inputs.size();
      const DeviceBase::CpuWorkerThreads& worker_threads =
          *context->device()->tensorflow_cpu_worker_threads();
      Shard(worker_threads.num_threads, worker_threads.workers,
            weights_inputs.size(), kCostPerUnit, do_work);
    }
  }

 private:
  Regularizations regularizations_;
};
REGISTER_KERNEL_BUILDER(Name("SdcaShrinkL1").Device(DEVICE_CPU), SdcaShrinkL1);

// Computes platform independent, compact and unique (with very high
// probability) representation of an example id. It shouldn't be put in
// persistent storage, as its implementation may change in the future.
//
// The current probability of at least one collision for 1B example_ids is
// approximately 10^-21 (ie 2^60 / 2^129).
class SdcaFprint : public OpKernel {
 public:
  explicit SdcaFprint(OpKernelConstruction* const context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* const context) override {
    const Tensor& input = context->input(0);
    OP_REQUIRES(context, TensorShapeUtils::IsVector(input.shape()),
                errors::InvalidArgument("Input must be a vector, got shape ",
                                        input.shape().DebugString()));
    Tensor* out;
    const int64 num_elements = input.NumElements();
    OP_REQUIRES_OK(context, context->allocate_output(
                                0, TensorShape({num_elements, 2}), &out));

    const auto in_values = input.flat<string>();
    auto out_values = out->matrix<int64>();

    for (int64 i = 0; i < num_elements; ++i) {
      const Fprint128 fprint = Fingerprint128(in_values(i));
      // Never return 0 or 1 as the first value of the hash to allow these to
      // safely be used as sentinel values (e.g. dense hash table empty key).
      out_values(i, 0) = TF_PREDICT_TRUE(fprint.low64 >= 2)
                             ? fprint.low64
                             : fprint.low64 + ~static_cast<uint64>(1);
      out_values(i, 1) = fprint.high64;
    }
  }
};
REGISTER_KERNEL_BUILDER(Name("SdcaFprint").Device(DEVICE_CPU), SdcaFprint);

}  // namespace tensorflow
