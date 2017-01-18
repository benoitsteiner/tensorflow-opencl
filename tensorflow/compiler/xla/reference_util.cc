/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/reference_util.h"

#include <array>

#include "tensorflow/compiler/xla/client/computation_builder.h"
#include "tensorflow/compiler/xla/service/cpu/runtime_single_threaded_matmul.h"
#include "tensorflow/compiler/xla/window_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/math/math_util.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

/* static */ std::unique_ptr<Array2D<float>> ReferenceUtil::TransposeArray2D(
    const Array2D<float>& operand) {
  auto result = MakeUnique<Array2D<float>>(operand.width(), operand.height());
  for (int64 w = 0; w < operand.width(); ++w) {
    for (int64 h = 0; h < operand.height(); ++h) {
      (*result)(w, h) = operand(h, w);
    }
  }

  return result;
}

/* static */ std::unique_ptr<Array2D<float>> ReferenceUtil::MatmulArray2D(
    const Array2D<float>& lhs, const Array2D<float>& rhs) {
  CHECK_EQ(lhs.width(), rhs.height());
  int m = lhs.height();
  int n = rhs.width();
  int k = lhs.width();
  auto result = MakeUnique<Array2D<float>>(m, n);
  // Because Eigen is a header-oriented library, make sure that the Eigen code
  // is the same as the code used by the CPU backend (otherwise the linker will
  // randomly pick *some* definition).
  __xla_cpu_runtime_EigenSingleThreadedMatMulF32(
      /*run_options_ptr=*/nullptr, result->data(), rhs.data(), lhs.data(), n, m,
      k,
      /*transpose_lhs=*/0,
      /*transpose_rhs=*/0);
  return result;
}

/* static */ std::unique_ptr<Array2D<double>> ReferenceUtil::MatmulArray2D(
    const Array2D<double>& lhs, const Array2D<double>& rhs) {
  CHECK_EQ(lhs.width(), rhs.height());
  int m = lhs.height();
  int n = rhs.width();
  int k = lhs.width();
  auto result = MakeUnique<Array2D<double>>(m, n);
  // Because Eigen is a header-oriented library, make sure that the Eigen code
  // is the same as the code used by the CPU backend (otherwise the linker will
  // randomly pick *some* definition).
  __xla_cpu_runtime_EigenSingleThreadedMatMulF64(
      /*run_options_ptr=*/nullptr, result->data(), rhs.data(), lhs.data(), n, m,
      k,
      /*transpose_lhs=*/0,
      /*transpose_rhs=*/0);
  return result;
}

/* static */ std::unique_ptr<Array2D<double>> ReferenceUtil::Array2DF32ToF64(
    const Array2D<float>& input) {
  auto result = MakeUnique<Array2D<double>>(input.height(), input.width());
  for (int64 rowno = 0; rowno < input.height(); ++rowno) {
    for (int64 colno = 0; colno < input.height(); ++colno) {
      (*result)(rowno, colno) = input(rowno, colno);
    }
  }
  return result;
}

/* static */ std::unique_ptr<Array4D<float>> ReferenceUtil::ConvArray4D(
    const Array4D<float>& lhs, const Array4D<float>& rhs,
    std::pair<int64, int64> kernel_stride, Padding padding) {
  return ConvArray4DGeneralDimensions(
      lhs, rhs, kernel_stride, padding,
      ComputationBuilder::CreateDefaultConvDimensionNumbers());
}

/* static */ int64 ReferenceUtil::WindowCount(int64 unpadded_width,
                                              int64 window_len, int64 stride,
                                              Padding padding) {
  if (padding == Padding::kValid) {
    return window_util::StridedBound(unpadded_width, window_len, stride);
  }
  return tensorflow::MathUtil::CeilOfRatio(unpadded_width, stride);
}

/* static  */ std::unique_ptr<Array4D<float>> ReferenceUtil::ReduceWindow4DAdd(
    const Array4D<float>& operand, float init,
    const tensorflow::gtl::ArraySlice<int64>& window,
    const tensorflow::gtl::ArraySlice<int64>& stride, Padding padding) {
  std::vector<int64> dim_lengths{operand.n1(), operand.n2(), operand.n3(),
                                 operand.n4()};
  auto padding_both = xla::MakePadding(dim_lengths, window, stride, padding);

  std::vector<int64> window_counts(window.size(), 0);
  std::vector<int64> pad_low(window.size(), 0);
  for (int64 i = 0; i < window.size(); ++i) {
    window_counts[i] =
        WindowCount(dim_lengths[i], window[i], stride[i], padding);
    pad_low[i] = padding_both[i].first;
  }
  auto result = MakeUnique<Array4D<float>>(window_counts[0], window_counts[1],
                                           window_counts[2], window_counts[3]);
  // Do a full 4D reduce window.
  for (int64 i0 = 0; i0 < window_counts[0]; ++i0) {
    for (int64 i1 = 0; i1 < window_counts[1]; ++i1) {
      for (int64 i2 = 0; i2 < window_counts[2]; ++i2) {
        for (int64 i3 = 0; i3 < window_counts[3]; ++i3) {
          int64 i0_base = i0 * stride[0] - pad_low[0];
          int64 i1_base = i1 * stride[1] - pad_low[1];
          int64 i2_base = i2 * stride[2] - pad_low[2];
          int64 i3_base = i3 * stride[3] - pad_low[3];

          float val = init;
          for (int64 i0_win = 0; i0_win < window[0]; ++i0_win) {
            for (int64 i1_win = 0; i1_win < window[1]; ++i1_win) {
              for (int64 i2_win = 0; i2_win < window[2]; ++i2_win) {
                for (int64 i3_win = 0; i3_win < window[3]; ++i3_win) {
                  if (i0_base + i0_win >= 0 && i1_base + i1_win >= 0 &&
                      i2_base + i2_win >= 0 && i3_base + i3_win >= 0 &&
                      i0_base + i0_win < operand.n1() &&
                      i1_base + i1_win < operand.n2() &&
                      i2_base + i2_win < operand.n3() &&
                      i3_base + i3_win < operand.n4()) {
                    val += operand(i0_base + i0_win, i1_base + i1_win,
                                   i2_base + i2_win, i3_base + i3_win);
                  }
                }
              }
            }
          }
          (*result)(i0, i1, i2, i3) = val;
        }
      }
    }
  }
  return result;
}

/* static  */ std::unique_ptr<Array4D<float>>
ReferenceUtil::SelectAndScatter4DGePlus(
    const Array4D<float>& operand, const Array4D<float>& source, float init,
    const tensorflow::gtl::ArraySlice<int64>& window,
    const tensorflow::gtl::ArraySlice<int64>& stride, bool same_padding) {
  Padding padding = same_padding ? Padding::kSame : Padding::kValid;
  auto result = MakeUnique<Array4D<float>>(operand.n1(), operand.n2(),
                                           operand.n3(), operand.n4());
  std::vector<int64> dim_lengths{operand.n1(), operand.n2(), operand.n3(),
                                 operand.n4()};
  auto padding_both = xla::MakePadding(dim_lengths, window, stride, padding);
  // Fill the output, with the initial value.
  result->Fill(init);

  std::vector<int64> window_counts(window.size(), 0);
  std::vector<int64> pad_low(window.size(), 0);
  for (int64 i = 0; i < window.size(); ++i) {
    window_counts[i] =
        WindowCount(dim_lengths[i], window[i], stride[i], padding);
    pad_low[i] = padding_both[i].first;
  }
  CHECK_EQ(window_counts[0], source.n1());
  CHECK_EQ(window_counts[1], source.n2());
  CHECK_EQ(window_counts[2], source.n3());
  CHECK_EQ(window_counts[3], source.n4());

  // Do a full 4D select and Scatter.
  for (int64 i0 = 0; i0 < window_counts[0]; ++i0) {
    for (int64 i1 = 0; i1 < window_counts[1]; ++i1) {
      for (int64 i2 = 0; i2 < window_counts[2]; ++i2) {
        for (int64 i3 = 0; i3 < window_counts[3]; ++i3) {
          // Now we are inside a window and need to find the max and the argmax.
          int64 i0_base = i0 * stride[0] - pad_low[0];
          int64 i1_base = i1 * stride[1] - pad_low[1];
          int64 i2_base = i2 * stride[2] - pad_low[2];
          int64 i3_base = i3 * stride[3] - pad_low[3];
          int64 scatter_0 = (i0_base >= 0) ? i0_base : 0;
          int64 scatter_1 = (i1_base >= 0) ? i1_base : 0;
          int64 scatter_2 = (i2_base >= 0) ? i2_base : 0;
          int64 scatter_3 = (i3_base >= 0) ? i3_base : 0;
          float val = operand(scatter_0, scatter_1, scatter_2, scatter_3);
          for (int64 i0_win = 0; i0_win < window[0]; ++i0_win) {
            for (int64 i1_win = 0; i1_win < window[1]; ++i1_win) {
              for (int64 i2_win = 0; i2_win < window[2]; ++i2_win) {
                for (int64 i3_win = 0; i3_win < window[3]; ++i3_win) {
                  if (i0_base + i0_win >= 0 && i1_base + i1_win >= 0 &&
                      i2_base + i2_win >= 0 && i3_base + i3_win >= 0 &&
                      i0_base + i0_win < operand.n1() &&
                      i1_base + i1_win < operand.n2() &&
                      i2_base + i2_win < operand.n3() &&
                      i3_base + i3_win < operand.n4()) {
                    float tmp = operand(i0_base + i0_win, i1_base + i1_win,
                                        i2_base + i2_win, i3_base + i3_win);
                    if (tmp >= val) {
                      val = tmp;
                      scatter_0 = i0_base + i0_win;
                      scatter_1 = i1_base + i1_win;
                      scatter_2 = i2_base + i2_win;
                      scatter_3 = i3_base + i3_win;
                    }
                  }
                }
              }
            }
          }
          (*result)(scatter_0, scatter_1, scatter_2, scatter_3) +=
              source(i0, i1, i2, i3);
        }
      }
    }
  }
  return result;
}

/* static */ std::unique_ptr<Array4D<float>>
ReferenceUtil::ConvArray4DGeneralDimensions(
    const Array4D<float>& lhs, const Array4D<float>& rhs,
    std::pair<int64, int64> kernel_stride, Padding padding,
    ConvolutionDimensionNumbers dimension_numbers) {
  return ConvArray4DGeneralDimensionsDilated(lhs, rhs, kernel_stride, padding,
                                             {1, 1}, {1, 1}, dimension_numbers);
}

/* static */ std::unique_ptr<Array4D<float>>
ReferenceUtil::ConvArray4DGeneralDimensionsDilated(
    const Array4D<float>& lhs, const Array4D<float>& rhs,
    std::pair<int64, int64> kernel_stride, Padding padding,
    std::pair<int64, int64> lhs_dilation, std::pair<int64, int64> rhs_dilation,
    ConvolutionDimensionNumbers dnums) {
  std::array<int64, 4> lhs_dimensions{{lhs.n1(), lhs.n2(), lhs.n3(), lhs.n4()}};
  std::array<int64, 4> rhs_dimensions{{rhs.n1(), rhs.n2(), rhs.n3(), rhs.n4()}};

  const int64 ksy = kernel_stride.first;
  const int64 ksx = kernel_stride.second;
  const int64 dy = lhs_dilation.first;
  const int64 dx = lhs_dilation.second;
  const int64 dky = rhs_dilation.first;
  const int64 dkx = rhs_dilation.second;
  CHECK_GE(dky, 1);
  CHECK_GE(dkx, 1);
  CHECK_GE(dy, 1);
  CHECK_GE(dx, 1);

  // Get all dimension sizes in lhs and rhs based on the given convolution
  // dimension configuration.
  const int64 ix = window_util::DilatedBound(
      lhs_dimensions[dnums.spatial_dimensions(1)], dx);
  const int64 iy = window_util::DilatedBound(
      lhs_dimensions[dnums.spatial_dimensions(0)], dy);
  const int64 iz = lhs_dimensions[dnums.feature_dimension()];
  const int64 samples = lhs_dimensions[dnums.batch_dimension()];
  const int64 kx = window_util::DilatedBound(
      rhs_dimensions[dnums.kernel_spatial_dimensions(1)], dkx);
  const int64 ky = window_util::DilatedBound(
      rhs_dimensions[dnums.kernel_spatial_dimensions(0)], dky);
  const int64 oz = rhs_dimensions[dnums.kernel_output_feature_dimension()];
  {
    const int64 kiz = rhs_dimensions[dnums.kernel_input_feature_dimension()];
    CHECK_EQ(kiz, iz);
  }

  if (padding == Padding::kSame) {
    // We reject same padding with kernel striding, since it's somewhat
    // nonsensical. We can always follow up to implement this with the desired
    // semantics if anybody actually uses it.
    CHECK_EQ(1, ksy);
    CHECK_EQ(1, ksx);
  }

  const int64 ox =
      padding == Padding::kSame ? ix : window_util::StridedBound(ix, kx, ksx);
  const int64 oy =
      padding == Padding::kSame ? iy : window_util::StridedBound(iy, ky, ksy);
  const int64 istartx =
      padding == Padding::kValid ? 0 : kx % 2 == 0 ? -(kx / 2 - 1) : -kx / 2;
  const int64 istarty =
      padding == Padding::kValid ? 0 : ky % 2 == 0 ? -(ky / 2 - 1) : -ky / 2;
  // Create the output result array and reset the values to 0.
  std::array<int64, 4> result_dimensions;
  result_dimensions[dnums.batch_dimension()] = samples;
  result_dimensions[dnums.feature_dimension()] = oz;
  result_dimensions[dnums.spatial_dimensions(0)] = oy;
  result_dimensions[dnums.spatial_dimensions(1)] = ox;
  auto result =
      MakeUnique<Array4D<float>>(result_dimensions[0], result_dimensions[1],
                                 result_dimensions[2], result_dimensions[3]);
  result->Fill(0.0);

  // Lambda to access the lhs operand at the given 4D index.
  const auto lhs_element = [&](int64 batch, int64 feature, int64 height,
                               int64 width) {
    if (height % dy != 0 || width % dx != 0) {
      return 0.0f;
    }

    std::array<int64, 4> index;
    index[dnums.batch_dimension()] = batch;
    index[dnums.feature_dimension()] = feature;
    index[dnums.spatial_dimensions(0)] = height / dy;
    index[dnums.spatial_dimensions(1)] = width / dx;
    return lhs(index[0], index[1], index[2], index[3]);
  };

  // Lambda to access the rhs operand at the given 4D index.
  const auto rhs_element = [&](int64 kernel_output_feature,
                               int64 kernel_input_feature, int64 height,
                               int64 width) {
    CHECK_EQ(height % dky, 0);
    CHECK_EQ(width % dkx, 0);
    std::array<int64, 4> index;
    index[dnums.kernel_output_feature_dimension()] = kernel_output_feature;
    index[dnums.kernel_input_feature_dimension()] = kernel_input_feature;
    index[dnums.kernel_spatial_dimensions(0)] = height / dky;
    index[dnums.kernel_spatial_dimensions(1)] = width / dkx;
    return rhs(index[0], index[1], index[2], index[3]);
  };

  // Lambda to access the result data at the given 4D index.
  const auto result_element = [&](int64 batch, int64 kernel_output_feature,
                                  int64 height, int64 width) -> float& {
    std::array<int64, 4> index;
    index[dnums.batch_dimension()] = batch;
    index[dnums.feature_dimension()] = kernel_output_feature;
    index[dnums.spatial_dimensions(0)] = height;
    index[dnums.spatial_dimensions(1)] = width;
    return (*result)(index[0], index[1], index[2], index[3]);
  };

  for (int64 oyi = 0; oyi < oy; ++oyi) {
    for (int64 oxi = 0; oxi < ox; ++oxi) {
      for (int64 sample = 0; sample < samples; ++sample) {
        for (int64 izi = 0; izi < iz; ++izi) {
          for (int64 ozi = 0; ozi < oz; ++ozi) {
            for (int64 kyi = 0; kyi < ky; kyi += dky) {
              for (int64 kxi = 0; kxi < kx; kxi += dkx) {
                int64 iyi = istarty + ksy * oyi + kyi;
                int64 ixi = istartx + ksx * oxi + kxi;
                float input = (iyi >= iy || ixi >= ix || iyi < 0 || ixi < 0)
                                  ? 0.0
                                  : lhs_element(sample, izi, iyi, ixi);
                float gain = rhs_element(ozi, izi, kyi, kxi);
                float addend = input * gain;
                result_element(sample, ozi, oyi, oxi) += addend;
              }
            }
          }
        }
      }
    }
  }
  return result;
}

/* static */ std::unique_ptr<std::vector<float>>
ReferenceUtil::ReduceToColArray2D(
    const Array2D<float>& matrix, float init,
    std::function<float(float, float)> reduce_function) {
  int64 rows = matrix.height();
  int64 cols = matrix.width();
  auto result = MakeUnique<std::vector<float>>();
  for (int64 i = 0; i < rows; ++i) {
    float acc = init;
    for (int64 j = 0; j < cols; ++j) {
      acc = reduce_function(acc, matrix(i, j));
    }
    result->push_back(acc);
  }
  return result;
}

/* static */ std::unique_ptr<std::vector<float>>
ReferenceUtil::ReduceToRowArray2D(
    const Array2D<float>& matrix, float init,
    std::function<float(float, float)> reduce_function) {
  int64 rows = matrix.height();
  int64 cols = matrix.width();
  auto result = MakeUnique<std::vector<float>>();
  for (int64 i = 0; i < cols; ++i) {
    float acc = init;
    for (int64 j = 0; j < rows; ++j) {
      acc = reduce_function(acc, matrix(j, i));
    }
    result->push_back(acc);
  }
  return result;
}

/*static*/ std::vector<float> ReferenceUtil::Reduce4DTo1D(
    const Array4D<float>& array, float init,
    tensorflow::gtl::ArraySlice<int64> dims,
    std::function<float(float, float)> reduce_function) {
  std::vector<float> result;
  CHECK_EQ(dims.size(), 3);
  const std::set<int64> dim_set(dims.begin(), dims.end());
  CHECK_EQ(dim_set.size(), 3);
  for (int64 a0 = 0; a0 == 0 || (!dim_set.count(0) && a0 < array.n1()); ++a0) {
    for (int64 a1 = 0; a1 == 0 || (!dim_set.count(1) && a1 < array.n2());
         ++a1) {
      for (int64 a2 = 0; a2 == 0 || (!dim_set.count(2) && a2 < array.n3());
           ++a2) {
        for (int64 a3 = 0; a3 == 0 || (!dim_set.count(3) && a3 < array.n4());
             ++a3) {
          float accumulator = init;
          for (int64 i0 = 0; i0 == 0 || (dim_set.count(0) && i0 < array.n1());
               ++i0) {
            for (int64 i1 = 0; i1 == 0 || (dim_set.count(1) && i1 < array.n2());
                 ++i1) {
              for (int64 i2 = 0;
                   i2 == 0 || (dim_set.count(2) && i2 < array.n3()); ++i2) {
                for (int64 i3 = 0;
                     i3 == 0 || (dim_set.count(3) && i3 < array.n4()); ++i3) {
                  accumulator = reduce_function(
                      accumulator, array(a0 + i0, a1 + i1, a2 + i2, a3 + i3));
                }
              }
            }
          }
          result.push_back(accumulator);
        }
      }
    }
  }
  return result;
}

/* static */ std::unique_ptr<Array2D<float>> ReferenceUtil::Reduce3DTo2D(
    const Array3D<float>& array, float init,
    tensorflow::gtl::ArraySlice<int64> dims,
    std::function<float(float, float)> reduce_function) {
  CHECK_EQ(dims.size(), 1);
  int64 rows = dims[0] == 0 ? array.n2() : array.n1();
  int64 cols = dims[0] == 2 ? array.n2() : array.n3();
  auto result = MakeUnique<Array2D<float>>(rows, cols);
  result->Fill(init);
  for (int i0 = 0; i0 < array.n1(); ++i0) {
    for (int i1 = 0; i1 < array.n2(); ++i1) {
      for (int i2 = 0; i2 < array.n3(); ++i2) {
        int64 row = dims[0] == 0 ? i1 : i0;
        int64 col = dims[0] == 2 ? i1 : i2;
        (*result)(row, col) =
            reduce_function((*result)(row, col), array(i0, i1, i2));
      }
    }
  }
  return result;
}

/* static */ std::unique_ptr<Array2D<float>> ReferenceUtil::MapArray2D(
    const Array2D<float>& matrix,
    const std::function<float(float)>& map_function) {
  int64 rows = matrix.height();
  int64 cols = matrix.width();
  auto result = MakeUnique<Array2D<float>>(rows, cols);
  for (int64 i = 0; i < rows; ++i) {
    for (int64 j = 0; j < cols; ++j) {
      (*result)(i, j) = map_function(matrix(i, j));
    }
  }
  return result;
}

/* static */ std::unique_ptr<Array2D<float>> ReferenceUtil::MapArray2D(
    const Array2D<float>& lhs, const Array2D<float>& rhs,
    const std::function<float(float, float)>& map_function) {
  CHECK_EQ(lhs.height(), rhs.height());
  CHECK_EQ(lhs.width(), rhs.width());
  int64 rows = lhs.height();
  int64 cols = rhs.width();
  auto result = MakeUnique<Array2D<float>>(rows, cols);
  for (int64 i = 0; i < rows; ++i) {
    for (int64 j = 0; j < cols; ++j) {
      (*result)(i, j) = map_function(lhs(i, j), rhs(i, j));
    }
  }
  return result;
}

/* static */ std::unique_ptr<Array2D<float>> ReferenceUtil::MapWithIndexArray2D(
    const Array2D<float>& matrix,
    const std::function<float(float, int64, int64)>& map_function) {
  int64 rows = matrix.height();
  int64 cols = matrix.width();
  auto result = MakeUnique<Array2D<float>>(rows, cols);
  for (int64 i = 0; i < rows; ++i) {
    for (int64 j = 0; j < cols; ++j) {
      (*result)(i, j) = map_function(matrix(i, j), i, j);
    }
  }
  return result;
}

/* static */ std::unique_ptr<Array2D<float>> ReferenceUtil::PadArray2D(
    const Array2D<float>& operand, const PaddingConfig& padding,
    const float pad) {
  int64 in0 = operand.n1();
  int64 high_padding0 = padding.dimensions(0).edge_padding_high();
  int64 low_padding0 = padding.dimensions(0).edge_padding_low();
  int64 interior_padding0 = padding.dimensions(0).interior_padding();
  int64 out0 =
      in0 + low_padding0 + high_padding0 + (in0 - 1) * interior_padding0;
  int64 in1 = operand.n2();
  int64 high_padding1 = padding.dimensions(1).edge_padding_high();
  int64 low_padding1 = padding.dimensions(1).edge_padding_low();
  int64 interior_padding1 = padding.dimensions(1).interior_padding();
  int64 out1 =
      in1 + low_padding1 + high_padding1 + (in1 - 1) * interior_padding1;
  auto result = MakeUnique<Array2D<float>>(out0, out1);
  result->Fill(pad);
  int64 i0 = 0;
  for (int64 o0 = low_padding0; o0 < out0 - high_padding0;
       o0 += interior_padding0 + 1) {
    int64 i1 = 0;
    for (int64 o1 = low_padding1; o1 < out1 - high_padding1;
         o1 += interior_padding1 + 1) {
      (*result)(o0, o1) = operand(i0, i1);
      ++i1;
    }
    ++i0;
  }
  return result;
}

}  // namespace xla
