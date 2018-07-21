// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2018 Andy Davis <andydavis@google.com>
// Copyright (C) 2018 Eugene Zhulenev <ezhulenev@google.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"

#include <set>

#include <Eigen/CXX11/Tensor>

using Eigen::Tensor;
using Eigen::Index;
using Eigen::RowMajor;
using Eigen::ColMajor;

template<typename T>
static const T& choose(int layout, const T& col, const T& row) {
  return layout == ColMajor ? col : row;
}

template <int Layout>
static void test_block_mapper_sanity()
{
  using T = int;
  using TensorBlock = internal::TensorBlock<T, Index, 2, Layout>;
  using TensorBlockMapper = internal::TensorBlockMapper<T, Index, 2, Layout>;

  DSizes<Index, 2> tensor_dims(100, 100);

  // Test uniform blocks.
  TensorBlockMapper uniform_block_mapper(
      tensor_dims, internal::TensorBlockShapeType::kUniformAllDims, 100);

  VERIFY_IS_EQUAL(uniform_block_mapper.total_block_count(), 100);
  VERIFY_IS_EQUAL(uniform_block_mapper.block_dims_total_size(), 100);

  // 10x10 blocks
  auto uniform_b0 = uniform_block_mapper.GetBlockForIndex(0, nullptr);
  VERIFY_IS_EQUAL(uniform_b0.block_sizes().at(0), 10);
  VERIFY_IS_EQUAL(uniform_b0.block_sizes().at(1), 10);
  // Depending on a layout we stride by cols rows.
  VERIFY_IS_EQUAL(uniform_b0.block_strides().at(0), choose(Layout, 1, 10));
  VERIFY_IS_EQUAL(uniform_b0.block_strides().at(1), choose(Layout, 10, 1));
  // Tensor strides depend only on a layout and not on the block size.
  VERIFY_IS_EQUAL(uniform_b0.tensor_strides().at(0), choose(Layout, 1, 100));
  VERIFY_IS_EQUAL(uniform_b0.tensor_strides().at(1), choose(Layout, 100, 1));

  // Test skewed to inner dims blocks.
  TensorBlockMapper skewed_block_mapper(
      tensor_dims, internal::TensorBlockShapeType::kSkewedInnerDims, 100);

  VERIFY_IS_EQUAL(skewed_block_mapper.total_block_count(), 100);
  VERIFY_IS_EQUAL(skewed_block_mapper.block_dims_total_size(), 100);

  // 1x100 (100x1) rows/cols depending on a tensor layout.
  auto skewed_b0 = skewed_block_mapper.GetBlockForIndex(0, nullptr);
  VERIFY_IS_EQUAL(skewed_b0.block_sizes().at(0), choose(Layout, 100, 1));
  VERIFY_IS_EQUAL(skewed_b0.block_sizes().at(1), choose(Layout, 1, 100));
  // Depending on a layout we stride by cols rows.
  VERIFY_IS_EQUAL(skewed_b0.block_strides().at(0), choose(Layout, 1, 100));
  VERIFY_IS_EQUAL(skewed_b0.block_strides().at(1), choose(Layout, 100, 1));
  // Tensor strides depend only on a layout and not on the block size.
  VERIFY_IS_EQUAL(skewed_b0.tensor_strides().at(0), choose(Layout, 1, 100));
  VERIFY_IS_EQUAL(skewed_b0.tensor_strides().at(1), choose(Layout, 100, 1));
}

// Given a TensorBlock "visit" every element accessible though it, and a keep an
// index in the visited set. Verify that every coeff accessed only once.
template <typename T, int Layout, int NumDims>
static void UpdateCoeffSet(
    const internal::TensorBlock<T, Index, 4, Layout>& block,
    Index first_coeff_index,
    int dim_index,
    std::set<Index>* visited_coeffs) {
  const DSizes<Index, NumDims> block_sizes = block.block_sizes();
  const DSizes<Index, NumDims> tensor_strides = block.tensor_strides();

  for (int i = 0; i < block_sizes[dim_index]; ++i) {
    if (tensor_strides[dim_index] == 1) {
      auto inserted = visited_coeffs->insert(first_coeff_index + i);
      VERIFY_IS_EQUAL(inserted.second, true);
    } else {
      int next_dim_index = dim_index + choose(Layout, -1, 1);
      UpdateCoeffSet<T, Layout, NumDims>(block, first_coeff_index,
                                         next_dim_index, visited_coeffs);
      first_coeff_index += tensor_strides[dim_index];
    }
  }
}

template <int Layout>
static void test_block_mapper_maps_every_element()
{
  using T = int;
  using TensorBlock = internal::TensorBlock<T, Index, 4, Layout>;
  using TensorBlockMapper = internal::TensorBlockMapper<T, Index, 4, Layout>;

  DSizes<Index, 4> dims(5, 7, 11, 17);

  auto total_coeffs = static_cast<int>(dims.TotalSize());

  // Keep track of elements indices available via block access.
  std::set<Index> coeff_set;

  // Try different combinations of block types and sizes.
  auto block_shape_type =
      internal::random<bool>()
          ? internal::TensorBlockShapeType::kUniformAllDims
          : internal::TensorBlockShapeType::kSkewedInnerDims;
  auto block_target_size = internal::random<int>(1, total_coeffs);
  TensorBlockMapper block_mapper(dims, block_shape_type, block_target_size);

  for (int i = 0; i < block_mapper.total_block_count(); ++i) {
    TensorBlock block = block_mapper.GetBlockForIndex(i, nullptr);
    UpdateCoeffSet<T, Layout, 4>(block, block.first_coeff_index(),
                                 choose(Layout, 3, 0), &coeff_set);
  }

  // Verify that every coefficient in the original Tensor is accessible through
  // TensorBlock only once.
  VERIFY_IS_EQUAL(coeff_set.size(), total_coeffs);
  VERIFY_IS_EQUAL(*coeff_set.begin(), static_cast<Index>(0));
  VERIFY_IS_EQUAL(*coeff_set.rbegin(), static_cast<Index>(total_coeffs - 1));
}

template <int Layout>
static void test_slice_block_mapper_maps_every_element()
{
  using T = int;
  using TensorBlock = internal::TensorBlock<T, Index, 4, Layout>;
  using TensorSliceBlockMapper =
      internal::TensorSliceBlockMapper<T, Index, 4, Layout>;

  DSizes<Index, 4> tensor_dims(5,7,11,17);
  DSizes<Index, 4> tensor_slice_offsets(1,3,5,7);
  DSizes<Index, 4> tensor_slice_extents(3,2,4,5);

  // Keep track of elements indices available via block access.
  std::set<Index> coeff_set;

  auto total_coeffs = static_cast<int>(tensor_slice_extents.TotalSize());

  // Try different combinations of block types and sizes.
  auto block_shape_type =
      internal::random<bool>()
      ? internal::TensorBlockShapeType::kUniformAllDims
      : internal::TensorBlockShapeType::kSkewedInnerDims;
  auto block_target_size = internal::random<int>(1, total_coeffs);

  // Pick a random dimension sizes for the tensor blocks.
  DSizes<Index, 4> block_sizes;
  for (int i = 0; i < 4; ++i) {
    block_sizes[i] = internal::random<int>(1, tensor_slice_extents[i]);
  }

  TensorSliceBlockMapper block_mapper(tensor_dims, tensor_slice_offsets,
                                      tensor_slice_extents, block_sizes,
                                      DimensionList<Index, 4>());

  for (int i = 0; i < block_mapper.total_block_count(); ++i) {
    TensorBlock block = block_mapper.GetBlockForIndex(i, NULL);
    UpdateCoeffSet<T, Layout, 4>(block, block.first_coeff_index(),
                                 choose(Layout, 3, 0), &coeff_set);
  }

  VERIFY_IS_EQUAL(coeff_set.size(), total_coeffs);
}

EIGEN_DECLARE_TEST(cxx11_tensor_assign) {
  CALL_SUBTEST(test_block_mapper_sanity<ColMajor>());
  CALL_SUBTEST(test_block_mapper_sanity<RowMajor>());
  CALL_SUBTEST(test_block_mapper_maps_every_element<ColMajor>());
  CALL_SUBTEST(test_block_mapper_maps_every_element<RowMajor>());
  CALL_SUBTEST(test_slice_block_mapper_maps_every_element<ColMajor>());
  CALL_SUBTEST(test_slice_block_mapper_maps_every_element<RowMajor>());
}
