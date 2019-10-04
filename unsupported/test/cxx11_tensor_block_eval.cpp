// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// clang-format off
#include "main.h"
#include <Eigen/CXX11/Tensor>
// clang-format on

using Eigen::internal::TensorBlockDescriptor;
using Eigen::internal::TensorExecutor;

// -------------------------------------------------------------------------- //
// Utility functions to generate random tensors, blocks, and evaluate them.

template <int NumDims>
static DSizes<Index, NumDims> RandomDims(Index min, Index max) {
  DSizes<Index, NumDims> dims;
  for (int i = 0; i < NumDims; ++i) {
    dims[i] = internal::random<Index>(min, max);
  }
  return DSizes<Index, NumDims>(dims);
}

// Block offsets and extents allows to construct a TensorSlicingOp corresponding
// to a TensorBlockDescriptor.
template <int NumDims>
struct TensorBlockParams {
  DSizes<Index, NumDims> offsets;
  DSizes<Index, NumDims> sizes;
  TensorBlockDescriptor<NumDims, Index> desc;
};

template <int Layout, int NumDims>
static TensorBlockParams<NumDims> RandomBlock(DSizes<Index, NumDims> dims,
                                              Index min, Index max) {
  // Choose random offsets and sizes along all tensor dimensions.
  DSizes<Index, NumDims> offsets(RandomDims<NumDims>(min, max));
  DSizes<Index, NumDims> sizes(RandomDims<NumDims>(min, max));

  // Make sure that offset + size do not overflow dims.
  for (int i = 0; i < NumDims; ++i) {
    offsets[i] = numext::mini(dims[i] - 1, offsets[i]);
    sizes[i] = numext::mini(sizes[i], dims[i] - offsets[i]);
  }

  Index offset = 0;
  DSizes<Index, NumDims> strides = Eigen::internal::strides<Layout>(dims);
  for (int i = 0; i < NumDims; ++i) {
    offset += strides[i] * offsets[i];
  }

  return {offsets, sizes, TensorBlockDescriptor<NumDims, Index>(offset, sizes)};
}

// Generate block with block sizes skewed towards inner dimensions. This type of
// block is required for evaluating broadcast expressions.
template <int Layout, int NumDims>
static TensorBlockParams<NumDims> SkewedInnerBlock(
    DSizes<Index, NumDims> dims) {
  using BlockMapper = internal::TensorBlockMapper<int, Index, NumDims, Layout>;
  BlockMapper block_mapper(dims,
                           internal::TensorBlockShapeType::kSkewedInnerDims,
                           internal::random<Index>(1, dims.TotalSize()));

  Index total_blocks = block_mapper.total_block_count();
  Index block_index = internal::random<Index>(0, total_blocks - 1);
  auto block = block_mapper.GetBlockForIndex(block_index, nullptr);
  DSizes<Index, NumDims> sizes = block.block_sizes();

  auto strides = internal::strides<Layout>(dims);
  DSizes<Index, NumDims> offsets;

  // Compute offsets for the first block coefficient.
  Index index = block.first_coeff_index();
  if (static_cast<int>(Layout) == static_cast<int>(ColMajor)) {
    for (int i = NumDims - 1; i > 0; --i) {
      const Index idx = index / strides[i];
      index -= idx * strides[i];
      offsets[i] = idx;
    }
    offsets[0] = index;
  } else {
    for (int i = 0; i < NumDims - 1; ++i) {
      const Index idx = index / strides[i];
      index -= idx * strides[i];
      offsets[i] = idx;
    }
    offsets[NumDims - 1] = index;
  }

  auto desc = TensorBlockDescriptor<NumDims>(block.first_coeff_index(), sizes);
  return {offsets, sizes, desc};
}

template <int NumDims>
static TensorBlockParams<NumDims> FixedSizeBlock(DSizes<Index, NumDims> dims) {
  DSizes<Index, NumDims> offsets;
  for (int i = 0; i < NumDims; ++i) offsets[i] = 0;

  return {offsets, dims, TensorBlockDescriptor<NumDims, Index>(0, dims)};
}

inline Eigen::IndexList<Index, Eigen::type2index<1>> NByOne(Index n) {
  Eigen::IndexList<Index, Eigen::type2index<1>> ret;
  ret.set(0, n);
  return ret;
}
inline Eigen::IndexList<Eigen::type2index<1>, Index> OneByM(Index m) {
  Eigen::IndexList<Eigen::type2index<1>, Index> ret;
  ret.set(1, m);
  return ret;
}

// -------------------------------------------------------------------------- //
// Verify that block expression evaluation produces the same result as a
// TensorSliceOp (reading a tensor block is same to taking a tensor slice).

template <typename T, int NumDims, int Layout, typename Expression,
          typename GenBlockParams>
static void VerifyBlockEvaluator(Expression expr, GenBlockParams gen_block) {
  using Device = DefaultDevice;
  auto d = Device();

  // Scratch memory allocator for block evaluation.
  typedef internal::TensorBlockScratchAllocator<Device> TensorBlockScratch;
  TensorBlockScratch scratch(d);

  // TensorEvaluator is needed to produce tensor blocks of the expression.
  auto eval = TensorEvaluator<const decltype(expr), Device>(expr, d);

  // Choose a random offsets, sizes and TensorBlockDescriptor.
  TensorBlockParams<NumDims> block_params = gen_block();

  // Evaluate TensorBlock expression into a tensor.
  Tensor<T, NumDims, Layout> block(block_params.desc.dimensions());

  // Maybe use this tensor as a block desc destination.
  Tensor<T, NumDims, Layout> dst(block_params.desc.dimensions());
  if (internal::random<bool>()) {
    block_params.desc.template AddDestinationBuffer(
        dst.data(), internal::strides<Layout>(dst.dimensions()),
        dst.dimensions().TotalSize() * sizeof(T));
  }

  auto tensor_block = eval.blockV2(block_params.desc, scratch);
  auto b_expr = tensor_block.expr();

  // We explicitly disable vectorization and tiling, to run a simple coefficient
  // wise assignment loop, because it's very simple and should be correct.
  using BlockAssign = TensorAssignOp<decltype(block), const decltype(b_expr)>;
  using BlockExecutor = TensorExecutor<const BlockAssign, Device, false,
                                       internal::TiledEvaluation::Off>;
  BlockExecutor::run(BlockAssign(block, b_expr), d);

  // Cleanup temporary buffers owned by a tensor block.
  tensor_block.cleanup();

  // Compute a Tensor slice corresponding to a Tensor block.
  Tensor<T, NumDims, Layout> slice(block_params.desc.dimensions());
  auto s_expr = expr.slice(block_params.offsets, block_params.sizes);

  // Explicitly use coefficient assignment to evaluate slice expression.
  using SliceAssign = TensorAssignOp<decltype(slice), const decltype(s_expr)>;
  using SliceExecutor = TensorExecutor<const SliceAssign, Device, false,
                                       internal::TiledEvaluation::Off>;
  SliceExecutor::run(SliceAssign(slice, s_expr), d);

  // Tensor block and tensor slice must be the same.
  for (Index i = 0; i < block.dimensions().TotalSize(); ++i) {
    VERIFY_IS_EQUAL(block.coeff(i), slice.coeff(i));
  }
}

// -------------------------------------------------------------------------- //

template <typename T, int NumDims, int Layout>
static void test_eval_tensor_block() {
  DSizes<Index, NumDims> dims = RandomDims<NumDims>(10, 20);
  Tensor<T, NumDims, Layout> input(dims);
  input.setRandom();

  // Identity tensor expression transformation.
  VerifyBlockEvaluator<T, NumDims, Layout>(
      input, [&dims]() { return RandomBlock<Layout>(dims, 1, 10); });
}

template <typename T, int NumDims, int Layout>
static void test_eval_tensor_unary_expr_block() {
  DSizes<Index, NumDims> dims = RandomDims<NumDims>(10, 20);
  Tensor<T, NumDims, Layout> input(dims);
  input.setRandom();

  VerifyBlockEvaluator<T, NumDims, Layout>(
      input.square(), [&dims]() { return RandomBlock<Layout>(dims, 1, 10); });
}

template <typename T, int NumDims, int Layout>
static void test_eval_tensor_binary_expr_block() {
  DSizes<Index, NumDims> dims = RandomDims<NumDims>(10, 20);
  Tensor<T, NumDims, Layout> lhs(dims), rhs(dims);
  lhs.setRandom();
  rhs.setRandom();

  VerifyBlockEvaluator<T, NumDims, Layout>(
      lhs + rhs, [&dims]() { return RandomBlock<Layout>(dims, 1, 10); });
}

template <typename T, int NumDims, int Layout>
static void test_eval_tensor_binary_with_unary_expr_block() {
  DSizes<Index, NumDims> dims = RandomDims<NumDims>(10, 20);
  Tensor<T, NumDims, Layout> lhs(dims), rhs(dims);
  lhs.setRandom();
  rhs.setRandom();

  VerifyBlockEvaluator<T, NumDims, Layout>(
      (lhs.square() + rhs.square()).sqrt(),
      [&dims]() { return RandomBlock<Layout>(dims, 1, 10); });
}

template <typename T, int NumDims, int Layout>
static void test_eval_tensor_broadcast() {
  DSizes<Index, NumDims> dims = RandomDims<NumDims>(1, 10);
  Tensor<T, NumDims, Layout> input(dims);
  input.setRandom();

  DSizes<Index, NumDims> bcast = RandomDims<NumDims>(1, 5);

  DSizes<Index, NumDims> bcasted_dims;
  for (int i = 0; i < NumDims; ++i) bcasted_dims[i] = dims[i] * bcast[i];

  VerifyBlockEvaluator<T, NumDims, Layout>(
      input.broadcast(bcast),
      [&bcasted_dims]() { return SkewedInnerBlock<Layout>(bcasted_dims); });

  VerifyBlockEvaluator<T, NumDims, Layout>(
      input.broadcast(bcast),
      [&bcasted_dims]() { return FixedSizeBlock(bcasted_dims); });

  // Check that desc.destination() memory is not shared between two broadcast
  // materializations.
  VerifyBlockEvaluator<T, NumDims, Layout>(
      input.broadcast(bcast) + input.square().broadcast(bcast),
      [&bcasted_dims]() { return SkewedInnerBlock<Layout>(bcasted_dims); });
}

template <typename T, int NumDims, int Layout>
static void test_eval_tensor_reshape() {
  DSizes<Index, NumDims> dims = RandomDims<NumDims>(1, 10);

  DSizes<Index, NumDims> shuffled = dims;
  std::shuffle(&shuffled[0], &shuffled[NumDims - 1], std::mt19937(g_seed));

  Tensor<T, NumDims, Layout> input(dims);
  input.setRandom();

  VerifyBlockEvaluator<T, NumDims, Layout>(
      input.reshape(shuffled),
      [&shuffled]() { return RandomBlock<Layout>(shuffled, 1, 10); });

  VerifyBlockEvaluator<T, NumDims, Layout>(
      input.reshape(shuffled),
      [&shuffled]() { return SkewedInnerBlock<Layout>(shuffled); });
}

template <typename T, int Layout>
static void test_eval_tensor_reshape_with_bcast() {
  Index dim = internal::random<Index>(1, 100);

  Tensor<T, 2, Layout> lhs(1, dim);
  Tensor<T, 2, Layout> rhs(dim, 1);
  lhs.setRandom();
  rhs.setRandom();

  auto reshapeLhs = NByOne(dim);
  auto reshapeRhs = OneByM(dim);

  auto bcastLhs = OneByM(dim);
  auto bcastRhs = NByOne(dim);

  DSizes<Index, 2> dims(dim, dim);

  VerifyBlockEvaluator<T, 2, Layout>(
      lhs.reshape(reshapeLhs).broadcast(bcastLhs) +
          rhs.reshape(reshapeRhs).broadcast(bcastRhs),
      [dims]() { return SkewedInnerBlock<Layout, 2>(dims); });
}

template <typename T, int NumDims, int Layout>
static void test_eval_tensor_cast() {
  DSizes<Index, NumDims> dims = RandomDims<NumDims>(10, 20);
  Tensor<T, NumDims, Layout> input(dims);
  input.setRandom();

  VerifyBlockEvaluator<T, NumDims, Layout>(
      input.template cast<int>().template cast<T>(),
      [&dims]() { return RandomBlock<Layout>(dims, 1, 10); });
}

template <typename T, int NumDims, int Layout>
static void test_eval_tensor_select() {
  DSizes<Index, NumDims> dims = RandomDims<NumDims>(10, 20);
  Tensor<T, NumDims, Layout> lhs(dims);
  Tensor<T, NumDims, Layout> rhs(dims);
  Tensor<bool, NumDims, Layout> cond(dims);
  lhs.setRandom();
  rhs.setRandom();
  cond.setRandom();

  VerifyBlockEvaluator<T, NumDims, Layout>(cond.select(lhs, rhs), [&dims]() {
    return RandomBlock<Layout>(dims, 1, 20);
  });
}

template <typename T, int NumDims, int Layout>
static void test_eval_tensor_padding() {
  const int inner_dim = Layout == static_cast<int>(ColMajor) ? 0 : NumDims - 1;

  DSizes<Index, NumDims> dims = RandomDims<NumDims>(10, 20);
  Tensor<T, NumDims, Layout> input(dims);
  input.setRandom();

  DSizes<Index, NumDims> pad_before = RandomDims<NumDims>(0, 4);
  DSizes<Index, NumDims> pad_after = RandomDims<NumDims>(0, 4);
  array<std::pair<Index, Index>, NumDims> paddings;
  for (int i = 0; i < NumDims; ++i) {
    paddings[i] = std::make_pair(pad_before[i], pad_after[i]);
  }

  // Test squeezing reads from inner dim.
  if (internal::random<bool>()) {
    pad_before[inner_dim] = 0;
    pad_after[inner_dim] = 0;
    paddings[inner_dim] = std::make_pair(0, 0);
  }

  DSizes<Index, NumDims> padded_dims;
  for (int i = 0; i < NumDims; ++i) {
    padded_dims[i] = dims[i] + pad_before[i] + pad_after[i];
  }

  VerifyBlockEvaluator<T, NumDims, Layout>(
      input.pad(paddings),
      [&padded_dims]() { return FixedSizeBlock(padded_dims); });

  VerifyBlockEvaluator<T, NumDims, Layout>(
      input.pad(paddings),
      [&padded_dims]() { return RandomBlock<Layout>(padded_dims, 1, 10); });

  VerifyBlockEvaluator<T, NumDims, Layout>(
      input.pad(paddings),
      [&padded_dims]() { return SkewedInnerBlock<Layout>(padded_dims); });
}

// -------------------------------------------------------------------------- //
// Verify that assigning block to a Tensor expression produces the same result
// as an assignment to TensorSliceOp (writing a block is is identical to
// assigning one tensor to a slice of another tensor).

template <typename T, int NumDims, int Layout, typename Expression,
          typename GenBlockParams>
static void VerifyBlockAssignment(Tensor<T, NumDims, Layout>& tensor,
                                  Expression expr, GenBlockParams gen_block) {
  using Device = DefaultDevice;
  auto d = Device();

  // We use tensor evaluator as a target for block and slice assignments.
  auto eval = TensorEvaluator<decltype(expr), Device>(expr, d);

  // Generate a random block, or choose a block that fits in full expression.
  TensorBlockParams<NumDims> block_params = gen_block();

  // Generate random data of the selected block size.
  Tensor<T, NumDims, Layout> block(block_params.desc.dimensions());
  block.setRandom();

  // ************************************************************************ //
  // (1) Assignment from a block.

  // Construct a materialize block from a random generated block tensor.
  internal::TensorMaterializedBlock<T, NumDims, Layout> blk(
      internal::TensorBlockKind::kView, block.data(), block.dimensions());

  // Reset all underlying tensor values to zero.
  tensor.setZero();

  // Use evaluator to write block into a tensor.
  eval.writeBlockV2(block_params.desc, blk);

  // Make a copy of the result after assignment.
  Tensor<T, NumDims, Layout> block_assigned = tensor;

  // ************************************************************************ //
  // (2) Assignment to a slice

  // Reset all underlying tensor values to zero.
  tensor.setZero();

  // Assign block to a slice of original expression
  auto s_expr = expr.slice(block_params.offsets, block_params.sizes);

  // Explicitly use coefficient assignment to evaluate slice expression.
  using SliceAssign = TensorAssignOp<decltype(s_expr), const decltype(block)>;
  using SliceExecutor = TensorExecutor<const SliceAssign, Device, false,
                                       internal::TiledEvaluation::Off>;
  SliceExecutor::run(SliceAssign(s_expr, block), d);

  // Make a copy of the result after assignment.
  Tensor<T, NumDims, Layout> slice_assigned = tensor;

  for (Index i = 0; i < tensor.dimensions().TotalSize(); ++i) {
    VERIFY_IS_EQUAL(block_assigned.coeff(i), slice_assigned.coeff(i));
  }
}

// -------------------------------------------------------------------------- //

template <typename T, int NumDims, int Layout>
static void test_assign_to_tensor() {
  DSizes<Index, NumDims> dims = RandomDims<NumDims>(10, 20);
  Tensor<T, NumDims, Layout> tensor(dims);

  TensorMap<Tensor<T, NumDims, Layout>> map(tensor.data(), dims);

  VerifyBlockAssignment<T, NumDims, Layout>(
      tensor, map, [&dims]() { return RandomBlock<Layout>(dims, 10, 20); });
  VerifyBlockAssignment<T, NumDims, Layout>(
      tensor, map, [&dims]() { return FixedSizeBlock(dims); });
}

template <typename T, int NumDims, int Layout>
static void test_assign_to_tensor_reshape() {
  DSizes<Index, NumDims> dims = RandomDims<NumDims>(10, 20);
  Tensor<T, NumDims, Layout> tensor(dims);

  TensorMap<Tensor<T, NumDims, Layout>> map(tensor.data(), dims);

  DSizes<Index, NumDims> shuffled = dims;
  std::shuffle(&shuffled[0], &shuffled[NumDims - 1], std::mt19937(g_seed));

  VerifyBlockAssignment<T, NumDims, Layout>(
      tensor, map.reshape(shuffled),
      [&shuffled]() { return RandomBlock<Layout>(shuffled, 1, 10); });

  VerifyBlockAssignment<T, NumDims, Layout>(
      tensor, map.reshape(shuffled),
      [&shuffled]() { return SkewedInnerBlock<Layout>(shuffled); });

  VerifyBlockAssignment<T, NumDims, Layout>(
      tensor, map.reshape(shuffled),
      [&shuffled]() { return FixedSizeBlock(shuffled); });
}

// -------------------------------------------------------------------------- //

#define CALL_SUBTESTS_DIMS_LAYOUTS(NAME)      \
  CALL_SUBTEST((NAME<float, 1, RowMajor>())); \
  CALL_SUBTEST((NAME<float, 2, RowMajor>())); \
  CALL_SUBTEST((NAME<float, 4, RowMajor>())); \
  CALL_SUBTEST((NAME<float, 5, RowMajor>())); \
  CALL_SUBTEST((NAME<float, 1, ColMajor>())); \
  CALL_SUBTEST((NAME<float, 2, ColMajor>())); \
  CALL_SUBTEST((NAME<float, 4, ColMajor>())); \
  CALL_SUBTEST((NAME<float, 5, ColMajor>()))

#define CALL_SUBTESTS_LAYOUTS(NAME)        \
  CALL_SUBTEST((NAME<float, RowMajor>())); \
  CALL_SUBTEST((NAME<float, ColMajor>()))

EIGEN_DECLARE_TEST(cxx11_tensor_block_eval) {
  // clang-format off
  CALL_SUBTESTS_DIMS_LAYOUTS(test_eval_tensor_block);
  CALL_SUBTESTS_DIMS_LAYOUTS(test_eval_tensor_unary_expr_block);
  CALL_SUBTESTS_DIMS_LAYOUTS(test_eval_tensor_binary_expr_block);
  CALL_SUBTESTS_DIMS_LAYOUTS(test_eval_tensor_binary_with_unary_expr_block);
  CALL_SUBTESTS_DIMS_LAYOUTS(test_eval_tensor_broadcast);
  CALL_SUBTESTS_DIMS_LAYOUTS(test_eval_tensor_reshape);
  CALL_SUBTESTS_DIMS_LAYOUTS(test_eval_tensor_cast);
  CALL_SUBTESTS_DIMS_LAYOUTS(test_eval_tensor_padding);

  CALL_SUBTESTS_LAYOUTS(test_eval_tensor_reshape_with_bcast);

  CALL_SUBTESTS_DIMS_LAYOUTS(test_assign_to_tensor);
  CALL_SUBTESTS_DIMS_LAYOUTS(test_assign_to_tensor_reshape);
  // clang-format on
}
