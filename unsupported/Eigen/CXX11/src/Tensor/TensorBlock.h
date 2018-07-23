// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2018 Andy Davis <andydavis@google.com>
// Copyright (C) 2018 Eugene Zhulenev <ezhulenev@google.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CXX11_TENSOR_TENSOR_BLOCK_H
#define EIGEN_CXX11_TENSOR_TENSOR_BLOCK_H

namespace Eigen {
namespace internal {

namespace {

// Helper template to choose between ColMajor and RowMajor values.
template <int Layout>
struct cond;

template <>
struct cond<ColMajor> {
  template <typename T>
  EIGEN_STRONG_INLINE const T& operator()(const T& col,
                                          const T& /*row*/) const {
    return col;
  }
};

template <>
struct cond<RowMajor> {
  template <typename T>
  EIGEN_STRONG_INLINE const T& operator()(const T& /*col*/,
                                          const T& row) const {
    return row;
  }
};

}  // namespace

/**
 * \class TensorBlockShapeType
 * \ingroup CXX11_Tensor_Module
 *
 * \brief Tensor block shape type.
 *
 * Tensor block shape type defines what are the shape preference for the blocks
 * extracted from the larger tensor.
 *
 * Example:
 *
 * We want to extract blocks of 100 elements from the large 100x100 tensor:
 *  - tensor: 100x100
 *  - target_block_size: 100
 *
 * TensorBlockShapeType:
 *  - kUniformAllDims: 100 blocks of size 10x10
 *  - kSkewedInnerDims: 100 blocks of size 100x1 (or 1x100 depending on a column
 *                      or row major layout)
 */
enum class TensorBlockShapeType {
  kUniformAllDims,
  kSkewedInnerDims,
};

/**
 * \class TensorBlock
 * \ingroup CXX11_Tensor_Module
 *
 * \brief Tensor block class.
 *
 * This class represents a tensor block specified by the index of the
 * first block coefficient, and the size of the block in each dimension.
 */
template <typename Scalar, typename Index, std::size_t NumDims, int Layout>
class TensorBlock {
 public:
  typedef DSizes<Index, NumDims> Dimensions;

  TensorBlock(const Index first_coeff_index, const Dimensions& block_sizes,
              const Dimensions& block_strides, const Dimensions& tensor_strides,
              Scalar* data)
      : m_first_coeff_index(first_coeff_index),
        m_block_sizes(block_sizes),
        m_block_strides(block_strides),
        m_tensor_strides(tensor_strides),
        m_data(data) {}

  Index first_coeff_index() const { return m_first_coeff_index; }

  const Dimensions& block_sizes() const { return m_block_sizes; }

  const Dimensions& block_strides() const { return m_block_strides; }

  const Dimensions& tensor_strides() const { return m_tensor_strides; }

  Scalar* data() { return m_data; }

  const Scalar* data() const { return m_data; }

 private:
  Index m_first_coeff_index;
  Dimensions m_block_sizes;
  Dimensions m_block_strides;
  Dimensions m_tensor_strides;
  Scalar* m_data;  // Not owned.
};

template <typename Scalar, typename Index, bool Vectorizable>
struct TensorBlockCopyOp {
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void Run(
      const Index num_coeff_to_copy, const Index dst_index,
      const Index dst_stride, Scalar* EIGEN_RESTRICT dst_data,
      const Index src_index, const Index src_stride,
      const Scalar* EIGEN_RESTRICT src_data) {
    for (Index i = 0; i < num_coeff_to_copy; ++i) {
      dst_data[dst_index + i * dst_stride] =
          src_data[src_index + i * src_stride];
    }
  }
};

// NOTE: Benchmarks run on an implementation of this that broke each of the
// loops in these conditionals into it's own template specialization (to
// avoid conditionals in the caller's loop) did not show an improvement.
template <typename Scalar, typename Index>
struct TensorBlockCopyOp<Scalar, Index, true> {
  typedef typename packet_traits<Scalar>::type Packet;
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void Run(
      const Index num_coeff_to_copy, const Index dst_index,
      const Index dst_stride, Scalar* EIGEN_RESTRICT dst_data,
      const Index src_index, const Index src_stride,
      const Scalar* EIGEN_RESTRICT src_data) {
    if (src_stride == 1) {
      const Index packet_size = internal::unpacket_traits<Packet>::size;
      const Index vectorized_size =
          (num_coeff_to_copy / packet_size) * packet_size;
      if (dst_stride == 1) {
        // LINEAR
        for (Index i = 0; i < vectorized_size; i += packet_size) {
          Packet p = internal::ploadu<Packet>(src_data + src_index + i);
          internal::pstoreu<Scalar, Packet>(dst_data + dst_index + i, p);
        }
        for (Index i = vectorized_size; i < num_coeff_to_copy; ++i) {
          dst_data[dst_index + i] = src_data[src_index + i];
        }
      } else {
        // SCATTER
        for (Index i = 0; i < vectorized_size; i += packet_size) {
          Packet p = internal::ploadu<Packet>(src_data + src_index + i);
          internal::pscatter<Scalar, Packet>(
              dst_data + dst_index + i * dst_stride, p, dst_stride);
        }
        for (Index i = vectorized_size; i < num_coeff_to_copy; ++i) {
          dst_data[dst_index + i * dst_stride] = src_data[src_index + i];
        }
      }
    } else if (src_stride == 0) {
      const Index packet_size = internal::unpacket_traits<Packet>::size;
      const Index vectorized_size =
          (num_coeff_to_copy / packet_size) * packet_size;
      if (dst_stride == 1) {
        // LINEAR
        for (Index i = 0; i < vectorized_size; i += packet_size) {
          Packet p = internal::pload1<Packet>(src_data + src_index);
          internal::pstoreu<Scalar, Packet>(dst_data + dst_index + i, p);
        }
        for (Index i = vectorized_size; i < num_coeff_to_copy; ++i) {
          dst_data[dst_index + i] = src_data[src_index];
        }
      } else {
        // SCATTER
        for (Index i = 0; i < vectorized_size; i += packet_size) {
          Packet p = internal::pload1<Packet>(src_data + src_index);
          internal::pscatter<Scalar, Packet>(
              dst_data + dst_index + i * dst_stride, p, dst_stride);
        }
        for (Index i = vectorized_size; i < num_coeff_to_copy; ++i) {
          dst_data[dst_index + i * dst_stride] = src_data[src_index];
        }
      }
    } else {
      if (dst_stride == 1) {
        // GATHER
        const Index packet_size = internal::unpacket_traits<Packet>::size;
        const Index vectorized_size =
            (num_coeff_to_copy / packet_size) * packet_size;
        for (Index i = 0; i < vectorized_size; i += packet_size) {
          Packet p = internal::pgather<Scalar, Packet>(
              src_data + src_index + i * src_stride, src_stride);
          internal::pstoreu<Scalar, Packet>(dst_data + dst_index + i, p);
        }
        for (Index i = vectorized_size; i < num_coeff_to_copy; ++i) {
          dst_data[dst_index + i] = src_data[src_index + i * src_stride];
        }
      } else {
        // RANDOM
        for (Index i = 0; i < num_coeff_to_copy; ++i) {
          dst_data[dst_index + i * dst_stride] =
              src_data[src_index + i * src_stride];
        }
      }
    }
  }
};

/**
 * \class TensorBlockIO
 * \ingroup CXX11_Tensor_Module
 *
 * \brief Tensor block IO class.
 *
 * This class is responsible for copying data between a tensor and a tensor
 * block.
 */
template <typename Scalar, typename Index, int NumDims, int Layout,
          bool Vectorizable, bool BlockRead>
class TensorBlockIO {
 public:
  typedef typename internal::TensorBlock<Scalar, Index, NumDims, Layout>
      TensorBlock;
  typedef typename internal::TensorBlockCopyOp<Scalar, Index, Vectorizable>
      TensorBlockCopyOp;

 protected:
  struct BlockIteratorState {
    Index input_stride;
    Index output_stride;
    Index input_span;
    Index output_span;
    Index size;
    Index count;
  };

  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void Copy(
      const TensorBlock& block, Index first_coeff_index,
      const array<Index, NumDims>& tensor_to_block_dim_map,
      const array<Index, NumDims>& tensor_strides, const Scalar* src_data,
      Scalar* dst_data) {
    // Find the innermost tensor dimension whose size is not 1. This is the
    // effective inner dim. If all dimensions are of size 1, then fallback to
    // using the actual innermost dim to avoid out-of-bound access.
    Index num_size_one_inner_dims = 0;
    for (int i = 0; i < NumDims; ++i) {
      const int dim = cond<Layout>()(i, NumDims - i - 1);
      if (block.block_sizes()[tensor_to_block_dim_map[dim]] != 1) {
        num_size_one_inner_dims = i;
        break;
      }
    }
    // Calculate strides and dimensions.
    const Index tensor_stride1_dim = cond<Layout>()(
        num_size_one_inner_dims, NumDims - num_size_one_inner_dims - 1);
    const Index block_dim_for_tensor_stride1_dim =
        NumDims == 0 ? 1 : tensor_to_block_dim_map[tensor_stride1_dim];
    size_t block_inner_dim_size =
        NumDims == 0 ? 1
                     : block.block_sizes()[block_dim_for_tensor_stride1_dim];
    for (int i = num_size_one_inner_dims + 1; i < NumDims; ++i) {
      const int dim = cond<Layout>()(i, NumDims - i - 1);
      const Index block_stride =
          block.block_strides()[tensor_to_block_dim_map[dim]];
      if (block_inner_dim_size == block_stride &&
          block_stride == tensor_strides[dim]) {
        block_inner_dim_size *=
            block.block_sizes()[tensor_to_block_dim_map[dim]];
        ++num_size_one_inner_dims;
      } else {
        break;
      }
    }

    Index inputIndex;
    Index outputIndex;
    Index input_stride;
    Index output_stride;

    // Setup strides to read/write along the tensor's stride1 dimension.
    if (BlockRead) {
      inputIndex = first_coeff_index;
      outputIndex = 0;
      input_stride = NumDims == 0 ? 1 : tensor_strides[tensor_stride1_dim];
      output_stride =
          NumDims == 0
              ? 1
              : block.block_strides()[block_dim_for_tensor_stride1_dim];
    } else {
      inputIndex = 0;
      outputIndex = first_coeff_index;
      input_stride =
          NumDims == 0
              ? 1
              : block.block_strides()[block_dim_for_tensor_stride1_dim];
      output_stride = NumDims == 0 ? 1 : tensor_strides[tensor_stride1_dim];
    }

    const int at_least_1_dim = NumDims <= 1 ? 1 : NumDims - 1;
    array<BlockIteratorState, at_least_1_dim> block_iter_state;

    // Initialize block iterator state. Squeeze away any dimension of size 1.
    int num_squeezed_dims = 0;
    for (int i = num_size_one_inner_dims; i < NumDims - 1; ++i) {
      const int dim = cond<Layout>()(i + 1, NumDims - i - 2);
      const Index size = block.block_sizes()[tensor_to_block_dim_map[dim]];
      if (size == 1) {
        continue;
      }
      block_iter_state[num_squeezed_dims].size = size;
      if (BlockRead) {
        block_iter_state[num_squeezed_dims].input_stride = tensor_strides[dim];
        block_iter_state[num_squeezed_dims].output_stride =
            block.block_strides()[tensor_to_block_dim_map[dim]];
      } else {
        block_iter_state[num_squeezed_dims].input_stride =
            block.block_strides()[tensor_to_block_dim_map[dim]];
        block_iter_state[num_squeezed_dims].output_stride = tensor_strides[dim];
      }
      block_iter_state[num_squeezed_dims].input_span =
          block_iter_state[num_squeezed_dims].input_stride *
          (block_iter_state[num_squeezed_dims].size - 1);
      block_iter_state[num_squeezed_dims].output_span =
          block_iter_state[num_squeezed_dims].output_stride *
          (block_iter_state[num_squeezed_dims].size - 1);
      block_iter_state[num_squeezed_dims].count = 0;
      ++num_squeezed_dims;
    }

    // Iterate copying data from src to dst.
    const Index block_total_size =
        NumDims == 0 ? 1 : block.block_sizes().TotalSize();
    for (Index i = 0; i < block_total_size; i += block_inner_dim_size) {
      TensorBlockCopyOp::Run(block_inner_dim_size, outputIndex, output_stride,
                             dst_data, inputIndex, input_stride, src_data);
      // Update index.
      for (int j = 0; j < num_squeezed_dims; ++j) {
        if (++block_iter_state[j].count < block_iter_state[j].size) {
          inputIndex += block_iter_state[j].input_stride;
          outputIndex += block_iter_state[j].output_stride;
          break;
        }
        block_iter_state[j].count = 0;
        inputIndex -= block_iter_state[j].input_span;
        outputIndex -= block_iter_state[j].output_span;
      }
    }
  }
};

/**
 * \class TensorBlockReader
 * \ingroup CXX11_Tensor_Module
 *
 * \brief Tensor block reader class.
 *
 * This class is responsible for reading a tensor block.
 *
 */
template <typename Scalar, typename Index, int NumDims, int Layout,
          bool Vectorizable>
class TensorBlockReader
    : public TensorBlockIO<Scalar, Index, NumDims, Layout, Vectorizable, true> {
 public:
  typedef typename internal::TensorBlock<Scalar, Index, NumDims, Layout>
      TensorBlock;
  typedef TensorBlockIO<Scalar, Index, NumDims, Layout, Vectorizable, true>
      Base;

  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void Run(
      TensorBlock* block, const Scalar* src_data) {
    array<Index, NumDims> tensor_to_block_dim_map;
    for (int i = 0; i < NumDims; ++i) {
      tensor_to_block_dim_map[i] = i;
    }
    Base::Copy(*block, block->first_coeff_index(), tensor_to_block_dim_map,
               block->tensor_strides(), src_data, block->data());
  }

  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void Run(
      TensorBlock* block, Index first_coeff_index,
      const array<Index, NumDims>& tensor_to_block_dim_map,
      const array<Index, NumDims>& tensor_strides, const Scalar* src_data) {
    Base::Copy(*block, first_coeff_index, tensor_to_block_dim_map,
               tensor_strides, src_data, block->data());
  }
};

/**
 * \class TensorBlockWriter
 * \ingroup CXX11_Tensor_Module
 *
 * \brief Tensor block writer class.
 *
 * This class is responsible for writing a tensor block.
 *
 */
template <typename Scalar, typename Index, int NumDims, int Layout,
          bool Vectorizable>
class TensorBlockWriter : public TensorBlockIO<Scalar, Index, NumDims, Layout,
                                               Vectorizable, false> {
 public:
  typedef typename internal::TensorBlock<Scalar, Index, NumDims, Layout>
      TensorBlock;
  typedef TensorBlockIO<Scalar, Index, NumDims, Layout, Vectorizable, false>
      Base;

  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void Run(
      const TensorBlock& block, Scalar* dst_data) {
    array<Index, NumDims> tensor_to_block_dim_map;
    for (int i = 0; i < NumDims; ++i) {
      tensor_to_block_dim_map[i] = i;
    }
    Base::Copy(block, block.first_coeff_index(), tensor_to_block_dim_map,
               block.tensor_strides(), block.data(), dst_data);
  }

  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void Run(
      const TensorBlock& block, Index first_coeff_index,
      const array<Index, NumDims>& tensor_to_block_dim_map,
      const array<Index, NumDims>& tensor_strides, Scalar* dst_data) {
    Base::Copy(block, first_coeff_index, tensor_to_block_dim_map,
               tensor_strides, block.data(), dst_data);
  }
};

/**
 * \class TensorBlockCwiseBinaryOp
 * \ingroup CXX11_Tensor_Module
 *
 * \brief Carries out a cwise binary op on a number of coefficients.
 *
 * This class reads strided inputs from left and right operands, and writes the
 * result of the cwise binary op to the strided output array.
 *
 */
template <bool Vectorizable>
struct TensorBlockCwiseBinaryOp {
  template <typename Index, typename BinaryFunctor, typename OutputScalar,
            typename LeftScalar, typename RightScalar>
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void Run(
      const BinaryFunctor& functor, const Index num_coeff,
      const Index output_index, const Index output_stride,
      OutputScalar* output_data, const Index left_index,
      const Index left_stride, const LeftScalar* left_data,
      const Index right_index, const Index right_stride,
      const RightScalar* right_data) {
    for (Index i = 0; i < num_coeff; ++i) {
      output_data[output_index + i * output_stride] =
          functor(left_data[left_index + i * left_stride],
                  right_data[right_index + i * right_stride]);
    }
  }
};

template <>
struct TensorBlockCwiseBinaryOp<true> {
  template <typename Index, typename BinaryFunctor, typename OutputScalar,
            typename LeftScalar, typename RightScalar>
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void Run(
      const BinaryFunctor& functor, const Index num_coeff,
      const Index output_index, const Index output_stride,
      OutputScalar* output_data, const Index left_index,
      const Index left_stride, const LeftScalar* left_data,
      const Index right_index, const Index right_stride,
      const RightScalar* right_data) {
    EIGEN_STATIC_ASSERT(functor_traits<BinaryFunctor>::PacketAccess,
                        YOU_MADE_A_PROGRAMMING_MISTAKE);
    typedef typename packet_traits<OutputScalar>::type OutputPacket;
    typedef typename packet_traits<LeftScalar>::type LeftPacket;
    typedef typename packet_traits<RightScalar>::type RightPacket;
    const Index packet_size = unpacket_traits<OutputPacket>::size;
    EIGEN_STATIC_ASSERT(unpacket_traits<LeftPacket>::size == packet_size,
                        YOU_MADE_A_PROGRAMMING_MISTAKE);
    EIGEN_STATIC_ASSERT(unpacket_traits<RightPacket>::size == packet_size,
                        YOU_MADE_A_PROGRAMMING_MISTAKE);
    const Index vectorized_size = (num_coeff / packet_size) * packet_size;
    if (output_stride != 1 || left_stride != 1 || right_stride != 1) {
      TensorBlockCwiseBinaryOp<false>::Run(
          functor, num_coeff, output_index, output_stride, output_data,
          left_index, left_stride, left_data, right_index, right_stride,
          right_data);
      return;
    }
    // Vectorization for the most common case.
    for (Index i = 0; i < vectorized_size; i += packet_size) {
      LeftPacket l = internal::ploadu<LeftPacket>(left_data + left_index + i);
      RightPacket r =
          internal::ploadu<RightPacket>(right_data + right_index + i);
      OutputPacket p = functor.packetOp(l, r);
      internal::pstoreu<OutputScalar, OutputPacket>(
          output_data + output_index + i, p);
    }
    for (Index i = vectorized_size; i < num_coeff; ++i) {
      output_data[output_index + i] =
          functor(left_data[left_index + i], right_data[right_index + i]);
    }
  }
};

/**
 * \class TensorBlockCwiseBinaryIO
 * \ingroup CXX11_Tensor_Module
 *
 * \brief Tensor block IO class for carrying out cwise binary ops.
 *
 * This class carries out the binary op on given blocks.
 *
 */
template <typename BinaryFunctor, typename Index, typename OutputScalar,
          int NumDims, int Layout>
struct TensorBlockCwiseBinaryIO {
  typedef typename internal::TensorBlock<OutputScalar, Index, NumDims,
                                         Layout>::Dimensions Dimensions;
  typedef internal::TensorBlockCwiseBinaryOp<
      functor_traits<BinaryFunctor>::PacketAccess>
      TensorBlockCwiseBinaryOp;

  struct BlockIteratorState {
    Index output_stride, output_span;
    Index left_stride, left_span;
    Index right_stride, right_span;
    Index size, count;
  };

  template <typename LeftScalar, typename RightScalar>
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void Run(
      const BinaryFunctor& functor, const Dimensions& block_sizes,
      const Dimensions& block_strides, OutputScalar* output_data,
      const array<Index, NumDims>& left_strides, const LeftScalar* left_data,
      const array<Index, NumDims>& right_strides,
      const RightScalar* right_data) {
    // Find the innermost dimension whose size is not 1. This is the effective
    // inner dim. If all dimensions are of size 1, fallback to using the actual
    // innermost dim to avoid out-of-bound access.
    int num_size_one_inner_dims = 0;
    for (int i = 0; i < NumDims; ++i) {
      const int dim = cond<Layout>()(i, NumDims - i - 1);
      if (block_sizes[dim] != 1) {
        num_size_one_inner_dims = i;
        break;
      }
    }
    // Calculate strides and dimensions.
    const int inner_dim =
        NumDims == 0 ? 1
                     : cond<Layout>()(num_size_one_inner_dims,
                                      NumDims - num_size_one_inner_dims - 1);
    Index inner_dim_size = NumDims == 0 ? 1 : block_sizes[inner_dim];
    for (int i = num_size_one_inner_dims + 1; i < NumDims; ++i) {
      const int dim = cond<Layout>()(i, NumDims - i - 1);
      // Merge multiple inner dims into one for larger inner dim size (i.e.
      // fewer calls to TensorBlockCwiseBinaryOp::Run()).
      if (inner_dim_size == block_strides[dim] &&
          block_strides[dim] == left_strides[dim] &&
          block_strides[dim] == right_strides[dim]) {
        inner_dim_size *= block_sizes[dim];
        ++num_size_one_inner_dims;
      } else {
        break;
      }
    }

    Index output_index = 0, left_index = 0, right_index = 0;
    const Index output_stride = NumDims == 0 ? 1 : block_strides[inner_dim];
    const Index left_stride = NumDims == 0 ? 1 : left_strides[inner_dim];
    const Index right_stride = NumDims == 0 ? 1 : right_strides[inner_dim];

    const int at_least_1_dim = NumDims <= 1 ? 1 : NumDims - 1;
    array<BlockIteratorState, at_least_1_dim> block_iter_state;

    // Initialize block iterator state. Squeeze away any dimension of size 1.
    int num_squeezed_dims = 0;
    for (int i = num_size_one_inner_dims; i < NumDims - 1; ++i) {
      const int dim = cond<Layout>()(i + 1, NumDims - i - 2);
      const Index size = block_sizes[dim];
      if (size == 1) {
        continue;
      }
      auto& state = block_iter_state[num_squeezed_dims];
      state.output_stride = block_strides[dim];
      state.left_stride = left_strides[dim];
      state.right_stride = right_strides[dim];
      state.size = size;
      state.output_span = state.output_stride * (size - 1);
      state.left_span = state.left_stride * (size - 1);
      state.right_span = state.right_stride * (size - 1);
      state.count = 0;
      ++num_squeezed_dims;
    }

    // Compute cwise binary op.
    const Index block_total_size = NumDims == 0 ? 1 : block_sizes.TotalSize();
    for (Index i = 0; i < block_total_size; i += inner_dim_size) {
      TensorBlockCwiseBinaryOp::Run(functor, inner_dim_size, output_index,
                                    output_stride, output_data, left_index,
                                    left_stride, left_data, right_index,
                                    right_stride, right_data);
      // Update index.
      for (int j = 0; j < num_squeezed_dims; ++j) {
        auto& state = block_iter_state[j];
        if (++state.count < state.size) {
          output_index += state.output_stride;
          left_index += state.left_stride;
          right_index += state.right_stride;
          break;
        }
        state.count = 0;
        output_index -= state.output_span;
        left_index -= state.left_span;
        right_index -= state.right_span;
      }
    }
  }
};

/**
 * \class TensorBlockMapper
 * \ingroup CXX11_Tensor_Module
 *
 * \brief Tensor block mapper class.
 *
 * This class is responsible for iterating over the blocks of a tensor.
 */
template <typename Scalar, typename Index, int NumDims, int Layout>
class TensorBlockMapper {
 public:
  typedef typename internal::TensorBlock<Scalar, Index, NumDims, Layout>
      TensorBlock;
  typedef DSizes<Index, NumDims> Dimensions;

  TensorBlockMapper(const Dimensions& dims,
                    const TensorBlockShapeType block_shape,
                    size_t min_target_size)
      : m_dimensions(dims),
        m_block_dim_sizes(BlockDimensions(dims, block_shape, min_target_size)) {
    // Calculate block counts by dimension and total block count.
    DSizes<Index, NumDims> block_count;
    for (size_t i = 0; i < block_count.rank(); ++i) {
      block_count[i] = divup(m_dimensions[i], m_block_dim_sizes[i]);
    }
    m_total_block_count = array_prod(block_count);

    // Calculate block strides (used for enumerating blocks).
    if (NumDims > 0) {
      if (static_cast<int>(Layout) == static_cast<int>(ColMajor)) {
        m_block_strides[0] = 1;
        m_tensor_strides[0] = 1;
        for (int i = 1; i < NumDims; ++i) {
          m_block_strides[i] = m_block_strides[i - 1] * block_count[i - 1];
          m_tensor_strides[i] = m_tensor_strides[i - 1] * m_dimensions[i - 1];
        }
      } else {
        m_block_strides[NumDims - 1] = 1;
        m_tensor_strides[NumDims - 1] = 1;
        for (int i = NumDims - 2; i >= 0; --i) {
          m_block_strides[i] = m_block_strides[i + 1] * block_count[i + 1];
          m_tensor_strides[i] = m_tensor_strides[i + 1] * m_dimensions[i + 1];
        }
      }
    }
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorBlock
  GetBlockForIndex(Index block_index, Scalar* data) const {
    Index first_coeff_index = 0;
    DSizes<Index, NumDims> coords;
    DSizes<Index, NumDims> sizes;
    DSizes<Index, NumDims> strides;
    if (NumDims > 0) {
      if (static_cast<int>(Layout) == static_cast<int>(ColMajor)) {
        for (int i = NumDims - 1; i > 0; --i) {
          const Index idx = block_index / m_block_strides[i];
          coords[i] = idx * m_block_dim_sizes[i];
          sizes[i] =
              numext::mini((m_dimensions[i] - coords[i]), m_block_dim_sizes[i]);
          block_index -= idx * m_block_strides[i];
          first_coeff_index += coords[i] * m_tensor_strides[i];
        }
        coords[0] = block_index * m_block_dim_sizes[0];
        sizes[0] =
            numext::mini((m_dimensions[0] - coords[0]), m_block_dim_sizes[0]);
        first_coeff_index += coords[0] * m_tensor_strides[0];

        strides[0] = 1;
        for (int i = 1; i < NumDims; ++i) {
          strides[i] = strides[i - 1] * sizes[i - 1];
        }
      } else {
        for (int i = 0; i < NumDims - 1; ++i) {
          const Index idx = block_index / m_block_strides[i];
          coords[i] = idx * m_block_dim_sizes[i];
          sizes[i] =
              numext::mini((m_dimensions[i] - coords[i]), m_block_dim_sizes[i]);
          block_index -= idx * m_block_strides[i];
          first_coeff_index += coords[i] * m_tensor_strides[i];
        }
        coords[NumDims - 1] = block_index * m_block_dim_sizes[NumDims - 1];
        sizes[NumDims - 1] =
            numext::mini((m_dimensions[NumDims - 1] - coords[NumDims - 1]),
                         m_block_dim_sizes[NumDims - 1]);
        first_coeff_index +=
            coords[NumDims - 1] * m_tensor_strides[NumDims - 1];

        strides[NumDims - 1] = 1;
        for (int i = NumDims - 2; i >= 0; --i) {
          strides[i] = strides[i + 1] * sizes[i + 1];
        }
      }
    }

    return TensorBlock(first_coeff_index, sizes, strides, m_tensor_strides,
                       data);
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Index total_block_count() const {
    return m_total_block_count;
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Index block_dims_total_size() const {
    return m_block_dim_sizes.TotalSize();
  }

 private:
  static Dimensions BlockDimensions(const Dimensions& tensor_dims,
                                    const TensorBlockShapeType block_shape,
                                    size_t min_target_size) {
    min_target_size = numext::maxi<size_t>(1, min_target_size);

    // If tensor fully fits into the target size, we'll treat it a single block.
    Dimensions block_dim_sizes = tensor_dims;

    if (tensor_dims.TotalSize() == 0) {
      // Corner case: one of the dimensions is zero. Logic below is too complex
      // to handle this case on a general basis, just use unit block size.
      // Note: we must not yield blocks with zero dimensions (recipe for
      // overflows/underflows, divisions by zero and NaNs later).
      for (int i = 0; i < NumDims; ++i) {
        block_dim_sizes[i] = 1;
      }
    } else if (block_dim_sizes.TotalSize() > min_target_size) {
      if (block_shape == TensorBlockShapeType::kUniformAllDims) {
        // Tensor will not fit within 'min_target_size' budget: calculate tensor
        // block dimension sizes based on "square" dimension size target.
        const size_t dim_size_target = static_cast<const size_t>(
            std::pow(static_cast<float>(min_target_size),
                     1.0 / static_cast<float>(block_dim_sizes.rank())));
        for (size_t i = 0; i < block_dim_sizes.rank(); ++i) {
          // TODO(andydavis) Adjust the inner most 'block_dim_size' to make it
          // a multiple of the packet size. Note that reducing
          // 'block_dim_size' in this manner can increase the number of
          // blocks, and so will amplify any per-block overhead.
          block_dim_sizes[i] = numext::mini(
              dim_size_target, static_cast<size_t>(tensor_dims[i]));
        }
        // Add any un-allocated coefficients to inner dimension(s).
        Index total_size = block_dim_sizes.TotalSize();
        for (int i = 0; i < NumDims; ++i) {
          const int dim = cond<Layout>()(i, NumDims - i - 1);
          if (block_dim_sizes[dim] < tensor_dims[dim]) {
            const Index total_size_other_dims =
                total_size / block_dim_sizes[dim];
            const Index alloc_avail =
                divup<Index>(min_target_size, total_size_other_dims);
            if (alloc_avail == block_dim_sizes[dim]) {
              // Insufficient excess coefficients to allocate.
              break;
            }
            block_dim_sizes[dim] = numext::mini(tensor_dims[dim], alloc_avail);
            total_size = total_size_other_dims * block_dim_sizes[dim];
          }
        }
      } else if (block_shape == TensorBlockShapeType::kSkewedInnerDims) {
        Index coeff_to_allocate = min_target_size;
        for (int i = 0; i < NumDims; ++i) {
          const int dim = cond<Layout>()(i, NumDims - i - 1);
          block_dim_sizes[dim] =
              numext::mini(coeff_to_allocate, tensor_dims[dim]);
          coeff_to_allocate =
              divup(coeff_to_allocate,
                    numext::maxi(static_cast<Index>(1), block_dim_sizes[dim]));
        }
        eigen_assert(coeff_to_allocate == 1);
      } else {
        eigen_assert(false);  // someone added new block shape type
      }
    }

    eigen_assert(
        block_dim_sizes.TotalSize() >=
        numext::mini<size_t>(min_target_size, tensor_dims.TotalSize()));

    return block_dim_sizes;
  }

  Dimensions m_dimensions;
  Dimensions m_block_dim_sizes;
  Dimensions m_block_strides;
  Dimensions m_tensor_strides;
  Index m_total_block_count;
};

/**
 * \class TensorSliceBlockMapper
 * \ingroup CXX11_Tensor_Module
 *
 * \brief Tensor slice block mapper class.
 *
 * This class is responsible for iterating over the blocks of
 * a slice of a tensor. Supports shuffling of the block strides
 * for callers that want to reduce strides for dimensions to be
 * processed together.
 *
 */
template <typename Scalar, typename Index, int NumDims, int Layout>
class TensorSliceBlockMapper {
 public:
  typedef typename internal::TensorBlock<Scalar, Index, NumDims, Layout>
      TensorBlock;
  typedef DSizes<Index, NumDims> Dimensions;

  TensorSliceBlockMapper(const Dimensions& tensor_dims,
                         const Dimensions& tensor_slice_offsets,
                         const Dimensions& tensor_slice_extents,
                         const Dimensions& block_dim_sizes,
                         const Dimensions& block_stride_order)
      : m_tensor_dimensions(tensor_dims),
        m_tensor_slice_offsets(tensor_slice_offsets),
        m_tensor_slice_extents(tensor_slice_extents),
        m_block_dim_sizes(block_dim_sizes),
        m_block_stride_order(block_stride_order),
        m_total_block_count(1) {
    // Calculate block counts by dimension and total block count.
    DSizes<Index, NumDims> block_count;
    for (size_t i = 0; i < block_count.rank(); ++i) {
      block_count[i] = divup(m_tensor_slice_extents[i], m_block_dim_sizes[i]);
    }
    m_total_block_count = array_prod(block_count);

    // Calculate block strides (used for enumerating blocks).
    if (static_cast<int>(Layout) == static_cast<int>(ColMajor)) {
      m_block_strides[0] = 1;
      m_tensor_strides[0] = 1;
      for (int i = 1; i < NumDims; ++i) {
        m_block_strides[i] = m_block_strides[i - 1] * block_count[i - 1];
        m_tensor_strides[i] =
            m_tensor_strides[i - 1] * m_tensor_dimensions[i - 1];
      }
    } else {
      m_block_strides[NumDims - 1] = 1;
      m_tensor_strides[NumDims - 1] = 1;
      for (int i = NumDims - 2; i >= 0; --i) {
        m_block_strides[i] = m_block_strides[i + 1] * block_count[i + 1];
        m_tensor_strides[i] =
            m_tensor_strides[i + 1] * m_tensor_dimensions[i + 1];
      }
    }
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorBlock
  GetBlockForIndex(Index block_index, Scalar* data) const {
    Index first_coeff_index = 0;
    DSizes<Index, NumDims> coords;
    DSizes<Index, NumDims> sizes;
    DSizes<Index, NumDims> strides;
    if (static_cast<int>(Layout) == static_cast<int>(ColMajor)) {
      for (int i = NumDims - 1; i > 0; --i) {
        const Index idx = block_index / m_block_strides[i];
        coords[i] = m_tensor_slice_offsets[i] + idx * m_block_dim_sizes[i];
        sizes[i] = numext::mini(
            m_tensor_slice_offsets[i] + m_tensor_slice_extents[i] - coords[i],
            m_block_dim_sizes[i]);
        block_index -= idx * m_block_strides[i];
        first_coeff_index += coords[i] * m_tensor_strides[i];
      }
      coords[0] =
          m_tensor_slice_offsets[0] + block_index * m_block_dim_sizes[0];
      sizes[0] = numext::mini(
          m_tensor_slice_offsets[0] + m_tensor_slice_extents[0] - coords[0],
          m_block_dim_sizes[0]);
      first_coeff_index += coords[0] * m_tensor_strides[0];

      Index prev_dim = m_block_stride_order[0];
      strides[prev_dim] = 1;
      for (int i = 1; i < NumDims; ++i) {
        const Index curr_dim = m_block_stride_order[i];
        strides[curr_dim] = strides[prev_dim] * sizes[prev_dim];
        prev_dim = curr_dim;
      }
    } else {
      for (int i = 0; i < NumDims - 1; ++i) {
        const Index idx = block_index / m_block_strides[i];
        coords[i] = m_tensor_slice_offsets[i] + idx * m_block_dim_sizes[i];
        sizes[i] = numext::mini(
            m_tensor_slice_offsets[i] + m_tensor_slice_extents[i] - coords[i],
            m_block_dim_sizes[i]);
        block_index -= idx * m_block_strides[i];
        first_coeff_index += coords[i] * m_tensor_strides[i];
      }
      coords[NumDims - 1] = m_tensor_slice_offsets[NumDims - 1] +
                            block_index * m_block_dim_sizes[NumDims - 1];
      sizes[NumDims - 1] = numext::mini(
          m_tensor_slice_offsets[NumDims - 1] +
              m_tensor_slice_extents[NumDims - 1] - coords[NumDims - 1],
          m_block_dim_sizes[NumDims - 1]);
      first_coeff_index += coords[NumDims - 1] * m_tensor_strides[NumDims - 1];

      Index prev_dim = m_block_stride_order[NumDims - 1];
      strides[prev_dim] = 1;
      for (int i = NumDims - 2; i >= 0; --i) {
        const Index curr_dim = m_block_stride_order[i];
        strides[curr_dim] = strides[prev_dim] * sizes[prev_dim];
        prev_dim = curr_dim;
      }
    }

    return TensorBlock(first_coeff_index, sizes, strides, m_tensor_strides,
                       data);
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Index total_block_count() const {
    return m_total_block_count;
  }

 private:
  Dimensions m_tensor_dimensions;
  Dimensions m_tensor_slice_offsets;
  Dimensions m_tensor_slice_extents;
  Dimensions m_tensor_strides;
  Dimensions m_block_dim_sizes;
  Dimensions m_block_stride_order;
  Dimensions m_block_strides;
  Index m_total_block_count;
};

}  // namespace internal

}  // namespace Eigen

#endif  // EIGEN_CXX11_TENSOR_TENSOR_BLOCK_H
