// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CXX11_TENSOR_TENSOR_IMAGE_PATCH_H
#define EIGEN_CXX11_TENSOR_TENSOR_IMAGE_PATCH_H

namespace Eigen {

/** \class TensorImagePatch
  * \ingroup CXX11_Tensor_Module
  *
  * \brief Patch extraction specialized for image processing.
  * This assumes that the input has a least 3 dimensions ordered as follow:
  *  1st dimension: channels (of size d)
  *  2nd dimension: rows (of size r)
  *  3rd dimension: columns (of size c)
  *  There can be additional dimensions such as time (for video) or batch (for
  * bulk processing after the first 3.
  * Calling the image patch code with patch_rows and patch_cols is equivalent
  * to calling the regular patch extraction code with parameters d, patch_rows,
  * patch_cols, and 1 for all the additional dimensions.
  */
namespace internal {
template<DenseIndex Rows, DenseIndex Cols, typename XprType>
struct traits<TensorImagePatchOp<Rows, Cols, XprType> > : public traits<XprType>
{
  typedef typename XprType::Scalar Scalar;
  typedef traits<XprType> XprTraits;
  typedef typename packet_traits<Scalar>::type Packet;
  typedef typename XprTraits::StorageKind StorageKind;
  typedef typename XprTraits::Index Index;
  typedef typename XprType::Nested Nested;
  typedef typename remove_reference<Nested>::type _Nested;
};

template<DenseIndex Rows, DenseIndex Cols, typename XprType>
struct eval<TensorImagePatchOp<Rows, Cols, XprType>, Eigen::Dense>
{
  typedef const TensorImagePatchOp<Rows, Cols, XprType>& type;
};

template<DenseIndex Rows, DenseIndex Cols, typename XprType>
struct nested<TensorImagePatchOp<Rows, Cols, XprType>, 1, typename eval<TensorImagePatchOp<Rows, Cols, XprType> >::type>
{
  typedef TensorImagePatchOp<Rows, Cols, XprType> type;
};

}  // end namespace internal



template<DenseIndex Rows, DenseIndex Cols, typename XprType>
class TensorImagePatchOp : public TensorBase<TensorImagePatchOp<Rows, Cols, XprType>, ReadOnlyAccessors>
{
  public:
  typedef typename Eigen::internal::traits<TensorImagePatchOp>::Scalar Scalar;
  typedef typename Eigen::internal::traits<TensorImagePatchOp>::Packet Packet;
  typedef typename Eigen::NumTraits<Scalar>::Real RealScalar;
  typedef typename XprType::CoeffReturnType CoeffReturnType;
  typedef typename XprType::PacketReturnType PacketReturnType;
  typedef typename Eigen::internal::nested<TensorImagePatchOp>::type Nested;
  typedef typename Eigen::internal::traits<TensorImagePatchOp>::StorageKind StorageKind;
  typedef typename Eigen::internal::traits<TensorImagePatchOp>::Index Index;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorImagePatchOp(const XprType& expr, DenseIndex patch_rows, DenseIndex patch_cols,
                                                           DenseIndex row_strides, DenseIndex col_strides)
      : m_xpr(expr), m_patch_rows(patch_rows), m_patch_cols(patch_cols),
        m_row_strides(row_strides), m_col_strides(col_strides){}

    EIGEN_DEVICE_FUNC
    DenseIndex patch_rows() const { return m_patch_rows; }
    EIGEN_DEVICE_FUNC
    DenseIndex patch_cols() const { return m_patch_cols; }
    EIGEN_DEVICE_FUNC
    DenseIndex row_strides() const { return m_row_strides; }
    EIGEN_DEVICE_FUNC
    DenseIndex col_strides() const { return m_col_strides; }

    EIGEN_DEVICE_FUNC
    const typename internal::remove_all<typename XprType::Nested>::type&
    expression() const { return m_xpr; }

  protected:
    typename XprType::Nested m_xpr;
    const DenseIndex m_patch_rows;
    const DenseIndex m_patch_cols;
    const DenseIndex m_row_strides;
    const DenseIndex m_col_strides;
};


// Eval as rvalue
template<DenseIndex Rows, DenseIndex Cols, typename ArgType, typename Device>
struct TensorEvaluator<const TensorImagePatchOp<Rows, Cols, ArgType>, Device>
{
  typedef TensorImagePatchOp<Rows, Cols, ArgType> XprType;
  typedef typename XprType::Index Index;
  static const int NumDims = internal::array_size<typename TensorEvaluator<ArgType, Device>::Dimensions>::value + 1;
  typedef DSizes<Index, NumDims> Dimensions;
  typedef typename XprType::Scalar Scalar;

  enum {
    IsAligned = false,
    PacketAccess = TensorEvaluator<ArgType, Device>::PacketAccess,
  };

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorEvaluator(const XprType& op, const Device& device)
      : m_impl(op.expression(), device)
  {
    EIGEN_STATIC_ASSERT(NumDims >= 4, YOU_MADE_A_PROGRAMMING_MISTAKE);

    const typename TensorEvaluator<ArgType, Device>::Dimensions& input_dims = m_impl.dimensions();
    m_dimensions[0] = input_dims[0];
    m_dimensions[1] = op.patch_rows();
    m_dimensions[2] = op.patch_cols();
    m_dimensions[3] = ceilf(static_cast<float>(input_dims[1]) / op.row_strides()) *
                      ceilf(static_cast<float>(input_dims[2]) / op.col_strides());
    for (int i = 4; i < NumDims; ++i) {
      m_dimensions[i] = input_dims[i-1];
    }

    m_colStride = m_dimensions[1];
    m_patchStride = m_colStride * m_dimensions[2] * m_dimensions[0];
    m_otherStride = m_patchStride * m_dimensions[3];

    m_inputRows = input_dims[1];
    m_inputCols = input_dims[2];

    m_rowInputStride = input_dims[0] * op.row_strides();
    m_colInputStride = input_dims[0] * input_dims[1] * op.col_strides();
    m_patchInputStride = input_dims[0] * input_dims[1] * input_dims[2];

    m_rowPaddingTop = op.patch_rows() / 2;
    m_colPaddingLeft = op.patch_cols() / 2;

    m_fastOtherStride = internal::TensorIntDivisor<Index>(m_otherStride);
    m_fastPatchStride = internal::TensorIntDivisor<Index>(m_patchStride);
    m_fastColStride = internal::TensorIntDivisor<Index>(m_colStride);
    m_fastInputRows = internal::TensorIntDivisor<Index>(m_inputRows);
    m_fastDimZero = internal::TensorIntDivisor<Index>(m_dimensions[0]);
  }

  typedef typename XprType::CoeffReturnType CoeffReturnType;
  typedef typename XprType::PacketReturnType PacketReturnType;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Dimensions& dimensions() const { return m_dimensions; }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool evalSubExprsIfNeeded(Scalar* /*data*/) {
    m_impl.evalSubExprsIfNeeded(NULL);
    return true;
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void cleanup() {
    m_impl.cleanup();
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE CoeffReturnType coeff(Index index) const
  {
    // Find the location of the first element of the patch.
    const Index patchIndex = index / m_fastPatchStride;

    // Find the offset of the element wrt the location of the first element.
    const Index patchOffset = (index - patchIndex * m_patchStride) / m_fastDimZero;

    const Index otherIndex = (NumDims == 4) ? 0 : index / m_fastOtherStride;
    const Index patch2DIndex = (NumDims == 4) ? patchIndex : (index - otherIndex * m_otherStride) / m_fastPatchStride;

    const Index colIndex = patch2DIndex / m_fastInputRows;
    const Index colOffset = patchOffset / m_fastColStride;

    const Index inputCol = colIndex + colOffset - m_colPaddingLeft;
    if (inputCol < 0 || inputCol >= m_inputCols) {
      return Scalar(0);
    }
    const Index rowIndex = patch2DIndex - colIndex * m_inputRows;  // m_rowStride is always 1
    const Index rowOffset = patchOffset - colOffset * m_colStride;

    const Index inputRow = rowIndex + rowOffset - m_rowPaddingTop;
    if (inputRow < 0 || inputRow >= m_inputRows) {
      return Scalar(0);
    }

    const Index depth = index - (index / m_fastDimZero) * m_dimensions[0];

    const Index inputIndex = depth + inputRow * m_rowInputStride + inputCol * m_colInputStride + otherIndex * m_patchInputStride;
    return m_impl.coeff(inputIndex);
  }

  template<int LoadMode>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE PacketReturnType packet(Index index) const
  {
    const Index packetSize = internal::unpacket_traits<PacketReturnType>::size;
    EIGEN_STATIC_ASSERT(packetSize > 1, YOU_MADE_A_PROGRAMMING_MISTAKE)
    eigen_assert(index+packetSize-1 < dimensions().TotalSize());

    const Index indices[2] = {index, index + packetSize - 1};
    const Index patchIndex = indices[0] / m_fastPatchStride;
    if (patchIndex != indices[1] / m_fastPatchStride) {
      return packetWithPossibleZero(index);
    }
    const Index otherIndex = (NumDims == 4) ? 0 : indices[0] / m_fastOtherStride;
    eigen_assert(otherIndex == indices[1] / m_fastOtherStride);

    // Find the offset of the element wrt the location of the first element.
    const Index patchOffsets[2] = {(indices[0] - patchIndex * m_patchStride) / m_fastDimZero,
                                   (indices[1] - patchIndex * m_patchStride) / m_fastDimZero};

    const Index patch2DIndex = (NumDims == 4) ? patchIndex : (indices[0] - otherIndex * m_otherStride) / m_fastPatchStride;
    eigen_assert(patch2DIndex == (indices[1] - otherIndex * m_otherStride) / m_fastPatchStride);

    const Index colIndex = patch2DIndex / m_fastInputRows;
    const Index colOffsets[2] = {patchOffsets[0] / m_fastColStride, patchOffsets[1] / m_fastColStride};

    const Index inputCols[2] = {colIndex + colOffsets[0] - m_colPaddingLeft, colIndex + colOffsets[1] - m_colPaddingLeft};
    if (inputCols[1] < 0 || inputCols[0] >= m_inputCols) {
      // all zeros
      return internal::pset1<PacketReturnType>(Scalar(0));
    }

    if (inputCols[0] == inputCols[1]) {
      const Index rowIndex = patch2DIndex - colIndex * m_inputRows;
      const Index rowOffsets[2] = {patchOffsets[0] - colOffsets[0]*m_colStride, patchOffsets[1] - colOffsets[1]*m_colStride};
      eigen_assert(rowOffsets[0] <= rowOffsets[1]);
      const Index inputRows[2] = {rowIndex + rowOffsets[0] - m_rowPaddingTop, rowIndex + rowOffsets[1] - m_rowPaddingTop};

      if (inputRows[1] < 0 || inputRows[0] >= m_inputRows) {
        // all zeros
        return internal::pset1<PacketReturnType>(Scalar(0));
      }

      if (inputRows[0] >= 0 && inputRows[1] < m_inputRows) {
        // no padding
        const Index depth = index - (index / m_fastDimZero) * m_dimensions[0];
        const Index inputIndex = depth + inputRows[0] * m_rowInputStride + inputCols[0] * m_colInputStride + otherIndex * m_patchInputStride;
        return m_impl.template packet<Unaligned>(inputIndex);
      }
    }

    return packetWithPossibleZero(index);
  }

  Scalar* data() const { return NULL; }

 protected:
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE PacketReturnType packetWithPossibleZero(Index index) const
  {
    const int packetSize = internal::unpacket_traits<PacketReturnType>::size;
    EIGEN_ALIGN_DEFAULT typename internal::remove_const<CoeffReturnType>::type values[packetSize];
    for (int i = 0; i < packetSize; ++i) {
      values[i] = coeff(index+i);
    }
    PacketReturnType rslt = internal::pload<PacketReturnType>(values);
    return rslt;
  }

  Dimensions m_dimensions;

  Index m_otherStride;
  Index m_patchStride;
  Index m_colStride;
  internal::TensorIntDivisor<Index> m_fastOtherStride;
  internal::TensorIntDivisor<Index> m_fastPatchStride;
  internal::TensorIntDivisor<Index> m_fastColStride;

  Index m_rowInputStride;
  Index m_colInputStride;
  Index m_patchInputStride;

  Index m_inputRows;
  Index m_inputCols;

  Index m_rowPaddingTop;
  Index m_colPaddingLeft;

  internal::TensorIntDivisor<Index> m_fastInputRows;
  internal::TensorIntDivisor<Index> m_fastDimZero;

  TensorEvaluator<ArgType, Device> m_impl;
};


} // end namespace Eigen

#endif // EIGEN_CXX11_TENSOR_TENSOR_IMAGE_PATCH_H
