// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CXX11_TENSOR_TENSOR_CONTRACTION_H
#define EIGEN_CXX11_TENSOR_TENSOR_CONTRACTION_H

namespace Eigen {

/** \class TensorContraction
  * \ingroup CXX11_Tensor_Module
  *
  * \brief Tensor contraction class.
  *
  *
  */
namespace internal {
template<typename Dimensions, typename LhsXprType, typename RhsXprType>
struct traits<TensorContractionOp<Dimensions, LhsXprType, RhsXprType> >
{
  // Type promotion to handle the case where the types of the lhs and the rhs are different.
  typedef typename internal::promote_storage_type<typename LhsXprType::Scalar,
                                                  typename RhsXprType::Scalar>::ret Scalar;
  typedef typename internal::packet_traits<Scalar>::type Packet;
  typedef typename promote_storage_type<typename traits<LhsXprType>::StorageKind,
                                        typename traits<RhsXprType>::StorageKind>::ret StorageKind;
  typedef typename promote_index_type<typename traits<LhsXprType>::Index,
                                      typename traits<RhsXprType>::Index>::type Index;
  typedef typename LhsXprType::Nested LhsNested;
  typedef typename RhsXprType::Nested RhsNested;
  typedef typename remove_reference<LhsNested>::type _LhsNested;
  typedef typename remove_reference<RhsNested>::type _RhsNested;
};

template<typename Dimensions, typename LhsXprType, typename RhsXprType>
struct eval<TensorContractionOp<Dimensions, LhsXprType, RhsXprType>, Eigen::Dense>
{
  typedef const TensorContractionOp<Dimensions, LhsXprType, RhsXprType>& type;
};

template<typename Dimensions, typename LhsXprType, typename RhsXprType>
struct nested<TensorContractionOp<Dimensions, LhsXprType, RhsXprType>, 1, typename eval<TensorContractionOp<Dimensions, LhsXprType, RhsXprType> >::type>
{
  typedef TensorContractionOp<Dimensions, LhsXprType, RhsXprType> type;
};

}  // end namespace internal



template<typename Indices, typename LhsXprType, typename RhsXprType>
class TensorContractionOp : public TensorBase<TensorContractionOp<Indices, LhsXprType, RhsXprType> >
{
  public:
  typedef typename Eigen::internal::traits<TensorContractionOp>::Scalar Scalar;
  typedef typename Eigen::internal::traits<TensorContractionOp>::Packet Packet;
  typedef typename Eigen::NumTraits<Scalar>::Real RealScalar;
  typedef typename internal::promote_storage_type<typename LhsXprType::CoeffReturnType,
                                                  typename RhsXprType::CoeffReturnType>::ret CoeffReturnType;
  typedef typename internal::promote_storage_type<typename LhsXprType::PacketReturnType,
                                                  typename RhsXprType::PacketReturnType>::ret PacketReturnType;
  typedef typename Eigen::internal::nested<TensorContractionOp>::type Nested;
  typedef typename Eigen::internal::traits<TensorContractionOp>::StorageKind StorageKind;
  typedef typename Eigen::internal::traits<TensorContractionOp>::Index Index;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorContractionOp(const LhsXprType& lhs, const RhsXprType& rhs, const Indices& dims)
      : m_lhs_xpr(lhs), m_rhs_xpr(rhs), m_indices(dims) {}

    EIGEN_DEVICE_FUNC
    const Indices& indices() const { return m_indices; }

    /** \returns the nested expressions */
    EIGEN_DEVICE_FUNC
    const typename internal::remove_all<typename LhsXprType::Nested>::type&
    lhsExpression() const { return m_lhs_xpr; }

    EIGEN_DEVICE_FUNC
    const typename internal::remove_all<typename RhsXprType::Nested>::type&
    rhsExpression() const { return m_rhs_xpr; }

  protected:
    typename LhsXprType::Nested m_lhs_xpr;
    typename RhsXprType::Nested m_rhs_xpr;
    const Indices m_indices;
};


template <size_t n> struct max_n_1 {
  static const size_t size = n;
};
template <> struct max_n_1<0> {
  static const size_t size = 1;
};


template<typename Indices, typename LeftArgType, typename RightArgType>
struct TensorEvaluator<const TensorContractionOp<Indices, LeftArgType, RightArgType> >
{
  typedef TensorContractionOp<Indices, LeftArgType, RightArgType> XprType;

  static const int NumDims = max_n_1<TensorEvaluator<LeftArgType>::Dimensions::count + TensorEvaluator<RightArgType>::Dimensions::count - 2 * Indices::size>::size;
  typedef typename XprType::Index Index;
  typedef DSizes<Index, NumDims> Dimensions;

  enum {
    IsAligned = TensorEvaluator<LeftArgType>::IsAligned & TensorEvaluator<RightArgType>::IsAligned,
    PacketAccess = /*TensorEvaluator<LeftArgType>::PacketAccess & TensorEvaluator<RightArgType>::PacketAccess */
                   false,
  };

  TensorEvaluator(const XprType& op)
      : m_leftImpl(op.lhsExpression()), m_rightImpl(op.rhsExpression())
  {
    Index index = 0;
    Index stride = 1;
    m_shiftright = 1;

    int skipped = 0;
    const typename TensorEvaluator<LeftArgType>::Dimensions& left_dims = m_leftImpl.dimensions();
    for (int i = 0; i < TensorEvaluator<LeftArgType>::Dimensions::count; ++i) {
      bool skip = false;
      for (int j = 0; j < Indices::size; ++j) {
        if (op.indices()[j].first == i) {
          skip = true;
          m_leftOffsets[2*skipped] = stride;
          m_leftOffsets[2*skipped+1] = stride * left_dims[i];
          m_stitchsize[skipped] = left_dims[i];
          break;
        }
      }
      if (!skip) {
        m_dimensions[index++] = left_dims[i];
        m_shiftright *= left_dims[i];
      } else {
        ++skipped;
      }
      stride *= left_dims[i];
    }

    stride = 1;
    skipped = 0;
    const typename TensorEvaluator<RightArgType>::Dimensions& right_dims = m_rightImpl.dimensions();
    for (int i = 0; i < TensorEvaluator<RightArgType>::Dimensions::count; ++i) {
      bool skip = false;
      for (int j = 0; j < Indices::size; ++j) {
        if (op.indices()[j].second == i) {
          skip = true;
          m_rightOffsets[2*skipped] = stride;
          m_rightOffsets[2*skipped+1] = stride * right_dims[i];
          break;
        }
      }
      if (!skip) {
        m_dimensions[index++] = right_dims[i];
      } else {
        ++skipped;
      }
      stride *= right_dims[i];
    }

    // Scalar case
    if (TensorEvaluator<LeftArgType>::Dimensions::count + TensorEvaluator<LeftArgType>::Dimensions::count == 2 * Indices::size) {
      m_dimensions[0] = 1;
    }
  }

  //  typedef typename XprType::Index Index;
  typedef typename XprType::CoeffReturnType CoeffReturnType;
  typedef typename XprType::PacketReturnType PacketReturnType;

  const Dimensions& dimensions() const { return m_dimensions; }

  void evalTo(typename XprType::Scalar* buffer) const {
    for (int i = 0; i < dimensions().TotalSize(); ++i) {
      buffer[i] += coeff(i);
    }
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE CoeffReturnType coeff(Index index) const
  {
    const Index startLeft = index % m_shiftright;
    const Index startRight = index / m_shiftright;
    CoeffReturnType result = CoeffReturnType(0);
    partialStitch(startLeft, startRight, 0, result);
    return result;
  }

  /* TODO: vectorization
  template<int LoadMode>
  EIGEN_DEVICE_FUNC PacketReturnType packet(Index index) const
  {
    assert(false);
  }*/

 private:
  EIGEN_DEVICE_FUNC void partialStitch(Index startLeft, Index startRight, int StitchIndex, CoeffReturnType& accum) const {
    Index firstLeft = (startLeft / m_leftOffsets[2*StitchIndex]) * m_leftOffsets[2*StitchIndex+1] + (startLeft % m_leftOffsets[2*StitchIndex]);
    Index firstRight = (startRight / m_rightOffsets[2*StitchIndex]) * m_rightOffsets[2*StitchIndex+1] + (startRight % m_rightOffsets[2*StitchIndex]);

    for (int j = 0; j < m_stitchsize[StitchIndex]; ++j) {
      const Index left = firstLeft+j*m_leftOffsets[2*StitchIndex];
      const Index right = firstRight+j*m_rightOffsets[2*StitchIndex];
      if (StitchIndex < Indices::size-1) {
        partialStitch(left, right, StitchIndex+1, accum);
      } else {
        accum += m_leftImpl.coeff(left) * m_rightImpl.coeff(right);
      }
    }
  }

 private:
  array<Index, 2*Indices::size> m_leftOffsets;
  array<Index, 2*Indices::size> m_rightOffsets;
  array<Index, Indices::size> m_stitchsize;
  Index m_shiftright;
  Dimensions m_dimensions;
  TensorEvaluator<LeftArgType> m_leftImpl;
  TensorEvaluator<RightArgType> m_rightImpl;
};


} // end namespace Eigen

#endif // EIGEN_CXX11_TENSOR_TENSOR_CONTRACTION_H
