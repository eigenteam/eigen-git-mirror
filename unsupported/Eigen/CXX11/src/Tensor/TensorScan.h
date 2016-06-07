// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2016 Igor Babuschkin <igor@babuschk.in>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CXX11_TENSOR_TENSOR_SCAN_H
#define EIGEN_CXX11_TENSOR_TENSOR_SCAN_H
namespace Eigen {

namespace internal {
template <typename Op, typename XprType>
struct traits<TensorScanOp<Op, XprType> >
    : public traits<XprType> {
  typedef typename XprType::Scalar Scalar;
  typedef traits<XprType> XprTraits;
  typedef typename XprTraits::StorageKind StorageKind;
  typedef typename XprType::Nested Nested;
  typedef typename remove_reference<Nested>::type _Nested;
  static const int NumDimensions = XprTraits::NumDimensions;
  static const int Layout = XprTraits::Layout;
};

template<typename Op, typename XprType>
struct eval<TensorScanOp<Op, XprType>, Eigen::Dense>
{
  typedef const TensorScanOp<Op, XprType>& type;
};

template<typename Op, typename XprType>
struct nested<TensorScanOp<Op, XprType>, 1,
            typename eval<TensorScanOp<Op, XprType> >::type>
{
  typedef TensorScanOp<Op, XprType> type;
};
} // end namespace internal

/** \class TensorScan
  * \ingroup CXX11_Tensor_Module
  *
  * \brief Tensor scan class.
  *
  */

template <typename Op, typename XprType>
class TensorScanOp
    : public TensorBase<TensorScanOp<Op, XprType>, ReadOnlyAccessors> {
public:
  typedef typename Eigen::internal::traits<TensorScanOp>::Scalar Scalar;
  typedef typename Eigen::NumTraits<Scalar>::Real RealScalar;
  typedef typename XprType::CoeffReturnType CoeffReturnType;
  typedef typename Eigen::internal::nested<TensorScanOp>::type Nested;
  typedef typename Eigen::internal::traits<TensorScanOp>::StorageKind StorageKind;
  typedef typename Eigen::internal::traits<TensorScanOp>::Index Index;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorScanOp(
      const XprType& expr, const Index& axis, const Op& op = Op())
      : m_expr(expr), m_axis(axis), m_accumulator(op) {}

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  const Index axis() const { return m_axis; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  const XprType& expression() const { return m_expr; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  const Op accumulator() const { return m_accumulator; }

protected:
  typename XprType::Nested m_expr;
  const Index m_axis;
  const Op m_accumulator;
};

// Eval as rvalue
template <typename Op, typename ArgType, typename Device>
struct TensorEvaluator<const TensorScanOp<Op, ArgType>, Device> {

  typedef TensorScanOp<Op, ArgType> XprType;
  typedef typename XprType::Index Index;
  static const int NumDims = internal::array_size<typename TensorEvaluator<ArgType, Device>::Dimensions>::value;
  typedef DSizes<Index, NumDims> Dimensions;
  typedef typename internal::remove_const<typename XprType::Scalar>::type Scalar;
  typedef typename XprType::CoeffReturnType CoeffReturnType;
  typedef typename PacketType<CoeffReturnType, Device>::type PacketReturnType;

  enum {
    IsAligned = false,
    PacketAccess = (internal::packet_traits<Scalar>::size > 1),
    BlockAccess = false,
    Layout = TensorEvaluator<ArgType, Device>::Layout,
    CoordAccess = false,
    RawAccess = true
  };

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorEvaluator(const XprType& op,
                                                        const Device& device)
      : m_impl(op.expression(), device),
        m_device(device),
        m_axis(op.axis()),
        m_accumulator(op.accumulator()),
        m_dimensions(m_impl.dimensions()),
        m_size(m_dimensions[m_axis]),
        m_stride(1),
        m_output(NULL) {

    // Accumulating a scalar isn't supported.
    EIGEN_STATIC_ASSERT((NumDims > 0), YOU_MADE_A_PROGRAMMING_MISTAKE);
    eigen_assert(m_axis >= 0 && m_axis < NumDims);

    // Compute stride of scan axis
    if (static_cast<int>(Layout) == static_cast<int>(ColMajor)) {
      for (int i = 0; i < m_axis; ++i) {
        m_stride = m_stride * m_dimensions[i];
      }
    } else {
      for (int i = NumDims - 1; i > m_axis; --i) {
        m_stride = m_stride * m_dimensions[i];
      }
    }
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Dimensions& dimensions() const {
      return m_dimensions;
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool evalSubExprsIfNeeded(Scalar* data) {
    m_impl.evalSubExprsIfNeeded(NULL);
    if (data) {
      accumulateTo(data);
      return false;
    } else {
      m_output = static_cast<CoeffReturnType*>(m_device.allocate(dimensions().TotalSize() * sizeof(Scalar)));
      accumulateTo(m_output);
      return true;
    }
  }

  template<int LoadMode>
  EIGEN_DEVICE_FUNC PacketReturnType packet(Index index) const {
    return internal::ploadt<PacketReturnType, LoadMode>(m_output + index);
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE CoeffReturnType* data() const
  {
    return m_output;
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE CoeffReturnType coeff(Index index) const
  {
    return m_output[index];
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorOpCost costPerCoeff(bool) const {
    return TensorOpCost(sizeof(CoeffReturnType), 0, 0);
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void cleanup() {
    if (m_output != NULL) {
      m_device.deallocate(m_output);
      m_output = NULL;
    }
    m_impl.cleanup();
  }

protected:
  TensorEvaluator<ArgType, Device> m_impl;
  const Device& m_device;
  const Index m_axis;
  Op m_accumulator;
  const Dimensions& m_dimensions;
  const Index& m_size;
  Index m_stride;
  CoeffReturnType* m_output;

  // TODO(ibab) Parallelize this single-threaded implementation if desired
  EIGEN_DEVICE_FUNC void accumulateTo(Scalar* data) {
    // We fix the index along the scan axis to 0 and perform an
    // scan per remaining entry. The iteration is split into two nested
    // loops to avoid an integer division by keeping track of each idx1 and idx2.
    for (Index idx1 = 0; idx1 < dimensions().TotalSize() / m_size; idx1 += m_stride) {
       for (Index idx2 = 0; idx2 < m_stride; idx2++) {
          // Calculate the starting offset for the scan
          Index offset = idx1 * m_size + idx2;

          // Compute the prefix sum along the axis, starting at the calculated offset
          CoeffReturnType accum = m_accumulator.initialize();
          for (Index idx3 = 0; idx3 < m_size; idx3++) {
            Index curr = offset + idx3 * m_stride;
            m_accumulator.reduce(m_impl.coeff(curr), &accum);
            data[curr] = m_accumulator.finalize(accum);
          }
       }
    }
  }
};

}  // end namespace Eigen

#endif  // EIGEN_CXX11_TENSOR_TENSOR_SCAN_H
