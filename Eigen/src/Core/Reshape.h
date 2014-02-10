// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2006-2010 Benoit Jacob <jacob.benoit.1@gmail.com>
// Copyright (C) 2014 yoco <peter.xiau@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_RESHAPE_H
#define EIGEN_RESHAPE_H

namespace Eigen {

/** \class Reshape
  * \ingroup Core_Module
  *
  * \brief Expression of a fixed-size or dynamic-size reshape
  *
  * \param XprType the type of the expression in which we are taking a reshape
  * \param ReshapeRows the number of rows of the reshape we are taking at compile time (optional)
  * \param ReshapeCols the number of columns of the reshape we are taking at compile time (optional)
  *
  * This class represents an expression of either a fixed-size or dynamic-size reshape. It is the return
  * type of DenseBase::reshape(Index,Index) and DenseBase::reshape<int,int>() and
  * most of the time this is the only way it is used.
  *
  * However, if you want to directly maniputate reshape expressions,
  * for instance if you want to write a function returning such an expression, you
  * will need to use this class.
  *
  * Here is an example illustrating the dynamic case:
  * \include class_Reshape.cpp
  * Output: \verbinclude class_Reshape.out
  *
  * \note Even though this expression has dynamic size, in the case where \a XprType
  * has fixed size, this expression inherits a fixed maximal size which means that evaluating
  * it does not cause a dynamic memory allocation.
  *
  * Here is an example illustrating the fixed-size case:
  * \include class_FixedReshape.cpp
  * Output: \verbinclude class_FixedReshape.out
  *
  * \sa DenseBase::reshape(Index,Index), DenseBase::reshape(), class VectorReshape
  */

namespace internal {
template<typename XprType, int ReshapeRows, int ReshapeCols>
struct traits<Reshape<XprType, ReshapeRows, ReshapeCols> > : traits<XprType>
{
  typedef typename traits<XprType>::Scalar Scalar;
  typedef typename traits<XprType>::StorageKind StorageKind;
  typedef typename traits<XprType>::XprKind XprKind;
  enum{
    MatrixRows = traits<XprType>::RowsAtCompileTime,
    MatrixCols = traits<XprType>::ColsAtCompileTime,
    RowsAtCompileTime = ReshapeRows,
    ColsAtCompileTime = ReshapeCols,
    MaxRowsAtCompileTime = ReshapeRows,
    MaxColsAtCompileTime = ReshapeCols,
    XprTypeIsRowMajor = (int(traits<XprType>::Flags) & RowMajorBit) != 0,
    IsRowMajor = (RowsAtCompileTime == 1 && ColsAtCompileTime != 1) ? 1
               : (ColsAtCompileTime == 1 && RowsAtCompileTime != 1) ? 0
               : XprTypeIsRowMajor,
    HasSameStorageOrderAsXprType = (IsRowMajor == XprTypeIsRowMajor),
    InnerSize = IsRowMajor ? int(ColsAtCompileTime) : int(RowsAtCompileTime),
    InnerStrideAtCompileTime = HasSameStorageOrderAsXprType
                             ? int(inner_stride_at_compile_time<XprType>::ret)
                             : int(outer_stride_at_compile_time<XprType>::ret),
    OuterStrideAtCompileTime = HasSameStorageOrderAsXprType
                             ? int(outer_stride_at_compile_time<XprType>::ret)
                             : int(inner_stride_at_compile_time<XprType>::ret),
    MaskPacketAccessBit = (InnerSize == Dynamic || (InnerSize % packet_traits<Scalar>::size) == 0)
                       && (InnerStrideAtCompileTime == 1)
                        ? PacketAccessBit : 0,
    MaskAlignedBit = ((OuterStrideAtCompileTime!=Dynamic) && (((OuterStrideAtCompileTime * int(sizeof(Scalar))) % 16) == 0)) ? AlignedBit : 0,
    FlagsLinearAccessBit = (RowsAtCompileTime == 1 || ColsAtCompileTime == 1) ? LinearAccessBit : 0,
    FlagsLvalueBit = is_lvalue<XprType>::value ? LvalueBit : 0,
    FlagsRowMajorBit = IsRowMajor ? RowMajorBit : 0,
    Flags0 = traits<XprType>::Flags & ( (HereditaryBits & ~RowMajorBit) |
                                        MaskPacketAccessBit |
                                        MaskAlignedBit)
                                    & ~DirectAccessBit,

    Flags = (Flags0 | FlagsLinearAccessBit | FlagsLvalueBit | FlagsRowMajorBit)
  };
};

template<typename XprType, int ReshapeRows=Dynamic, int ReshapeCols=Dynamic,
         bool HasDirectAccess = internal::has_direct_access<XprType>::ret> class ReshapeImpl_dense;

} // end namespace internal

template<typename XprType, int ReshapeRows, int ReshapeCols, typename StorageKind> class ReshapeImpl;

template<typename XprType, int ReshapeRows, int ReshapeCols> class Reshape
  : public ReshapeImpl<XprType, ReshapeRows, ReshapeCols, typename internal::traits<XprType>::StorageKind>
{
    typedef ReshapeImpl<XprType, ReshapeRows, ReshapeCols, typename internal::traits<XprType>::StorageKind> Impl;
  public:
    //typedef typename Impl::Base Base;
    typedef Impl Base;
    EIGEN_GENERIC_PUBLIC_INTERFACE(Reshape)
    EIGEN_INHERIT_ASSIGNMENT_OPERATORS(Reshape)

    /** Fixed-size constructor
      */
    EIGEN_DEVICE_FUNC
    inline Reshape(XprType& xpr)
      : Impl(xpr)
    {
      EIGEN_STATIC_ASSERT(RowsAtCompileTime!=Dynamic && ColsAtCompileTime!=Dynamic,THIS_METHOD_IS_ONLY_FOR_FIXED_SIZE)
      eigen_assert(ReshapeRows * ReshapeCols == xpr.rows() * xpr.cols());
    }

    /** Dynamic-size constructor
      */
    EIGEN_DEVICE_FUNC
    inline Reshape(XprType& xpr,
          Index reshapeRows, Index reshapeCols)
      : Impl(xpr, reshapeRows, reshapeCols)
    {
      eigen_assert((RowsAtCompileTime==Dynamic || RowsAtCompileTime==reshapeRows)
          && (ColsAtCompileTime==Dynamic || ColsAtCompileTime==reshapeCols));
      eigen_assert(reshapeRows * reshapeCols == xpr.rows() * xpr.cols());
    }
};

// The generic default implementation for dense reshape simplu forward to the internal::ReshapeImpl_dense
// that must be specialized for direct and non-direct access...
template<typename XprType, int ReshapeRows, int ReshapeCols>
class ReshapeImpl<XprType, ReshapeRows, ReshapeCols, Dense>
  : public internal::ReshapeImpl_dense<XprType, ReshapeRows, ReshapeCols>
{
    typedef internal::ReshapeImpl_dense<XprType, ReshapeRows, ReshapeCols> Impl;
    typedef typename XprType::Index Index;
  public:
    typedef Impl Base;
    EIGEN_INHERIT_ASSIGNMENT_OPERATORS(ReshapeImpl)
    EIGEN_DEVICE_FUNC inline ReshapeImpl(XprType& xpr) : Impl(xpr) {}
    EIGEN_DEVICE_FUNC inline ReshapeImpl(XprType& xpr, Index reshapeRows, Index reshapeCols)
      : Impl(xpr, reshapeRows, reshapeCols) {}
};

namespace internal {

/** \internal Internal implementation of dense Reshapes in the general case. */
template<typename XprType, int ReshapeRows, int ReshapeCols, bool HasDirectAccess> class ReshapeImpl_dense
  : public internal::dense_xpr_base<Reshape<XprType, ReshapeRows, ReshapeCols> >::type
{
    typedef Reshape<XprType, ReshapeRows, ReshapeCols> ReshapeType;
  public:

    typedef typename internal::dense_xpr_base<ReshapeType>::type Base;
    EIGEN_DENSE_PUBLIC_INTERFACE(ReshapeType)
    EIGEN_INHERIT_ASSIGNMENT_OPERATORS(ReshapeImpl_dense)

    class InnerIterator;

    /** Fixed-size constructor
      */
    EIGEN_DEVICE_FUNC
    inline ReshapeImpl_dense(XprType& xpr)
      : m_xpr(xpr), m_reshapeRows(ReshapeRows), m_reshapeCols(ReshapeCols)
    {}

    /** Dynamic-size constructor
      */
    EIGEN_DEVICE_FUNC
    inline ReshapeImpl_dense(XprType& xpr,
          Index reshapeRows, Index reshapeCols)
      : m_xpr(xpr), m_reshapeRows(reshapeRows), m_reshapeCols(reshapeCols)
    {}

    EIGEN_DEVICE_FUNC inline Index rows() const { return m_reshapeRows.value(); }
    EIGEN_DEVICE_FUNC inline Index cols() const { return m_reshapeCols.value(); }

    typedef std::pair<Index, Index> RowCol;

    inline RowCol index_remap(Index rowId, Index colId) const {
      const Index nth_elem_idx = colId * m_reshapeRows.value() + rowId;
      const Index actual_col = nth_elem_idx / m_xpr.rows();
      const Index actual_row = nth_elem_idx % m_xpr.rows();
      return RowCol(actual_row, actual_col);
    }

    EIGEN_DEVICE_FUNC
    inline Scalar& coeffRef(Index rowId, Index colId)
    {
      EIGEN_STATIC_ASSERT_LVALUE(XprType)
      const RowCol row_col = index_remap(rowId, colId);
      return m_xpr.const_cast_derived().coeffRef(row_col.first, row_col.second);
    }

    EIGEN_DEVICE_FUNC
    inline const Scalar& coeffRef(Index rowId, Index colId) const
    {
      const RowCol row_col = index_remap(rowId, colId);
      return m_xpr.derived().coeffRef(row_col.first, row_col.second);
    }

    EIGEN_DEVICE_FUNC
    EIGEN_STRONG_INLINE const CoeffReturnType coeff(Index rowId, Index colId) const
    {
      const RowCol row_col = index_remap(rowId, colId);
      return m_xpr.coeff(row_col.first, row_col.second);
    }

    EIGEN_DEVICE_FUNC
    inline Scalar& coeffRef(Index index)
    {
      EIGEN_STATIC_ASSERT_LVALUE(XprType)
      const RowCol row_col = index_remap(RowsAtCompileTime == 1 ? 0 : index,
                                         RowsAtCompileTime == 1 ? index : 0);
      return m_xpr.const_cast_derived().coeffRef(row_col.first, row_col.second);

    }

    EIGEN_DEVICE_FUNC
    inline const Scalar& coeffRef(Index index) const
    {
      const RowCol row_col = index_remap(RowsAtCompileTime == 1 ? 0 : index,
                                         RowsAtCompileTime == 1 ? index : 0);
      return m_xpr.const_cast_derived().coeffRef(row_col.first, row_col.second);
    }

    EIGEN_DEVICE_FUNC
    inline const CoeffReturnType coeff(Index index) const
    {
      const RowCol row_col = index_remap(RowsAtCompileTime == 1 ? 0 : index,
                                         RowsAtCompileTime == 1 ? index : 0);
      return m_xpr.coeff(row_col.first, row_col.second);
    }

    EIGEN_DEVICE_FUNC
    template<int LoadMode>
    inline PacketScalar packet(Index rowId, Index colId) const
    {
      const RowCol row_col = index_remap(rowId, colId);
      return m_xpr.template packet<Unaligned>(row_col.first, row_col.second);

    }

    template<int LoadMode>
    inline void writePacket(Index rowId, Index colId, const PacketScalar& val)
    {
      const RowCol row_col = index_remap(rowId, colId);
      m_xpr.const_cast_derived().template writePacket<Unaligned>
              (row_col.first, row_col.second, val);
    }

    template<int LoadMode>
    inline PacketScalar packet(Index index) const
    {
      const RowCol row_col = index_remap(RowsAtCompileTime == 1 ? 0 : index,
                                         RowsAtCompileTime == 1 ? index : 0);
      return m_xpr.template packet<Unaligned>(row_col.first, row_col.second);
    }

    template<int LoadMode>
    inline void writePacket(Index index, const PacketScalar& val)
    {
      const RowCol row_col = index_remap(RowsAtCompileTime == 1 ? 0 : index,
                                         RowsAtCompileTime == 1 ? index : 0);
      return m_xpr.template packet<Unaligned>(row_col.first, row_col.second, val);
    }

    #ifdef EIGEN_PARSED_BY_DOXYGEN
    /** \sa MapBase::data() */
    EIGEN_DEVICE_FUNC inline const Scalar* data() const;
    EIGEN_DEVICE_FUNC inline Index innerStride() const;
    EIGEN_DEVICE_FUNC inline Index outerStride() const;
    #endif

    EIGEN_DEVICE_FUNC
    const typename internal::remove_all<typename XprType::Nested>::type& nestedExpression() const
    {
      return m_xpr;
    }

  protected:

    const typename XprType::Nested m_xpr;
    const internal::variable_if_dynamic<Index, RowsAtCompileTime> m_reshapeRows;
    const internal::variable_if_dynamic<Index, ColsAtCompileTime> m_reshapeCols;
};

///** \internal Internal implementation of dense Reshapes in the direct access case.*/
//template<typename XprType, int ReshapeRows, int ReshapeCols>
//class ReshapeImpl_dense<XprType,ReshapeRows,ReshapeCols, true>
//  : public MapBase<Reshape<XprType, ReshapeRows, ReshapeCols> >
//{
//    typedef Reshape<XprType, ReshapeRows, ReshapeCols> ReshapeType;
//  public:
//
//    typedef MapBase<ReshapeType> Base;
//    EIGEN_DENSE_PUBLIC_INTERFACE(ReshapeType)
//    EIGEN_INHERIT_ASSIGNMENT_OPERATORS(ReshapeImpl_dense)
//
//    /** Column or Row constructor
//      */
//    EIGEN_DEVICE_FUNC
//    inline ReshapeImpl_dense(XprType& xpr, Index i)
//      : Base(internal::const_cast_ptr(&xpr.coeffRef(
//              (ReshapeRows==1) && (ReshapeCols==XprType::ColsAtCompileTime) ? i : 0,
//              (ReshapeRows==XprType::RowsAtCompileTime) && (ReshapeCols==1) ? i : 0)),
//             ReshapeRows==1 ? 1 : xpr.rows(),
//             ReshapeCols==1 ? 1 : xpr.cols()),
//        m_xpr(xpr)
//    {
//      init();
//    }
//
//    /** Fixed-size constructor
//      */
//    EIGEN_DEVICE_FUNC
//    inline ReshapeImpl_dense(XprType& xpr)
//      : Base(internal::const_cast_ptr(&xpr.coeffRef(0, 0))), m_xpr(xpr)
//    {
//      init();
//    }
//
//    /** Dynamic-size constructor
//      */
//    EIGEN_DEVICE_FUNC
//    inline ReshapeImpl_dense(XprType& xpr,
//          Index reshapeRows, Index reshapeCols)
//      : Base(internal::const_cast_ptr(&xpr.coeffRef(0, 0)), reshapeRows, reshapeCols),
//        m_xpr(xpr)
//    {
//      init();
//    }
//
//    EIGEN_DEVICE_FUNC
//    const typename internal::remove_all<typename XprType::Nested>::type& nestedExpression() const
//    { 
//      return m_xpr; 
//    }
//      
//    EIGEN_DEVICE_FUNC
//    /** \sa MapBase::innerStride() */
//    inline Index innerStride() const
//    {
//      return internal::traits<ReshapeType>::HasSameStorageOrderAsXprType
//             ? m_xpr.innerStride()
//             : m_xpr.outerStride();
//    }
//
//    EIGEN_DEVICE_FUNC
//    /** \sa MapBase::outerStride() */
//    inline Index outerStride() const
//    {
//      return m_outerStride;
//    }
//
//  #ifndef __SUNPRO_CC
//  // FIXME sunstudio is not friendly with the above friend...
//  // META-FIXME there is no 'friend' keyword around here. Is this obsolete?
//  protected:
//  #endif
//
//    #ifndef EIGEN_PARSED_BY_DOXYGEN
//    /** \internal used by allowAligned() */
//    EIGEN_DEVICE_FUNC
//    inline ReshapeImpl_dense(XprType& xpr, const Scalar* data, Index reshapeRows, Index reshapeCols)
//      : Base(data, reshapeRows, reshapeCols), m_xpr(xpr)
//    {
//      init();
//    }
//    #endif
//
//  protected:
//    EIGEN_DEVICE_FUNC
//    void init()
//    {
//      m_outerStride = internal::traits<ReshapeType>::HasSameStorageOrderAsXprType
//                    ? m_xpr.outerStride()
//                    : m_xpr.innerStride();
//    }
//
//    typename XprType::Nested m_xpr;
//    Index m_outerStride;
//};

} // end namespace internal

} // end namespace Eigen

#endif // EIGEN_RESHAPE_H
