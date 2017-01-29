// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2017 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2014 yoco <peter.xiau@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_RESHAPED_H
#define EIGEN_RESHAPED_H

namespace Eigen {

/** \class Reshapedd
  * \ingroup Core_Module
  *
  * \brief Expression of a fixed-size or dynamic-size reshape
  *
  * \tparam XprType the type of the expression in which we are taking a reshape
  * \tparam Rows the number of rows of the reshape we are taking at compile time (optional)
  * \tparam Cols the number of columns of the reshape we are taking at compile time (optional)
  * \tparam Order
  *
  * This class represents an expression of either a fixed-size or dynamic-size reshape. It is the return
  * type of DenseBase::reshaped(Index,Index) and DenseBase::reshape<int,int>() and
  * most of the time this is the only way it is used.
  *
  * However, if you want to directly maniputate reshape expressions,
  * for instance if you want to write a function returning such an expression, you
  * will need to use this class.
  *
  * Here is an example illustrating the dynamic case:
  * \include class_Reshaped.cpp
  * Output: \verbinclude class_Reshaped.out
  *
  * \note Even though this expression has dynamic size, in the case where \a XprType
  * has fixed size, this expression inherits a fixed maximal size which means that evaluating
  * it does not cause a dynamic memory allocation.
  *
  * Here is an example illustrating the fixed-size case:
  * \include class_FixedReshaped.cpp
  * Output: \verbinclude class_FixedReshaped.out
  *
  * \sa DenseBase::reshaped(Index,Index), DenseBase::reshaped(), class VectorReshaped
  */

namespace internal {
template<typename XprType, int Rows, int Cols, int Order>
struct traits<Reshaped<XprType, Rows, Cols, Order> > : traits<XprType>
{
  typedef typename traits<XprType>::Scalar Scalar;
  typedef typename traits<XprType>::StorageKind StorageKind;
  typedef typename traits<XprType>::XprKind XprKind;
  enum{
    MatrixRows = traits<XprType>::RowsAtCompileTime,
    MatrixCols = traits<XprType>::ColsAtCompileTime,
    RowsAtCompileTime = Rows,
    ColsAtCompileTime = Cols,
    MaxRowsAtCompileTime = Rows,
    MaxColsAtCompileTime = Cols,
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
    //MaskAlignedBit = ((OuterStrideAtCompileTime!=Dynamic) && (((OuterStrideAtCompileTime * int(sizeof(Scalar))) % 16) == 0)) ? AlignedBit : 0,
    FlagsLinearAccessBit = (RowsAtCompileTime == 1 || ColsAtCompileTime == 1) ? LinearAccessBit : 0,
    FlagsLvalueBit = is_lvalue<XprType>::value ? LvalueBit : 0,
    FlagsRowMajorBit = IsRowMajor ? RowMajorBit : 0,
    Flags0 = traits<XprType>::Flags & ( (HereditaryBits & ~RowMajorBit) | MaskPacketAccessBit)
                                    & ~DirectAccessBit,

    Flags = (Flags0 | FlagsLinearAccessBit | FlagsLvalueBit | FlagsRowMajorBit)
  };
};

template<typename XprType, int Rows=Dynamic, int Cols=Dynamic, int Order = 0,
         bool HasDirectAccess = internal::has_direct_access<XprType>::ret> class ReshapedImpl_dense;

} // end namespace internal

template<typename XprType, int Rows, int Cols, int Order, typename StorageKind> class ReshapedImpl;

template<typename XprType, int Rows, int Cols, int Order> class Reshaped
  : public ReshapedImpl<XprType, Rows, Cols, Order, typename internal::traits<XprType>::StorageKind>
{
    typedef ReshapedImpl<XprType, Rows, Cols, Order, typename internal::traits<XprType>::StorageKind> Impl;
  public:
    //typedef typename Impl::Base Base;
    typedef Impl Base;
    EIGEN_GENERIC_PUBLIC_INTERFACE(Reshaped)
    EIGEN_INHERIT_ASSIGNMENT_OPERATORS(Reshaped)

    /** Fixed-size constructor
      */
    EIGEN_DEVICE_FUNC
    inline Reshaped(XprType& xpr)
      : Impl(xpr)
    {
      EIGEN_STATIC_ASSERT(RowsAtCompileTime!=Dynamic && ColsAtCompileTime!=Dynamic,THIS_METHOD_IS_ONLY_FOR_FIXED_SIZE)
      eigen_assert(Rows * Cols == xpr.rows() * xpr.cols());
    }

    /** Dynamic-size constructor
      */
    EIGEN_DEVICE_FUNC
    inline Reshaped(XprType& xpr,
          Index reshapeRows, Index reshapeCols)
      : Impl(xpr, reshapeRows, reshapeCols)
    {
      eigen_assert((RowsAtCompileTime==Dynamic || RowsAtCompileTime==reshapeRows)
          && (ColsAtCompileTime==Dynamic || ColsAtCompileTime==reshapeCols));
      eigen_assert(reshapeRows * reshapeCols == xpr.rows() * xpr.cols());
    }
};

// The generic default implementation for dense reshape simplu forward to the internal::ReshapedImpl_dense
// that must be specialized for direct and non-direct access...
template<typename XprType, int Rows, int Cols, int Order>
class ReshapedImpl<XprType, Rows, Cols, Order, Dense>
  : public internal::ReshapedImpl_dense<XprType, Rows, Cols, Order>
{
    typedef internal::ReshapedImpl_dense<XprType, Rows, Cols, Order> Impl;
  public:
    typedef Impl Base;
    EIGEN_INHERIT_ASSIGNMENT_OPERATORS(ReshapedImpl)
    EIGEN_DEVICE_FUNC inline ReshapedImpl(XprType& xpr) : Impl(xpr) {}
    EIGEN_DEVICE_FUNC inline ReshapedImpl(XprType& xpr, Index reshapeRows, Index reshapeCols)
      : Impl(xpr, reshapeRows, reshapeCols) {}
};

namespace internal {

/** \internal Internal implementation of dense Reshapeds in the general case. */
template<typename XprType, int Rows, int Cols, int Order, bool HasDirectAccess> class ReshapedImpl_dense
  : public internal::dense_xpr_base<Reshaped<XprType, Rows, Cols, Order> >::type
{
    typedef Reshaped<XprType, Rows, Cols, Order> ReshapedType;
  public:

    typedef typename internal::dense_xpr_base<ReshapedType>::type Base;
    EIGEN_DENSE_PUBLIC_INTERFACE(ReshapedType)
    EIGEN_INHERIT_ASSIGNMENT_OPERATORS(ReshapedImpl_dense)

    typedef typename internal::ref_selector<XprType>::non_const_type MatrixTypeNested;
    typedef typename internal::remove_all<XprType>::type NestedExpression;

    class InnerIterator;

    /** Fixed-size constructor
      */
    EIGEN_DEVICE_FUNC
    inline ReshapedImpl_dense(XprType& xpr)
      : m_xpr(xpr), m_rows(Rows), m_cols(Cols)
    {}

    /** Dynamic-size constructor
      */
    EIGEN_DEVICE_FUNC
    inline ReshapedImpl_dense(XprType& xpr,
          Index nRows, Index nCols)
      : m_xpr(xpr), m_rows(nRows), m_cols(nCols)
    {}

    EIGEN_DEVICE_FUNC Index rows() const { return m_rows; }
    EIGEN_DEVICE_FUNC Index cols() const { return m_cols; }

    #ifdef EIGEN_PARSED_BY_DOXYGEN
    /** \sa MapBase::data() */
    EIGEN_DEVICE_FUNC inline const Scalar* data() const;
    EIGEN_DEVICE_FUNC inline Index innerStride() const;
    EIGEN_DEVICE_FUNC inline Index outerStride() const;
    #endif

    /** \returns the nested expression */
    EIGEN_DEVICE_FUNC
    const typename internal::remove_all<XprType>::type&
    nestedExpression() const { return m_xpr; }

    /** \returns the nested expression */
    EIGEN_DEVICE_FUNC
    typename internal::remove_reference<XprType>::type&
    nestedExpression() { return m_xpr.const_cast_derived(); }

  protected:

    MatrixTypeNested m_xpr;
    const internal::variable_if_dynamic<Index, RowsAtCompileTime> m_rows;
    const internal::variable_if_dynamic<Index, ColsAtCompileTime> m_cols;
};


template<typename ArgType, int Rows, int Cols, int Order>
struct unary_evaluator<Reshaped<ArgType, Rows, Cols, Order>, IndexBased>
  : evaluator_base<Reshaped<ArgType, Rows, Cols, Order> >
{
  typedef Reshaped<ArgType, Rows, Cols, Order> XprType;

  enum {
    CoeffReadCost = evaluator<ArgType>::CoeffReadCost /* TODO + cost of index computations */,

    Flags = (evaluator<ArgType>::Flags & (HereditaryBits /*| LinearAccessBit | DirectAccessBit*/)),

    Alignment = 0
  };

  EIGEN_DEVICE_FUNC explicit unary_evaluator(const XprType& xpr) : m_argImpl(xpr.nestedExpression()), m_xpr(xpr)
  {
    EIGEN_INTERNAL_CHECK_COST_VALUE(CoeffReadCost);
  }

  typedef typename XprType::Scalar Scalar;
  typedef typename XprType::CoeffReturnType CoeffReturnType;

  typedef std::pair<Index, Index> RowCol;

  inline RowCol index_remap(Index rowId, Index colId) const {
    const Index nth_elem_idx = colId * m_xpr.rows() + rowId;
    const Index actual_col = nth_elem_idx / m_xpr.nestedExpression().rows();
    const Index actual_row = nth_elem_idx % m_xpr.nestedExpression().rows();
    return RowCol(actual_row, actual_col);
  }

  EIGEN_DEVICE_FUNC
  inline Scalar& coeffRef(Index rowId, Index colId)
  {
    EIGEN_STATIC_ASSERT_LVALUE(XprType)
    const RowCol row_col = index_remap(rowId, colId);
    return m_argImpl.coeffRef(row_col.first, row_col.second);
  }

  EIGEN_DEVICE_FUNC
  inline const Scalar& coeffRef(Index rowId, Index colId) const
  {
    const RowCol row_col = index_remap(rowId, colId);
    return m_argImpl.coeffRef(row_col.first, row_col.second);
  }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE const CoeffReturnType coeff(Index rowId, Index colId) const
  {
    const RowCol row_col = index_remap(rowId, colId);
    return m_argImpl.coeff(row_col.first, row_col.second);
  }

  EIGEN_DEVICE_FUNC
  inline Scalar& coeffRef(Index index)
  {
    EIGEN_STATIC_ASSERT_LVALUE(XprType)
    const RowCol row_col = index_remap(Rows == 1 ? 0 : index,
                                       Rows == 1 ? index : 0);
    return m_argImpl.coeffRef(row_col.first, row_col.second);

  }

  EIGEN_DEVICE_FUNC
  inline const Scalar& coeffRef(Index index) const
  {
    const RowCol row_col = index_remap(Rows == 1 ? 0 : index,
                                       Rows == 1 ? index : 0);
    return m_argImpl.coeffRef(row_col.first, row_col.second);
  }

  EIGEN_DEVICE_FUNC
  inline const CoeffReturnType coeff(Index index) const
  {
    const RowCol row_col = index_remap(Rows == 1 ? 0 : index,
                                       Rows == 1 ? index : 0);
    return m_argImpl.coeff(row_col.first, row_col.second);
  }
#if 0
  EIGEN_DEVICE_FUNC
  template<int LoadMode>
  inline PacketScalar packet(Index rowId, Index colId) const
  {
    const RowCol row_col = index_remap(rowId, colId);
    return m_argImpl.template packet<Unaligned>(row_col.first, row_col.second);

  }

  template<int LoadMode>
  EIGEN_DEVICE_FUNC
  inline void writePacket(Index rowId, Index colId, const PacketScalar& val)
  {
    const RowCol row_col = index_remap(rowId, colId);
    m_argImpl.const_cast_derived().template writePacket<Unaligned>
            (row_col.first, row_col.second, val);
  }

  template<int LoadMode>
  EIGEN_DEVICE_FUNC
  inline PacketScalar packet(Index index) const
  {
    const RowCol row_col = index_remap(RowsAtCompileTime == 1 ? 0 : index,
                                        RowsAtCompileTime == 1 ? index : 0);
    return m_argImpl.template packet<Unaligned>(row_col.first, row_col.second);
  }

  template<int LoadMode>
  EIGEN_DEVICE_FUNC
  inline void writePacket(Index index, const PacketScalar& val)
  {
    const RowCol row_col = index_remap(RowsAtCompileTime == 1 ? 0 : index,
                                        RowsAtCompileTime == 1 ? index : 0);
    return m_argImpl.template packet<Unaligned>(row_col.first, row_col.second, val);
  }
#endif
protected:

  evaluator<ArgType> m_argImpl;
  const XprType& m_xpr;

};


///** \internal Internal implementation of dense Reshapeds in the direct access case.*/
//template<typename XprType, int Rows, int Cols, int Order>
//class ReshapedImpl_dense<XprType,ReshapedRows,ReshapedCols, true>
//  : public MapBase<Reshaped<XprType, Rows, Cols, Order> >
//{
//    typedef Reshaped<XprType, Rows, Cols, Order> ReshapedType;
//  public:
//
//    typedef MapBase<ReshapedType> Base;
//    EIGEN_DENSE_PUBLIC_INTERFACE(ReshapedType)
//    EIGEN_INHERIT_ASSIGNMENT_OPERATORS(ReshapedImpl_dense)
//
//    /** Column or Row constructor
//      */
//    EIGEN_DEVICE_FUNC
//    inline ReshapedImpl_dense(XprType& xpr, Index i)
//      : Base(internal::const_cast_ptr(&xpr.coeffRef(
//              (ReshapedRows==1) && (ReshapedCols==XprType::ColsAtCompileTime) ? i : 0,
//              (ReshapedRows==XprType::RowsAtCompileTime) && (ReshapedCols==1) ? i : 0)),
//             ReshapedRows==1 ? 1 : xpr.rows(),
//             ReshapedCols==1 ? 1 : xpr.cols()),
//        m_xpr(xpr)
//    {
//      init();
//    }
//
//    /** Fixed-size constructor
//      */
//    EIGEN_DEVICE_FUNC
//    inline ReshapedImpl_dense(XprType& xpr)
//      : Base(internal::const_cast_ptr(&xpr.coeffRef(0, 0))), m_xpr(xpr)
//    {
//      init();
//    }
//
//    /** Dynamic-size constructor
//      */
//    EIGEN_DEVICE_FUNC
//    inline ReshapedImpl_dense(XprType& xpr,
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
//      return internal::traits<ReshapedType>::HasSameStorageOrderAsXprType
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
//    inline ReshapedImpl_dense(XprType& xpr, const Scalar* data, Index reshapeRows, Index reshapeCols)
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
//      m_outerStride = internal::traits<ReshapedType>::HasSameStorageOrderAsXprType
//                    ? m_xpr.outerStride()
//                    : m_xpr.innerStride();
//    }
//
//    typename XprType::Nested m_xpr;
//    Index m_outerStride;
//};

} // end namespace internal

/** \returns a dynamic-size expression of a reshape in *this.
  *
  * \param reshapeRows the number of rows in the reshape
  * \param reshapeCols the number of columns in the reshape
  *
  * Example: \include MatrixBase_reshape_int_int.cpp
  * Output: \verbinclude MatrixBase_reshape_int_int.out
  *
  * \note Even though the returned expression has dynamic size, in the case
  * when it is applied to a fixed-size matrix, it inherits a fixed maximal size,
  * which means that evaluating it does not cause a dynamic memory allocation.
  *
  * \sa class Reshape, reshaped()
  */
template<typename Derived>
EIGEN_DEVICE_FUNC
inline Reshaped<Derived> DenseBase<Derived>::reshaped(Index reshapeRows, Index reshapeCols)
{
  return Reshaped<Derived>(derived(), reshapeRows, reshapeCols);
}

/** This is the const version of reshaped(Index,Index). */
template<typename Derived>
EIGEN_DEVICE_FUNC
inline const Reshaped<const Derived> DenseBase<Derived>::reshaped(Index reshapeRows, Index reshapeCols) const
{
  return Reshaped<const Derived>(derived(), reshapeRows, reshapeCols);
}

/** \returns a fixed-size expression of a reshape in *this.
  *
  * The template parameters \a ReshapeRows and \a ReshapeCols are the number of
  * rows and columns in the reshape.
  *
  * Example: \include MatrixBase_reshape.cpp
  * Output: \verbinclude MatrixBase_reshape.out
  *
  * \note since reshape is a templated member, the keyword template has to be used
  * if the matrix type is also a template parameter: \code m.template reshape<3,3>(); \endcode
  *
  * \sa class Reshape, reshaped(Index,Index)
  */
template<typename Derived>
template<int ReshapeRows, int ReshapeCols>
EIGEN_DEVICE_FUNC
inline Reshaped<Derived, ReshapeRows, ReshapeCols> DenseBase<Derived>::reshaped()
{
  return Reshaped<Derived, ReshapeRows, ReshapeCols>(derived());
}

/** This is the const version of reshape<>(Index, Index). */
template<typename Derived>
template<int ReshapeRows, int ReshapeCols>
EIGEN_DEVICE_FUNC
inline const Reshaped<const Derived, ReshapeRows, ReshapeCols> DenseBase<Derived>::reshaped() const
{
  return Reshaped<const Derived, ReshapeRows, ReshapeCols>(derived());
}

} // end namespace Eigen

#endif // EIGEN_RESHAPED_H
