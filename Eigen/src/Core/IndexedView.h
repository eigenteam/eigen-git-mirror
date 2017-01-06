// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2017 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_INDEXED_VIEW_H
#define EIGEN_INDEXED_VIEW_H

namespace Eigen {

namespace internal {

template<typename XprType, typename RowIndices, typename ColIndices>
struct traits<IndexedView<XprType, RowIndices, ColIndices> >
 : traits<XprType>
{
  enum {
    RowsAtCompileTime = get_compile_time_size<RowIndices>::value,
    ColsAtCompileTime = get_compile_time_size<ColIndices>::value,
    MaxRowsAtCompileTime = RowsAtCompileTime != Dynamic ? int(RowsAtCompileTime) : int(traits<XprType>::MaxRowsAtCompileTime),
    MaxColsAtCompileTime = ColsAtCompileTime != Dynamic ? int(ColsAtCompileTime) : int(traits<XprType>::MaxColsAtCompileTime),

    XprTypeIsRowMajor = (int(traits<XprType>::Flags)&RowMajorBit) != 0,
    IsRowMajor = (MaxRowsAtCompileTime==1&&MaxColsAtCompileTime!=1) ? 1
               : (MaxColsAtCompileTime==1&&MaxRowsAtCompileTime!=1) ? 0
               : XprTypeIsRowMajor,

    FlagsRowMajorBit = IsRowMajor ? RowMajorBit : 0,
    FlagsLvalueBit = is_lvalue<XprType>::value ? LvalueBit : 0,
    Flags = (traits<XprType>::Flags & HereditaryBits) | FlagsLvalueBit | FlagsRowMajorBit,
    //MatrixTypeInnerStride =  inner_stride_at_compile_time<XprType>::ret,
    InnerStrideAtCompileTime = int(Dynamic),
    OuterStrideAtCompileTime = int(Dynamic)
  };
};

}

template<typename XprType, typename RowIndices, typename ColIndices, typename StorageKind>
class IndexedViewImpl;

// Expression of a generic slice
template<typename XprType, typename RowIndices, typename ColIndices>
class IndexedView : public IndexedViewImpl<XprType, RowIndices, ColIndices, typename internal::traits<XprType>::StorageKind>
{
public:
  typedef typename IndexedViewImpl<XprType, RowIndices, ColIndices, typename internal::traits<XprType>::StorageKind>::Base Base;
  EIGEN_GENERIC_PUBLIC_INTERFACE(IndexedView)

  typedef typename internal::ref_selector<XprType>::non_const_type MatrixTypeNested;
  typedef typename internal::remove_all<XprType>::type NestedExpression;

  template<typename T0, typename T1>
  IndexedView(XprType& xpr, const T0& rowIndices, const T1& colIndices)
    : m_xpr(xpr), m_rowIndices(rowIndices), m_colIndices(colIndices)
  {}
  Index rows() const { return m_rowIndices.size(); }
  Index cols() const { return m_colIndices.size(); }

  /** \returns the nested expression */
  const typename internal::remove_all<XprType>::type&
  nestedExpression() const { return m_xpr; }

  /** \returns the nested expression */
  typename internal::remove_reference<XprType>::type&
  nestedExpression() { return m_xpr.const_cast_derived(); }

  const RowIndices& rowIndices() const { return m_rowIndices; }
  const ColIndices& colIndices() const { return m_colIndices; }

//   std::pair<Index,Index> index(Index i, Index j) const {
//     return std::pair<Index,Index>(m_rowIndices[i], m_colIndices[j]);
//   }
//
//   void print() const {
//     for(Index i=0; i<rows(); ++i)
//     {
//       for(Index j=0; j<cols(); ++j)
//       {
//         std::pair<Index,Index> k = index(i,j);
//         std::cout << '(' << k.first << ',' << k.second  << ") ";
//       }
//       std::cout << '\n';
//     }
//   }
protected:
  MatrixTypeNested m_xpr;
  RowIndices m_rowIndices;
  ColIndices m_colIndices;
};


// Generic API dispatcher
template<typename XprType, typename RowIndices, typename ColIndices, typename StorageKind>
class IndexedViewImpl
  : public internal::generic_xpr_base<IndexedView<XprType, RowIndices, ColIndices> >::type
{
public:
  typedef typename internal::generic_xpr_base<IndexedView<XprType, RowIndices, ColIndices> >::type Base;
};

namespace internal {


template<typename ArgType, typename RowIndices, typename ColIndices>
struct unary_evaluator<IndexedView<ArgType, RowIndices, ColIndices>, IndexBased>
  : evaluator_base<IndexedView<ArgType, RowIndices, ColIndices> >
{
  typedef IndexedView<ArgType, RowIndices, ColIndices> XprType;

  enum {
    CoeffReadCost = evaluator<ArgType>::CoeffReadCost /* + cost of row/col index */,

    Flags = (evaluator<ArgType>::Flags & (HereditaryBits /*| LinearAccessBit | DirectAccessBit*/)),

    Alignment = 0
  };

  EIGEN_DEVICE_FUNC explicit unary_evaluator(const XprType& xpr) : m_argImpl(xpr.nestedExpression()), m_xpr(xpr)
  {
    EIGEN_INTERNAL_CHECK_COST_VALUE(CoeffReadCost);
  }

  typedef typename XprType::Scalar Scalar;
  typedef typename XprType::CoeffReturnType CoeffReturnType;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  CoeffReturnType coeff(Index row, Index col) const
  {
    return m_argImpl.coeff(m_xpr.rowIndices()[row], m_xpr.colIndices()[col]);
  }

protected:

  evaluator<ArgType> m_argImpl;
  const XprType& m_xpr;

};

} // end namespace internal

} // end namespace Eigen

#endif // EIGEN_INDEXED_VIEW_H
