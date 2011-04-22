// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2011 Benoit Jacob <jacob.benoit.1@gmail.com>
// Copyright (C) 2011 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2011 Jitse Niesen <jitse@maths.leeds.ac.uk>
//
// Eigen is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 3 of the License, or (at your option) any later version.
//
// Alternatively, you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of
// the License, or (at your option) any later version.
//
// Eigen is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License or the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License and a copy of the GNU General Public License along with
// Eigen. If not, see <http://www.gnu.org/licenses/>.


#ifndef EIGEN_COREEVALUATORS_H
#define EIGEN_COREEVALUATORS_H

namespace internal {
  
template<typename T>
struct evaluator_impl {};

template<typename T>
struct evaluator
{
  typedef evaluator_impl<T> type;
};

// TODO: Think about const-correctness

template<typename T>
struct evaluator<const T>
{
  typedef evaluator_impl<T> type;
};

// -------------------- Transpose --------------------

template<typename ExpressionType>
struct evaluator_impl<Transpose<ExpressionType> >
{
  typedef Transpose<ExpressionType> TransposeType;
  evaluator_impl(const TransposeType& t) : m_argImpl(t.nestedExpression()) {}

  typedef typename TransposeType::Index Index;

  typename TransposeType::CoeffReturnType coeff(Index i, Index j) const
  {
    return m_argImpl.coeff(j, i);
  }

  typename TransposeType::CoeffReturnType coeff(Index index) const
  {
    return m_argImpl.coeff(index);
  }

  typename TransposeType::Scalar& coeffRef(Index i, Index j)
  {
    return m_argImpl.coeffRef(j, i);
  }

  typename TransposeType::Scalar& coeffRef(Index index)
  {
    return m_argImpl.coeffRef(index);
  }

  // TODO: Difference between PacketScalar and PacketReturnType?
  template<int LoadMode>
  const typename ExpressionType::PacketScalar packet(Index row, Index col) const
  {
    return m_argImpl.template packet<LoadMode>(col, row);
  }

  template<int LoadMode>
  const typename ExpressionType::PacketScalar packet(Index index) const
  {
    return m_argImpl.template packet<LoadMode>(index);
  }

  template<int StoreMode> 
  void writePacket(Index row, Index col, const typename ExpressionType::PacketScalar& x)
  {
    m_argImpl.template writePacket<StoreMode>(col, row, x);
  }

  template<int StoreMode> 
  void writePacket(Index index, const typename ExpressionType::PacketScalar& x)
  {
    m_argImpl.template writePacket<StoreMode>(index, x);
  }

protected:
  typename evaluator<ExpressionType>::type m_argImpl;
};

// -------------------- Matrix and Array --------------------
//
// evaluator_impl<PlainObjectBase> is a common base class for the
// Matrix and Array evaluators.

template<typename Derived>
struct evaluator_impl<PlainObjectBase<Derived> >
{
  typedef PlainObjectBase<Derived> PlainObjectType;

  evaluator_impl(const PlainObjectType& m) : m_plainObject(m) {}

  typedef typename PlainObjectType::Index Index;
  typedef typename PlainObjectType::Scalar Scalar;
  typedef typename PlainObjectType::CoeffReturnType CoeffReturnType;
  typedef typename PlainObjectType::PacketScalar PacketScalar;
  typedef typename PlainObjectType::PacketReturnType PacketReturnType;

  CoeffReturnType coeff(Index i, Index j) const
  {
    return m_plainObject.coeff(i, j);
  }

  CoeffReturnType coeff(Index index) const
  {
    return m_plainObject.coeff(index);
  }

  Scalar& coeffRef(Index i, Index j)
  {
    return m_plainObject.const_cast_derived().coeffRef(i, j);
  }

  Scalar& coeffRef(Index index)
  {
    return m_plainObject.const_cast_derived().coeffRef(index);
  }

  template<int LoadMode> 
  PacketReturnType packet(Index row, Index col) const
  {
    return m_plainObject.template packet<LoadMode>(row, col);
  }

  template<int LoadMode> 
  PacketReturnType packet(Index index) const
  {
    return m_plainObject.template packet<LoadMode>(index);
  }

  template<int StoreMode> 
  void writePacket(Index row, Index col, const PacketScalar& x)
  {
    m_plainObject.const_cast_derived().template writePacket<StoreMode>(row, col, x);
  }

  template<int StoreMode> 
  void writePacket(Index index, const PacketScalar& x)
  {
    m_plainObject.const_cast_derived().template writePacket<StoreMode>(index, x);
  }

protected:
  const PlainObjectType &m_plainObject;
};

template<typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
struct evaluator_impl<Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols> >
  : evaluator_impl<PlainObjectBase<Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols> > >
{
  typedef Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols> MatrixType;

  evaluator_impl(const MatrixType& m) 
    : evaluator_impl<PlainObjectBase<MatrixType> >(m) 
  { }
};

template<typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
struct evaluator_impl<Array<Scalar, Rows, Cols, Options, MaxRows, MaxCols> >
  : evaluator_impl<PlainObjectBase<Array<Scalar, Rows, Cols, Options, MaxRows, MaxCols> > >
{
  typedef Array<Scalar, Rows, Cols, Options, MaxRows, MaxCols> ArrayType;

  evaluator_impl(const ArrayType& m) 
    : evaluator_impl<PlainObjectBase<ArrayType> >(m) 
  { }
};

// -------------------- CwiseNullaryOp --------------------

template<typename NullaryOp, typename PlainObjectType>
struct evaluator_impl<CwiseNullaryOp<NullaryOp,PlainObjectType> >
{
  typedef CwiseNullaryOp<NullaryOp,PlainObjectType> NullaryOpType;

  evaluator_impl(const NullaryOpType& n) 
    : m_functor(n.functor()) 
  { }

  typedef typename NullaryOpType::Index Index;
  typedef typename NullaryOpType::CoeffReturnType CoeffReturnType;
  typedef typename NullaryOpType::PacketScalar PacketScalar;

  CoeffReturnType coeff(Index row, Index col) const
  {
    return m_functor(row, col);
  }

  CoeffReturnType coeff(Index index) const
  {
    return m_functor(index);
  }

  template<int LoadMode>
  PacketScalar packet(Index row, Index col) const
  {
    return m_functor.packetOp(row, col);
  }

  template<int LoadMode>
  PacketScalar packet(Index index) const
  {
    return m_functor.packetOp(index);
  }

protected:
  const NullaryOp m_functor;
};

// -------------------- CwiseUnaryOp --------------------

template<typename UnaryOp, typename ArgType>
struct evaluator_impl<CwiseUnaryOp<UnaryOp, ArgType> >
{
  typedef CwiseUnaryOp<UnaryOp, ArgType> UnaryOpType;

  evaluator_impl(const UnaryOpType& op) 
    : m_functor(op.functor()), 
      m_argImpl(op.nestedExpression()) 
  { }

  typedef typename UnaryOpType::Index Index;
  typedef typename UnaryOpType::CoeffReturnType CoeffReturnType;
  typedef typename UnaryOpType::PacketScalar PacketScalar;

  CoeffReturnType coeff(Index row, Index col) const
  {
    return m_functor(m_argImpl.coeff(row, col));
  }

  CoeffReturnType coeff(Index index) const
  {
    return m_functor(m_argImpl.coeff(index));
  }

  template<int LoadMode>
  PacketScalar packet(Index row, Index col) const
  {
    return m_functor.packetOp(m_argImpl.template packet<LoadMode>(row, col));
  }

  template<int LoadMode>
  PacketScalar packet(Index index) const
  {
    return m_functor.packetOp(m_argImpl.template packet<LoadMode>(index));
  }

protected:
  const UnaryOp m_functor;
  typename evaluator<ArgType>::type m_argImpl;
};

// -------------------- CwiseBinaryOp --------------------

template<typename BinaryOp, typename Lhs, typename Rhs>
struct evaluator_impl<CwiseBinaryOp<BinaryOp, Lhs, Rhs> >
{
  typedef CwiseBinaryOp<BinaryOp, Lhs, Rhs> BinaryOpType;

  evaluator_impl(const BinaryOpType& xpr) 
    : m_functor(xpr.functor()),
      m_lhsImpl(xpr.lhs()), 
      m_rhsImpl(xpr.rhs())  
  { }

  typedef typename BinaryOpType::Index Index;
  typedef typename BinaryOpType::CoeffReturnType CoeffReturnType;
  typedef typename BinaryOpType::PacketScalar PacketScalar;

  CoeffReturnType coeff(Index row, Index col) const
  {
    return m_functor(m_lhsImpl.coeff(row, col), m_rhsImpl.coeff(row, col));
  }

  CoeffReturnType coeff(Index index) const
  {
    return m_functor(m_lhsImpl.coeff(index), m_rhsImpl.coeff(index));
  }

  template<int LoadMode>
  PacketScalar packet(Index row, Index col) const
  {
    return m_functor.packetOp(m_lhsImpl.template packet<LoadMode>(row, col),
			      m_rhsImpl.template packet<LoadMode>(row, col));
  }

  template<int LoadMode>
  PacketScalar packet(Index index) const
  {
    return m_functor.packetOp(m_lhsImpl.template packet<LoadMode>(index),
			      m_rhsImpl.template packet<LoadMode>(index));
  }

protected:
  const BinaryOp m_functor;
  typename evaluator<Lhs>::type m_lhsImpl;
  typename evaluator<Rhs>::type m_rhsImpl;
};

// -------------------- CwiseUnaryView --------------------

template<typename UnaryOp, typename ArgType>
struct evaluator_impl<CwiseUnaryView<UnaryOp, ArgType> >
{
  typedef CwiseUnaryView<UnaryOp, ArgType> CwiseUnaryViewType;

  evaluator_impl(const CwiseUnaryViewType& op) 
    : m_unaryOp(op.functor()), 
      m_argImpl(op.nestedExpression()) 
  { }

  typedef typename CwiseUnaryViewType::Index Index;
  typedef typename CwiseUnaryViewType::Scalar Scalar;
  typedef typename CwiseUnaryViewType::CoeffReturnType CoeffReturnType;

  CoeffReturnType coeff(Index row, Index col) const
  {
    return m_unaryOp(m_argImpl.coeff(row, col));
  }

  CoeffReturnType coeff(Index index) const
  {
    return m_unaryOp(m_argImpl.coeff(index));
  }

  Scalar& coeffRef(Index row, Index col)
  {
    return m_unaryOp(m_argImpl.coeffRef(row, col));
  }

  Scalar& coeffRef(Index index)
  {
    return m_unaryOp(m_argImpl.coeffRef(index));
  }

protected:
  const UnaryOp& m_unaryOp;
  typename evaluator<ArgType>::type m_argImpl;
};

// -------------------- Product --------------------

template<typename Lhs, typename Rhs>
struct evaluator_impl<Product<Lhs,Rhs> > : public evaluator<typename Product<Lhs,Rhs>::PlainObject>::type
{
  typedef Product<Lhs,Rhs> XprType;
  typedef typename XprType::PlainObject PlainObject;
  typedef typename evaluator<PlainObject>::type evaluator_base;
  
//   enum {
//     EvaluateLhs = ;
//     EvaluateRhs = ;
//   };
  
  evaluator_impl(const XprType& product) : evaluator_base(m_result)
  {
    // here we process the left and right hand sides with a specialized evaluator
    // perhaps this step should be done by the TreeOptimizer to get a canonical tree and reduce evaluator instanciations
    // typename product_operand_evaluator<Lhs>::type m_lhsImpl(product.lhs());
    // typename product_operand_evaluator<Rhs>::type m_rhsImpl(product.rhs());
  
    // TODO do not rely on previous product mechanism !!
    m_result.resize(product.rows(), product.cols());
    m_result.noalias() = product.lhs() * product.rhs();
  }
  
protected:  
  PlainObject m_result;
};

// -------------------- Map --------------------

template<typename Derived, int AccessorsType>
struct evaluator_impl<MapBase<Derived, AccessorsType> >
{
  typedef MapBase<Derived, AccessorsType> MapType;
  typedef typename MapType::PointerType PointerType;
  typedef typename MapType::Index Index;
  typedef typename MapType::Scalar Scalar;
  typedef typename MapType::CoeffReturnType CoeffReturnType;
  typedef typename MapType::PacketScalar PacketScalar;
  typedef typename MapType::PacketReturnType PacketReturnType;
  
  evaluator_impl(const MapType& map) 
    : m_data(const_cast<PointerType>(map.data())),  
      m_rowStride(map.rowStride()),
      m_colStride(map.colStride())
  { }
 
  enum {
    RowsAtCompileTime = MapType::RowsAtCompileTime
  };
 
  CoeffReturnType coeff(Index row, Index col) const 
  { 
    return m_data[col * m_colStride + row * m_rowStride];
  }
  
  CoeffReturnType coeff(Index index) const 
  { 
    return coeff(RowsAtCompileTime == 1 ? 0 : index,
		 RowsAtCompileTime == 1 ? index : 0);
  }

  Scalar& coeffRef(Index row, Index col) 
  { 
    return m_data[col * m_colStride + row * m_rowStride];
  }
  
  Scalar& coeffRef(Index index) 
  { 
    return coeffRef(RowsAtCompileTime == 1 ? 0 : index,
		    RowsAtCompileTime == 1 ? index : 0);
  }
 
  template<int LoadMode> 
  PacketReturnType packet(Index row, Index col) const 
  { 
    PointerType ptr = m_data + row * m_rowStride + col * m_colStride;
    return internal::ploadt<PacketScalar, LoadMode>(ptr);
  }

  template<int LoadMode> 
  PacketReturnType packet(Index index) const 
  { 
    return packet<LoadMode>(RowsAtCompileTime == 1 ? 0 : index,
			    RowsAtCompileTime == 1 ? index : 0);
  }
  
  template<int StoreMode> 
  void writePacket(Index row, Index col, const PacketScalar& x) 
  { 
    PointerType ptr = m_data + row * m_rowStride + col * m_colStride;
    return internal::pstoret<Scalar, PacketScalar, StoreMode>(ptr, x);
  }
  
  template<int StoreMode> 
  void writePacket(Index index, const PacketScalar& x) 
  { 
    return writePacket<StoreMode>(RowsAtCompileTime == 1 ? 0 : index,
				  RowsAtCompileTime == 1 ? index : 0,
				  x);
  }
 
protected:
  PointerType m_data;
  int m_rowStride;
  int m_colStride;
};

template<typename PlainObjectType, int MapOptions, typename StrideType> 
struct evaluator_impl<Map<PlainObjectType, MapOptions, StrideType> >
  : public evaluator_impl<MapBase<Map<PlainObjectType, MapOptions, StrideType> > >
{
  typedef Map<PlainObjectType, MapOptions, StrideType> MapType;

  evaluator_impl(const MapType& map) 
    : evaluator_impl<MapBase<MapType> >(map) 
  { }
};

// -------------------- Block --------------------

template<typename XprType, int BlockRows, int BlockCols, bool InnerPanel> 
struct evaluator_impl<Block<XprType, BlockRows, BlockCols, InnerPanel, /* HasDirectAccess */ false> >
{
  typedef Block<XprType, BlockRows, BlockCols, InnerPanel, false> BlockType;

  evaluator_impl(const BlockType& block) 
    : m_argImpl(block.nestedExpression()), 
      m_startRow(block.startRow()), 
      m_startCol(block.startCol()) 
  { }
 
  typedef typename BlockType::Index Index;
  typedef typename BlockType::Scalar Scalar;
  typedef typename BlockType::CoeffReturnType CoeffReturnType;
  typedef typename BlockType::PacketScalar PacketScalar;
  typedef typename BlockType::PacketReturnType PacketReturnType;

  enum {
    RowsAtCompileTime = BlockType::RowsAtCompileTime
  };
 
  CoeffReturnType coeff(Index row, Index col) const 
  { 
    return m_argImpl.coeff(m_startRow + row, m_startCol + col); 
  }
  
  CoeffReturnType coeff(Index index) const 
  { 
    return coeff(RowsAtCompileTime == 1 ? 0 : index,
		 RowsAtCompileTime == 1 ? index : 0);
  }

  Scalar& coeffRef(Index row, Index col) 
  { 
    return m_argImpl.coeffRef(m_startRow + row, m_startCol + col); 
  }
  
  Scalar& coeffRef(Index index) 
  { 
    return coeffRef(RowsAtCompileTime == 1 ? 0 : index,
		    RowsAtCompileTime == 1 ? index : 0);
  }
 
  template<int LoadMode> 
  PacketReturnType packet(Index row, Index col) const 
  { 
    return m_argImpl.template packet<LoadMode>(m_startRow + row, m_startCol + col); 
  }

  template<int LoadMode> 
  PacketReturnType packet(Index index) const 
  { 
    return packet<LoadMode>(RowsAtCompileTime == 1 ? 0 : index,
			    RowsAtCompileTime == 1 ? index : 0);
  }
  
  template<int StoreMode> 
  void writePacket(Index row, Index col, const PacketScalar& x) 
  { 
    return m_argImpl.template writePacket<StoreMode>(m_startRow + row, m_startCol + col, x); 
  }
  
  template<int StoreMode> 
  void writePacket(Index index, const PacketScalar& x) 
  { 
    return writePacket<StoreMode>(RowsAtCompileTime == 1 ? 0 : index,
				  RowsAtCompileTime == 1 ? index : 0,
				  x);
  }
 
protected:
  typename evaluator<XprType>::type m_argImpl;

  // TODO: Get rid of m_startRow, m_startCol if known at compile time
  Index m_startRow; 
  Index m_startCol;
};

// TODO: This evaluator does not actually use the child evaluator; 
// all action is via the data() as returned by the Block expression.

template<typename XprType, int BlockRows, int BlockCols, bool InnerPanel> 
struct evaluator_impl<Block<XprType, BlockRows, BlockCols, InnerPanel, /* HasDirectAccess */ true> >
  : evaluator_impl<MapBase<Block<XprType, BlockRows, BlockCols, InnerPanel, true> > >
{
  typedef Block<XprType, BlockRows, BlockCols, InnerPanel, true> BlockType;

  evaluator_impl(const BlockType& block) 
    : evaluator_impl<MapBase<BlockType> >(block) 
  { }
};


// -------------------- Select --------------------

template<typename ConditionMatrixType, typename ThenMatrixType, typename ElseMatrixType>
struct evaluator_impl<Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType> >
{
  typedef Select<ConditionMatrixType, ThenMatrixType, ElseMatrixType> SelectType;

  evaluator_impl(const SelectType& select) 
    : m_conditionImpl(select.conditionMatrix()),
      m_thenImpl(select.thenMatrix()),
      m_elseImpl(select.elseMatrix())
  { }
 
  typedef typename SelectType::Index Index;
  typedef typename SelectType::CoeffReturnType CoeffReturnType;

  CoeffReturnType coeff(Index row, Index col) const
  {
    if (m_conditionImpl.coeff(row, col))
      return m_thenImpl.coeff(row, col);
    else
      return m_elseImpl.coeff(row, col);
  }

  CoeffReturnType coeff(Index index) const
  {
    if (m_conditionImpl.coeff(index))
      return m_thenImpl.coeff(index);
    else
      return m_elseImpl.coeff(index);
  }
 
protected:
  typename evaluator<ConditionMatrixType>::type m_conditionImpl;
  typename evaluator<ThenMatrixType>::type m_thenImpl;
  typename evaluator<ElseMatrixType>::type m_elseImpl;
};


// -------------------- Replicate --------------------

template<typename XprType, int RowFactor, int ColFactor> 
struct evaluator_impl<Replicate<XprType, RowFactor, ColFactor> >
{
  typedef Replicate<XprType, RowFactor, ColFactor> ReplicateType;

  evaluator_impl(const ReplicateType& replicate) 
    : m_argImpl(replicate.nestedExpression()),
      m_rows(replicate.nestedExpression().rows()),
      m_cols(replicate.nestedExpression().cols())
  { }
 
  typedef typename ReplicateType::Index Index;
  typedef typename ReplicateType::CoeffReturnType CoeffReturnType;
  typedef typename ReplicateType::PacketReturnType PacketReturnType;

  CoeffReturnType coeff(Index row, Index col) const
  {
    // try to avoid using modulo; this is a pure optimization strategy
    const Index actual_row = internal::traits<XprType>::RowsAtCompileTime==1 ? 0
                           : RowFactor==1 ? row
                           : row % m_rows;
    const Index actual_col = internal::traits<XprType>::ColsAtCompileTime==1 ? 0
                           : ColFactor==1 ? col
                           : col % m_cols;
    
    return m_argImpl.coeff(actual_row, actual_col);
  }

  template<int LoadMode>
  PacketReturnType packet(Index row, Index col) const
  {
    const Index actual_row = internal::traits<XprType>::RowsAtCompileTime==1 ? 0
                           : RowFactor==1 ? row
                           : row % m_rows;
    const Index actual_col = internal::traits<XprType>::ColsAtCompileTime==1 ? 0
                           : ColFactor==1 ? col
                           : col % m_cols;

    return m_argImpl.template packet<LoadMode>(actual_row, actual_col);
  }
 
protected:
  typename evaluator<XprType>::type m_argImpl;
  Index m_rows; // TODO: Get rid of this if known at compile time
  Index m_cols;
};


// -------------------- PartialReduxExpr --------------------
//
// This is a wrapper around the expression object. 
// TODO: Find out how to write a proper evaluator without duplicating
//       the row() and col() member functions.

template< typename XprType, typename MemberOp, int Direction>
struct evaluator_impl<PartialReduxExpr<XprType, MemberOp, Direction> >
{
  typedef PartialReduxExpr<XprType, MemberOp, Direction> PartialReduxExprType;

  evaluator_impl(const PartialReduxExprType expr)
    : m_expr(expr)
  { }

  typedef typename PartialReduxExprType::Index Index;
  typedef typename PartialReduxExprType::CoeffReturnType CoeffReturnType;
 
  CoeffReturnType coeff(Index row, Index col) const 
  { 
    return m_expr.coeff(row, col);
  }
  
  CoeffReturnType coeff(Index index) const 
  { 
    return m_expr.coeff(index);
  }

protected:
  const PartialReduxExprType& m_expr;
};


// -------------------- MatrixWrapper and ArrayWrapper --------------------
//
// evaluator_impl_wrapper_base<T> is a common base class for the
// MatrixWrapper and ArrayWrapper evaluators.

template<typename ArgType>
struct evaluator_impl_wrapper_base
{
  evaluator_impl_wrapper_base(const ArgType& arg) : m_argImpl(arg) {}

  typedef typename ArgType::Index Index;
  typedef typename ArgType::Scalar Scalar;
  typedef typename ArgType::CoeffReturnType CoeffReturnType;
  typedef typename ArgType::PacketScalar PacketScalar;
  typedef typename ArgType::PacketReturnType PacketReturnType;

  CoeffReturnType coeff(Index row, Index col) const
  {
    return m_argImpl.coeff(row, col);
  }

  CoeffReturnType coeff(Index index) const
  {
    return m_argImpl.coeff(index);
  }

  Scalar& coeffRef(Index row, Index col)
  {
    return m_argImpl.coeffRef(row, col);
  }

  Scalar& coeffRef(Index index)
  {
    return m_argImpl.coeffRef(index);
  }

  template<int LoadMode> 
  PacketReturnType packet(Index row, Index col) const
  {
    return m_argImpl.template packet<LoadMode>(row, col);
  }

  template<int LoadMode> 
  PacketReturnType packet(Index index) const
  {
    return m_argImpl.template packet<LoadMode>(index);
  }

  template<int StoreMode> 
  void writePacket(Index row, Index col, const PacketScalar& x)
  {
    m_argImpl.template writePacket<StoreMode>(row, col, x);
  }

  template<int StoreMode> 
  void writePacket(Index index, const PacketScalar& x)
  {
    m_argImpl.template writePacket<StoreMode>(index, x);
  }

protected:
  typename evaluator<ArgType>::type m_argImpl;
};

template<typename ArgType>
struct evaluator_impl<MatrixWrapper<ArgType> >
  : evaluator_impl_wrapper_base<ArgType>
{
  evaluator_impl(const MatrixWrapper<ArgType>& wrapper) 
    : evaluator_impl_wrapper_base<ArgType>(wrapper.nestedExpression())
  { }
};

template<typename ArgType>
struct evaluator_impl<ArrayWrapper<ArgType> >
  : evaluator_impl_wrapper_base<ArgType>
{
  evaluator_impl(const ArrayWrapper<ArgType>& wrapper) 
    : evaluator_impl_wrapper_base<ArgType>(wrapper.nestedExpression())
  { }
};


// -------------------- Reverse --------------------

// defined in Reverse.h:
template<typename PacketScalar, bool ReversePacket> struct reverse_packet_cond;

template<typename ArgType, int Direction>
struct evaluator_impl<Reverse<ArgType, Direction> >
{
  typedef Reverse<ArgType, Direction> ReverseType;

  evaluator_impl(const ReverseType& reverse) 
    : m_argImpl(reverse.nestedExpression()),
      m_rows(reverse.nestedExpression().rows()),
      m_cols(reverse.nestedExpression().cols())
  { }
 
  typedef typename ReverseType::Index Index;
  typedef typename ReverseType::Scalar Scalar;
  typedef typename ReverseType::CoeffReturnType CoeffReturnType;
  typedef typename ReverseType::PacketScalar PacketScalar;
  typedef typename ReverseType::PacketReturnType PacketReturnType;

  enum {
    PacketSize = internal::packet_traits<Scalar>::size,
    IsRowMajor = ReverseType::IsRowMajor,
    IsColMajor = !IsRowMajor,
    ReverseRow = (Direction == Vertical)   || (Direction == BothDirections),
    ReverseCol = (Direction == Horizontal) || (Direction == BothDirections),
    OffsetRow  = ReverseRow && IsColMajor ? PacketSize : 1,
    OffsetCol  = ReverseCol && IsRowMajor ? PacketSize : 1,
    ReversePacket = (Direction == BothDirections)
                    || ((Direction == Vertical)   && IsColMajor)
                    || ((Direction == Horizontal) && IsRowMajor)
  };
  typedef internal::reverse_packet_cond<PacketScalar,ReversePacket> reverse_packet;

  CoeffReturnType coeff(Index row, Index col) const
  {
    return m_argImpl.coeff(ReverseRow ? m_rows - row - 1 : row,
			   ReverseCol ? m_cols - col - 1 : col);
  }

  CoeffReturnType coeff(Index index) const
  {
    return m_argImpl.coeff(m_rows * m_cols - index - 1);
  }

  Scalar& coeffRef(Index row, Index col)
  {
    return m_argImpl.coeffRef(ReverseRow ? m_rows - row - 1 : row,
			      ReverseCol ? m_cols - col - 1 : col);
  }

  Scalar& coeffRef(Index index)
  {
    return m_argImpl.coeffRef(m_rows * m_cols - index - 1);
  }

  template<int LoadMode>
  PacketScalar packet(Index row, Index col) const
  {
    return reverse_packet::run(m_argImpl.template packet<LoadMode>(
                                  ReverseRow ? m_rows - row - OffsetRow : row,
                                  ReverseCol ? m_cols - col - OffsetCol : col));
  }

  template<int LoadMode>
  PacketScalar packet(Index index) const
  {
    return preverse(m_argImpl.template packet<LoadMode>(m_rows * m_cols - index - PacketSize));
  }

  template<int LoadMode>
  void writePacket(Index row, Index col, const PacketScalar& x)
  {
    m_argImpl.template writePacket<LoadMode>(
                                  ReverseRow ? m_rows - row - OffsetRow : row,
                                  ReverseCol ? m_cols - col - OffsetCol : col,
                                  reverse_packet::run(x));
  }

  template<int LoadMode>
  void writePacket(Index index, const PacketScalar& x)
  {
    m_argImpl.template writePacket<LoadMode>(m_rows * m_cols - index - PacketSize, preverse(x));
  }
 
protected:
  typename evaluator<ArgType>::type m_argImpl;
  Index m_rows; // TODO: Don't use if known at compile time or not needed
  Index m_cols;
};


} // namespace internal

#endif // EIGEN_COREEVALUATORS_H
