// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2014 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_SPARSETRANSPOSE_H
#define EIGEN_SPARSETRANSPOSE_H

namespace Eigen { 

#ifndef EIGEN_TEST_EVALUATORS
template<typename MatrixType> class TransposeImpl<MatrixType,Sparse>
  : public SparseMatrixBase<Transpose<MatrixType> >
{
    typedef typename internal::remove_all<typename MatrixType::Nested>::type _MatrixTypeNested;
  public:

    EIGEN_SPARSE_PUBLIC_INTERFACE(Transpose<MatrixType> )

    class InnerIterator;
    class ReverseInnerIterator;

    inline Index nonZeros() const { return derived().nestedExpression().nonZeros(); }
};

// NOTE: VC10 trigger an ICE if don't put typename TransposeImpl<MatrixType,Sparse>:: in front of Index,
// a typedef typename TransposeImpl<MatrixType,Sparse>::Index Index;
// does not fix the issue.
// An alternative is to define the nested class in the parent class itself.
template<typename MatrixType> class TransposeImpl<MatrixType,Sparse>::InnerIterator
  : public _MatrixTypeNested::InnerIterator
{
    typedef typename _MatrixTypeNested::InnerIterator Base;
    typedef typename TransposeImpl::Index Index;
  public:

    EIGEN_STRONG_INLINE InnerIterator(const TransposeImpl& trans, typename TransposeImpl<MatrixType,Sparse>::Index outer)
      : Base(trans.derived().nestedExpression(), outer)
    {}
    Index row() const { return Base::col(); }
    Index col() const { return Base::row(); }
};

template<typename MatrixType> class TransposeImpl<MatrixType,Sparse>::ReverseInnerIterator
  : public _MatrixTypeNested::ReverseInnerIterator
{
    typedef typename _MatrixTypeNested::ReverseInnerIterator Base;
    typedef typename TransposeImpl::Index Index;
  public:

    EIGEN_STRONG_INLINE ReverseInnerIterator(const TransposeImpl& xpr, typename TransposeImpl<MatrixType,Sparse>::Index outer)
      : Base(xpr.derived().nestedExpression(), outer)
    {}
    Index row() const { return Base::col(); }
    Index col() const { return Base::row(); }
};

#else // EIGEN_TEST_EVALUATORS

namespace internal {
  
template<typename ArgType>
struct unary_evaluator<Transpose<ArgType>, IteratorBased>
  : public evaluator_base<Transpose<ArgType> >
{
    typedef typename evaluator<ArgType>::InnerIterator        EvalIterator;
    typedef typename evaluator<ArgType>::ReverseInnerIterator EvalReverseIterator;
  public:
    typedef Transpose<ArgType> XprType;
    typedef typename XprType::Index Index;

    class InnerIterator : public EvalIterator
    {
    public:
      EIGEN_STRONG_INLINE InnerIterator(const unary_evaluator& unaryOp, typename XprType::Index outer)
        : EvalIterator(unaryOp.m_argImpl,outer)
      {}
      
      Index row() const { return EvalIterator::col(); }
      Index col() const { return EvalIterator::row(); }
    };
    
    class ReverseInnerIterator : public EvalReverseIterator
    {
    public:
      EIGEN_STRONG_INLINE ReverseInnerIterator(const unary_evaluator& unaryOp, typename XprType::Index outer)
        : EvalReverseIterator(unaryOp.m_argImpl,outer)
      {}
      
      Index row() const { return EvalReverseIterator::col(); }
      Index col() const { return EvalReverseIterator::row(); }
    };
    
    enum {
      CoeffReadCost = evaluator<ArgType>::CoeffReadCost,
      Flags = XprType::Flags
    };
    
    unary_evaluator(const XprType& op) :m_argImpl(op.nestedExpression()) {}

  protected:
    typename evaluator<ArgType>::nestedType m_argImpl;
};

} // end namespace internal

#endif // EIGEN_TEST_EVALUATORS

} // end namespace Eigen

#endif // EIGEN_SPARSETRANSPOSE_H
