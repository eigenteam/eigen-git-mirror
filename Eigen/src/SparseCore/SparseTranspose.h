// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2015 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_SPARSETRANSPOSE_H
#define EIGEN_SPARSETRANSPOSE_H

namespace Eigen { 

namespace internal {
  template<typename MatrixType,int CompressedAccess=int(MatrixType::Flags&CompressedAccessBit)>
  class SparseTransposeImpl
    : public SparseMatrixBase<Transpose<MatrixType> >
  {};
  
  template<typename MatrixType>
  class SparseTransposeImpl<MatrixType,CompressedAccessBit>
    : public SparseCompressedBase<Transpose<MatrixType> >
  {
    typedef SparseCompressedBase<Transpose<MatrixType> > Base;
  public:
    using Base::derived;
    typedef typename Base::Scalar Scalar;
    typedef typename Base::Index Index;
    
    inline const Scalar* valuePtr() const { return derived().nestedExpression().valuePtr(); }
    inline const Index* innerIndexPtr() const { return derived().nestedExpression().innerIndexPtr(); }
    inline const Index* outerIndexPtr() const { return derived().nestedExpression().outerIndexPtr(); }
    inline const Index* innerNonZeroPtr() const { return derived().nestedExpression().innerNonZeroPtr(); }
    
    inline Scalar* valuePtr() { return derived().nestedExpression().valuePtr(); }
    inline Index* innerIndexPtr() { return derived().nestedExpression().innerIndexPtr(); }
    inline Index* outerIndexPtr() { return derived().nestedExpression().outerIndexPtr(); }
    inline Index* innerNonZeroPtr() { return derived().nestedExpression().innerNonZeroPtr(); }
  };
}
  
// Implement nonZeros() for transpose. I'm not sure that's the best approach for that.
// Perhaps it should be implemented in Transpose<> itself.
template<typename MatrixType> class TransposeImpl<MatrixType,Sparse>
  : public internal::SparseTransposeImpl<MatrixType>
{
  protected:
    typedef internal::SparseTransposeImpl<MatrixType> Base;
  public:
    inline typename MatrixType::Index nonZeros() const { return Base::derived().nestedExpression().nonZeros(); }
};

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
    
    explicit unary_evaluator(const XprType& op) :m_argImpl(op.nestedExpression()) {}

  protected:
    typename evaluator<ArgType>::nestedType m_argImpl;
};

} // end namespace internal

} // end namespace Eigen

#endif // EIGEN_SPARSETRANSPOSE_H
