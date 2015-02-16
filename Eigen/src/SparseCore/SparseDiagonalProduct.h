// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009-2014 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_SPARSE_DIAGONAL_PRODUCT_H
#define EIGEN_SPARSE_DIAGONAL_PRODUCT_H

namespace Eigen { 

// The product of a diagonal matrix with a sparse matrix can be easily
// implemented using expression template.
// We have two consider very different cases:
// 1 - diag * row-major sparse
//     => each inner vector <=> scalar * sparse vector product
//     => so we can reuse CwiseUnaryOp::InnerIterator
// 2 - diag * col-major sparse
//     => each inner vector <=> densevector * sparse vector cwise product
//     => again, we can reuse specialization of CwiseBinaryOp::InnerIterator
//        for that particular case
// The two other cases are symmetric.

namespace internal {

enum {
  SDP_AsScalarProduct,
  SDP_AsCwiseProduct
};
  
template<typename SparseXprType, typename DiagonalCoeffType, int SDP_Tag>
struct sparse_diagonal_product_evaluator;

template<typename Lhs, typename Rhs, int ProductTag>
struct product_evaluator<Product<Lhs, Rhs, DefaultProduct>, ProductTag, DiagonalShape, SparseShape, typename traits<Lhs>::Scalar, typename traits<Rhs>::Scalar> 
  : public sparse_diagonal_product_evaluator<Rhs, typename Lhs::DiagonalVectorType, Rhs::Flags&RowMajorBit?SDP_AsScalarProduct:SDP_AsCwiseProduct>
{
  typedef Product<Lhs, Rhs, DefaultProduct> XprType;
  typedef evaluator<XprType> type;
  typedef evaluator<XprType> nestedType;
  enum { CoeffReadCost = Dynamic, Flags = Rhs::Flags&RowMajorBit }; // FIXME CoeffReadCost & Flags
  
  typedef sparse_diagonal_product_evaluator<Rhs, typename Lhs::DiagonalVectorType, Rhs::Flags&RowMajorBit?SDP_AsScalarProduct:SDP_AsCwiseProduct> Base;
  explicit product_evaluator(const XprType& xpr) : Base(xpr.rhs(), xpr.lhs().diagonal()) {}
};

template<typename Lhs, typename Rhs, int ProductTag>
struct product_evaluator<Product<Lhs, Rhs, DefaultProduct>, ProductTag, SparseShape, DiagonalShape, typename traits<Lhs>::Scalar, typename traits<Rhs>::Scalar> 
  : public sparse_diagonal_product_evaluator<Lhs, Transpose<const typename Rhs::DiagonalVectorType>, Lhs::Flags&RowMajorBit?SDP_AsCwiseProduct:SDP_AsScalarProduct>
{
  typedef Product<Lhs, Rhs, DefaultProduct> XprType;
  typedef evaluator<XprType> type;
  typedef evaluator<XprType> nestedType;
  enum { CoeffReadCost = Dynamic, Flags = Lhs::Flags&RowMajorBit }; // FIXME CoeffReadCost & Flags
  
  typedef sparse_diagonal_product_evaluator<Lhs, Transpose<const typename Rhs::DiagonalVectorType>, Lhs::Flags&RowMajorBit?SDP_AsCwiseProduct:SDP_AsScalarProduct> Base;
  explicit product_evaluator(const XprType& xpr) : Base(xpr.lhs(), xpr.rhs().diagonal().transpose()) {}
};

template<typename SparseXprType, typename DiagonalCoeffType>
struct sparse_diagonal_product_evaluator<SparseXprType, DiagonalCoeffType, SDP_AsScalarProduct> 
{
protected:
  typedef typename evaluator<SparseXprType>::InnerIterator SparseXprInnerIterator;
  typedef typename SparseXprType::Scalar Scalar;
  
public:
  class InnerIterator : public SparseXprInnerIterator
  {
  public:
    InnerIterator(const sparse_diagonal_product_evaluator &xprEval, Index outer)
      : SparseXprInnerIterator(xprEval.m_sparseXprImpl, outer),
        m_coeff(xprEval.m_diagCoeffImpl.coeff(outer))
    {}
    
    EIGEN_STRONG_INLINE Scalar value() const { return m_coeff * SparseXprInnerIterator::value(); }
  protected:
    typename DiagonalCoeffType::Scalar m_coeff;
  };
  
  sparse_diagonal_product_evaluator(const SparseXprType &sparseXpr, const DiagonalCoeffType &diagCoeff)
    : m_sparseXprImpl(sparseXpr), m_diagCoeffImpl(diagCoeff)
  {}
    
protected:
  typename evaluator<SparseXprType>::nestedType m_sparseXprImpl;
  typename evaluator<DiagonalCoeffType>::nestedType m_diagCoeffImpl;
};


template<typename SparseXprType, typename DiagCoeffType>
struct sparse_diagonal_product_evaluator<SparseXprType, DiagCoeffType, SDP_AsCwiseProduct> 
{
  typedef typename SparseXprType::Scalar Scalar;
  
  typedef CwiseBinaryOp<scalar_product_op<Scalar>,
                        const typename SparseXprType::ConstInnerVectorReturnType,
                        const DiagCoeffType> CwiseProductType;
                        
  typedef typename evaluator<CwiseProductType>::type CwiseProductEval;
  typedef typename evaluator<CwiseProductType>::InnerIterator CwiseProductIterator;
  
  class InnerIterator
  {
  public:
    InnerIterator(const sparse_diagonal_product_evaluator &xprEval, Index outer)
      : m_cwiseEval(xprEval.m_sparseXprNested.innerVector(outer).cwiseProduct(xprEval.m_diagCoeffNested)),
        m_cwiseIter(m_cwiseEval, 0),
        m_outer(outer)
    {}
    
    inline Scalar value() const { return m_cwiseIter.value(); }
    inline Index index() const  { return m_cwiseIter.index(); }
    inline Index outer() const  { return m_outer; }
    inline Index col() const    { return SparseXprType::IsRowMajor ? m_cwiseIter.index() : m_outer; }
    inline Index row() const    { return SparseXprType::IsRowMajor ? m_outer : m_cwiseIter.index(); }
    
    EIGEN_STRONG_INLINE InnerIterator& operator++()
    { ++m_cwiseIter; return *this; }
    inline operator bool() const  { return m_cwiseIter; }
    
  protected:
    CwiseProductEval m_cwiseEval;
    CwiseProductIterator m_cwiseIter;
    Index m_outer;
  };
  
  sparse_diagonal_product_evaluator(const SparseXprType &sparseXpr, const DiagCoeffType &diagCoeff)
    : m_sparseXprNested(sparseXpr), m_diagCoeffNested(diagCoeff)
  {}
    
protected:
  typename nested_eval<SparseXprType,1>::type m_sparseXprNested;
  typename nested_eval<DiagCoeffType,SparseXprType::IsRowMajor ? SparseXprType::RowsAtCompileTime
                                                               : SparseXprType::ColsAtCompileTime>::type m_diagCoeffNested;
};

} // end namespace internal

} // end namespace Eigen

#endif // EIGEN_SPARSE_DIAGONAL_PRODUCT_H
