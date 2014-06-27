// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Gael Guennebaud <gael.guennebaud@inria.fr>
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

#ifndef EIGEN_TEST_EVALUATORS
  
namespace internal {
  
template<typename Lhs, typename Rhs>
struct traits<SparseDiagonalProduct<Lhs, Rhs> >
{
  typedef typename remove_all<Lhs>::type _Lhs;
  typedef typename remove_all<Rhs>::type _Rhs;
  typedef typename _Lhs::Scalar Scalar;
  typedef typename promote_index_type<typename traits<Lhs>::Index,
                                         typename traits<Rhs>::Index>::type Index;
  typedef Sparse StorageKind;
  typedef MatrixXpr XprKind;
  enum {
    RowsAtCompileTime = _Lhs::RowsAtCompileTime,
    ColsAtCompileTime = _Rhs::ColsAtCompileTime,

    MaxRowsAtCompileTime = _Lhs::MaxRowsAtCompileTime,
    MaxColsAtCompileTime = _Rhs::MaxColsAtCompileTime,

    SparseFlags = is_diagonal<_Lhs>::ret ? int(_Rhs::Flags) : int(_Lhs::Flags),
    Flags = (SparseFlags&RowMajorBit),
    CoeffReadCost = Dynamic
  };
};

enum {SDP_IsDiagonal, SDP_IsSparseRowMajor, SDP_IsSparseColMajor};
template<typename Lhs, typename Rhs, typename SparseDiagonalProductType, int RhsMode, int LhsMode>
class sparse_diagonal_product_inner_iterator_selector;

} // end namespace internal

template<typename Lhs, typename Rhs>
class SparseDiagonalProduct
  : public SparseMatrixBase<SparseDiagonalProduct<Lhs,Rhs> >,
    internal::no_assignment_operator
{
    typedef typename Lhs::Nested LhsNested;
    typedef typename Rhs::Nested RhsNested;

    typedef typename internal::remove_all<LhsNested>::type _LhsNested;
    typedef typename internal::remove_all<RhsNested>::type _RhsNested;

    enum {
      LhsMode = internal::is_diagonal<_LhsNested>::ret ? internal::SDP_IsDiagonal
              : (_LhsNested::Flags&RowMajorBit) ? internal::SDP_IsSparseRowMajor : internal::SDP_IsSparseColMajor,
      RhsMode = internal::is_diagonal<_RhsNested>::ret ? internal::SDP_IsDiagonal
              : (_RhsNested::Flags&RowMajorBit) ? internal::SDP_IsSparseRowMajor : internal::SDP_IsSparseColMajor
    };

  public:

    EIGEN_SPARSE_PUBLIC_INTERFACE(SparseDiagonalProduct)

    typedef internal::sparse_diagonal_product_inner_iterator_selector
                      <_LhsNested,_RhsNested,SparseDiagonalProduct,LhsMode,RhsMode> InnerIterator;
    
    // We do not want ReverseInnerIterator for diagonal-sparse products,
    // but this dummy declaration is needed to make diag * sparse * diag compile.
    class ReverseInnerIterator;

    EIGEN_STRONG_INLINE SparseDiagonalProduct(const Lhs& lhs, const Rhs& rhs)
      : m_lhs(lhs), m_rhs(rhs)
    {
      eigen_assert(lhs.cols() == rhs.rows() && "invalid sparse matrix * diagonal matrix product");
    }

    EIGEN_STRONG_INLINE Index rows() const { return m_lhs.rows(); }
    EIGEN_STRONG_INLINE Index cols() const { return m_rhs.cols(); }

    EIGEN_STRONG_INLINE const _LhsNested& lhs() const { return m_lhs; }
    EIGEN_STRONG_INLINE const _RhsNested& rhs() const { return m_rhs; }

  protected:
    LhsNested m_lhs;
    RhsNested m_rhs;
};
#endif

namespace internal {

#ifndef EIGEN_TEST_EVALUATORS


  
template<typename Lhs, typename Rhs, typename SparseDiagonalProductType>
class sparse_diagonal_product_inner_iterator_selector
<Lhs,Rhs,SparseDiagonalProductType,SDP_IsDiagonal,SDP_IsSparseRowMajor>
  : public CwiseUnaryOp<scalar_multiple_op<typename Lhs::Scalar>,const Rhs>::InnerIterator
{
    typedef typename CwiseUnaryOp<scalar_multiple_op<typename Lhs::Scalar>,const Rhs>::InnerIterator Base;
    typedef typename Lhs::Index Index;
  public:
    inline sparse_diagonal_product_inner_iterator_selector(
              const SparseDiagonalProductType& expr, Index outer)
      : Base(expr.rhs()*(expr.lhs().diagonal().coeff(outer)), outer)
    {}
};

template<typename Lhs, typename Rhs, typename SparseDiagonalProductType>
class sparse_diagonal_product_inner_iterator_selector
<Lhs,Rhs,SparseDiagonalProductType,SDP_IsDiagonal,SDP_IsSparseColMajor>
  : public CwiseBinaryOp<
      scalar_product_op<typename Lhs::Scalar>,
      const typename Rhs::ConstInnerVectorReturnType,
      const typename Lhs::DiagonalVectorType>::InnerIterator
{
    typedef typename CwiseBinaryOp<
      scalar_product_op<typename Lhs::Scalar>,
      const typename Rhs::ConstInnerVectorReturnType,
      const typename Lhs::DiagonalVectorType>::InnerIterator Base;
    typedef typename Lhs::Index Index;
    Index m_outer;
  public:
    inline sparse_diagonal_product_inner_iterator_selector(
              const SparseDiagonalProductType& expr, Index outer)
      : Base(expr.rhs().innerVector(outer) .cwiseProduct(expr.lhs().diagonal()), 0), m_outer(outer)
    {}
    
    inline Index outer() const { return m_outer; }
    inline Index col() const { return m_outer; }
};

template<typename Lhs, typename Rhs, typename SparseDiagonalProductType>
class sparse_diagonal_product_inner_iterator_selector
<Lhs,Rhs,SparseDiagonalProductType,SDP_IsSparseColMajor,SDP_IsDiagonal>
  : public CwiseUnaryOp<scalar_multiple_op<typename Rhs::Scalar>,const Lhs>::InnerIterator
{
    typedef typename CwiseUnaryOp<scalar_multiple_op<typename Rhs::Scalar>,const Lhs>::InnerIterator Base;
    typedef typename Lhs::Index Index;
  public:
    inline sparse_diagonal_product_inner_iterator_selector(
              const SparseDiagonalProductType& expr, Index outer)
      : Base(expr.lhs()*expr.rhs().diagonal().coeff(outer), outer)
    {}
};

template<typename Lhs, typename Rhs, typename SparseDiagonalProductType>
class sparse_diagonal_product_inner_iterator_selector
<Lhs,Rhs,SparseDiagonalProductType,SDP_IsSparseRowMajor,SDP_IsDiagonal>
  : public CwiseBinaryOp<
      scalar_product_op<typename Rhs::Scalar>,
      const typename Lhs::ConstInnerVectorReturnType,
      const Transpose<const typename Rhs::DiagonalVectorType> >::InnerIterator
{
    typedef typename CwiseBinaryOp<
      scalar_product_op<typename Rhs::Scalar>,
      const typename Lhs::ConstInnerVectorReturnType,
      const Transpose<const typename Rhs::DiagonalVectorType> >::InnerIterator Base;
    typedef typename Lhs::Index Index;
    Index m_outer;
  public:
    inline sparse_diagonal_product_inner_iterator_selector(
              const SparseDiagonalProductType& expr, Index outer)
      : Base(expr.lhs().innerVector(outer) .cwiseProduct(expr.rhs().diagonal().transpose()), 0), m_outer(outer)
    {}
    
    inline Index outer() const { return m_outer; }
    inline Index row() const { return m_outer; }
};

#else // EIGEN_TEST_EVALUATORS
enum {
  SDP_AsScalarProduct,
  SDP_AsCwiseProduct
};
  
template<typename SparseXprType, typename DiagonalCoeffType, int SDP_Tag>
struct sparse_diagonal_product_evaluator;

template<typename Lhs, typename Rhs, int Options, int ProductTag>
struct product_evaluator<Product<Lhs, Rhs, Options>, ProductTag, DiagonalShape, SparseShape, typename Lhs::Scalar, typename Rhs::Scalar> 
  : public sparse_diagonal_product_evaluator<Rhs, typename Lhs::DiagonalVectorType, Rhs::Flags&RowMajorBit?SDP_AsScalarProduct:SDP_AsCwiseProduct>
{
  typedef Product<Lhs, Rhs, Options> XprType;
  typedef evaluator<XprType> type;
  typedef evaluator<XprType> nestedType;
  enum { CoeffReadCost = Dynamic, Flags = Rhs::Flags&RowMajorBit }; // FIXME CoeffReadCost & Flags
  
  typedef sparse_diagonal_product_evaluator<Rhs, typename Lhs::DiagonalVectorType, Rhs::Flags&RowMajorBit?SDP_AsScalarProduct:SDP_AsCwiseProduct> Base;
  product_evaluator(const XprType& xpr) : Base(xpr.rhs(), xpr.lhs().diagonal()) {}
};

template<typename Lhs, typename Rhs, int Options, int ProductTag>
struct product_evaluator<Product<Lhs, Rhs, Options>, ProductTag, SparseShape, DiagonalShape, typename Lhs::Scalar, typename Rhs::Scalar> 
  : public sparse_diagonal_product_evaluator<Lhs, Transpose<const typename Rhs::DiagonalVectorType>, Lhs::Flags&RowMajorBit?SDP_AsCwiseProduct:SDP_AsScalarProduct>
{
  typedef Product<Lhs, Rhs, Options> XprType;
  typedef evaluator<XprType> type;
  typedef evaluator<XprType> nestedType;
  enum { CoeffReadCost = Dynamic, Flags = Lhs::Flags&RowMajorBit }; // FIXME CoeffReadCost & Flags
  
  typedef sparse_diagonal_product_evaluator<Lhs, Transpose<const typename Rhs::DiagonalVectorType>, Lhs::Flags&RowMajorBit?SDP_AsCwiseProduct:SDP_AsScalarProduct> Base;
  product_evaluator(const XprType& xpr) : Base(xpr.lhs(), xpr.rhs().diagonal()) {}
};

template<typename SparseXprType, typename DiagonalCoeffType>
struct sparse_diagonal_product_evaluator<SparseXprType, DiagonalCoeffType, SDP_AsScalarProduct> 
{
protected:
  typedef typename evaluator<SparseXprType>::InnerIterator SparseXprInnerIterator;
  typedef typename SparseXprType::Scalar Scalar;
  typedef typename SparseXprType::Index Index;
  
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
  typedef typename SparseXprType::Index Index;
  
  typedef CwiseBinaryOp<scalar_product_op<Scalar>,
                        const typename SparseXprType::ConstInnerVectorReturnType,
                        const DiagCoeffType> CwiseProductType;
                        
  typedef typename evaluator<CwiseProductType>::type CwiseProductEval;
  typedef typename evaluator<CwiseProductType>::InnerIterator CwiseProductIterator;
  
  class InnerIterator : public CwiseProductIterator
  {
  public:
    InnerIterator(const sparse_diagonal_product_evaluator &xprEval, Index outer)
      : CwiseProductIterator(CwiseProductEval(xprEval.m_sparseXprNested.innerVector(outer).cwiseProduct(xprEval.m_diagCoeffNested)),0),
        m_cwiseEval(xprEval.m_sparseXprNested.innerVector(outer).cwiseProduct(xprEval.m_diagCoeffNested)),
        m_outer(outer)
    {
      ::new (static_cast<CwiseProductIterator*>(this)) CwiseProductIterator(m_cwiseEval,0);
    }
    
    inline Index outer() const  { return m_outer; }
    inline Index col() const    { return SparseXprType::IsRowMajor ? CwiseProductIterator::index() : m_outer; }
    inline Index row() const    { return SparseXprType::IsRowMajor ? m_outer : CwiseProductIterator::index(); }
    
  protected:
    Index m_outer;
    CwiseProductEval m_cwiseEval;
  };
  
  sparse_diagonal_product_evaluator(const SparseXprType &sparseXpr, const DiagCoeffType &diagCoeff)
    : m_sparseXprNested(sparseXpr), m_diagCoeffNested(diagCoeff)
  {}
    
protected:
  typename nested_eval<SparseXprType,1>::type m_sparseXprNested;
  typename nested_eval<DiagCoeffType,SparseXprType::IsRowMajor ? SparseXprType::RowsAtCompileTime
                                                               : SparseXprType::ColsAtCompileTime>::type m_diagCoeffNested;
};

#endif // EIGEN_TEST_EVALUATORS


} // end namespace internal

#ifndef EIGEN_TEST_EVALUATORS

// SparseMatrixBase functions
template<typename Derived>
template<typename OtherDerived>
const SparseDiagonalProduct<Derived,OtherDerived>
SparseMatrixBase<Derived>::operator*(const DiagonalBase<OtherDerived> &other) const
{
  return SparseDiagonalProduct<Derived,OtherDerived>(this->derived(), other.derived());
}
#endif // EIGEN_TEST_EVALUATORS

} // end namespace Eigen

#endif // EIGEN_SPARSE_DIAGONAL_PRODUCT_H
