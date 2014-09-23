// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2014 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_SPARSE_CWISE_BINARY_OP_H
#define EIGEN_SPARSE_CWISE_BINARY_OP_H

namespace Eigen { 

// Here we have to handle 3 cases:
//  1 - sparse op dense
//  2 - dense op sparse
//  3 - sparse op sparse
// We also need to implement a 4th iterator for:
//  4 - dense op dense
// Finally, we also need to distinguish between the product and other operations :
//                configuration      returned mode
//  1 - sparse op dense    product      sparse
//                         generic      dense
//  2 - dense op sparse    product      sparse
//                         generic      dense
//  3 - sparse op sparse   product      sparse
//                         generic      sparse
//  4 - dense op dense     product      dense
//                         generic      dense

namespace internal {

template<typename BinaryOp, typename Lhs, typename Rhs, typename Derived,
  typename _LhsStorageMode = typename traits<Lhs>::StorageKind,
  typename _RhsStorageMode = typename traits<Rhs>::StorageKind>
class sparse_cwise_binary_op_inner_iterator_selector;

} // end namespace internal

namespace internal {

  
// Generic "sparse OP sparse"
template<typename BinaryOp, typename Lhs, typename Rhs>
struct binary_evaluator<CwiseBinaryOp<BinaryOp, Lhs, Rhs>, IteratorBased, IteratorBased>
  : evaluator_base<CwiseBinaryOp<BinaryOp, Lhs, Rhs> >
{
protected:
  typedef typename evaluator<Lhs>::InnerIterator  LhsIterator;
  typedef typename evaluator<Rhs>::InnerIterator  RhsIterator;
public:
  typedef CwiseBinaryOp<BinaryOp, Lhs, Rhs> XprType;

  class ReverseInnerIterator;
  class InnerIterator
  {
    typedef typename traits<XprType>::Scalar Scalar;
    typedef typename XprType::Index Index;

  public:
    
    EIGEN_STRONG_INLINE InnerIterator(const binary_evaluator& aEval, Index outer)
      : m_lhsIter(aEval.m_lhsImpl,outer), m_rhsIter(aEval.m_rhsImpl,outer), m_functor(aEval.m_functor)
    {
      this->operator++();
    }

    EIGEN_STRONG_INLINE InnerIterator& operator++()
    {
      if (m_lhsIter && m_rhsIter && (m_lhsIter.index() == m_rhsIter.index()))
      {
        m_id = m_lhsIter.index();
        m_value = m_functor(m_lhsIter.value(), m_rhsIter.value());
        ++m_lhsIter;
        ++m_rhsIter;
      }
      else if (m_lhsIter && (!m_rhsIter || (m_lhsIter.index() < m_rhsIter.index())))
      {
        m_id = m_lhsIter.index();
        m_value = m_functor(m_lhsIter.value(), Scalar(0));
        ++m_lhsIter;
      }
      else if (m_rhsIter && (!m_lhsIter || (m_lhsIter.index() > m_rhsIter.index())))
      {
        m_id = m_rhsIter.index();
        m_value = m_functor(Scalar(0), m_rhsIter.value());
        ++m_rhsIter;
      }
      else
      {
        m_value = 0; // this is to avoid a compilation warning
        m_id = -1;
      }
      return *this;
    }

    EIGEN_STRONG_INLINE Scalar value() const { return m_value; }

    EIGEN_STRONG_INLINE Index index() const { return m_id; }
    EIGEN_STRONG_INLINE Index row() const { return Lhs::IsRowMajor ? m_lhsIter.row() : index(); }
    EIGEN_STRONG_INLINE Index col() const { return Lhs::IsRowMajor ? index() : m_lhsIter.col(); }

    EIGEN_STRONG_INLINE operator bool() const { return m_id>=0; }

  protected:
    LhsIterator m_lhsIter;
    RhsIterator m_rhsIter;
    const BinaryOp& m_functor;
    Scalar m_value;
    Index m_id;
  };
  
  
  enum {
    CoeffReadCost = evaluator<Lhs>::CoeffReadCost + evaluator<Rhs>::CoeffReadCost + functor_traits<BinaryOp>::Cost,
    Flags = XprType::Flags
  };
  
  explicit binary_evaluator(const XprType& xpr)
    : m_functor(xpr.functor()),
      m_lhsImpl(xpr.lhs()), 
      m_rhsImpl(xpr.rhs())  
  { }

protected:
  const BinaryOp m_functor;
  typename evaluator<Lhs>::nestedType m_lhsImpl;
  typename evaluator<Rhs>::nestedType m_rhsImpl;
};

// "sparse .* sparse"
template<typename T, typename Lhs, typename Rhs>
struct binary_evaluator<CwiseBinaryOp<scalar_product_op<T>, Lhs, Rhs>, IteratorBased, IteratorBased>
  : evaluator_base<CwiseBinaryOp<scalar_product_op<T>, Lhs, Rhs> >
{
protected:
  typedef scalar_product_op<T> BinaryOp;
  typedef typename evaluator<Lhs>::InnerIterator  LhsIterator;
  typedef typename evaluator<Rhs>::InnerIterator  RhsIterator;
public:
  typedef CwiseBinaryOp<BinaryOp, Lhs, Rhs> XprType;

  class ReverseInnerIterator;
  class InnerIterator
  {
    typedef typename traits<XprType>::Scalar Scalar;
    typedef typename XprType::Index Index;

  public:
    
    EIGEN_STRONG_INLINE InnerIterator(const binary_evaluator& aEval, Index outer)
      : m_lhsIter(aEval.m_lhsImpl,outer), m_rhsIter(aEval.m_rhsImpl,outer), m_functor(aEval.m_functor)
    {
      while (m_lhsIter && m_rhsIter && (m_lhsIter.index() != m_rhsIter.index()))
      {
        if (m_lhsIter.index() < m_rhsIter.index())
          ++m_lhsIter;
        else
          ++m_rhsIter;
      }
    }

    EIGEN_STRONG_INLINE InnerIterator& operator++()
    {
      ++m_lhsIter;
      ++m_rhsIter;
      while (m_lhsIter && m_rhsIter && (m_lhsIter.index() != m_rhsIter.index()))
      {
        if (m_lhsIter.index() < m_rhsIter.index())
          ++m_lhsIter;
        else
          ++m_rhsIter;
      }
      return *this;
    }
    
    EIGEN_STRONG_INLINE Scalar value() const { return m_functor(m_lhsIter.value(), m_rhsIter.value()); }

    EIGEN_STRONG_INLINE Index index() const { return m_lhsIter.index(); }
    EIGEN_STRONG_INLINE Index row() const { return m_lhsIter.row(); }
    EIGEN_STRONG_INLINE Index col() const { return m_lhsIter.col(); }

    EIGEN_STRONG_INLINE operator bool() const { return (m_lhsIter && m_rhsIter); }

  protected:
    LhsIterator m_lhsIter;
    RhsIterator m_rhsIter;
    const BinaryOp& m_functor;
  };
  
  
  enum {
    CoeffReadCost = evaluator<Lhs>::CoeffReadCost + evaluator<Rhs>::CoeffReadCost + functor_traits<BinaryOp>::Cost,
    Flags = XprType::Flags
  };
  
  explicit binary_evaluator(const XprType& xpr)
    : m_functor(xpr.functor()),
      m_lhsImpl(xpr.lhs()), 
      m_rhsImpl(xpr.rhs())  
  { }

protected:
  const BinaryOp m_functor;
  typename evaluator<Lhs>::nestedType m_lhsImpl;
  typename evaluator<Rhs>::nestedType m_rhsImpl;
};

// "dense .* sparse"
template<typename T, typename Lhs, typename Rhs>
struct binary_evaluator<CwiseBinaryOp<scalar_product_op<T>, Lhs, Rhs>, IndexBased, IteratorBased>
  : evaluator_base<CwiseBinaryOp<scalar_product_op<T>, Lhs, Rhs> >
{
protected:
  typedef scalar_product_op<T> BinaryOp;
  typedef typename evaluator<Lhs>::type  LhsEvaluator;
  typedef typename evaluator<Rhs>::InnerIterator  RhsIterator;
public:
  typedef CwiseBinaryOp<BinaryOp, Lhs, Rhs> XprType;

  class ReverseInnerIterator;
  class InnerIterator
  {
    typedef typename traits<XprType>::Scalar Scalar;
    typedef typename XprType::Index Index;
    enum { IsRowMajor = (int(Rhs::Flags)&RowMajorBit)==RowMajorBit };

  public:
    
    EIGEN_STRONG_INLINE InnerIterator(const binary_evaluator& aEval, Index outer)
      : m_lhsEval(aEval.m_lhsImpl), m_rhsIter(aEval.m_rhsImpl,outer), m_functor(aEval.m_functor), m_outer(outer)
    {}

    EIGEN_STRONG_INLINE InnerIterator& operator++()
    {
      ++m_rhsIter;
      return *this;
    }

    EIGEN_STRONG_INLINE Scalar value() const
    { return m_functor(m_lhsEval.coeff(IsRowMajor?m_outer:m_rhsIter.index(),IsRowMajor?m_rhsIter.index():m_outer), m_rhsIter.value()); }

    EIGEN_STRONG_INLINE Index index() const { return m_rhsIter.index(); }
    EIGEN_STRONG_INLINE Index row() const { return m_rhsIter.row(); }
    EIGEN_STRONG_INLINE Index col() const { return m_rhsIter.col(); }

    EIGEN_STRONG_INLINE operator bool() const { return m_rhsIter; }

  protected:
    const LhsEvaluator &m_lhsEval;
    RhsIterator m_rhsIter;
    const BinaryOp& m_functor;
    const Index m_outer;
  };
  
  
  enum {
    CoeffReadCost = evaluator<Lhs>::CoeffReadCost + evaluator<Rhs>::CoeffReadCost + functor_traits<BinaryOp>::Cost,
    Flags = XprType::Flags
  };
  
  explicit binary_evaluator(const XprType& xpr)
    : m_functor(xpr.functor()),
      m_lhsImpl(xpr.lhs()), 
      m_rhsImpl(xpr.rhs())  
  { }

protected:
  const BinaryOp m_functor;
  typename evaluator<Lhs>::nestedType m_lhsImpl;
  typename evaluator<Rhs>::nestedType m_rhsImpl;
};

// "sparse .* dense"
template<typename T, typename Lhs, typename Rhs>
struct binary_evaluator<CwiseBinaryOp<scalar_product_op<T>, Lhs, Rhs>, IteratorBased, IndexBased>
  : evaluator_base<CwiseBinaryOp<scalar_product_op<T>, Lhs, Rhs> >
{
protected:
  typedef scalar_product_op<T> BinaryOp;
  typedef typename evaluator<Lhs>::InnerIterator  LhsIterator;
  typedef typename evaluator<Rhs>::type  RhsEvaluator;
public:
  typedef CwiseBinaryOp<BinaryOp, Lhs, Rhs> XprType;

  class ReverseInnerIterator;
  class InnerIterator
  {
    typedef typename traits<XprType>::Scalar Scalar;
    typedef typename XprType::Index Index;
    enum { IsRowMajor = (int(Lhs::Flags)&RowMajorBit)==RowMajorBit };

  public:
    
    EIGEN_STRONG_INLINE InnerIterator(const binary_evaluator& aEval, Index outer)
      : m_lhsIter(aEval.m_lhsImpl,outer), m_rhsEval(aEval.m_rhsImpl), m_functor(aEval.m_functor), m_outer(outer)
    {}

    EIGEN_STRONG_INLINE InnerIterator& operator++()
    {
      ++m_lhsIter;
      return *this;
    }

    EIGEN_STRONG_INLINE Scalar value() const
    { return m_functor(m_lhsIter.value(),
                       m_rhsEval.coeff(IsRowMajor?m_outer:m_lhsIter.index(),IsRowMajor?m_lhsIter.index():m_outer)); }

    EIGEN_STRONG_INLINE Index index() const { return m_lhsIter.index(); }
    EIGEN_STRONG_INLINE Index row() const { return m_lhsIter.row(); }
    EIGEN_STRONG_INLINE Index col() const { return m_lhsIter.col(); }

    EIGEN_STRONG_INLINE operator bool() const { return m_lhsIter; }

  protected:
    LhsIterator m_lhsIter;
    const RhsEvaluator &m_rhsEval;
    const BinaryOp& m_functor;
    const Index m_outer;
  };
  
  
  enum {
    CoeffReadCost = evaluator<Lhs>::CoeffReadCost + evaluator<Rhs>::CoeffReadCost + functor_traits<BinaryOp>::Cost,
    Flags = XprType::Flags
  };
  
  explicit binary_evaluator(const XprType& xpr)
    : m_functor(xpr.functor()),
      m_lhsImpl(xpr.lhs()), 
      m_rhsImpl(xpr.rhs())  
  { }

protected:
  const BinaryOp m_functor;
  typename evaluator<Lhs>::nestedType m_lhsImpl;
  typename evaluator<Rhs>::nestedType m_rhsImpl;
};

}

/***************************************************************************
* Implementation of SparseMatrixBase and SparseCwise functions/operators
***************************************************************************/

template<typename Derived>
template<typename OtherDerived>
EIGEN_STRONG_INLINE Derived &
SparseMatrixBase<Derived>::operator-=(const SparseMatrixBase<OtherDerived> &other)
{
  return derived() = derived() - other.derived();
}

template<typename Derived>
template<typename OtherDerived>
EIGEN_STRONG_INLINE Derived &
SparseMatrixBase<Derived>::operator+=(const SparseMatrixBase<OtherDerived>& other)
{
  return derived() = derived() + other.derived();
}

template<typename Derived>
template<typename OtherDerived>
EIGEN_STRONG_INLINE const EIGEN_SPARSE_CWISE_PRODUCT_RETURN_TYPE
SparseMatrixBase<Derived>::cwiseProduct(const MatrixBase<OtherDerived> &other) const
{
  return EIGEN_SPARSE_CWISE_PRODUCT_RETURN_TYPE(derived(), other.derived());
}

} // end namespace Eigen

#endif // EIGEN_SPARSE_CWISE_BINARY_OP_H
