// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2014 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_SPARSE_CWISE_UNARY_OP_H
#define EIGEN_SPARSE_CWISE_UNARY_OP_H

namespace Eigen { 
  
#ifndef EIGEN_TEST_EVALUATORS

template<typename UnaryOp, typename MatrixType>
class CwiseUnaryOpImpl<UnaryOp,MatrixType,Sparse>
  : public SparseMatrixBase<CwiseUnaryOp<UnaryOp, MatrixType> >
{
  public:

    typedef CwiseUnaryOp<UnaryOp, MatrixType> Derived;
    EIGEN_SPARSE_PUBLIC_INTERFACE(Derived)

    class InnerIterator;
    class ReverseInnerIterator;
    
  protected:
    typedef typename internal::traits<Derived>::_XprTypeNested _MatrixTypeNested;
    typedef typename _MatrixTypeNested::InnerIterator MatrixTypeIterator;
    typedef typename _MatrixTypeNested::ReverseInnerIterator MatrixTypeReverseIterator;
};

template<typename UnaryOp, typename MatrixType>
class CwiseUnaryOpImpl<UnaryOp,MatrixType,Sparse>::InnerIterator
    : public CwiseUnaryOpImpl<UnaryOp,MatrixType,Sparse>::MatrixTypeIterator
{
    typedef typename CwiseUnaryOpImpl::Scalar Scalar;
    typedef typename CwiseUnaryOpImpl<UnaryOp,MatrixType,Sparse>::MatrixTypeIterator Base;
  public:

    EIGEN_STRONG_INLINE InnerIterator(const CwiseUnaryOpImpl& unaryOp, typename CwiseUnaryOpImpl::Index outer)
      : Base(unaryOp.derived().nestedExpression(),outer), m_functor(unaryOp.derived().functor())
    {}

    EIGEN_STRONG_INLINE InnerIterator& operator++()
    { Base::operator++(); return *this; }

    EIGEN_STRONG_INLINE typename CwiseUnaryOpImpl::Scalar value() const { return m_functor(Base::value()); }

  protected:
    const UnaryOp m_functor;
  private:
    typename CwiseUnaryOpImpl::Scalar& valueRef();
};

template<typename UnaryOp, typename MatrixType>
class CwiseUnaryOpImpl<UnaryOp,MatrixType,Sparse>::ReverseInnerIterator
    : public CwiseUnaryOpImpl<UnaryOp,MatrixType,Sparse>::MatrixTypeReverseIterator
{
    typedef typename CwiseUnaryOpImpl::Scalar Scalar;
    typedef typename CwiseUnaryOpImpl<UnaryOp,MatrixType,Sparse>::MatrixTypeReverseIterator Base;
  public:

    EIGEN_STRONG_INLINE ReverseInnerIterator(const CwiseUnaryOpImpl& unaryOp, typename CwiseUnaryOpImpl::Index outer)
      : Base(unaryOp.derived().nestedExpression(),outer), m_functor(unaryOp.derived().functor())
    {}

    EIGEN_STRONG_INLINE ReverseInnerIterator& operator--()
    { Base::operator--(); return *this; }

    EIGEN_STRONG_INLINE typename CwiseUnaryOpImpl::Scalar value() const { return m_functor(Base::value()); }

  protected:
    const UnaryOp m_functor;
  private:
    typename CwiseUnaryOpImpl::Scalar& valueRef();
};

template<typename ViewOp, typename MatrixType>
class CwiseUnaryViewImpl<ViewOp,MatrixType,Sparse>
  : public SparseMatrixBase<CwiseUnaryView<ViewOp, MatrixType> >
{
  public:

    class InnerIterator;
    class ReverseInnerIterator;

    typedef CwiseUnaryView<ViewOp, MatrixType> Derived;
    EIGEN_SPARSE_PUBLIC_INTERFACE(Derived)

  protected:
    typedef typename internal::traits<Derived>::_MatrixTypeNested _MatrixTypeNested;
    typedef typename _MatrixTypeNested::InnerIterator MatrixTypeIterator;
    typedef typename _MatrixTypeNested::ReverseInnerIterator MatrixTypeReverseIterator;
};

template<typename ViewOp, typename MatrixType>
class CwiseUnaryViewImpl<ViewOp,MatrixType,Sparse>::InnerIterator
    : public CwiseUnaryViewImpl<ViewOp,MatrixType,Sparse>::MatrixTypeIterator
{
    typedef typename CwiseUnaryViewImpl::Scalar Scalar;
    typedef typename CwiseUnaryViewImpl<ViewOp,MatrixType,Sparse>::MatrixTypeIterator Base;
  public:

    EIGEN_STRONG_INLINE InnerIterator(const CwiseUnaryViewImpl& unaryOp, typename CwiseUnaryViewImpl::Index outer)
      : Base(unaryOp.derived().nestedExpression(),outer), m_functor(unaryOp.derived().functor())
    {}

    EIGEN_STRONG_INLINE InnerIterator& operator++()
    { Base::operator++(); return *this; }

    EIGEN_STRONG_INLINE typename CwiseUnaryViewImpl::Scalar value() const { return m_functor(Base::value()); }
    EIGEN_STRONG_INLINE typename CwiseUnaryViewImpl::Scalar& valueRef() { return m_functor(Base::valueRef()); }

  protected:
    const ViewOp m_functor;
};

template<typename ViewOp, typename MatrixType>
class CwiseUnaryViewImpl<ViewOp,MatrixType,Sparse>::ReverseInnerIterator
    : public CwiseUnaryViewImpl<ViewOp,MatrixType,Sparse>::MatrixTypeReverseIterator
{
    typedef typename CwiseUnaryViewImpl::Scalar Scalar;
    typedef typename CwiseUnaryViewImpl<ViewOp,MatrixType,Sparse>::MatrixTypeReverseIterator Base;
  public:

    EIGEN_STRONG_INLINE ReverseInnerIterator(const CwiseUnaryViewImpl& unaryOp, typename CwiseUnaryViewImpl::Index outer)
      : Base(unaryOp.derived().nestedExpression(),outer), m_functor(unaryOp.derived().functor())
    {}

    EIGEN_STRONG_INLINE ReverseInnerIterator& operator--()
    { Base::operator--(); return *this; }

    EIGEN_STRONG_INLINE typename CwiseUnaryViewImpl::Scalar value() const { return m_functor(Base::value()); }
    EIGEN_STRONG_INLINE typename CwiseUnaryViewImpl::Scalar& valueRef() { return m_functor(Base::valueRef()); }

  protected:
    const ViewOp m_functor;
};

#else // EIGEN_TEST_EVALUATORS

namespace internal {
  
template<typename UnaryOp, typename ArgType>
struct unary_evaluator<CwiseUnaryOp<UnaryOp,ArgType>, IteratorBased>
  : public evaluator_base<CwiseUnaryOp<UnaryOp,ArgType> >
{
  public:
    typedef CwiseUnaryOp<UnaryOp, ArgType> XprType;

    class InnerIterator;
    class ReverseInnerIterator;
    
    enum {
      CoeffReadCost = evaluator<ArgType>::CoeffReadCost + functor_traits<UnaryOp>::Cost,
      Flags = XprType::Flags
    };
    
    unary_evaluator(const XprType& op) : m_functor(op.functor()), m_argImpl(op.nestedExpression()) {}

  protected:
    typedef typename evaluator<ArgType>::InnerIterator        EvalIterator;
    typedef typename evaluator<ArgType>::ReverseInnerIterator EvalReverseIterator;
    
    const UnaryOp m_functor;
    typename evaluator<ArgType>::nestedType m_argImpl;
};

template<typename UnaryOp, typename ArgType>
class unary_evaluator<CwiseUnaryOp<UnaryOp,ArgType>, IteratorBased>::InnerIterator
    : public unary_evaluator<CwiseUnaryOp<UnaryOp,ArgType>, IteratorBased>::EvalIterator
{
    typedef typename XprType::Scalar Scalar;
    typedef typename unary_evaluator<CwiseUnaryOp<UnaryOp,ArgType>, IteratorBased>::EvalIterator Base;
  public:

    EIGEN_STRONG_INLINE InnerIterator(const unary_evaluator& unaryOp, typename XprType::Index outer)
      : Base(unaryOp.m_argImpl,outer), m_functor(unaryOp.m_functor)
    {}

    EIGEN_STRONG_INLINE InnerIterator& operator++()
    { Base::operator++(); return *this; }

    EIGEN_STRONG_INLINE Scalar value() const { return m_functor(Base::value()); }

  protected:
    const UnaryOp m_functor;
  private:
    Scalar& valueRef();
};

template<typename UnaryOp, typename ArgType>
class unary_evaluator<CwiseUnaryOp<UnaryOp,ArgType>, IteratorBased>::ReverseInnerIterator
    : public unary_evaluator<CwiseUnaryOp<UnaryOp,ArgType>, IteratorBased>::EvalReverseIterator
{
    typedef typename XprType::Scalar Scalar;
    typedef typename unary_evaluator<CwiseUnaryOp<UnaryOp,ArgType>, IteratorBased>::EvalReverseIterator Base;
  public:

    EIGEN_STRONG_INLINE ReverseInnerIterator(const XprType& unaryOp, typename XprType::Index outer)
      : Base(unaryOp.derived().nestedExpression(),outer), m_functor(unaryOp.derived().functor())
    {}

    EIGEN_STRONG_INLINE ReverseInnerIterator& operator--()
    { Base::operator--(); return *this; }

    EIGEN_STRONG_INLINE Scalar value() const { return m_functor(Base::value()); }

  protected:
    const UnaryOp m_functor;
  private:
    Scalar& valueRef();
};


// template<typename ViewOp, typename MatrixType>
// class CwiseUnaryViewImpl<ViewOp,MatrixType,Sparse>
//   : public SparseMatrixBase<CwiseUnaryView<ViewOp, MatrixType> >
// {
//   public:
// 
//     class InnerIterator;
//     class ReverseInnerIterator;
// 
//     typedef CwiseUnaryView<ViewOp, MatrixType> Derived;
//     EIGEN_SPARSE_PUBLIC_INTERFACE(Derived)
// 
//   protected:
//     typedef typename internal::traits<Derived>::_MatrixTypeNested _MatrixTypeNested;
//     typedef typename _MatrixTypeNested::InnerIterator MatrixTypeIterator;
//     typedef typename _MatrixTypeNested::ReverseInnerIterator MatrixTypeReverseIterator;
// };
// 
// template<typename ViewOp, typename MatrixType>
// class CwiseUnaryViewImpl<ViewOp,MatrixType,Sparse>::InnerIterator
//     : public CwiseUnaryViewImpl<ViewOp,MatrixType,Sparse>::MatrixTypeIterator
// {
//     typedef typename CwiseUnaryViewImpl::Scalar Scalar;
//     typedef typename CwiseUnaryViewImpl<ViewOp,MatrixType,Sparse>::MatrixTypeIterator Base;
//   public:
// 
//     EIGEN_STRONG_INLINE InnerIterator(const CwiseUnaryViewImpl& unaryOp, typename CwiseUnaryViewImpl::Index outer)
//       : Base(unaryOp.derived().nestedExpression(),outer), m_functor(unaryOp.derived().functor())
//     {}
// 
//     EIGEN_STRONG_INLINE InnerIterator& operator++()
//     { Base::operator++(); return *this; }
// 
//     EIGEN_STRONG_INLINE typename CwiseUnaryViewImpl::Scalar value() const { return m_functor(Base::value()); }
//     EIGEN_STRONG_INLINE typename CwiseUnaryViewImpl::Scalar& valueRef() { return m_functor(Base::valueRef()); }
// 
//   protected:
//     const ViewOp m_functor;
// };
// 
// template<typename ViewOp, typename MatrixType>
// class CwiseUnaryViewImpl<ViewOp,MatrixType,Sparse>::ReverseInnerIterator
//     : public CwiseUnaryViewImpl<ViewOp,MatrixType,Sparse>::MatrixTypeReverseIterator
// {
//     typedef typename CwiseUnaryViewImpl::Scalar Scalar;
//     typedef typename CwiseUnaryViewImpl<ViewOp,MatrixType,Sparse>::MatrixTypeReverseIterator Base;
//   public:
// 
//     EIGEN_STRONG_INLINE ReverseInnerIterator(const CwiseUnaryViewImpl& unaryOp, typename CwiseUnaryViewImpl::Index outer)
//       : Base(unaryOp.derived().nestedExpression(),outer), m_functor(unaryOp.derived().functor())
//     {}
// 
//     EIGEN_STRONG_INLINE ReverseInnerIterator& operator--()
//     { Base::operator--(); return *this; }
// 
//     EIGEN_STRONG_INLINE typename CwiseUnaryViewImpl::Scalar value() const { return m_functor(Base::value()); }
//     EIGEN_STRONG_INLINE typename CwiseUnaryViewImpl::Scalar& valueRef() { return m_functor(Base::valueRef()); }
// 
//   protected:
//     const ViewOp m_functor;
// };

} // end namespace internal

#endif // EIGEN_TEST_EVALUATORS

template<typename Derived>
EIGEN_STRONG_INLINE Derived&
SparseMatrixBase<Derived>::operator*=(const Scalar& other)
{
  for (Index j=0; j<outerSize(); ++j)
    for (typename Derived::InnerIterator i(derived(),j); i; ++i)
      i.valueRef() *= other;
  return derived();
}

template<typename Derived>
EIGEN_STRONG_INLINE Derived&
SparseMatrixBase<Derived>::operator/=(const Scalar& other)
{
  for (Index j=0; j<outerSize(); ++j)
    for (typename Derived::InnerIterator i(derived(),j); i; ++i)
      i.valueRef() /= other;
  return derived();
}

} // end namespace Eigen

#endif // EIGEN_SPARSE_CWISE_UNARY_OP_H
