// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2014 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_SPARSEDENSEPRODUCT_H
#define EIGEN_SPARSEDENSEPRODUCT_H

namespace Eigen { 

namespace internal {

template <> struct product_promote_storage_type<Sparse,Dense, OuterProduct> { typedef Sparse ret; };
template <> struct product_promote_storage_type<Dense,Sparse, OuterProduct> { typedef Sparse ret; };

template<typename SparseLhsType, typename DenseRhsType, typename DenseResType,
         typename AlphaType,
         int LhsStorageOrder = ((SparseLhsType::Flags&RowMajorBit)==RowMajorBit) ? RowMajor : ColMajor,
         bool ColPerCol = ((DenseRhsType::Flags&RowMajorBit)==0) || DenseRhsType::ColsAtCompileTime==1>
struct sparse_time_dense_product_impl;

template<typename SparseLhsType, typename DenseRhsType, typename DenseResType>
struct sparse_time_dense_product_impl<SparseLhsType,DenseRhsType,DenseResType, typename DenseResType::Scalar, RowMajor, true>
{
  typedef typename internal::remove_all<SparseLhsType>::type Lhs;
  typedef typename internal::remove_all<DenseRhsType>::type Rhs;
  typedef typename internal::remove_all<DenseResType>::type Res;
  typedef typename Lhs::Index Index;
#ifndef EIGEN_TEST_EVALUATORS
  typedef typename Lhs::InnerIterator LhsInnerIterator;
#else
  typedef typename evaluator<Lhs>::InnerIterator LhsInnerIterator;
#endif
  static void run(const SparseLhsType& lhs, const DenseRhsType& rhs, DenseResType& res, const typename Res::Scalar& alpha)
  {
#ifndef EIGEN_TEST_EVALUATORS
  const Lhs &lhsEval(lhs);
#else
  typename evaluator<Lhs>::type lhsEval(lhs);
#endif
    for(Index c=0; c<rhs.cols(); ++c)
    {
      Index n = lhs.outerSize();
      for(Index j=0; j<n; ++j)
      {
        typename Res::Scalar tmp(0);
        for(LhsInnerIterator it(lhsEval,j); it ;++it)
          tmp += it.value() * rhs.coeff(it.index(),c);
        res.coeffRef(j,c) = alpha * tmp;
      }
    }
  }
};

template<typename T1, typename T2/*, int _Options, typename _StrideType*/>
struct scalar_product_traits<T1, Ref<T2/*, _Options, _StrideType*/> >
{
  enum {
    Defined = 1
  };
  typedef typename CwiseUnaryOp<scalar_multiple2_op<T1, typename T2::Scalar>, T2>::PlainObject ReturnType;
};
template<typename SparseLhsType, typename DenseRhsType, typename DenseResType, typename AlphaType>
struct sparse_time_dense_product_impl<SparseLhsType,DenseRhsType,DenseResType, AlphaType, ColMajor, true>
{
  typedef typename internal::remove_all<SparseLhsType>::type Lhs;
  typedef typename internal::remove_all<DenseRhsType>::type Rhs;
  typedef typename internal::remove_all<DenseResType>::type Res;
  typedef typename Lhs::Index Index;
#ifndef EIGEN_TEST_EVALUATORS
  typedef typename Lhs::InnerIterator LhsInnerIterator;
#else
  typedef typename evaluator<Lhs>::InnerIterator LhsInnerIterator;
#endif
  static void run(const SparseLhsType& lhs, const DenseRhsType& rhs, DenseResType& res, const AlphaType& alpha)
  {
#ifndef EIGEN_TEST_EVALUATORS
  const Lhs &lhsEval(lhs);
#else
  typename evaluator<Lhs>::type lhsEval(lhs);
#endif
    for(Index c=0; c<rhs.cols(); ++c)
    {
      for(Index j=0; j<lhs.outerSize(); ++j)
      {
//        typename Res::Scalar rhs_j = alpha * rhs.coeff(j,c);
        typename internal::scalar_product_traits<AlphaType, typename Rhs::Scalar>::ReturnType rhs_j(alpha * rhs.coeff(j,c));
        for(LhsInnerIterator it(lhsEval,j); it ;++it)
          res.coeffRef(it.index(),c) += it.value() * rhs_j;
      }
    }
  }
};

template<typename SparseLhsType, typename DenseRhsType, typename DenseResType>
struct sparse_time_dense_product_impl<SparseLhsType,DenseRhsType,DenseResType, typename DenseResType::Scalar, RowMajor, false>
{
  typedef typename internal::remove_all<SparseLhsType>::type Lhs;
  typedef typename internal::remove_all<DenseRhsType>::type Rhs;
  typedef typename internal::remove_all<DenseResType>::type Res;
  typedef typename Lhs::Index Index;
#ifndef EIGEN_TEST_EVALUATORS
  typedef typename Lhs::InnerIterator LhsInnerIterator;
#else
  typedef typename evaluator<Lhs>::InnerIterator LhsInnerIterator;
#endif
  static void run(const SparseLhsType& lhs, const DenseRhsType& rhs, DenseResType& res, const typename Res::Scalar& alpha)
  {
#ifndef EIGEN_TEST_EVALUATORS
  const Lhs &lhsEval(lhs);
#else
  typename evaluator<Lhs>::type lhsEval(lhs);
#endif
    for(Index j=0; j<lhs.outerSize(); ++j)
    {
      typename Res::RowXpr res_j(res.row(j));
      for(LhsInnerIterator it(lhsEval,j); it ;++it)
        res_j += (alpha*it.value()) * rhs.row(it.index());
    }
  }
};

template<typename SparseLhsType, typename DenseRhsType, typename DenseResType>
struct sparse_time_dense_product_impl<SparseLhsType,DenseRhsType,DenseResType, typename DenseResType::Scalar, ColMajor, false>
{
  typedef typename internal::remove_all<SparseLhsType>::type Lhs;
  typedef typename internal::remove_all<DenseRhsType>::type Rhs;
  typedef typename internal::remove_all<DenseResType>::type Res;
  typedef typename Lhs::Index Index;
#ifndef EIGEN_TEST_EVALUATORS
  typedef typename Lhs::InnerIterator LhsInnerIterator;
#else
  typedef typename evaluator<Lhs>::InnerIterator LhsInnerIterator;
#endif
  static void run(const SparseLhsType& lhs, const DenseRhsType& rhs, DenseResType& res, const typename Res::Scalar& alpha)
  {
#ifndef EIGEN_TEST_EVALUATORS
  const Lhs &lhsEval(lhs);
#else
  typename evaluator<Lhs>::type lhsEval(lhs);
#endif
    for(Index j=0; j<lhs.outerSize(); ++j)
    {
      typename Rhs::ConstRowXpr rhs_j(rhs.row(j));
      for(LhsInnerIterator it(lhsEval,j); it ;++it)
        res.row(it.index()) += (alpha*it.value()) * rhs_j;
    }
  }
};

template<typename SparseLhsType, typename DenseRhsType, typename DenseResType,typename AlphaType>
inline void sparse_time_dense_product(const SparseLhsType& lhs, const DenseRhsType& rhs, DenseResType& res, const AlphaType& alpha)
{
  sparse_time_dense_product_impl<SparseLhsType,DenseRhsType,DenseResType, AlphaType>::run(lhs, rhs, res, alpha);
}

} // end namespace internal

#ifndef EIGEN_TEST_EVALUATORS
template<typename Lhs, typename Rhs, int InnerSize> struct SparseDenseProductReturnType
{
  typedef SparseTimeDenseProduct<Lhs,Rhs> Type;
};

template<typename Lhs, typename Rhs> struct SparseDenseProductReturnType<Lhs,Rhs,1>
{
  typedef typename internal::conditional<
    Lhs::IsRowMajor,
    SparseDenseOuterProduct<Rhs,Lhs,true>,
    SparseDenseOuterProduct<Lhs,Rhs,false> >::type Type;
};

template<typename Lhs, typename Rhs, int InnerSize> struct DenseSparseProductReturnType
{
  typedef DenseTimeSparseProduct<Lhs,Rhs> Type;
};

template<typename Lhs, typename Rhs> struct DenseSparseProductReturnType<Lhs,Rhs,1>
{
  typedef typename internal::conditional<
    Rhs::IsRowMajor,
    SparseDenseOuterProduct<Rhs,Lhs,true>,
    SparseDenseOuterProduct<Lhs,Rhs,false> >::type Type;
};

namespace internal {

template<typename Lhs, typename Rhs, bool Tr>
struct traits<SparseDenseOuterProduct<Lhs,Rhs,Tr> >
{
  typedef Sparse StorageKind;
  typedef typename scalar_product_traits<typename traits<Lhs>::Scalar,
                                         typename traits<Rhs>::Scalar>::ReturnType Scalar;
  typedef typename Lhs::Index Index;
  typedef typename Lhs::Nested LhsNested;
  typedef typename Rhs::Nested RhsNested;
  typedef typename remove_all<LhsNested>::type _LhsNested;
  typedef typename remove_all<RhsNested>::type _RhsNested;

  enum {
    LhsCoeffReadCost = traits<_LhsNested>::CoeffReadCost,
    RhsCoeffReadCost = traits<_RhsNested>::CoeffReadCost,

    RowsAtCompileTime    = Tr ? int(traits<Rhs>::RowsAtCompileTime)     : int(traits<Lhs>::RowsAtCompileTime),
    ColsAtCompileTime    = Tr ? int(traits<Lhs>::ColsAtCompileTime)     : int(traits<Rhs>::ColsAtCompileTime),
    MaxRowsAtCompileTime = Tr ? int(traits<Rhs>::MaxRowsAtCompileTime)  : int(traits<Lhs>::MaxRowsAtCompileTime),
    MaxColsAtCompileTime = Tr ? int(traits<Lhs>::MaxColsAtCompileTime)  : int(traits<Rhs>::MaxColsAtCompileTime),

    Flags = Tr ? RowMajorBit : 0,

    CoeffReadCost = LhsCoeffReadCost + RhsCoeffReadCost + NumTraits<Scalar>::MulCost
  };
};

} // end namespace internal

template<typename Lhs, typename Rhs, bool Tr>
class SparseDenseOuterProduct
 : public SparseMatrixBase<SparseDenseOuterProduct<Lhs,Rhs,Tr> >
{
  public:

    typedef SparseMatrixBase<SparseDenseOuterProduct> Base;
    EIGEN_DENSE_PUBLIC_INTERFACE(SparseDenseOuterProduct)
    typedef internal::traits<SparseDenseOuterProduct> Traits;

  private:

    typedef typename Traits::LhsNested LhsNested;
    typedef typename Traits::RhsNested RhsNested;
    typedef typename Traits::_LhsNested _LhsNested;
    typedef typename Traits::_RhsNested _RhsNested;

  public:

    class InnerIterator;

    EIGEN_STRONG_INLINE SparseDenseOuterProduct(const Lhs& lhs, const Rhs& rhs)
      : m_lhs(lhs), m_rhs(rhs)
    {
      EIGEN_STATIC_ASSERT(!Tr,YOU_MADE_A_PROGRAMMING_MISTAKE);
    }

    EIGEN_STRONG_INLINE SparseDenseOuterProduct(const Rhs& rhs, const Lhs& lhs)
      : m_lhs(lhs), m_rhs(rhs)
    {
      EIGEN_STATIC_ASSERT(Tr,YOU_MADE_A_PROGRAMMING_MISTAKE);
    }

    EIGEN_STRONG_INLINE Index rows() const { return Tr ? Index(m_rhs.rows()) : m_lhs.rows(); }
    EIGEN_STRONG_INLINE Index cols() const { return Tr ? m_lhs.cols() : Index(m_rhs.cols()); }

    EIGEN_STRONG_INLINE const _LhsNested& lhs() const { return m_lhs; }
    EIGEN_STRONG_INLINE const _RhsNested& rhs() const { return m_rhs; }

  protected:
    LhsNested m_lhs;
    RhsNested m_rhs;
};

template<typename Lhs, typename Rhs, bool Transpose>
class SparseDenseOuterProduct<Lhs,Rhs,Transpose>::InnerIterator : public _LhsNested::InnerIterator
{
    typedef typename _LhsNested::InnerIterator Base;
    typedef typename SparseDenseOuterProduct::Index Index;
  public:
    EIGEN_STRONG_INLINE InnerIterator(const SparseDenseOuterProduct& prod, Index outer)
      : Base(prod.lhs(), 0), m_outer(outer), m_empty(false), m_factor(get(prod.rhs(), outer, typename internal::traits<Rhs>::StorageKind() ))
    {}

    inline Index outer() const { return m_outer; }
    inline Index row() const { return Transpose ? m_outer : Base::index(); }
    inline Index col() const { return Transpose ? Base::index() : m_outer; }

    inline Scalar value() const { return Base::value() * m_factor; }
    inline operator bool() const { return Base::operator bool() && !m_empty; }

  protected:
    Scalar get(const _RhsNested &rhs, Index outer, Dense = Dense()) const
    {
      return rhs.coeff(outer);
    }
    
    Scalar get(const _RhsNested &rhs, Index outer, Sparse = Sparse())
    {
      typename Traits::_RhsNested::InnerIterator it(rhs, outer);
      if (it && it.index()==0 && it.value()!=Scalar(0))
        return it.value();
      m_empty = true;
      return Scalar(0);
    }
    
    Index m_outer;
    bool m_empty;
    Scalar m_factor;
};

namespace internal {
template<typename Lhs, typename Rhs>
struct traits<SparseTimeDenseProduct<Lhs,Rhs> >
 : traits<ProductBase<SparseTimeDenseProduct<Lhs,Rhs>, Lhs, Rhs> >
{
  typedef Dense StorageKind;
  typedef MatrixXpr XprKind;
};

} // end namespace internal

template<typename Lhs, typename Rhs>
class SparseTimeDenseProduct
  : public ProductBase<SparseTimeDenseProduct<Lhs,Rhs>, Lhs, Rhs>
{
  public:
    EIGEN_PRODUCT_PUBLIC_INTERFACE(SparseTimeDenseProduct)

    SparseTimeDenseProduct(const Lhs& lhs, const Rhs& rhs) : Base(lhs,rhs)
    {}

    template<typename Dest> void scaleAndAddTo(Dest& dest, const Scalar& alpha) const
    {
      internal::sparse_time_dense_product(m_lhs, m_rhs, dest, alpha);
    }

  private:
    SparseTimeDenseProduct& operator=(const SparseTimeDenseProduct&);
};


// dense = dense * sparse
namespace internal {
template<typename Lhs, typename Rhs>
struct traits<DenseTimeSparseProduct<Lhs,Rhs> >
 : traits<ProductBase<DenseTimeSparseProduct<Lhs,Rhs>, Lhs, Rhs> >
{
  typedef Dense StorageKind;
};
} // end namespace internal

template<typename Lhs, typename Rhs>
class DenseTimeSparseProduct
  : public ProductBase<DenseTimeSparseProduct<Lhs,Rhs>, Lhs, Rhs>
{
  public:
    EIGEN_PRODUCT_PUBLIC_INTERFACE(DenseTimeSparseProduct)

    DenseTimeSparseProduct(const Lhs& lhs, const Rhs& rhs) : Base(lhs,rhs)
    {}

    template<typename Dest> void scaleAndAddTo(Dest& dest, const Scalar& alpha) const
    {
      Transpose<const _LhsNested> lhs_t(m_lhs);
      Transpose<const _RhsNested> rhs_t(m_rhs);
      Transpose<Dest> dest_t(dest);
      internal::sparse_time_dense_product(rhs_t, lhs_t, dest_t, alpha);
    }

  private:
    DenseTimeSparseProduct& operator=(const DenseTimeSparseProduct&);
};

// sparse * dense
template<typename Derived>
template<typename OtherDerived>
inline const typename SparseDenseProductReturnType<Derived,OtherDerived>::Type
SparseMatrixBase<Derived>::operator*(const MatrixBase<OtherDerived> &other) const
{
  return typename SparseDenseProductReturnType<Derived,OtherDerived>::Type(derived(), other.derived());
}
#endif // EIGEN_TEST_EVALUATORS

#ifdef EIGEN_TEST_EVALUATORS

namespace internal {

template<typename Lhs, typename Rhs, int ProductType>
struct generic_product_impl<Lhs, Rhs, SparseShape, DenseShape, ProductType>
{
  template<typename Dest>
  static void evalTo(Dest& dst, const Lhs& lhs, const Rhs& rhs)
  {
    typedef typename nested_eval<Lhs,Dynamic>::type LhsNested;
    typedef typename nested_eval<Rhs,Dynamic>::type RhsNested;
    LhsNested lhsNested(lhs);
    RhsNested rhsNested(rhs);
    
    dst.setZero();
    internal::sparse_time_dense_product(lhsNested, rhsNested, dst, typename Dest::Scalar(1));
  }
};

template<typename Lhs, typename Rhs, int ProductType>
struct generic_product_impl<Lhs, Rhs, DenseShape, SparseShape, ProductType>
{
  template<typename Dest>
  static void evalTo(Dest& dst, const Lhs& lhs, const Rhs& rhs)
  {
    typedef typename nested_eval<Lhs,Dynamic>::type LhsNested;
    typedef typename nested_eval<Rhs,Dynamic>::type RhsNested;
    LhsNested lhsNested(lhs);
    RhsNested rhsNested(rhs);
    
    dst.setZero();
    // transpoe everything
    Transpose<Dest> dstT(dst);
    internal::sparse_time_dense_product(rhsNested.transpose(), lhsNested.transpose(), dstT, typename Dest::Scalar(1));
  }
};

template<typename Lhs, typename Rhs, int ProductTag>
struct product_evaluator<Product<Lhs, Rhs, DefaultProduct>, ProductTag, SparseShape, DenseShape, typename Lhs::Scalar, typename Rhs::Scalar> 
  : public evaluator<typename Product<Lhs, Rhs, DefaultProduct>::PlainObject>::type
{
  typedef Product<Lhs, Rhs, DefaultProduct> XprType;
  typedef typename XprType::PlainObject PlainObject;
  typedef typename evaluator<PlainObject>::type Base;

  product_evaluator(const XprType& xpr)
    : m_result(xpr.rows(), xpr.cols())
  {
    ::new (static_cast<Base*>(this)) Base(m_result);
    generic_product_impl<Lhs, Rhs, SparseShape, DenseShape, ProductTag>::evalTo(m_result, xpr.lhs(), xpr.rhs());
  }
  
protected:  
  PlainObject m_result;
};

template<typename Lhs, typename Rhs, int ProductTag>
struct product_evaluator<Product<Lhs, Rhs, DefaultProduct>, ProductTag, DenseShape, SparseShape, typename Lhs::Scalar, typename Rhs::Scalar> 
  : public evaluator<typename Product<Lhs, Rhs, DefaultProduct>::PlainObject>::type
{
  typedef Product<Lhs, Rhs, DefaultProduct> XprType;
  typedef typename XprType::PlainObject PlainObject;
  typedef typename evaluator<PlainObject>::type Base;

  product_evaluator(const XprType& xpr)
    : m_result(xpr.rows(), xpr.cols())
  {
    ::new (static_cast<Base*>(this)) Base(m_result);
    generic_product_impl<Lhs, Rhs, DenseShape, SparseShape, ProductTag>::evalTo(m_result, xpr.lhs(), xpr.rhs());
  }
  
protected:  
  PlainObject m_result;
};

template<typename LhsT, typename RhsT, bool Transpose>
struct sparse_dense_outer_product_evaluator
{
protected:
  typedef typename conditional<Transpose,RhsT,LhsT>::type Lhs1;
  typedef typename conditional<Transpose,LhsT,RhsT>::type Rhs;
  typedef Product<LhsT,RhsT> ProdXprType;
  
  // if the actual left-hand side is a dense vector,
  // then build a sparse-view so that we can seamlessly iterator over it.
  typedef typename conditional<is_same<typename internal::traits<Lhs1>::StorageKind,Sparse>::value,
            Lhs1, SparseView<Lhs1> >::type Lhs;
  typedef typename conditional<is_same<typename internal::traits<Lhs1>::StorageKind,Sparse>::value,
            Lhs1 const&, SparseView<Lhs1> >::type LhsArg;
            
  typedef typename evaluator<Lhs>::type LhsEval;
  typedef typename evaluator<Rhs>::type RhsEval;
  typedef typename evaluator<Lhs>::InnerIterator LhsIterator;
  typedef typename ProdXprType::Scalar Scalar;
  typedef typename ProdXprType::Index Index;
  
public:
  enum {
    Flags = Transpose ? RowMajorBit : 0,
    CoeffReadCost = Dynamic
  };
  
  class InnerIterator : public LhsIterator
  {
  public:
    InnerIterator(const sparse_dense_outer_product_evaluator &xprEval, Index outer)
      : LhsIterator(xprEval.m_lhsXprImpl, 0),
        m_outer(outer),
        m_empty(false),
        m_factor(get(xprEval.m_rhsXprImpl, outer, typename internal::traits<Rhs>::StorageKind() ))
    {}
    
    EIGEN_STRONG_INLINE Index outer() const { return m_outer; }
    EIGEN_STRONG_INLINE Index row()   const { return Transpose ? m_outer : LhsIterator::index(); }
    EIGEN_STRONG_INLINE Index col()   const { return Transpose ? LhsIterator::index() : m_outer; }

    EIGEN_STRONG_INLINE Scalar value() const { return LhsIterator::value() * m_factor; }
    EIGEN_STRONG_INLINE operator bool() const { return LhsIterator::operator bool() && (!m_empty); }
    
    
  protected:
    Scalar get(const RhsEval &rhs, Index outer, Dense = Dense()) const
    {
      return rhs.coeff(outer);
    }
    
    Scalar get(const RhsEval &rhs, Index outer, Sparse = Sparse())
    {
      typename RhsEval::InnerIterator it(rhs, outer);
      if (it && it.index()==0 && it.value()!=Scalar(0))
        return it.value();
      m_empty = true;
      return Scalar(0);
    }
    
    Index m_outer;
    bool m_empty;
    Scalar m_factor;
  };
  
  sparse_dense_outer_product_evaluator(const Lhs &lhs, const Rhs &rhs)
    : m_lhs(lhs), m_lhsXprImpl(m_lhs), m_rhsXprImpl(rhs)
  {}
  
  // transpose case
  sparse_dense_outer_product_evaluator(const Rhs &rhs, const Lhs1 &lhs)
    : m_lhs(lhs), m_lhsXprImpl(m_lhs), m_rhsXprImpl(rhs)
  {}
    
protected:
  const LhsArg m_lhs;
  typename evaluator<Lhs>::nestedType m_lhsXprImpl;
  typename evaluator<Rhs>::nestedType m_rhsXprImpl;
};

// sparse * dense outer product
template<typename Lhs, typename Rhs>
struct product_evaluator<Product<Lhs, Rhs, DefaultProduct>, OuterProduct, SparseShape, DenseShape, typename Lhs::Scalar, typename Rhs::Scalar> 
  : sparse_dense_outer_product_evaluator<Lhs,Rhs, Lhs::IsRowMajor>
{
  typedef sparse_dense_outer_product_evaluator<Lhs,Rhs, Lhs::IsRowMajor> Base;
  
  typedef Product<Lhs, Rhs> XprType;
  typedef typename XprType::PlainObject PlainObject;

  product_evaluator(const XprType& xpr)
    : Base(xpr.lhs(), xpr.rhs())
  {}
  
};

template<typename Lhs, typename Rhs>
struct product_evaluator<Product<Lhs, Rhs, DefaultProduct>, OuterProduct, DenseShape, SparseShape, typename Lhs::Scalar, typename Rhs::Scalar> 
  : sparse_dense_outer_product_evaluator<Lhs,Rhs, Rhs::IsRowMajor>
{
  typedef sparse_dense_outer_product_evaluator<Lhs,Rhs, Rhs::IsRowMajor> Base;
  
  typedef Product<Lhs, Rhs> XprType;
  typedef typename XprType::PlainObject PlainObject;

  product_evaluator(const XprType& xpr)
    : Base(xpr.lhs(), xpr.rhs())
  {}
  
};

} // end namespace internal

#endif // EIGEN_TEST_EVALUATORS

} // end namespace Eigen

#endif // EIGEN_SPARSEDENSEPRODUCT_H
