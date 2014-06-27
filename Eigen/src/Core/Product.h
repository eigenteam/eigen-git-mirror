// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2011 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_PRODUCT_H
#define EIGEN_PRODUCT_H

namespace Eigen {

template<typename Lhs, typename Rhs, int Option, typename StorageKind> class ProductImpl;

/** \class Product
  * \ingroup Core_Module
  *
  * \brief Expression of the product of two arbitrary matrices or vectors
  *
  * \param Lhs the type of the left-hand side expression
  * \param Rhs the type of the right-hand side expression
  *
  * This class represents an expression of the product of two arbitrary matrices.
  * 
  * The other template parameters are:
  * \tparam Option     can be DefaultProduct or LazyProduct
  *
  */


namespace internal {

// Determine the scalar of Product<Lhs, Rhs>. This is normally the same as Lhs::Scalar times
// Rhs::Scalar, but product with permutation matrices inherit the scalar of the other factor.
template<typename Lhs, typename Rhs, typename LhsShape = typename evaluator_traits<Lhs>::Shape, 
         typename RhsShape = typename evaluator_traits<Rhs>::Shape >
struct product_result_scalar
{
  typedef typename scalar_product_traits<typename Lhs::Scalar, typename Rhs::Scalar>::ReturnType Scalar;
};

template<typename Lhs, typename Rhs, typename RhsShape>
struct product_result_scalar<Lhs, Rhs, PermutationShape, RhsShape>
{
  typedef typename Rhs::Scalar Scalar;
};

template<typename Lhs, typename Rhs, typename LhsShape>
  struct product_result_scalar<Lhs, Rhs, LhsShape, PermutationShape>
{
  typedef typename Lhs::Scalar Scalar;
};

template<typename Lhs, typename Rhs, int Option>
struct traits<Product<Lhs, Rhs, Option> >
{
  typedef typename remove_all<Lhs>::type LhsCleaned;
  typedef typename remove_all<Rhs>::type RhsCleaned;
  
  typedef MatrixXpr XprKind;
  
  typedef typename product_result_scalar<LhsCleaned,RhsCleaned>::Scalar Scalar;
  typedef typename promote_storage_type<typename traits<LhsCleaned>::StorageKind,
                                           typename traits<RhsCleaned>::StorageKind>::ret StorageKind;
  typedef typename promote_index_type<typename traits<LhsCleaned>::Index,
                                         typename traits<RhsCleaned>::Index>::type Index;
  
  enum {
    RowsAtCompileTime    = LhsCleaned::RowsAtCompileTime,
    ColsAtCompileTime    = RhsCleaned::ColsAtCompileTime,
    MaxRowsAtCompileTime = LhsCleaned::MaxRowsAtCompileTime,
    MaxColsAtCompileTime = RhsCleaned::MaxColsAtCompileTime,
    
    // FIXME: only needed by GeneralMatrixMatrixTriangular
    InnerSize = EIGEN_SIZE_MIN_PREFER_FIXED(LhsCleaned::ColsAtCompileTime, RhsCleaned::RowsAtCompileTime),
    
#ifndef EIGEN_TEST_EVALUATORS
    // dummy, for evaluators unit test only
    CoeffReadCost = Dynamic,
#endif
    
    // The storage order is somewhat arbitrary here. The correct one will be determined through the evaluator.
    Flags = (   MaxRowsAtCompileTime==1
             || ((LhsCleaned::Flags&NoPreferredStorageOrderBit) && (RhsCleaned::Flags&RowMajorBit))
             || ((RhsCleaned::Flags&NoPreferredStorageOrderBit) && (LhsCleaned::Flags&RowMajorBit)) )
          ? RowMajorBit : (MaxColsAtCompileTime==1 ? 0 : NoPreferredStorageOrderBit)
  };
};

} // end namespace internal


template<typename _Lhs, typename _Rhs, int Option>
class Product : public ProductImpl<_Lhs,_Rhs,Option,
                                   typename internal::promote_storage_type<typename internal::traits<_Lhs>::StorageKind,
                                                                           typename internal::traits<_Rhs>::StorageKind>::ret>
{
  public:
    
    typedef _Lhs Lhs;
    typedef _Rhs Rhs;
    
    typedef typename ProductImpl<
        Lhs, Rhs, Option,
        typename internal::promote_storage_type<typename Lhs::StorageKind,
                                                typename Rhs::StorageKind>::ret>::Base Base;
    EIGEN_GENERIC_PUBLIC_INTERFACE(Product)

    typedef typename internal::nested<Lhs>::type LhsNested;
    typedef typename internal::nested<Rhs>::type RhsNested;
    typedef typename internal::remove_all<LhsNested>::type LhsNestedCleaned;
    typedef typename internal::remove_all<RhsNested>::type RhsNestedCleaned;

    Product(const Lhs& lhs, const Rhs& rhs) : m_lhs(lhs), m_rhs(rhs)
    {
      eigen_assert(lhs.cols() == rhs.rows()
        && "invalid matrix product"
        && "if you wanted a coeff-wise or a dot product use the respective explicit functions");
    }

    inline Index rows() const { return m_lhs.rows(); }
    inline Index cols() const { return m_rhs.cols(); }

    const LhsNestedCleaned& lhs() const { return m_lhs; }
    const RhsNestedCleaned& rhs() const { return m_rhs; }

  protected:

    LhsNested m_lhs;
    RhsNested m_rhs;
};

namespace internal {
  
template<typename Lhs, typename Rhs, int Option, int ProductTag = internal::product_type<Lhs,Rhs>::ret>
class dense_product_base
 : public internal::dense_xpr_base<Product<Lhs,Rhs,Option> >::type
{};

/** Convertion to scalar for inner-products */
template<typename Lhs, typename Rhs, int Option>
class dense_product_base<Lhs, Rhs, Option, InnerProduct>
 : public internal::dense_xpr_base<Product<Lhs,Rhs,Option> >::type
{
  typedef Product<Lhs,Rhs,Option> ProductXpr;
  typedef typename internal::dense_xpr_base<ProductXpr>::type Base;
public:
  using Base::derived;
  typedef typename Base::Scalar Scalar;
  typedef typename Base::Index Index;
  
  operator const Scalar() const
  {
    return typename internal::evaluator<ProductXpr>::type(derived()).coeff(0,0);
  }
};

} // namespace internal

#ifdef EIGEN_TEST_EVALUATORS
// Generic API dispatcher
template<typename Lhs, typename Rhs, int Option, typename StorageKind>
class ProductImpl : public internal::generic_xpr_base<Product<Lhs,Rhs,Option>, MatrixXpr, StorageKind>::type
{
  public:
    typedef typename internal::generic_xpr_base<Product<Lhs,Rhs,Option>, MatrixXpr, StorageKind>::type Base;
};
#endif

template<typename Lhs, typename Rhs, int Option>
class ProductImpl<Lhs,Rhs,Option,Dense>
  : public internal::dense_product_base<Lhs,Rhs,Option>
{
    typedef Product<Lhs, Rhs, Option> Derived;
    
  public:
    
    typedef typename internal::dense_product_base<Lhs, Rhs, Option> Base;
    EIGEN_DENSE_PUBLIC_INTERFACE(Derived)
  protected:
    enum {
      IsOneByOne = (RowsAtCompileTime == 1 || RowsAtCompileTime == Dynamic) && 
                   (ColsAtCompileTime == 1 || ColsAtCompileTime == Dynamic),
      EnableCoeff = IsOneByOne || Option==LazyProduct
    };
    
  public:
  
    Scalar coeff(Index row, Index col) const
    {
      EIGEN_STATIC_ASSERT(EnableCoeff, THIS_METHOD_IS_ONLY_FOR_INNER_OR_LAZY_PRODUCTS);
      eigen_assert( (Option==LazyProduct) || (this->rows() == 1 && this->cols() == 1) );
      
      return typename internal::evaluator<Derived>::type(derived()).coeff(row,col);
    }

    Scalar coeff(Index i) const
    {
      EIGEN_STATIC_ASSERT(EnableCoeff, THIS_METHOD_IS_ONLY_FOR_INNER_OR_LAZY_PRODUCTS);
      eigen_assert( (Option==LazyProduct) || (this->rows() == 1 && this->cols() == 1) );
      
      return typename internal::evaluator<Derived>::type(derived()).coeff(i);
    }
    
  
};

/***************************************************************************
* Implementation of matrix base methods
***************************************************************************/


/** \internal used to test the evaluator only
  */
template<typename Lhs,typename Rhs>
const Product<Lhs,Rhs>
prod(const Lhs& lhs, const Rhs& rhs)
{
  return Product<Lhs,Rhs>(lhs,rhs);
}

/** \internal used to test the evaluator only
  */
template<typename Lhs,typename Rhs>
const Product<Lhs,Rhs,LazyProduct>
lazyprod(const Lhs& lhs, const Rhs& rhs)
{
  return Product<Lhs,Rhs,LazyProduct>(lhs,rhs);
}

} // end namespace Eigen

#endif // EIGEN_PRODUCT_H
