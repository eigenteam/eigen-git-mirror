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

// Use ProductReturnType to get correct traits, in particular vectorization flags
namespace internal {
template<typename Lhs, typename Rhs, int Option>
struct traits<Product<Lhs, Rhs, Option> >
  : traits<CoeffBasedProduct<Lhs, Rhs, NestByRefBit> >
{ 
  // We want A+B*C to be of type Product<Matrix, Sum> and not Product<Matrix, Matrix>
  // TODO: This flag should eventually go in a separate evaluator traits class
  enum {
    Flags = traits<CoeffBasedProduct<Lhs, Rhs, NestByRefBit> >::Flags & ~(EvalBeforeNestingBit | DirectAccessBit)
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
    
    

    typedef typename Lhs::Nested LhsNested;
    typedef typename Rhs::Nested RhsNested;
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

template<typename Lhs, typename Rhs, int Option>
class ProductImpl<Lhs,Rhs,Option,Dense>
  : public internal::dense_product_base<Lhs,Rhs,Option>
{
    typedef Product<Lhs, Rhs> Derived;
  public:
    
    typedef typename internal::dense_product_base<Lhs, Rhs, Option> Base;
    EIGEN_DENSE_PUBLIC_INTERFACE(Derived)
    
    Scalar coeff(Index row, Index col) const
    {
      EIGEN_STATIC_ASSERT_SIZE_1x1(Derived)
      eigen_assert(this->rows() == 1 && this->cols() == 1);
      
      return typename internal::evaluator<Derived>::type(derived()).coeff(row,col);
    }

    Scalar coeff(Index i) const
    {
      EIGEN_STATIC_ASSERT_SIZE_1x1(Derived)
      eigen_assert(this->rows() == 1 && this->cols() == 1);
      
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
