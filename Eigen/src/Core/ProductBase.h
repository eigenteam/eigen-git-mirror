// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009-2010 Gael Guennebaud <g.gael@free.fr>
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

#ifndef EIGEN_PRODUCTBASE_H
#define EIGEN_PRODUCTBASE_H

enum { OuterProduct, InnerProduct, UnrolledProduct, GemvProduct, GemmProduct };

/** \class ProductBase
  *
  */
template<typename Derived, typename _Lhs, typename _Rhs>
struct ei_traits<ProductBase<Derived,_Lhs,_Rhs> > //: ei_traits<typename ei_cleantype<_Lhs>::type>
{
  typedef DenseStorageMatrix DenseStorageType;
  typedef typename ei_cleantype<_Lhs>::type Lhs;
  typedef typename ei_cleantype<_Rhs>::type Rhs;
  typedef typename ei_scalar_product_traits<typename Lhs::Scalar, typename Rhs::Scalar>::ReturnType Scalar;
  typedef typename ei_promote_storage_type<typename ei_traits<Lhs>::StorageType,
                                           typename ei_traits<Rhs>::StorageType>::ret StorageType;
  enum {
    RowsAtCompileTime = ei_traits<Lhs>::RowsAtCompileTime,
    ColsAtCompileTime = ei_traits<Rhs>::ColsAtCompileTime,
    MaxRowsAtCompileTime = ei_traits<Lhs>::MaxRowsAtCompileTime,
    MaxColsAtCompileTime = ei_traits<Rhs>::MaxColsAtCompileTime,
    Flags = EvalBeforeNestingBit | EvalBeforeAssigningBit,
    CoeffReadCost = 0 // FIXME why is it needed ?
  };
};

#define EIGEN_PRODUCT_PUBLIC_INTERFACE(Derived) \
  typedef ProductBase<Derived, Lhs, Rhs > Base; \
  EIGEN_DENSE_PUBLIC_INTERFACE(Derived) \
  typedef typename Base::LhsNested LhsNested; \
  typedef typename Base::_LhsNested _LhsNested; \
  typedef typename Base::LhsBlasTraits LhsBlasTraits; \
  typedef typename Base::ActualLhsType ActualLhsType; \
  typedef typename Base::_ActualLhsType _ActualLhsType; \
  typedef typename Base::RhsNested RhsNested; \
  typedef typename Base::_RhsNested _RhsNested; \
  typedef typename Base::RhsBlasTraits RhsBlasTraits; \
  typedef typename Base::ActualRhsType ActualRhsType; \
  typedef typename Base::_ActualRhsType _ActualRhsType; \
  using Base::m_lhs; \
  using Base::m_rhs;

template<typename Derived, typename Lhs, typename Rhs>
class ProductBase : public MatrixBase<Derived>
{
  public:
    typedef MatrixBase<Derived> Base;
    EIGEN_DENSE_PUBLIC_INTERFACE(ProductBase)
  protected:
    typedef typename Lhs::Nested LhsNested;
    typedef typename ei_cleantype<LhsNested>::type _LhsNested;
    typedef ei_blas_traits<_LhsNested> LhsBlasTraits;
    typedef typename LhsBlasTraits::DirectLinearAccessType ActualLhsType;
    typedef typename ei_cleantype<ActualLhsType>::type _ActualLhsType;

    typedef typename Rhs::Nested RhsNested;
    typedef typename ei_cleantype<RhsNested>::type _RhsNested;
    typedef ei_blas_traits<_RhsNested> RhsBlasTraits;
    typedef typename RhsBlasTraits::DirectLinearAccessType ActualRhsType;
    typedef typename ei_cleantype<ActualRhsType>::type _ActualRhsType;

    typedef typename ProductReturnType<Lhs,Rhs,UnrolledProduct>::Type CoeffBaseProductType;
    typedef Flagged<CoeffBaseProductType,0,EvalBeforeNestingBit> LazyCoeffBaseProductType;
  public:

    typedef typename Base::PlainMatrixType PlainMatrixType;

    ProductBase(const Lhs& lhs, const Rhs& rhs)
      : m_lhs(lhs), m_rhs(rhs)
    {
      ei_assert(lhs.cols() == rhs.rows()
        && "invalid matrix product"
        && "if you wanted a coeff-wise or a dot product use the respective explicit functions");
    }

    inline int rows() const { return m_lhs.rows(); }
    inline int cols() const { return m_rhs.cols(); }

    template<typename Dest>
    inline void evalTo(Dest& dst) const { dst.setZero(); scaleAndAddTo(dst,1); }

    template<typename Dest>
    inline void addTo(Dest& dst) const { scaleAndAddTo(dst,1); }

    template<typename Dest>
    inline void subTo(Dest& dst) const { scaleAndAddTo(dst,-1); }

    template<typename Dest>
    inline void scaleAndAddTo(Dest& dst,Scalar alpha) const { derived().scaleAndAddTo(dst,alpha); }

    EIGEN_DEPRECATED const Flagged<ProductBase, 0, EvalBeforeAssigningBit> lazy() const
    { return *this; }

    const _LhsNested& lhs() const { return m_lhs; }
    const _RhsNested& rhs() const { return m_rhs; }

    const Diagonal<LazyCoeffBaseProductType,0> diagonal() const
    { return Diagonal<LazyCoeffBaseProductType,0>(CoeffBaseProductType(m_lhs, m_rhs)); }

    Diagonal<LazyCoeffBaseProductType,0> diagonal()
    { return Diagonal<LazyCoeffBaseProductType,0>(CoeffBaseProductType(m_lhs, m_rhs)); }

    template<int Index>
    const Diagonal<LazyCoeffBaseProductType,Index> diagonal() const
    { return Diagonal<LazyCoeffBaseProductType,Index>(CoeffBaseProductType(m_lhs, m_rhs)); }

    template<int Index>
    Diagonal<LazyCoeffBaseProductType,Index> diagonal()
    { return Diagonal<LazyCoeffBaseProductType,Index>(CoeffBaseProductType(m_lhs, m_rhs)); }

    const Diagonal<LazyCoeffBaseProductType,Dynamic> diagonal(int index) const
    { return Diagonal<LazyCoeffBaseProductType,Dynamic>(LazyCoeffBaseProductType(CoeffBaseProductType(m_lhs, m_rhs))).diagonal(index); }

    Diagonal<LazyCoeffBaseProductType,Dynamic> diagonal(int index)
    { return Diagonal<LazyCoeffBaseProductType,Dynamic>(LazyCoeffBaseProductType(CoeffBaseProductType(m_lhs, m_rhs))).diagonal(index); }

  protected:

    const LhsNested m_lhs;
    const RhsNested m_rhs;

  private:

    // discard coeff methods
    void coeff(int,int) const;
    void coeffRef(int,int);
    void coeff(int) const;
    void coeffRef(int);
};

template<typename NestedProduct>
class ScaledProduct;

// Note that these two operator* functions are not defined as member
// functions of ProductBase, because, otherwise we would have to
// define all overloads defined in MatrixBase. Furthermore, Using
// "using Base::operator*" would not work with MSVC.
//
// Also note that here we accept any compatible scalar types
template<typename Derived,typename Lhs,typename Rhs>
const ScaledProduct<Derived>
operator*(const ProductBase<Derived,Lhs,Rhs>& prod, typename Derived::Scalar x)
{ return ScaledProduct<Derived>(prod.derived(), x); }

template<typename Derived,typename Lhs,typename Rhs>
typename ei_enable_if<!ei_is_same_type<typename Derived::Scalar,typename Derived::RealScalar>::ret,
                      const ScaledProduct<Derived> >::type
operator*(const ProductBase<Derived,Lhs,Rhs>& prod, typename Derived::RealScalar x)
{ return ScaledProduct<Derived>(prod.derived(), x); }


template<typename Derived,typename Lhs,typename Rhs>
const ScaledProduct<Derived>
operator*(typename Derived::Scalar x,const ProductBase<Derived,Lhs,Rhs>& prod)
{ return ScaledProduct<Derived>(prod.derived(), x); }

template<typename Derived,typename Lhs,typename Rhs>
typename ei_enable_if<!ei_is_same_type<typename Derived::Scalar,typename Derived::RealScalar>::ret,
                      const ScaledProduct<Derived> >::type
operator*(typename Derived::RealScalar x,const ProductBase<Derived,Lhs,Rhs>& prod)
{ return ScaledProduct<Derived>(prod.derived(), x); }


template<typename NestedProduct>
struct ei_traits<ScaledProduct<NestedProduct> >
 : ei_traits<ProductBase<ScaledProduct<NestedProduct>,
                         typename NestedProduct::_LhsNested,
                         typename NestedProduct::_RhsNested> >
{
  typedef typename ei_traits<NestedProduct>::StorageType StorageType;
};

template<typename NestedProduct>
class ScaledProduct
  : public ProductBase<ScaledProduct<NestedProduct>,
                       typename NestedProduct::_LhsNested,
                       typename NestedProduct::_RhsNested>
{
  public:
    typedef ProductBase<ScaledProduct<NestedProduct>,
                       typename NestedProduct::_LhsNested,
                       typename NestedProduct::_RhsNested> Base;
    typedef typename Base::Scalar Scalar;
//     EIGEN_PRODUCT_PUBLIC_INTERFACE(ScaledProduct)

    ScaledProduct(const NestedProduct& prod, Scalar x)
    : Base(prod.lhs(),prod.rhs()), m_prod(prod), m_alpha(x) {}

    template<typename Dest>
    inline void evalTo(Dest& dst) const { dst.setZero(); scaleAndAddTo(dst,m_alpha); }

    template<typename Dest>
    inline void addTo(Dest& dst) const { scaleAndAddTo(dst,m_alpha); }

    template<typename Dest>
    inline void subTo(Dest& dst) const { scaleAndAddTo(dst,-m_alpha); }

    template<typename Dest>
    inline void scaleAndAddTo(Dest& dst,Scalar alpha) const { m_prod.derived().scaleAndAddTo(dst,alpha); }

  protected:
    const NestedProduct& m_prod;
    Scalar m_alpha;
};

/** \internal
  * Overloaded to perform an efficient C = (A*B).lazy() */
template<typename Derived>
template<typename ProductDerived, typename Lhs, typename Rhs>
Derived& MatrixBase<Derived>::lazyAssign(const ProductBase<ProductDerived, Lhs,Rhs>& other)
{
  other.derived().evalTo(derived());
  return derived();
}


#endif // EIGEN_PRODUCTBASE_H
