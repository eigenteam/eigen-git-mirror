// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Gael Guennebaud <g.gael@free.fr>
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

/** \class ProductBase
  *
  */
template<typename Derived, typename _Lhs, typename _Rhs>
struct ei_traits<ProductBase<Derived,_Lhs,_Rhs> >
{
  typedef typename ei_cleantype<_Lhs>::type Lhs;
  typedef typename ei_cleantype<_Rhs>::type Rhs;
  typedef typename ei_traits<Lhs>::Scalar Scalar;
  enum {
    RowsAtCompileTime = ei_traits<Lhs>::RowsAtCompileTime,
    ColsAtCompileTime = ei_traits<Rhs>::ColsAtCompileTime,
    MaxRowsAtCompileTime = ei_traits<Lhs>::MaxRowsAtCompileTime,
    MaxColsAtCompileTime = ei_traits<Rhs>::MaxColsAtCompileTime,
    Flags = EvalBeforeNestingBit | EvalBeforeAssigningBit,
    CoeffReadCost = 0 // FIXME why is it needed ?
  };
};

// enforce evaluation before nesting
template<typename Derived, typename Lhs, typename Rhs,int N,typename EvalType>
struct ei_nested<ProductBase<Derived,Lhs,Rhs>, N, EvalType>
{
  typedef EvalType type;
};

#define EIGEN_PRODUCT_PUBLIC_INTERFACE(Derived) \
  typedef ProductBase<Derived, Lhs, Rhs > ProductBaseType; \
  _EIGEN_GENERIC_PUBLIC_INTERFACE(Derived, ProductBaseType) \
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
    _EIGEN_GENERIC_PUBLIC_INTERFACE(ProductBase,MatrixBase<Derived>)

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

    using Base::derived;
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
    inline void evalTo(Dest& dst) const { dst.setZero(); addTo(dst,1); }

    template<typename Dest>
    inline void addTo(Dest& dst) const { addTo(dst,1); }

    template<typename Dest>
    inline void subTo(Dest& dst) const { addTo(dst,-1); }

    template<typename Dest>
    inline void addTo(Dest& dst,Scalar alpha) const { derived().addTo(dst,alpha); }

    PlainMatrixType eval() const
    {
      PlainMatrixType res(rows(), cols());
      res.setZero();
      evalTo(res);
      return res;
    }

    const Flagged<ProductBase, 0, EvalBeforeNestingBit | EvalBeforeAssigningBit> lazy() const
    {
      return *this;
    }

    const _LhsNested& lhs() const { return m_lhs; }
    const _RhsNested& rhs() const { return m_rhs; }

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

/** \internal
  * Overloaded to perform an efficient C = (A*B).lazy() */
template<typename Derived>
template<typename ProductDerived, typename Lhs, typename Rhs>
Derived& MatrixBase<Derived>::lazyAssign(const ProductBase<ProductDerived, Lhs,Rhs>& other)
{
  other.evalTo(derived()); return derived();
}

/** \internal
  * Overloaded to perform an efficient C += (A*B).lazy() */
template<typename Derived>
template<typename ProductDerived, typename Lhs, typename Rhs>
Derived& MatrixBase<Derived>::operator+=(const Flagged<ProductBase<ProductDerived, Lhs,Rhs>, 0,
                                                       EvalBeforeNestingBit | EvalBeforeAssigningBit>& other)
{
  other._expression().addTo(derived()); return derived();
}

/** \internal
  * Overloaded to perform an efficient C -= (A*B).lazy() */
template<typename Derived>
template<typename ProductDerived, typename Lhs, typename Rhs>
Derived& MatrixBase<Derived>::operator-=(const Flagged<ProductBase<ProductDerived, Lhs,Rhs>, 0,
                                                       EvalBeforeNestingBit | EvalBeforeAssigningBit>& other)
{
  other._expression().subTo(derived()); return derived();
}

#endif // EIGEN_PRODUCTBASE_H
