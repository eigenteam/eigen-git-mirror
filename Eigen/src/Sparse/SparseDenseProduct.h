// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2010 Gael Guennebaud <gael.guennebaud@inria.fr>
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

#ifndef EIGEN_SPARSEDENSEPRODUCT_H
#define EIGEN_SPARSEDENSEPRODUCT_H

template<typename Lhs, typename Rhs>
struct ei_traits<SparseTimeDenseProduct<Lhs,Rhs> >
 : ei_traits<ProductBase<SparseTimeDenseProduct<Lhs,Rhs>, Lhs, Rhs> >
{
  typedef Dense StorageKind;
  typedef MatrixXpr XprKind;
};

template<typename Lhs, typename Rhs>
class SparseTimeDenseProduct
  : public ProductBase<SparseTimeDenseProduct<Lhs,Rhs>, Lhs, Rhs>
{
  public:
    EIGEN_PRODUCT_PUBLIC_INTERFACE(SparseTimeDenseProduct)

    SparseTimeDenseProduct(const Lhs& lhs, const Rhs& rhs) : Base(lhs,rhs)
    {}

    template<typename Dest> void scaleAndAddTo(Dest& dest, Scalar alpha) const
    {
      typedef typename ei_cleantype<Lhs>::type _Lhs;
      typedef typename ei_cleantype<Rhs>::type _Rhs;
      typedef typename _Lhs::InnerIterator LhsInnerIterator;
      enum { LhsIsRowMajor = (_Lhs::Flags&RowMajorBit)==RowMajorBit };
      for(Index j=0; j<m_lhs.outerSize(); ++j)
      {
        typename Rhs::Scalar rhs_j = alpha * m_rhs.coeff(j,0);
        Block<Dest,1,Dest::ColsAtCompileTime> dest_j(dest.row(LhsIsRowMajor ? j : 0));
        for(LhsInnerIterator it(m_lhs,j); it ;++it)
        {
          if(LhsIsRowMajor)                   dest_j += (alpha*it.value()) * m_rhs.row(it.index());
          else if(Rhs::ColsAtCompileTime==1)  dest.coeffRef(it.index()) += it.value() * rhs_j;
          else                                dest.row(it.index()) += (alpha*it.value()) * m_rhs.row(j);
        }
      }
    }

  private:
    SparseTimeDenseProduct& operator=(const SparseTimeDenseProduct&);
};


// dense = dense * sparse
template<typename Lhs, typename Rhs>
struct ei_traits<DenseTimeSparseProduct<Lhs,Rhs> >
 : ei_traits<ProductBase<DenseTimeSparseProduct<Lhs,Rhs>, Lhs, Rhs> >
{
  typedef Dense StorageKind;
};

template<typename Lhs, typename Rhs>
class DenseTimeSparseProduct
  : public ProductBase<DenseTimeSparseProduct<Lhs,Rhs>, Lhs, Rhs>
{
  public:
    EIGEN_PRODUCT_PUBLIC_INTERFACE(DenseTimeSparseProduct)

    DenseTimeSparseProduct(const Lhs& lhs, const Rhs& rhs) : Base(lhs,rhs)
    {}

    template<typename Dest> void scaleAndAddTo(Dest& dest, Scalar alpha) const
    {
      typedef typename ei_cleantype<Rhs>::type _Rhs;
      typedef typename _Rhs::InnerIterator RhsInnerIterator;
      enum { RhsIsRowMajor = (_Rhs::Flags&RowMajorBit)==RowMajorBit };
      for(Index j=0; j<m_rhs.outerSize(); ++j)
        for(RhsInnerIterator i(m_rhs,j); i; ++i)
          dest.col(RhsIsRowMajor ? i.index() : j) += (alpha*i.value()) * m_lhs.col(RhsIsRowMajor ? j : i.index());
    }

  private:
    DenseTimeSparseProduct& operator=(const DenseTimeSparseProduct&);
};

// sparse * dense
template<typename Derived>
template<typename OtherDerived>
inline const SparseTimeDenseProduct<Derived,OtherDerived>
SparseMatrixBase<Derived>::operator*(const MatrixBase<OtherDerived> &other) const
{
  return SparseTimeDenseProduct<Derived,OtherDerived>(derived(), other.derived());
}

#endif // EIGEN_SPARSEDENSEPRODUCT_H
