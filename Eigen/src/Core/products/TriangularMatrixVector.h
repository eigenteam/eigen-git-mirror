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

#ifndef EIGEN_TRIANGULARMATRIXVECTOR_H
#define EIGEN_TRIANGULARMATRIXVECTOR_H

template<typename MatrixType, typename Rhs, typename Result,
         int Mode, bool ConjLhs, bool ConjRhs, int StorageOrder>
struct ei_product_triangular_vector_selector;

template<typename Lhs, typename Rhs, typename Result, int Mode, bool ConjLhs, bool ConjRhs>
struct ei_product_triangular_vector_selector<Lhs,Rhs,Result,Mode,ConjLhs,ConjRhs,ColMajor>
{
  typedef typename Rhs::Scalar Scalar;
  enum {
    IsLowerTriangular = ((Mode&LowerTriangularBit)==LowerTriangularBit),
    HasUnitDiag = (Mode & UnitDiagBit)==UnitDiagBit
  };
  static EIGEN_DONT_INLINE  void run(const Lhs& lhs, const Rhs& rhs, Result& res, typename ei_traits<Lhs>::Scalar alpha)
  {
    static const int PanelWidth = EIGEN_TUNE_TRIANGULAR_PANEL_WIDTH;
    typename ei_conj_expr_if<ConjLhs,Lhs>::ret cjLhs(lhs);
    typename ei_conj_expr_if<ConjRhs,Rhs>::ret cjRhs(rhs);

    int size = lhs.cols();
    for (int pi=0; pi<size; pi+=PanelWidth)
    {
      int actualPanelWidth = std::min(PanelWidth, size-pi);
      for (int k=0; k<actualPanelWidth; ++k)
      {
        int i = pi + k;
        int s = IsLowerTriangular ? (HasUnitDiag ? i+1 : i ) : pi;
        int r = IsLowerTriangular ? actualPanelWidth-k : k+1;
        if ((!HasUnitDiag) || (--r)>0)
          res.segment(s,r) += (alpha * cjRhs.coeff(i)) * cjLhs.col(i).segment(s,r);
        if (HasUnitDiag)
          res.coeffRef(i) += alpha * cjRhs.coeff(i);
      }
      int r = IsLowerTriangular ? size - pi - actualPanelWidth : pi;
      if (r>0)
      {
        int s = IsLowerTriangular ? pi+actualPanelWidth : 0;
        ei_cache_friendly_product_colmajor_times_vector<ConjLhs,ConjRhs>(
            r,
            &(lhs.const_cast_derived().coeffRef(s,pi)), lhs.stride(),
            rhs.segment(pi, actualPanelWidth),
            &(res.coeffRef(s)),
            alpha);
      }
    }
  }
};

template<typename Lhs, typename Rhs, typename Result, int Mode, bool ConjLhs, bool ConjRhs>
struct ei_product_triangular_vector_selector<Lhs,Rhs,Result,Mode,ConjLhs,ConjRhs,RowMajor>
{
  typedef typename Rhs::Scalar Scalar;
  enum {
    IsLowerTriangular = ((Mode&LowerTriangularBit)==LowerTriangularBit),
    HasUnitDiag = (Mode & UnitDiagBit)==UnitDiagBit
  };
  static void run(const Lhs& lhs, const Rhs& rhs, Result& res, typename ei_traits<Lhs>::Scalar alpha)
  {
    static const int PanelWidth = EIGEN_TUNE_TRIANGULAR_PANEL_WIDTH;
    typename ei_conj_expr_if<ConjLhs,Lhs>::ret cjLhs(lhs);
    typename ei_conj_expr_if<ConjRhs,Rhs>::ret cjRhs(rhs);
    int size = lhs.cols();
    for (int pi=0; pi<size; pi+=PanelWidth)
    {
      int actualPanelWidth = std::min(PanelWidth, size-pi);
      for (int k=0; k<actualPanelWidth; ++k)
      {
        int i = pi + k;
        int s = IsLowerTriangular ? pi  : (HasUnitDiag ? i+1 : i);
        int r = IsLowerTriangular ? k+1 : actualPanelWidth-k;
        if ((!HasUnitDiag) || (--r)>0)
          res.coeffRef(i) += alpha * (cjLhs.row(i).segment(s,r).cwise() * cjRhs.segment(s,r).transpose()).sum();
        if (HasUnitDiag)
          res.coeffRef(i) += alpha * cjRhs.coeff(i);
      }
      int r = IsLowerTriangular ? pi : size - pi - actualPanelWidth;
      if (r>0)
      {
        int s = IsLowerTriangular ? 0 : pi + actualPanelWidth;
        Block<Result,Dynamic,1> target(res,pi,0,actualPanelWidth,1);
        ei_cache_friendly_product_rowmajor_times_vector<ConjLhs,ConjRhs>(
            &(lhs.const_cast_derived().coeffRef(pi,s)), lhs.stride(),
            &(rhs.const_cast_derived().coeffRef(s)), r,
            target, alpha);
      }
    }
  }
};

/***************************************************************************
* Wrapper to ei_product_triangular_vector
***************************************************************************/

template<int Mode, /*bool LhsIsTriangular, */typename Lhs, typename Rhs>
struct ei_traits<TriangularProduct<Mode,true,Lhs,false,Rhs,true> >
 : ei_traits<ProductBase<TriangularProduct<Mode,true,Lhs,false,Rhs,true>, Lhs, Rhs> >
{};

template<int Mode, /*bool LhsIsTriangular, */typename Lhs, typename Rhs>
struct TriangularProduct<Mode,true,Lhs,false,Rhs,true>
  : public ProductBase<TriangularProduct<Mode,true,Lhs,false,Rhs,true>, Lhs, Rhs >
{
  EIGEN_PRODUCT_PUBLIC_INTERFACE(TriangularProduct)

  TriangularProduct(const Lhs& lhs, const Rhs& rhs) : Base(lhs,rhs) {}

  template<typename Dest> void scaleAndAddTo(Dest& dst, Scalar alpha) const
  {
    ei_assert(dst.rows()==m_lhs.rows() && dst.cols()==m_rhs.cols());

    const ActualLhsType lhs = LhsBlasTraits::extract(m_lhs);
    const ActualRhsType rhs = RhsBlasTraits::extract(m_rhs);

    Scalar actualAlpha = alpha * LhsBlasTraits::extractScalarFactor(m_lhs)
                               * RhsBlasTraits::extractScalarFactor(m_rhs);

    ei_product_triangular_vector_selector
      <_ActualLhsType,_ActualRhsType,Dest,
       Mode,
       LhsBlasTraits::NeedToConjugate,
       RhsBlasTraits::NeedToConjugate,
       (int(ei_traits<Lhs>::Flags)&RowMajorBit) ? RowMajor : ColMajor>
      ::run(lhs,rhs,dst,actualAlpha);
  }
};

#endif // EIGEN_TRIANGULARMATRIXVECTOR_H
