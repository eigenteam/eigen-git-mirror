// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
// Copyright (C) 2008 Gael Guennebaud <g.gael@free.fr>
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

#ifndef EIGEN_SPARSETRIANGULARSOLVER_H
#define EIGEN_SPARSETRIANGULARSOLVER_H

template<typename Lhs, typename Rhs,
  int TriangularPart = (int(Lhs::Flags) & LowerTriangularBit)
                     ? Lower
                     : (int(Lhs::Flags) & UpperTriangularBit)
                     ? Upper
                     : -1,
  int StorageOrder = int(Lhs::Flags) & RowMajorBit ? RowMajor : ColMajor
  >
struct ei_sparse_trisolve_selector;

// forward substitution, row-major
template<typename Lhs, typename Rhs>
struct ei_sparse_trisolve_selector<Lhs,Rhs,Lower,RowMajor>
{
  typedef typename Rhs::Scalar Scalar;
  static void run(const Lhs& lhs, const Rhs& rhs, Rhs& res)
  {
    for(int col=0 ; col<rhs.cols() ; ++col)
    {
      for(int i=0; i<lhs.rows(); ++i)
      {
        Scalar tmp = rhs.coeff(i,col);
        Scalar lastVal = 0;
        int lastIndex = 0;
        for(typename Lhs::InnerIterator it(lhs, i); it; ++it)
        {
          lastVal = it.value();
          lastIndex = it.index();
          tmp -= lastVal * res.coeff(lastIndex,col);
        }
        if (Lhs::Flags & UnitDiagBit)
          res.coeffRef(i,col) = tmp;
        else
        {
          ei_assert(lastIndex==i);
          res.coeffRef(i,col) = tmp/lastVal;
        }
      }
    }
  }
};

// backward substitution, row-major
template<typename Lhs, typename Rhs>
struct ei_sparse_trisolve_selector<Lhs,Rhs,Upper,RowMajor>
{
  typedef typename Rhs::Scalar Scalar;
  static void run(const Lhs& lhs, const Rhs& rhs, Rhs& res)
  {
    for(int col=0 ; col<rhs.cols() ; ++col)
    {
      for(int i=lhs.rows()-1 ; i>=0 ; --i)
      {
        Scalar tmp = rhs.coeff(i,col);
        typename Lhs::InnerIterator it(lhs, i);
        for(++it; it; ++it)
        {
          tmp -= it.value() * res.coeff(it.index(),col);
        }

        if (Lhs::Flags & UnitDiagBit)
          res.coeffRef(i,col) = tmp;
        else
        {
          typename Lhs::InnerIterator it(lhs, i);
          ei_assert(it.index() == i);
          res.coeffRef(i,col) = tmp/it.value();
        }
      }
    }
  }
};

// forward substitution, col-major
template<typename Lhs, typename Rhs>
struct ei_sparse_trisolve_selector<Lhs,Rhs,Lower,ColMajor>
{
  typedef typename Rhs::Scalar Scalar;
  static void run(const Lhs& lhs, const Rhs& rhs, Rhs& res)
  {
    // NOTE we could avoid this copy using an in-place API
    res = rhs;
    for(int col=0 ; col<rhs.cols() ; ++col)
    {
      for(int i=0; i<lhs.cols(); ++i)
      {
        typename Lhs::InnerIterator it(lhs, i);
        if(!(Lhs::Flags & UnitDiagBit))
        {
          ei_assert(it.index()==i);
          res.coeffRef(i,col) /= it.value();
        }
        Scalar tmp = res.coeffRef(i,col);
        for(++it; it; ++it)
          res.coeffRef(it.index(), col) -= tmp * it.value();
      }
    }
  }
};

// backward substitution, col-major
template<typename Lhs, typename Rhs>
struct ei_sparse_trisolve_selector<Lhs,Rhs,Upper,ColMajor>
{
  typedef typename Rhs::Scalar Scalar;
  static void run(const Lhs& lhs, const Rhs& rhs, Rhs& res)
  {
    // NOTE we could avoid this copy using an in-place API
    res = rhs;
    for(int col=0 ; col<rhs.cols() ; ++col)
    {
      for(int i=lhs.cols()-1; i>=0; --i)
      {
        if(!(Lhs::Flags & UnitDiagBit))
        {
          // FIXME lhs.coeff(i,i) might not be always efficient while it must simply be the
          // last element of the column !
          res.coeffRef(i,col) /= lhs.coeff(i,i);
        }
        Scalar tmp = res.coeffRef(i,col);
        typename Lhs::InnerIterator it(lhs, i);
        for(; it && it.index()<i; ++it)
          res.coeffRef(it.index(), col) -= tmp * it.value();
      }
    }
  }
};

template<typename Derived>
template<typename OtherDerived>
OtherDerived SparseMatrixBase<Derived>::solveTriangular(const MatrixBase<OtherDerived>& other) const
{
  ei_assert(derived().cols() == other.rows());
  ei_assert(!(Flags & ZeroDiagBit));
  ei_assert(Flags & (UpperTriangularBit|LowerTriangularBit));

  OtherDerived res(other.rows(), other.cols());
  ei_sparse_trisolve_selector<Derived, OtherDerived>::run(derived(), other.derived(), res);
  return res;
}

#endif // EIGEN_SPARSETRIANGULARSOLVER_H
