// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
// Copyright (C) 2006-2007 Benoit Jacob <jacob@math.jussieu.fr>
//
// Eigen is free software; you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software
// Foundation; either version 2 or (at your option) any later version.
//
// Eigen is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
// details.
//
// You should have received a copy of the GNU General Public License along
// with Eigen; if not, write to the Free Software Foundation, Inc., 51
// Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
//
// As a special exception, if other files instantiate templates or use macros
// or functions from this file, or you compile this file and link it
// with other works to produce a work based on this file, this file does not
// by itself cause the resulting work to be covered by the GNU General Public
// License. This exception does not invalidate any other reasons why a work
// based on this file might be covered by the GNU General Public License.

#ifndef EIGEN_COEFFS_H
#define EIGEN_COEFFS_H

template<typename Scalar, typename Derived>
Scalar MatrixBase<Scalar, Derived>
  ::coeff(int row, int col, AssertLevel assertLevel = InternalDebugging) const
{
  eigen_assert(assertLevel, row >= 0 && row < rows()
			    && col >= 0 && col < cols());
  return static_cast<const Derived *>(this)->_coeff(row, col);
}

template<typename Scalar, typename Derived>
Scalar MatrixBase<Scalar, Derived>
  ::operator()(int row, int col) const { return coeff(row, col, UserDebugging); }

template<typename Scalar, typename Derived>
Scalar& MatrixBase<Scalar, Derived>
  ::coeffRef(int row, int col, AssertLevel assertLevel = InternalDebugging)
{
  eigen_assert(assertLevel, row >= 0 && row < rows()
			    && col >= 0 && col < cols());
  return static_cast<Derived *>(this)->_coeffRef(row, col);
}

template<typename Scalar, typename Derived>
Scalar& MatrixBase<Scalar, Derived>
  ::operator()(int row, int col) { return coeffRef(row, col, UserDebugging); }

template<typename Scalar, typename Derived>
Scalar MatrixBase<Scalar, Derived>
  ::coeff(int index, AssertLevel assertLevel = InternalDebugging) const
{
  eigen_assert(assertLevel, IsVector);
  if(RowsAtCompileTime == 1)
  {
    eigen_assert(assertLevel, index >= 0 && index < cols());
    return coeff(0, index);
  }
  else
  {
    eigen_assert(assertLevel, index >= 0 && index < rows());
    return coeff(index, 0);
  }
}

template<typename Scalar, typename Derived>
Scalar MatrixBase<Scalar, Derived>
  ::operator[](int index) const { return coeff(index, UserDebugging); }

template<typename Scalar, typename Derived>
Scalar& MatrixBase<Scalar, Derived>
  ::coeffRef(int index, AssertLevel assertLevel = InternalDebugging)
{
  eigen_assert(assertLevel, IsVector);
  if(RowsAtCompileTime == 1)
  {
    eigen_assert(assertLevel, index >= 0 && index < cols());
    return coeffRef(0, index);
  }
  else
  {
    eigen_assert(assertLevel, index >= 0 && index < rows());
    return coeffRef(index, 0);
  }
}

template<typename Scalar, typename Derived>
Scalar& MatrixBase<Scalar, Derived>
  ::operator[](int index) { return coeffRef(index, UserDebugging); }

template<typename Scalar, typename Derived>
Scalar MatrixBase<Scalar, Derived>
  ::x() const { return coeff(0, UserDebugging); }

template<typename Scalar, typename Derived>
Scalar MatrixBase<Scalar, Derived>
  ::y() const { return coeff(1, UserDebugging); }

template<typename Scalar, typename Derived>
Scalar MatrixBase<Scalar, Derived>
  ::z() const { return coeff(2, UserDebugging); }

template<typename Scalar, typename Derived>
Scalar MatrixBase<Scalar, Derived>
  ::w() const { return coeff(3, UserDebugging); }

template<typename Scalar, typename Derived>
Scalar& MatrixBase<Scalar, Derived>
  ::x() { return coeffRef(0, UserDebugging); }

template<typename Scalar, typename Derived>
Scalar& MatrixBase<Scalar, Derived>
  ::y() { return coeffRef(1, UserDebugging); }

template<typename Scalar, typename Derived>
Scalar& MatrixBase<Scalar, Derived>
  ::z() { return coeffRef(2, UserDebugging); }

template<typename Scalar, typename Derived>
Scalar& MatrixBase<Scalar, Derived>
  ::w() { return coeffRef(3, UserDebugging); }


#endif // EIGEN_COEFFS_H
