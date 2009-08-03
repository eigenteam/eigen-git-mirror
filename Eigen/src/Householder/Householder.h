// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Benoit Jacob <jacob.benoit.1@gmail.com>
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

#ifndef EIGEN_HOUSEHOLDER_H
#define EIGEN_HOUSEHOLDER_H

template<int n> struct ei_decrement_size
{
  enum {
    ret = (n==1 || n==Dynamic) ? n : n-1
  };
};

template<typename Derived>
template<typename EssentialPart>
void MatrixBase<Derived>::makeHouseholder(
  EssentialPart *essential,
  RealScalar *beta) const
{
  EIGEN_STATIC_ASSERT_VECTOR_ONLY(EssentialPart)
  RealScalar _squaredNorm = squaredNorm();
  Scalar c0;
  if(ei_abs2(coeff(0)) <= ei_abs2(precision<Scalar>()) * _squaredNorm)
  {
    c0 = ei_sqrt(_squaredNorm);
  }
  else
  {
    Scalar sign = coeff(0) / ei_abs(coeff(0));
    c0 = coeff(0) + sign * ei_sqrt(_squaredNorm);
  }
  *essential = end(size()-1) / c0; // FIXME take advantage of fixed size
  const RealScalar c0abs2 = ei_abs2(c0);
  *beta = RealScalar(2) * c0abs2 / (c0abs2 + _squaredNorm - ei_abs2(coeff(0)));
}

template<typename Derived>
template<typename EssentialPart>
void MatrixBase<Derived>::applyHouseholderOnTheLeft(
  const EssentialPart& essential,
  const RealScalar& beta)
{
  Matrix<Scalar, 1, ColsAtCompileTime, PlainMatrixType::Options, 1, MaxColsAtCompileTime> tmp(cols());
  tmp = row(0) + essential.adjoint() * block(1,0,rows()-1,cols());
  // FIXME take advantage of fixed size
  // FIXME play with lazy()
  // FIXME maybe not a good idea to use matrix product
  row(0) -= beta * tmp;
  block(1,0,rows()-1,cols()) -= beta * essential * tmp;
}

template<typename Derived>
template<typename EssentialPart>
void MatrixBase<Derived>::applyHouseholderOnTheRight(
  const EssentialPart& essential,
  const RealScalar& beta)
{
  Matrix<Scalar, RowsAtCompileTime, 1, PlainMatrixType::Options, MaxRowsAtCompileTime, 1> tmp(rows());
  tmp = col(0) + block(0,1,rows(),cols()-1) * essential.conjugate();
  // FIXME take advantage of fixed size
  // FIXME play with lazy()
  // FIXME maybe not a good idea to use matrix product
  col(0) -= beta * tmp;
  block(0,1,rows(),cols()-1) -= beta * tmp * essential.transpose();
}

#endif // EIGEN_HOUSEHOLDER_H
