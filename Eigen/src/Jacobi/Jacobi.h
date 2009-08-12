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

#ifndef EIGEN_JACOBI_H
#define EIGEN_JACOBI_H

template<typename Derived>
void MatrixBase<Derived>::applyJacobiOnTheLeft(int p, int q, Scalar c, Scalar s)
{
  for(int i = 0; i < cols(); ++i)
  {
    Scalar tmp = coeff(p,i);
    coeffRef(p,i) = c * tmp - s * coeff(q,i);
    coeffRef(q,i) = s * tmp + c * coeff(q,i);
  }
}

template<typename Derived>
void MatrixBase<Derived>::applyJacobiOnTheRight(int p, int q, Scalar c, Scalar s)
{
  for(int i = 0; i < rows(); ++i)
  {
    Scalar tmp = coeff(i,p);
    coeffRef(i,p) = c * tmp - s * coeff(i,q);
    coeffRef(i,q) = s * tmp + c * coeff(i,q);
  }
}

template<typename Scalar>
bool ei_makeJacobi(Scalar x, Scalar y, Scalar z, Scalar *c, Scalar *s)
{
  if(y == 0)
  {
    *c = Scalar(1);
    *s = Scalar(0);
    return false;
  }
  else
  {
    Scalar tau = (z - x) / (2 * y);
    Scalar w = ei_sqrt(1 + ei_abs2(tau));
    Scalar t;
    if(tau>0)
      t = Scalar(1) / (tau + w);
    else
      t = Scalar(1) / (tau - w);
    *c = Scalar(1) / ei_sqrt(1 + ei_abs2(t));
    *s = *c * t;
    return true;
  }
}

template<typename Derived>
inline bool MatrixBase<Derived>::makeJacobi(int p, int q, Scalar *c, Scalar *s) const
{
  return ei_makeJacobi(coeff(p,p), coeff(p,q), coeff(q,q), c, s);
}

template<typename Derived>
inline bool MatrixBase<Derived>::makeJacobiForAtA(int p, int q, Scalar *c, Scalar *s) const
{
  return ei_makeJacobi(ei_abs2(coeff(p,p)) + ei_abs2(coeff(q,p)),
                       ei_conj(coeff(p,p))*coeff(p,q) + ei_conj(coeff(q,p))*coeff(q,q),
                       ei_abs2(coeff(p,q)) + ei_abs2(coeff(q,q)),
                       c,s);
}

template<typename Derived>
inline bool MatrixBase<Derived>::makeJacobiForAAt(int p, int q, Scalar *c, Scalar *s) const
{
  return ei_makeJacobi(ei_abs2(coeff(p,p)) + ei_abs2(coeff(p,q)),
                       ei_conj(coeff(q,p))*coeff(p,p) + ei_conj(coeff(q,q))*coeff(p,q),
                       ei_abs2(coeff(q,p)) + ei_abs2(coeff(q,q)),
                       c,s);
}


#endif // EIGEN_JACOBI_H
