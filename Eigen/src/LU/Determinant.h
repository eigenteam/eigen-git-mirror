// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
// Copyright (C) 2008 Benoit Jacob <jacob@math.jussieu.fr>
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

#ifndef EIGEN_DETERMINANT_H
#define EIGEN_DETERMINANT_H

template<typename Derived>
const typename Derived::Scalar ei_bruteforce_det3_helper
(const MatrixBase<Derived>& matrix, int a, int b, int c)
{
  return matrix.coeff(0,a)
         * (matrix.coeff(1,b) * matrix.coeff(2,c) - matrix.coeff(1,c) * matrix.coeff(2,b));
}

template<typename Derived>
const typename Derived::Scalar ei_bruteforce_det4_helper
(const MatrixBase<Derived>& matrix, int j, int k, int m, int n)
{
  return (matrix.coeff(j,0) * matrix.coeff(k,1) - matrix.coeff(k,0) * matrix.coeff(j,1))
       * (matrix.coeff(m,2) * matrix.coeff(n,3) - matrix.coeff(n,2) * matrix.coeff(m,3));
}

template<typename Derived>
const typename Derived::Scalar ei_bruteforce_det(const MatrixBase<Derived>& m)
{
  switch(Derived::RowsAtCompileTime)
  {
    case 1:
      return m.coeff(0,0);
    case 2:
      return m.coeff(0,0) * m.coeff(1,1) - m.coeff(1,0) * m.coeff(0,1);
    case 3:
      return ei_bruteforce_det3_helper(m,0,1,2)
           - ei_bruteforce_det3_helper(m,1,0,2)
           + ei_bruteforce_det3_helper(m,2,0,1);
    case 4:
      // trick by Martin Costabel to compute 4x4 det with only 30 muls
      return ei_bruteforce_det4_helper(m,0,1,2,3)
           - ei_bruteforce_det4_helper(m,0,2,1,3)
           + ei_bruteforce_det4_helper(m,0,3,1,2)
           + ei_bruteforce_det4_helper(m,1,2,0,3)
           - ei_bruteforce_det4_helper(m,1,3,0,2)
           + ei_bruteforce_det4_helper(m,2,3,0,1);
    default:
      assert(false);
  }
}

template<typename Derived>
typename ei_traits<Derived>::Scalar MatrixBase<Derived>::determinant() const
{
  assert(rows() == cols());
  if (Derived::Flags & (UpperTriangularBit | LowerTriangularBit))
  {
    if (Derived::Flags & UnitDiagBit)
      return 1;
    else if (Derived::Flags & ZeroDiagBit)
      return 0;
    else
      return derived().diagonal().redux(ei_scalar_product_op<Scalar>());
  }
  else if(rows() <= 4) return ei_bruteforce_det(derived());
  else assert(false); // unimplemented for now
}

#endif // EIGEN_DETERMINANT_H
