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

#ifndef EIGEN_INVERSEPRODUCT_H
#define EIGEN_INVERSEPRODUCT_H

/** \returns the product of the inverse of *this with \a other.
  *
  * This function computes the inverse-matrix matrix product inverse(*this) * \a other
  * It works as a forward (resp. backward) substitution if *this is an upper (resp. lower)
  * triangular matrix.
  */
template<typename Derived>
template<typename OtherDerived>
typename OtherDerived::Eval MatrixBase<Derived>::inverseProduct(const MatrixBase<OtherDerived>& other) const
{
  assert(cols() == other.rows());
  assert(!(Flags & ZeroDiagBit));
  assert(Flags & (UpperTriangularBit|LowerTriangularBit));

  typename OtherDerived::Eval res(other.rows(), other.cols());

  for(int c=0 ; c<other.cols() ; ++c)
  {
    if(Flags & LowerTriangularBit)
    {
      // forward substitution
      if(Flags & UnitDiagBit)
	res.coeffRef(0,c) = other.coeff(0,c);
      else
	res.coeffRef(0,c) = other.coeff(0,c)/coeff(0, 0);
      for(int i=1; i<rows(); ++i)
      {
	Scalar tmp = other.coeff(i,c) - ((this->row(i).start(i)) * res.col(c).start(i)).coeff(0,0);
	if (Flags & UnitDiagBit)
	  res.coeffRef(i,c) = tmp;
	else
	  res.coeffRef(i,c) = tmp/coeff(i,i);
      }
    }
    else
    {
      // backward substitution
      if(Flags & UnitDiagBit)
	res.coeffRef(cols()-1,c) = other.coeff(cols()-1,c);
      else
	res.coeffRef(cols()-1,c) = other.coeff(cols()-1, c)/coeff(rows()-1, cols()-1);
      for(int i=rows()-2 ; i>=0 ; --i)
      {
	Scalar tmp = other.coeff(i,c)
                     - ((this->row(i).end(cols()-i-1)) * res.col(c).end(cols()-i-1)).coeff(0,0);
	if (Flags & UnitDiagBit)
	  res.coeffRef(i,c) = tmp;
	else
	  res.coeffRef(i,c) = tmp/coeff(i,i);
      }
    }
  }
  return res;
}

#endif // EIGEN_INVERSEPRODUCT_H
