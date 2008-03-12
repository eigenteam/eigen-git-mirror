// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
// Copyright (C) 2006-2008 Benoit Jacob <jacob@math.jussieu.fr>
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

#ifndef EIGEN_SWAP_H
#define EIGEN_SWAP_H

template<typename Derived>
template<typename OtherDerived>
void MatrixBase<Derived>::swap(const MatrixBase<OtherDerived>& other)
{
  MatrixBase<OtherDerived> *_other = const_cast<MatrixBase<OtherDerived>*>(&other);
  if(SizeAtCompileTime == Dynamic)
  {
    Scalar tmp;
    if(IsVectorAtCompileTime)
    {
      assert(OtherDerived::IsVectorAtCompileTime && size() == _other->size());
      for(int i = 0; i < size(); i++)
      {
        tmp = coeff(i);
        coeffRef(i) = _other->coeff(i);
        _other->coeffRef(i) = tmp;
      }
    }
    else
      for(int j = 0; j < cols(); j++)
        for(int i = 0; i < rows(); i++)
        {
          tmp = coeff(i, j);
          coeffRef(i, j) = _other->coeff(i, j);
          _other->coeffRef(i, j) = tmp;
        }
  }
  else // SizeAtCompileTime != Dynamic
  {
    typename Eval<Derived>::MatrixType buf(*this);
    *this = other;
    *_other = buf;
  }
}

#endif // EIGEN_SWAP_H
