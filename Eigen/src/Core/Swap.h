// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
// Copyright (C) 2006-2008 Benoit Jacob <jacob@math.jussieu.fr>
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

#ifndef EIGEN_SWAP_H
#define EIGEN_SWAP_H

template<typename Scalar, typename Derived>
template<typename OtherDerived>
void MatrixBase<Scalar, Derived>::swap(const MatrixBase<Scalar, OtherDerived>& other)
{
  MatrixBase<Scalar, OtherDerived> *_other = const_cast<MatrixBase<Scalar, OtherDerived>*>(&other);
  if(Traits::SizeAtCompileTime == Dynamic)
  {
    Scalar tmp;
    if(Traits::IsVectorAtCompileTime)
    {
      assert(OtherDerived::Traits::IsVectorAtCompileTime && size() == _other->size());
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
