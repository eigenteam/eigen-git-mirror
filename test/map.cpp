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

#include "main.h"

namespace Eigen {

template<typename VectorType> void tmap(const VectorType& m)
{
  typedef typename VectorType::Scalar Scalar;
  
  int size = m.coeffs();
  
  // test Map.h
  Scalar* array1 = new Scalar[size];
  Scalar* array2 = new Scalar[size];
  VectorType::map(array1, size) = VectorType::random(size);
  VectorType::map(array2, size) = VectorType::map(array1, size);
  VectorType ma1 = VectorType::map(array1, size);
  VectorType ma2 = VectorType::map(array2, size);
  VERIFY_IS_APPROX(ma1, ma2);
  VERIFY_IS_APPROX(ma1, VectorType(array2, size));
  delete[] array1;
  delete[] array2;
}

void EigenTest::testMap()
{
  for(int i = 0; i < m_repeat; i++) {
    tmap(Matrix<float, 1, 1>());
    tmap(Vector4d());
    tmap(RowVector4f());
    tmap(VectorXcf(8));
    tmap(VectorXi(12));
  }
}

} // namespace Eigen
