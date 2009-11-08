// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2006-2008 Benoit Jacob <jacob.benoit.1@gmail.com>
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

#include "main.h"

template<typename VectorType> void map_class(const VectorType& m)
{
  typedef typename VectorType::Scalar Scalar;

  int size = m.size();

  // test Map.h
  Scalar* array1 = ei_aligned_new<Scalar>(size);
  Scalar* array2 = ei_aligned_new<Scalar>(size);
  Scalar* array3 = new Scalar[size+1];
  Scalar* array3unaligned = size_t(array3)%16 == 0 ? array3+1 : array3;
  
  Map<VectorType, Aligned>(array1, size) = VectorType::Random(size);
  Map<VectorType, Aligned>(array2, size) = Map<VectorType,Aligned>(array1, size);
  Map<VectorType>(array3unaligned, size) = Map<VectorType>(array1, size);
  VectorType ma1 = Map<VectorType, Aligned>(array1, size);
  VectorType ma2 = Map<VectorType, Aligned>(array2, size);
  VectorType ma3 = Map<VectorType>(array3unaligned, size);
  VERIFY_IS_APPROX(ma1, ma2);
  VERIFY_IS_APPROX(ma1, ma3);
  VERIFY_RAISES_ASSERT((Map<VectorType,Aligned>(array3unaligned, size)));

  ei_aligned_delete(array1, size);
  ei_aligned_delete(array2, size);
  delete[] array3;
}

template<typename VectorType> void map_static_methods(const VectorType& m)
{
  typedef typename VectorType::Scalar Scalar;

  int size = m.size();

  // test Map.h
  Scalar* array1 = ei_aligned_new<Scalar>(size);
  Scalar* array2 = ei_aligned_new<Scalar>(size);
  Scalar* array3 = new Scalar[size+1];
  Scalar* array3unaligned = size_t(array3)%16 == 0 ? array3+1 : array3;
  
  VectorType::MapAligned(array1, size) = VectorType::Random(size);
  VectorType::Map(array2, size) = VectorType::Map(array1, size);
  VectorType::Map(array3unaligned, size) = VectorType::Map(array1, size);
  VectorType ma1 = VectorType::Map(array1, size);
  VectorType ma2 = VectorType::MapAligned(array2, size);
  VectorType ma3 = VectorType::Map(array3unaligned, size);
  VERIFY_IS_APPROX(ma1, ma2);
  VERIFY_IS_APPROX(ma1, ma3);
  
  ei_aligned_delete(array1, size);
  ei_aligned_delete(array2, size);
  delete[] array3;
}


void test_map()
{
  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST( map_class(Matrix<float, 1, 1>()) );
    CALL_SUBTEST( map_class(Vector4d()) );
    CALL_SUBTEST( map_class(RowVector4f()) );
    CALL_SUBTEST( map_class(VectorXcf(8)) );
    CALL_SUBTEST( map_class(VectorXi(12)) );

    CALL_SUBTEST( map_static_methods(Matrix<double, 1, 1>()) );
    CALL_SUBTEST( map_static_methods(Vector3f()) );
    CALL_SUBTEST( map_static_methods(RowVector3d()) );
    CALL_SUBTEST( map_static_methods(VectorXcd(8)) );
    CALL_SUBTEST( map_static_methods(VectorXf(12)) );
  }
}
