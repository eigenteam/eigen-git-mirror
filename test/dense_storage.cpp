// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2010 Hauke Heibel <hauke.heibel@gmail.com>
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

#include <Eigen/Core>

template <typename T, int Rows, int Cols>
void dense_storage_copy()
{
  static const int Size = ((Rows==Dynamic || Cols==Dynamic) ? Dynamic : Rows*Cols);
  typedef DenseStorage<T,Size, Rows,Cols, 0> DenseStorageType;
  
  const int rows = (Rows==Dynamic) ? 4 : Rows;
  const int cols = (Cols==Dynamic) ? 3 : Cols;
  const int size = rows*cols;
  DenseStorageType reference(size, rows, cols);
  T* raw_reference = reference.data();
  for (int i=0; i<size; ++i)
    raw_reference[i] = static_cast<T>(i);
    
  DenseStorageType copied_reference(reference);
  const T* raw_copied_reference = copied_reference.data();
  for (int i=0; i<size; ++i)
    VERIFY_IS_EQUAL(raw_reference[i], raw_copied_reference[i]);
}

template <typename T, int Rows, int Cols>
void dense_storage_assignment()
{
  static const int Size = ((Rows==Dynamic || Cols==Dynamic) ? Dynamic : Rows*Cols);
  typedef DenseStorage<T,Size, Rows,Cols, 0> DenseStorageType;
  
  const int rows = (Rows==Dynamic) ? 4 : Rows;
  const int cols = (Cols==Dynamic) ? 3 : Cols;
  const int size = rows*cols;
  DenseStorageType reference(size, rows, cols);
  T* raw_reference = reference.data();
  for (int i=0; i<size; ++i)
    raw_reference[i] = static_cast<T>(i);
    
  DenseStorageType copied_reference;
  copied_reference = reference;
  const T* raw_copied_reference = copied_reference.data();
  for (int i=0; i<size; ++i)
    VERIFY_IS_EQUAL(raw_reference[i], raw_copied_reference[i]);
}

void test_dense_storage()
{
  dense_storage_copy<int,Dynamic,Dynamic>();  
  dense_storage_copy<int,Dynamic,3>();
  dense_storage_copy<int,4,Dynamic>();
  dense_storage_copy<int,4,3>();

  dense_storage_copy<float,Dynamic,Dynamic>();
  dense_storage_copy<float,Dynamic,3>();
  dense_storage_copy<float,4,Dynamic>();  
  dense_storage_copy<float,4,3>();
  
  dense_storage_assignment<int,Dynamic,Dynamic>();  
  dense_storage_assignment<int,Dynamic,3>();
  dense_storage_assignment<int,4,Dynamic>();
  dense_storage_assignment<int,4,3>();

  dense_storage_assignment<float,Dynamic,Dynamic>();
  dense_storage_assignment<float,Dynamic,3>();
  dense_storage_assignment<float,4,Dynamic>();  
  dense_storage_assignment<float,4,3>();  
}
