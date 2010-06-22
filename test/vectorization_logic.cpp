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

#include "main.h"
#include <typeinfo>

template<typename Dst, typename Src>
bool test_assign(const Dst&, const Src&, int vectorization, int unrolling)
{
  return ei_assign_traits<Dst,Src>::Vectorization==vectorization
    && ei_assign_traits<Dst,Src>::Unrolling==unrolling;
}

template<typename Xpr>
bool test_sum(const Xpr&, int vectorization, int unrolling)
{
  return ei_sum_traits<Xpr>::Vectorization==vectorization
    && ei_sum_traits<Xpr>::Unrolling==unrolling;
}

void test_vectorization_logic()
{

#ifdef EIGEN_VECTORIZE

#ifdef  EIGEN_DEFAULT_TO_ROW_MAJOR
  VERIFY(test_assign(Vector4f(),Vector4f(),
    LinearVectorization,CompleteUnrolling));
  VERIFY(test_assign(Vector4f(),Vector4f()+Vector4f(),
    LinearVectorization,CompleteUnrolling));
  VERIFY(test_assign(Vector4f(),Vector4f().cwise() * Vector4f(),
    LinearVectorization,CompleteUnrolling));
#else
  VERIFY(test_assign(Vector4f(),Vector4f(),
    InnerVectorization,CompleteUnrolling));
  VERIFY(test_assign(Vector4f(),Vector4f()+Vector4f(),
    InnerVectorization,CompleteUnrolling));
  VERIFY(test_assign(Vector4f(),Vector4f().cwise() * Vector4f(),
    InnerVectorization,CompleteUnrolling));
#endif

  VERIFY(test_assign(Matrix4f(),Matrix4f(),
    InnerVectorization,CompleteUnrolling));
  VERIFY(test_assign(Matrix4f(),Matrix4f()+Matrix4f(),
    InnerVectorization,CompleteUnrolling));
  VERIFY(test_assign(Matrix4f(),Matrix4f().cwise() * Matrix4f(),
    InnerVectorization,CompleteUnrolling));

  VERIFY(test_assign(Matrix<float,16,16>(),Matrix<float,16,16>()+Matrix<float,16,16>(),
    InnerVectorization,InnerUnrolling));

  VERIFY(test_assign(Matrix<float,16,16,DontAlign>(),Matrix<float,16,16>()+Matrix<float,16,16>(),
    NoVectorization,InnerUnrolling));

  VERIFY(test_assign(Matrix<float,6,2>(),Matrix<float,6,2>().cwise() / Matrix<float,6,2>(),
    LinearVectorization,CompleteUnrolling));

  VERIFY(test_assign(Matrix<float,17,17>(),Matrix<float,17,17>()+Matrix<float,17,17>(),
    NoVectorization,InnerUnrolling));

  VERIFY(test_assign(Matrix<float,4,4>(),Matrix<float,17,17>().block<4,4>(2,3)+Matrix<float,17,17>().block<4,4>(10,4),
    NoVectorization,CompleteUnrolling));

  VERIFY(test_assign(MatrixXf(10,10),MatrixXf(20,20).block(10,10,2,3),
    SliceVectorization,NoUnrolling));

  VERIFY(test_assign(VectorXf(10),VectorXf(10)+VectorXf(10),
    LinearVectorization,NoUnrolling));

  VERIFY(test_sum(VectorXf(10),
    LinearVectorization,NoUnrolling));

  VERIFY(test_sum(Matrix<float,5,2>(),
    NoVectorization,CompleteUnrolling));
  
  VERIFY(test_sum(Matrix<float,6,2>(),
    LinearVectorization,CompleteUnrolling));

  VERIFY(test_sum(Matrix<float,16,16>(),
    LinearVectorization,NoUnrolling));

  VERIFY(test_sum(Matrix<float,16,16>().block<4,4>(1,2),
    NoVectorization,CompleteUnrolling));

#ifndef EIGEN_DEFAULT_TO_ROW_MAJOR
  VERIFY(test_sum(Matrix<float,16,16>().block<8,1>(1,2),
    LinearVectorization,CompleteUnrolling));
#endif

  VERIFY(test_sum(Matrix<double,7,3>(),
    NoVectorization,CompleteUnrolling));

#endif // EIGEN_VECTORIZE

}
