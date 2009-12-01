// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Hauke Heibel <hauke.heibel@gmail.com>
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

#define EIGEN_OLD_NESTED

#include "Eigen/Core"
#include "Eigen/Array"
#include "Eigen/Geometry"

#include "Bench/BenchTimer.h"

using namespace Eigen;

struct Transform2D
{
  static void run(int num_runs)
  {
    const Matrix2d T = Matrix2d::Random();
    const Vector2d t = Vector2d::Random();
    const Matrix2Xd pts = Matrix2Xd::Random(2,100);

    Matrix2Xd res;
    for (int i=0; i<num_runs; ++i)
    {
      run(res, T, pts, t);
    }
  }

  EIGEN_DONT_INLINE static void run(Matrix2Xd& res, const Matrix2d& T, const Matrix2Xd& pts, const Vector2d& t)
  {
    res = T * pts + Replicate<Vector2d,1,100>(t);
  }
};

struct ColwiseTransform2D
{
  static void run(int num_runs)
  {
    const Matrix2d T = Matrix2d::Random();
    const Vector2d t = Vector2d::Random();
    const Matrix2Xd pts = Matrix2Xd::Random(2,100);

    Matrix2Xd res;
    for (int i=0; i<num_runs; ++i)
    {
      run(res, T, pts, t);
    }
  }

  EIGEN_DONT_INLINE static void run(Matrix2Xd& res, const Matrix2d& T, const Matrix2Xd& pts, const Vector2d& t)
  {
    res = T * pts + Replicate<Vector2d,1,100>(t);
  }
};

struct LinearCombination
{
  typedef Eigen::Matrix<double,2,4> Matrix2x4d;

  static void run(int num_runs)
  {
    const Matrix2Xd pts = Matrix2Xd::Random(2,100);
    const Matrix2x4d coefs = Matrix2x4d::Random();

    Matrix2x4d linear_combined = Matrix2x4d::Zero();
    for (int i=0; i<num_runs; ++i)
    {
      for (int r=0; r<coefs.rows(); ++r)
      {
        for (int c=0; c<pts.cols()-coefs.cols()+1; ++c)
        {
          run(linear_combined, pts, coefs, r, c);
        }
      }
    }
  }

  EIGEN_DONT_INLINE static void run(Matrix2x4d& res, const Matrix2Xd& pts, const Matrix2x4d& coefs, int r, int c)
  {
    res += pts.block(0,c,2,coefs.cols()).cwise() * Replicate<Matrix2x4d::RowXpr,2,1>(coefs.row(r));
  }
};

template <typename VectorType>
struct VectorAddition
{
  typedef VectorType ReturnType;
  EIGEN_DONT_INLINE static VectorType run(int)
  {
    VectorType a,b,c,d;
    return a+b+c+d;
  }
};

template <typename MatrixType>
struct MatrixProduct
{
  typedef MatrixType ReturnType;
  EIGEN_DONT_INLINE static MatrixType run(int num_runs)
  {
    MatrixType a,b;
    return a*b;
  }
};

template <typename MatrixType>
struct MatrixScaling
{
  typedef MatrixType ReturnType;
  EIGEN_DONT_INLINE static MatrixType run(int num_runs)
  {
      typename ei_traits<MatrixType>::Scalar s;
      MatrixType a,b;
      return s*a;
  }
};

template<typename TestFunction>
EIGEN_DONT_INLINE void run(int num_runs)
{
  for (int outer_runs=0; outer_runs<30; ++outer_runs)
  {
    //BenchTimer timer;
    //const double start = timer.getTime();
    {
      TestFunction::run(num_runs);
    }
    //const double stop = timer.getTime();
    //std::cout << (stop-start)*1000.0 << " ms" << std::endl;
  }
}

template<typename TestFunction>
EIGEN_DONT_INLINE void run_direct(int num_runs = 1)
{
  for (int outer_runs=0; outer_runs<30; ++outer_runs)
  {
    // required to prevent that the compiler replaces the run-call by nop
    typename TestFunction::ReturnType return_type;
    for (int i=0; i<num_runs; ++i)
    {
      return_type += TestFunction::run(num_runs);
    }
  }
}

void test_nesting_profiling()
{
  const int num_runs = 100000;

  BenchTimer timer;
  const double start = timer.getTime();
  {
    // leads to better run-time
    run<Transform2D>(num_runs);
    run<ColwiseTransform2D>(num_runs);
    run<LinearCombination>(num_runs);
  }
  const double stop = timer.getTime();
  std::cout << (stop-start)*1000.0 << " ms" << std::endl;

  // leads to identical assembly
  run_direct< MatrixProduct<Matrix2d> >();
  run_direct< MatrixProduct<Matrix3d> >();
  run_direct< MatrixProduct<Matrix4d> >();

  // leads to identical assembly
  run_direct< MatrixScaling<Matrix2d> >();
  run_direct< MatrixScaling<Matrix3d> >();
  run_direct< MatrixScaling<Matrix4d> >();

  // leads to better assembly
  run_direct< VectorAddition<Vector4f> >();
  run_direct< VectorAddition<Vector4d> >();
  run_direct< VectorAddition<Vector4i> >();
}
