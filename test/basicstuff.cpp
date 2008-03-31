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

#include "main.h"

namespace Eigen {

template<typename MatrixType> void basicStuff(const MatrixType& m)
{
  typedef typename MatrixType::Scalar Scalar;
  typedef Matrix<Scalar, MatrixType::RowsAtCompileTime, 1> VectorType;

  int rows = m.rows();
  int cols = m.cols();

  // this test relies a lot on Random.h, and there's not much more that we can do
  // to test it, hence I consider that we will have tested Random.h
  MatrixType m1 = MatrixType::random(rows, cols),
             m2 = MatrixType::random(rows, cols),
             m3(rows, cols),
             mzero = MatrixType::zero(rows, cols),
             identity = Matrix<Scalar, MatrixType::RowsAtCompileTime, MatrixType::RowsAtCompileTime>
                              ::identity(rows, rows),
             square = Matrix<Scalar, MatrixType::RowsAtCompileTime, MatrixType::RowsAtCompileTime>
                              ::random(rows, rows);
  VectorType v1 = VectorType::random(rows),
             v2 = VectorType::random(rows),
             vzero = VectorType::zero(rows);

  int r = ei_random<int>(0, rows-1),
      c = ei_random<int>(0, cols-1);

  VERIFY_IS_APPROX(               v1,    v1);
  VERIFY_IS_NOT_APPROX(           v1,    2*v1);
  VERIFY_IS_MUCH_SMALLER_THAN(    vzero, v1);
  if(NumTraits<Scalar>::HasFloatingPoint)
    VERIFY_IS_MUCH_SMALLER_THAN(  vzero, v1.norm());
  VERIFY_IS_NOT_MUCH_SMALLER_THAN(v1,    v1);
  VERIFY_IS_APPROX(               vzero, v1-v1);
  VERIFY_IS_APPROX(               m1,    m1);
  VERIFY_IS_NOT_APPROX(           m1,    2*m1);
  VERIFY_IS_MUCH_SMALLER_THAN(    mzero, m1);
  VERIFY_IS_NOT_MUCH_SMALLER_THAN(m1,    m1);
  VERIFY_IS_APPROX(               mzero, m1-m1);

  // always test operator() on each read-only expression class,
  // in order to check const-qualifiers.
  // indeed, if an expression class (here Zero) is meant to be read-only,
  // hence has no _write() method, the corresponding MatrixBase method (here zero())
  // should return a const-qualified object so that it is the const-qualified
  // operator() that gets called, which in turn calls _read().
  VERIFY_IS_MUCH_SMALLER_THAN(MatrixType::zero(rows,cols)(r,c), static_cast<Scalar>(1));

  // now test copying a row-vector into a (column-)vector and conversely.
  square.col(r) = square.row(r).eval();
  Matrix<Scalar, 1, MatrixType::RowsAtCompileTime> rv(rows);
  Matrix<Scalar, MatrixType::RowsAtCompileTime, 1> cv(rows);
  rv = square.row(r);
  cv = square.col(r);
  VERIFY_IS_APPROX(rv, cv.transpose());

  if(cols!=1 && rows!=1 && MatrixType::SizeAtCompileTime!=Dynamic)
  {
    VERIFY_RAISES_ASSERT(m1 = (m2.block(0,0, rows-1, cols-1)));
  }
}

void EigenTest::testBasicStuff()
{
  for(int i = 0; i < m_repeat; i++) {
    basicStuff(Matrix<float, 1, 1>());
    basicStuff(Matrix4d());
    basicStuff(MatrixXcf(3, 3));
    basicStuff(MatrixXi(8, 12));
    basicStuff(MatrixXcd(20, 20));
    basicStuff(Matrix<float, 100, 100>());
  }

  // some additional basic tests
  {
    Matrix3d m3;
    Matrix4d m4;
    VERIFY_RAISES_ASSERT(m4 = m3);

    VERIFY_RAISES_ASSERT( (m3 << 1, 2, 3, 4, 5, 6, 7, 8) );
    VERIFY_RAISES_ASSERT( (m3 << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10) );

    double data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};

    m3 = Matrix3d::random();
    m3 << 1, 2, 3, 4, 5, 6, 7, 8, 9;
    VERIFY_IS_APPROX(m3, (Matrix<double,3,3,RowMajorBit>::map(data)) );

    Vector3d vec[3];
    vec[0] << 1, 4, 7;
    vec[1] << 2, 5, 8;
    vec[2] << 3, 6, 9;
    m3 = Matrix3d::random();
    m3 << vec[0], vec[1], vec[2];
    VERIFY_IS_APPROX(m3, (Matrix<double,3,3,RowMajorBit>::map(data)) );

    vec[0] << 1, 2, 3;
    vec[1] << 4, 5, 6;
    vec[2] << 7, 8, 9;
    m3 = Matrix3d::random();
    m3 << vec[0].transpose(),
          4, 5, 6,
          vec[2].transpose();
    VERIFY_IS_APPROX(m3, (Matrix<double,3,3,RowMajorBit>::map(data)) );
  }
}

} // namespace Eigen
