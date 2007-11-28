// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
// Copyright (C) 2006-2007 Benoit Jacob <jacob@math.jussieu.fr>
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

template<typename MatrixType> void basicStuff(const MatrixType& m)
{
  /* this test covers the following files:
     1) Explicitly (see comments below):
     Random.h Zero.h Identity.h Fuzzy.h Sum.h Difference.h
     Opposite.h Product.h ScalarMultiple.h FromArray.h
     
     2) Implicitly (the core stuff):
     MatrixBase.h Matrix.h MatrixStorage.h CopyHelper.h MatrixRef.h
     NumTraits.h Util.h
  */

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
                              ::identity(rows),
             square = Matrix<Scalar, MatrixType::RowsAtCompileTime, MatrixType::RowsAtCompileTime>
                              ::random(rows, rows);
  VectorType v1 = VectorType::random(rows),
             v2 = VectorType::random(rows),
             vzero = VectorType::zero(rows);

  Scalar s1 = NumTraits<Scalar>::random(),
         s2 = NumTraits<Scalar>::random();
  
  // test Fuzzy.h and Zero.h.
  QVERIFY(v1.isApprox(v1));
  QVERIFY(!v1.isApprox(2*v1));
  QVERIFY(vzero.isMuchSmallerThan(v1));
  QVERIFY(vzero.isMuchSmallerThan(v1.norm()));
  QVERIFY(!v1.isMuchSmallerThan(v1));
  QVERIFY(vzero.isApprox(v1-v1));
  QVERIFY(m1.isApprox(m1));
  QVERIFY(!m1.isApprox(2*m1));
  QVERIFY(mzero.isMuchSmallerThan(m1));
  QVERIFY(!m1.isMuchSmallerThan(m1));
  QVERIFY(mzero.isApprox(m1-m1));
  
  // test the linear structure, i.e. the following files:
  // Sum.h Difference.h Opposite.h ScalarMultiple.h
  QVERIFY((m1+m1).isApprox(2 * m1));
  QVERIFY((m1+m2-m1).isApprox(m2));
  QVERIFY((-m2+m1+m2).isApprox(m1));
  QVERIFY((m1 * s1).isApprox(s1 * m1));
  QVERIFY(((m1 + m2) * s1).isApprox(s1 * m1 + s1 * m2));
  QVERIFY(((s1 + s2) * m1).isApprox(m1 * s1 + m1 * s2));
  QVERIFY(((m1 - m2) * s1).isApprox(s1 * m1 - s1 * m2));
  QVERIFY(((s1 - s2) * m1).isApprox(m1 * s1 - m1 * s2));
  QVERIFY(((-m1 + m2) * s1).isApprox(-s1 * m1 + s1 * m2));
  QVERIFY(((-s1 + s2) * m1).isApprox(-m1 * s1 + m1 * s2));
  m3 = m2;
  QVERIFY((m3 += m1).isApprox(m1 + m2));
  m3 = m2;
  QVERIFY((m3 -= m1).isApprox(-m1 + m2));
  m3 = m2;
  QVERIFY((m3 *= s1).isApprox(s1 * m2));
  m3 = m2;
  if(NumTraits<Scalar>::HasFloatingPoint
     && s1 != static_cast<Scalar>(0))
    QVERIFY((m3 /= s1).isApprox(m2 / s1));
  
  // begin testing Product.h: only associativity for now
  // (we use Transpose.h but this doesn't count as a test for it)
  QVERIFY(((m1 * m1.transpose()) * m2).isApprox(m1 * (m1.transpose() * m2)));
  m3 = m1;
  m3 *= (m1.transpose() * m2);
  QVERIFY(m3.isApprox(m1 * (m1.transpose() * m2)));
  QVERIFY(m3.isApprox(m1.lazyProduct(m1.transpose() * m2)));
  
  // continue testing Product.h: distributivity
  QVERIFY((square * (m1 + m2)).isApprox(square * m1 + square * m2));
  QVERIFY((square * (m1 - m2)).isApprox(square * m1 - square * m2));
  
  // continue testing Product.h: compatibility with ScalarMultiple.h
  QVERIFY((s1 * (square * m1)).isApprox((s1 * square) * m1));
  QVERIFY((s1 * (square * m1)).isApprox(square * (m1 * s1)));
  
  // continue testing Product.h: lazyProduct
  QVERIFY(square.lazyProduct(m1).isApprox(square * m1));
  
  // test Product.h together with Identity.h. This does test Identity.h.
  QVERIFY(m1.isApprox(identity * m1));
  QVERIFY(v1.isApprox(identity * v1));
  
  // test FromArray.h
  Scalar* array1 = new Scalar[rows];
  Scalar* array2 = new Scalar[rows];
  Matrix<Scalar, Dynamic, 1>::fromArray(array1, rows) = Matrix<Scalar, Dynamic, 1>::random(rows);
  Matrix<Scalar, Dynamic, 1>::fromArray(array2, rows)
    = Matrix<Scalar, Dynamic, 1>::fromArray(array1, rows);
  bool b = Matrix<Scalar, Dynamic, 1>::fromArray(array1, rows)
             .isApprox(Matrix<Scalar, Dynamic, 1>::fromArray(array2, rows));
  QVERIFY(b);
  delete[] array1;
  delete[] array2;
}

void EigenTest::testBasicStuff()
{
  basicStuff(Matrix<float, 1, 1>());
  basicStuff(Matrix<complex<double>, 4, 4>());
  basicStuff(MatrixXcf(3, 3));
  basicStuff(MatrixXi(8, 12));
  basicStuff(MatrixXd(20, 20));
}
