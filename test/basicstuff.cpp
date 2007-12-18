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

namespace Eigen {

template<typename MatrixType> void basicStuff(const MatrixType& m)
{
  /* this test covers the following files:
     1) Explicitly (see comments below):
     Random.h Zero.h Identity.h Fuzzy.h Sum.h Difference.h
     Opposite.h Product.h ScalarMultiple.h Map.h
     
     2) Implicitly (the core stuff):
     MatrixBase.h Matrix.h MatrixStorage.h CopyHelper.h MatrixRef.h
     NumTraits.h Util.h MathFunctions.h OperatorEquals.h Coeffs.h
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

  Scalar s1 = random<Scalar>(),
         s2 = random<Scalar>();
  
  int r = random<int>(0, rows-1),
      c = random<int>(0, cols-1);
  
  // test Fuzzy.h and Zero.h.
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
  
  // test the linear structure, i.e. the following files:
  // Sum.h Difference.h Opposite.h ScalarMultiple.h
  VERIFY_IS_APPROX(-(-m1),                  m1);
  VERIFY_IS_APPROX(m1+m1,                   2*m1);
  VERIFY_IS_APPROX(m1+m2-m1,                m2);
  VERIFY_IS_APPROX(-m2+m1+m2,               m1);
  VERIFY_IS_APPROX(m1*s1,                   s1*m1);
  VERIFY_IS_APPROX((m1+m2)*s1,              s1*m1+s1*m2);
  VERIFY_IS_APPROX((s1+s2)*m1,              m1*s1+m1*s2);
  VERIFY_IS_APPROX((m1-m2)*s1,              s1*m1-s1*m2);
  VERIFY_IS_APPROX((s1-s2)*m1,              m1*s1-m1*s2);
  VERIFY_IS_APPROX((-m1+m2)*s1,             -s1*m1+s1*m2);
  VERIFY_IS_APPROX((-s1+s2)*m1,             -m1*s1+m1*s2);
  m3 = m2; m3 += m1;
  VERIFY_IS_APPROX(m3,                      m1+m2);
  m3 = m2; m3 -= m1;
  VERIFY_IS_APPROX(m3,                      m2-m1);
  m3 = m2; m3 *= s1;
  VERIFY_IS_APPROX(m3,                      s1*m2);
  if(NumTraits<Scalar>::HasFloatingPoint)
  {
    m3 = m2; m3 /= s1;
    VERIFY_IS_APPROX(m3,                    m2/s1);
  }
  
  // again, test operator() to check const-qualification
  VERIFY_IS_APPROX((-m1)(r,c), -(m1(r,c)));
  VERIFY_IS_APPROX((m1-m2)(r,c), (m1(r,c))-(m2(r,c)));
  VERIFY_IS_APPROX((m1+m2)(r,c), (m1(r,c))+(m2(r,c)));
  VERIFY_IS_APPROX((s1*m1)(r,c), s1*(m1(r,c)));
  VERIFY_IS_APPROX((m1*s1)(r,c), (m1(r,c))*s1);
  if(NumTraits<Scalar>::HasFloatingPoint)
    VERIFY_IS_APPROX((m1/s1)(r,c), (m1(r,c))/s1);
  
  // begin testing Product.h: only associativity for now
  // (we use Transpose.h but this doesn't count as a test for it)
  VERIFY_IS_APPROX((m1*m1.transpose())*m2,  m1*(m1.transpose()*m2));
  m3 = m1;
  m3 *= (m1.transpose() * m2);
  VERIFY_IS_APPROX(m3,                      m1*(m1.transpose()*m2));
  VERIFY_IS_APPROX(m3,                      m1.lazyProduct(m1.transpose()*m2));
  
  // continue testing Product.h: distributivity
  VERIFY_IS_APPROX(square*(m1 + m2),        square*m1+square*m2);
  VERIFY_IS_APPROX(square*(m1 - m2),        square*m1-square*m2);
  
  // continue testing Product.h: compatibility with ScalarMultiple.h
  VERIFY_IS_APPROX(s1*(square*m1),          (s1*square)*m1);
  VERIFY_IS_APPROX(s1*(square*m1),          square*(m1*s1));
  
  // continue testing Product.h: lazyProduct
  VERIFY_IS_APPROX(square.lazyProduct(m1),  square*m1);
  // again, test operator() to check const-qualification
  s1 += square.lazyProduct(m1)(r,c);
  
  // test Product.h together with Identity.h
  VERIFY_IS_APPROX(m1,                      identity*m1);
  VERIFY_IS_APPROX(v1,                      identity*v1);
  // again, test operator() to check const-qualification
  VERIFY_IS_APPROX(MatrixType::identity(std::max(rows,cols))(r,c), static_cast<Scalar>(r==c));
  
  // test Map.h
  Scalar* array1 = new Scalar[rows];
  Scalar* array2 = new Scalar[rows];
  typedef Matrix<Scalar, Dynamic, 1> VectorX;
  VectorX::map(array1, rows) = VectorX::random(rows);
  VectorX::map(array2, rows) = VectorX::map(array1, rows);
  VectorX ma1 = VectorX::map(array1, rows);
  VectorX ma2 = VectorX::map(array2, rows);
  VERIFY_IS_APPROX(ma1, ma2);
  VERIFY_IS_APPROX(ma1, VectorX(array2, rows));
  
  delete[] array1;
  delete[] array2;
}

void EigenTest::testBasicStuff()
{
  for(int i = 0; i < m_repeat; i++) {
    basicStuff(Matrix<float, 1, 1>());
    basicStuff(Matrix4d());
    basicStuff(MatrixXcf(3, 3));
    basicStuff(MatrixXi(8, 12));
    basicStuff(MatrixXcd(20, 20));
  }
}

} // namespace Eigen
