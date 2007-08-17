/* This file is part of Eigen, a C++ template library for linear algebra
 * Copyright (C) 2007 Benoit Jacob <jacob@math.jussieu.fr>
 *
 * Based on Tvmet source code, http://tvmet.sourceforge.net,
 * Copyright (C) 2001 - 2003 Olaf Petzold <opetzold@users.sourceforge.net>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 * $Id: SelfTest.cc,v 1.1 2004/04/24 11:55:15 opetzold Exp $
 */

#include "main.h"

template<typename T>
bool operator==(const tvmet::Vector<T, 3> & v1, const tvmet::Vector<T, 3> & v2)
{
  bool ret = true;
  for(int i = 0; i < 3; i++) if(v1(i) != v2(i)) ret = false;
  return ret;
}

template<typename T>
bool operator==(const tvmet::Matrix<T, 3, 3> & v1, const tvmet::Matrix<T, 3, 3> & v2)
{
  bool ret = true;
  for(int i = 0; i < 3; i++)
    for(int j = 0; j < 3; j++)
      if(v1(i,j) != v2(i,j)) ret = false;
  return ret;
}

template<typename T> struct TestConstructors
{
  typedef tvmet::Vector<T, 3> vector_type;
  typedef tvmet::Matrix<T, 3, 3> matrix_type;

  void vector_ctor2()
  {
    T data[] = {1,2,3};
    vector_type v(data);
    TEST(v == v1);
  }
  
  void vector_ctor5()
  {
    vector_type v(v1 - v1); // expression
    TEST(v == vZero);
  }
  
  void vector_operator_eq()
  {
    vector_type v;
    v = v1;
    TEST(v == v1);
  }
  
  void vector_copy_ctor()
  {
    vector_type v(v1);
    TEST(v == v1);
  }
  
  void matrix_ctor2()
  {
    T data[] = { 1, 2, 3,
                 4, 5, 6,
                 7, 8, 9 };
    matrix_type m(data);
    TEST(m == m1);
  }
  
  void matrix_ctor4()
  {
    matrix_type m(m1 - m1); // expression
    TEST(m == mZero);
  }
  
  void matrix_operator_eq()
  {
    matrix_type m;
    m = m1;
    TEST(m == m1);
  }
  
  void matrix_copy_ctor()
  {
    matrix_type m(m1);
    TEST(m == m1);
  }
  
  TestConstructors()
  {
    vZero = 0,0,0;
    vOne = 1,1,1;
    mZero = 0,0,0,
            0,0,0,
            0,0,0;
    mOne = 1,1,1,
           1,1,1,
           1,1,1;
    v1 = 1,2,3;
    m1 = 1,4,7,
         2,5,8,
         3,6,9;

    vector_ctor2();
    vector_ctor5();
    vector_operator_eq();
    vector_copy_ctor();
  
    matrix_ctor2();
    matrix_ctor4();
    matrix_operator_eq();
    matrix_copy_ctor();
  }
  
private:
  vector_type vZero;
  vector_type vOne;
  vector_type v1;

  matrix_type mZero;
  matrix_type mOne;
  matrix_type m1;
};

void TvmetTestSuite::testConstructors()
{
  TestConstructors<int>();
  TestConstructors<float>();
  TestConstructors<double>();
  TestConstructors<std::complex<int> >();
  TestConstructors<std::complex<float> >();
  TestConstructors<std::complex<double> >();
}
