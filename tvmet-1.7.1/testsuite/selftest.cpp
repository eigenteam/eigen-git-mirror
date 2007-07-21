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

template<typename T, int n> static void basics1()
{
  const Vector<T, n> vZero(0);
  const Vector<T, n> vOne(1);
  const Matrix<T, n, n> mZero(0);
  const Matrix<T, n, n> mOne(1);
  
  for(int i = 0; i < n; i++) {
    QVERIFY(vZero(i) == T(0));
    QVERIFY(vOne(i) == T(1));
  }
  
  for(int i = 0; i < n; i++) {
    for(int j = 0; j < n; j++) {
      QVERIFY(mZero(i,j) == T(0));
      QVERIFY(mOne(i,j) == T(1));
    }
  }
}

template<typename T> static void basics2()
{
  // test the CommaInitializer
  Vector<T, 3> v1;
  v1 = 1,2,3;
  Matrix<T, 3, 3> m1;
  m1 = 1,4,7,
       2,5,8,
       3,6,9;
  
  QVERIFY(v1(0) == T(1) && v1(1) == T(2) && v1(2) == T(3));

  QVERIFY(m1(0,0) == T(1) && m1(0,1) == T(4) && m1(0,2) == T(7) &&
	  m1(1,0) == T(2) && m1(1,1) == T(5) && m1(1,2) == T(8) &&
	  m1(2,0) == T(3) && m1(2,1) == T(6) && m1(2,2) == T(9));
}

void TvmetTestSuite::selfTest()
{
  basics1<int, 1> ();
  basics1<int, 3> ();
  basics1<float, 4> ();
  basics1<double, 4> ();
  basics2<int> ();
  basics2<double> ();
  
  basics1<complex<float>, 4> ();
  basics1<complex<double>, 4> ();
  basics2<complex<int> > ();
  basics2<complex<double> > ();
}
