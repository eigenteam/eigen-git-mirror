// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
// Copyright (C) 2009 Mark Borgerding mark a borgerding net
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
#ifdef EIGEN_TEST_FUNC
#  include "main.h"
#else 
#  include <iostream>
#  define CALL_SUBTEST(x) x
#  define VERIFY(x) x
#  define test_Complex main
#endif

#include <unsupported/Eigen/Complex>
#include <vector>

using namespace std;
using namespace Eigen;

template <typename T>
void take_std( std::complex<T> * dst, int n )
{
    cout << dst[n-1] << endl;
}


template <typename T>
void syntax()
{
    // this works fine
    Matrix< Complex<T>, 9, 1>  a;
    std::complex<T> * pa = &a[0];
    Complex<T> * pa2 = &a[0];
    take_std( pa,9);

    // this does not work, but I wish it would
    // take_std(&a[0];)
    // this does
    take_std( (std::complex<T> *)&a[0],9);

    // this does not work, but it would be really nice
    //vector< Complex<T> > a; 
    // (on my gcc 4.4.1 )
    // std::vector assumes operator& returns a POD pointer

    // this works fine
    Complex<T> b[9];
    std::complex<T> * pb = &b[0]; // this works fine

    take_std( pb,9);
}

void test_Complex()
{
  CALL_SUBTEST( syntax<float>() );
  CALL_SUBTEST( syntax<double>() );
  CALL_SUBTEST( syntax<long double>() );
} 
