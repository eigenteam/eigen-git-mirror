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
#include "main.h"
#else 
#include <iostream>

#define CALL_SUBTEST(x) x
#define VERIFY(x) x
#define test_Complex main
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
    vector< Complex<T> > a;
    a.resize( 9 );
    //Complex<T> a[9];
    Complex<T> b[9];

    std::complex<T> * pa = &a[0]; // this works fine
    std::complex<T> * pb = &b[0]; // this works fine
    //VERIFY()
    // this does not compile: 
    //  take_std( &a[0] , a.size() );

    take_std( pa,9);
    take_std( pb,9);
    //take_std( static_cast<std::complex<T> *>( &a[0] ) , a.size() );
    //take_std( &b[0] , 9 );
}

int test_Complex()
{
  CALL_SUBTEST( syntax<float>() );
  //CALL_SUBTEST( syntax<double>() );
  //CALL_SUBTEST( syntax<long double>() );
} 
