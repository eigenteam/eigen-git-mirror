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

#include "main.h"
#include <unsupported/Eigen/FFT.h>

//#include <iostream>
//#include <cstdlib>
//#include <typeinfo>

using namespace std;

template <class T>
void test_fft(int nfft)
{
    typedef typename Eigen::FFT<T>::Complex Complex;

    //cout << "type:" << typeid(T).name() << " nfft:" << nfft;

    FFT<T> fft;

    vector<Complex> inbuf(nfft);
    vector<Complex> buf3(nfft);
    vector<Complex> outbuf(nfft);
    for (int k=0;k<nfft;++k)
        inbuf[k]= Complex( 
                (T)(rand()/(double)RAND_MAX - .5),
                (T)(rand()/(double)RAND_MAX - .5) );
    fft.fwd( &outbuf[0] , &inbuf[0] ,nfft);
    fft.inv( &buf3[0] , &outbuf[0] ,nfft);

    long double totalpower=0;
    long double difpower=0;
    for (int k0=0;k0<nfft;++k0) {
        complex<long double> acc = 0;
        long double phinc = 2*k0* M_PIl / nfft;
        for (int k1=0;k1<nfft;++k1) {
            complex<long double> x(inbuf[k1].real(),inbuf[k1].imag()); 
            acc += x * exp( complex<long double>(0,-k1*phinc) );
        }
        totalpower += norm(acc);
        complex<long double> x(outbuf[k0].real(),outbuf[k0].imag()); 
        complex<long double> dif = acc - x;
        difpower += norm(dif);
    }
    long double rmse = sqrt(difpower/totalpower);
    VERIFY( rmse < 1e-5 );// gross check

    totalpower=0;
    difpower=0;
    for (int k=0;k<nfft;++k) {
        totalpower += norm( inbuf[k] );
        difpower += norm(inbuf[k] - buf3[k]);
    }
    rmse = sqrt(difpower/totalpower);
    VERIFY( rmse < 1e-5 );// gross check
}

void test_FFT()
{
  CALL_SUBTEST(( test_fft<float>(32) )); CALL_SUBTEST(( test_fft<double>(32) )); CALL_SUBTEST(( test_fft<long double>(32) ));
  CALL_SUBTEST(( test_fft<float>(1024) )); CALL_SUBTEST(( test_fft<double>(1024) )); CALL_SUBTEST(( test_fft<long double>(1024) ));
  CALL_SUBTEST(( test_fft<float>(2*3*4*5*7) )); CALL_SUBTEST(( test_fft<double>(2*3*4*5*7) )); CALL_SUBTEST(( test_fft<long double>(2*3*4*5*7) ));
}
