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

#include <complex>
#include <vector>
#include <Eigen/Core>
#include <bench/BenchTimer.h>
#ifdef USE_FFTW
#include <fftw3.h>
#endif

#include <unsupported/Eigen/FFT>

using namespace Eigen;
using namespace std;


template <typename T>
string nameof();

template <> string nameof<float>() {return "float";}
template <> string nameof<double>() {return "double";}
template <> string nameof<long double>() {return "long double";}

#ifndef TYPE
#define TYPE float
#endif

#ifndef NFFT
#define NFFT 1024
#endif
#ifndef NDATA
#define NDATA 1000000
#endif

using namespace Eigen;

template <typename T>
void bench(int nfft,bool fwd)
{
    typedef typename NumTraits<T>::Real Scalar;
    typedef typename std::complex<Scalar> Complex;
    int nits = NDATA/nfft;
    vector<T> inbuf(nfft);
    vector<Complex > outbuf(nfft);
    FFT< Scalar > fft;

    fft.fwd( outbuf , inbuf);

    BenchTimer timer;
    timer.reset();
    for (int k=0;k<8;++k) {
        timer.start();
        for(int i = 0; i < nits; i++)
            if (fwd)
                fft.fwd( outbuf , inbuf);
            else
                fft.inv(inbuf,outbuf);
        timer.stop();
    }

    cout << nameof<Scalar>() << " ";
    double mflops = 5.*nfft*log2((double)nfft) / (1e6 * timer.value() / (double)nits );
    if ( NumTraits<T>::IsComplex ) {
        cout << "complex";
    }else{
        cout << "real   ";
        mflops /= 2;
    }

    if (fwd)
        cout << " fwd";
    else
        cout << " inv";

    cout << " NFFT=" << nfft << "  " << (double(1e-6*nfft*nits)/timer.value()) << " MS/s  " << mflops << "MFLOPS\n";
}

int main(int argc,char ** argv)
{
    bench<complex<float> >(NFFT,true);
    bench<complex<float> >(NFFT,false);
    bench<float>(NFFT,true);
    bench<float>(NFFT,false);
    bench<complex<double> >(NFFT,true);
    bench<complex<double> >(NFFT,false);
    bench<double>(NFFT,true);
    bench<double>(NFFT,false);
    bench<complex<long double> >(NFFT,true);
    bench<complex<long double> >(NFFT,false);
    bench<long double>(NFFT,true);
    bench<long double>(NFFT,false);
    return 0;
}
