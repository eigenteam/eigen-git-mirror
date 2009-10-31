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
#include <unsupported/Eigen/FFT>

using namespace std;

float norm(float x) {return x*x;}
double norm(double x) {return x*x;}
long double norm(long double x) {return x*x;}

template < typename T>
complex<long double>  promote(complex<T> x) { return complex<long double>(x.real(),x.imag()); }

complex<long double>  promote(float x) { return complex<long double>( x); }
complex<long double>  promote(double x) { return complex<long double>( x); }
complex<long double>  promote(long double x) { return complex<long double>( x); }
    

    template <typename VectorType1,typename VectorType2>
    long double fft_rmse( const VectorType1 & fftbuf,const VectorType2 & timebuf)
    {
        long double totalpower=0;
        long double difpower=0;
        cerr <<"idx\ttruth\t\tvalue\t|dif|=\n";
        for (size_t k0=0;k0<size_t(fftbuf.size());++k0) {
            complex<long double> acc = 0;
            long double phinc = -2.*k0* M_PIl / timebuf.size();
            for (size_t k1=0;k1<size_t(timebuf.size());++k1) {
                acc +=  promote( timebuf[k1] ) * exp( complex<long double>(0,k1*phinc) );
            }
            totalpower += norm(acc);
            complex<long double> x = promote(fftbuf[k0]); 
            complex<long double> dif = acc - x;
            difpower += norm(dif);
            cerr << k0 << "\t" << acc << "\t" <<  x << "\t" << sqrt(norm(dif)) << endl;
        }
        cerr << "rmse:" << sqrt(difpower/totalpower) << endl;
        return sqrt(difpower/totalpower);
    }

    template <typename VectorType1,typename VectorType2>
    long double dif_rmse( const VectorType1& buf1,const VectorType2& buf2)
    {
        long double totalpower=0;
        long double difpower=0;
        size_t n = min( buf1.size(),buf2.size() );
        for (size_t k=0;k<n;++k) {
            totalpower += (norm( buf1[k] ) + norm(buf2[k]) )/2.;
            difpower += norm(buf1[k] - buf2[k]);
        }
        return sqrt(difpower/totalpower);
    }

enum { StdVectorContainer, EigenVectorContainer };

template<int Container, typename Scalar> struct VectorType;

template<typename Scalar> struct VectorType<StdVectorContainer,Scalar>
{
  typedef vector<Scalar> type;
};

template<typename Scalar> struct VectorType<EigenVectorContainer,Scalar>
{
  typedef Matrix<Scalar,Dynamic,1> type;
};

template <int Container, typename T>
void test_scalar_generic(int nfft)
{
    typedef typename FFT<T>::Complex Complex;
    typedef typename FFT<T>::Scalar Scalar;
    typedef typename VectorType<Container,Scalar>::type ScalarVector;
    typedef typename VectorType<Container,Complex>::type ComplexVector;

    FFT<T> fft;
    ScalarVector inbuf(nfft);
    ComplexVector outbuf;
    for (int k=0;k<nfft;++k)
        inbuf[k]= (T)(rand()/(double)RAND_MAX - .5);

    // make sure it DOESN'T give the right full spectrum answer
    // if we've asked for half-spectrum
    fft.SetFlag(fft.HalfSpectrum );
    fft.fwd( outbuf,inbuf);
    VERIFY(outbuf.size() == (nfft>>1)+1);
    VERIFY( fft_rmse(outbuf,inbuf) < test_precision<T>()  );// gross check

    fft.ClearFlag(fft.HalfSpectrum );
    fft.fwd( outbuf,inbuf);
    VERIFY( fft_rmse(outbuf,inbuf) < test_precision<T>()  );// gross check

    ScalarVector buf3;
    fft.inv( buf3 , outbuf);
    VERIFY( dif_rmse(inbuf,buf3) < test_precision<T>()  );// gross check

    // verify that the Unscaled flag takes effect
    ComplexVector buf4;
    fft.SetFlag(fft.Unscaled);
    fft.inv( buf4 , outbuf);
    for (int k=0;k<nfft;++k)
        buf4[k] *= T(1./nfft);
    VERIFY( dif_rmse(inbuf,buf4) < test_precision<T>()  );// gross check

    // verify that ClearFlag works
    fft.ClearFlag(fft.Unscaled);
    fft.inv( buf3 , outbuf);
    VERIFY( dif_rmse(inbuf,buf3) < test_precision<T>()  );// gross check
}

template <typename T>
void test_scalar(int nfft)
{
  test_scalar_generic<StdVectorContainer,T>(nfft);
  test_scalar_generic<EigenVectorContainer,T>(nfft);
}

template <int Container, typename T>
void test_complex_generic(int nfft)
{
    typedef typename FFT<T>::Complex Complex;
    typedef typename VectorType<Container,Complex>::type ComplexVector;

    FFT<T> fft;

    ComplexVector inbuf(nfft);
    ComplexVector outbuf;
    ComplexVector buf3;
    for (int k=0;k<nfft;++k)
        inbuf[k]= Complex( (T)(rand()/(double)RAND_MAX - .5), (T)(rand()/(double)RAND_MAX - .5) );
    fft.fwd( outbuf , inbuf);

    VERIFY( fft_rmse(outbuf,inbuf) < test_precision<T>()  );// gross check

    fft.inv( buf3 , outbuf);

    VERIFY( dif_rmse(inbuf,buf3) < test_precision<T>()  );// gross check

    // verify that the Unscaled flag takes effect
    ComplexVector buf4;
    fft.SetFlag(fft.Unscaled);
    fft.inv( buf4 , outbuf);
    for (int k=0;k<nfft;++k)
        buf4[k] *= T(1./nfft);
    VERIFY( dif_rmse(inbuf,buf4) < test_precision<T>()  );// gross check

    // verify that ClearFlag works
    fft.ClearFlag(fft.Unscaled);
    fft.inv( buf3 , outbuf);
    VERIFY( dif_rmse(inbuf,buf3) < test_precision<T>()  );// gross check
}

template <typename T>
void test_complex(int nfft)
{
  test_complex_generic<StdVectorContainer,T>(nfft);
  test_complex_generic<EigenVectorContainer,T>(nfft);
}

void test_FFT()
{

  CALL_SUBTEST( test_complex<float>(32) );
  CALL_SUBTEST( test_complex<double>(32) );
  CALL_SUBTEST( test_complex<long double>(32) );
  
  CALL_SUBTEST( test_complex<float>(256) );
  CALL_SUBTEST( test_complex<double>(256) );
  CALL_SUBTEST( test_complex<long double>(256) );
  
  CALL_SUBTEST( test_complex<float>(3*8) );
  CALL_SUBTEST( test_complex<double>(3*8) );
  CALL_SUBTEST( test_complex<long double>(3*8) );
  
  CALL_SUBTEST( test_complex<float>(5*32) );
  CALL_SUBTEST( test_complex<double>(5*32) );
  CALL_SUBTEST( test_complex<long double>(5*32) );
  
  CALL_SUBTEST( test_complex<float>(2*3*4) );
  CALL_SUBTEST( test_complex<double>(2*3*4) );
  CALL_SUBTEST( test_complex<long double>(2*3*4) );
  
  CALL_SUBTEST( test_complex<float>(2*3*4*5) );
  CALL_SUBTEST( test_complex<double>(2*3*4*5) );
  CALL_SUBTEST( test_complex<long double>(2*3*4*5) );
  
  CALL_SUBTEST( test_complex<float>(2*3*4*5*7) );
  CALL_SUBTEST( test_complex<double>(2*3*4*5*7) );
  CALL_SUBTEST( test_complex<long double>(2*3*4*5*7) );



  CALL_SUBTEST( test_scalar<float>(32) );
  CALL_SUBTEST( test_scalar<double>(32) );
  CALL_SUBTEST( test_scalar<long double>(32) );
  
  CALL_SUBTEST( test_scalar<float>(45) );
  CALL_SUBTEST( test_scalar<double>(45) );
  CALL_SUBTEST( test_scalar<long double>(45) );
  
  CALL_SUBTEST( test_scalar<float>(50) );
  CALL_SUBTEST( test_scalar<double>(50) );
  CALL_SUBTEST( test_scalar<long double>(50) );
  
  CALL_SUBTEST( test_scalar<float>(256) );
  CALL_SUBTEST( test_scalar<double>(256) );
  CALL_SUBTEST( test_scalar<long double>(256) );
  
  CALL_SUBTEST( test_scalar<float>(2*3*4*5*7) );
  CALL_SUBTEST( test_scalar<double>(2*3*4*5*7) );
  CALL_SUBTEST( test_scalar<long double>(2*3*4*5*7) );
}
