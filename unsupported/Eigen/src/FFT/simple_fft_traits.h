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
#include <iostream>

namespace Eigen {

  template <typename _Scalar>
  struct simple_fft_traits
  {
    typedef _Scalar Scalar;
    typedef std::complex<Scalar> Complex;
    simple_fft_traits() : m_nfft(0) {} 

    template <typename _Src>
    void fwd( Complex * dst,const _Src *src,int nfft)
    {
        prepare(nfft,false);
        work(0, dst, src, 1,1);
    }

    // real-to-complex forward FFT
    // perform two FFTs of src even and src odd
    // then twiddle to recombine them into the half-spectrum format
    // then fill in the conjugate symmetric half
    void fwd( Complex * dst,const Scalar * src,int nfft) 
    {
        if ( nfft&1 ) {
            // use generic mode for odd
            prepare(nfft,false);
            work(0, dst, src, 1,1);
        }else{
            int ncfft = nfft>>1;
            int ncfft2 = nfft>>2;
            // use optimized mode for even real
            fwd( dst, reinterpret_cast<const Complex*> (src),ncfft);
            make_real_twiddles(nfft);
            Complex dc = dst[0].real() +  dst[0].imag();
            Complex nyquist = dst[0].real() -  dst[0].imag();
            int k;
            for ( k=1;k <= ncfft2 ; ++k ) {
                Complex fpk = dst[k];
                Complex fpnk = conj(dst[ncfft-k]);
                Complex f1k = fpk + fpnk;
                Complex f2k = fpk - fpnk;
                //Complex tw = f2k * exp( Complex(0,-3.14159265358979323846264338327 * ((double) (k) / ncfft + .5) ) );
                Complex tw= f2k * m_realTwiddles[k-1];

                dst[k] =  (f1k + tw) * Scalar(.5);
                dst[ncfft-k] =  conj(f1k -tw)*Scalar(.5);
            }
 
            // place conjugate-symmetric half at the end for completeness
            // TODO: make this configurable ( opt-out )
            for ( k=1;k < ncfft ; ++k )
                dst[nfft-k] = conj(dst[k]);

            dst[0] = dc;
            dst[ncfft] = nyquist;
        }
    }

    // half-complex to scalar
    void inv( Scalar * dst,const Complex * src,int nfft) 
    {
        // TODO add optimized version for even numbers
        std::vector<Complex> tmp(nfft);
        inv(&tmp[0],src,nfft);
        for (int k=0;k<nfft;++k)
            dst[k] = tmp[k].real();
    }

    void inv(Complex * dst,const Complex  *src,int nfft)
    {
        prepare(nfft,true);
        work(0, dst, src, 1,1);
        scale(dst, Scalar(1)/m_nfft );
    }

    void prepare(int nfft,bool inverse)
    {
        make_twiddles(nfft,inverse);
        factorize(nfft);
    }

    void make_real_twiddles(int nfft)
    {
        int ncfft2 = nfft>>2;
        if ( m_realTwiddles.size() != ncfft2) {
            m_realTwiddles.resize(ncfft2);
            int ncfft= nfft>>1;
            for (int k=1;k<=ncfft2;++k) 
                m_realTwiddles[k-1] = exp( Complex(0,-3.14159265358979323846264338327 * ((double) (k) / ncfft + .5) ) );
        }
    }

    void make_twiddles(int nfft,bool inverse)
    {
        if ( m_twiddles.size() == nfft) {
            // reuse the twiddles, conjugate if necessary
            if (inverse != m_inverse)
                for (int i=0;i<nfft;++i)
                    m_twiddles[i] = conj( m_twiddles[i] );
        }else{
            m_twiddles.resize(nfft);
            Scalar phinc =  (inverse?2:-2)* acos( (Scalar) -1)  / nfft;
            for (int i=0;i<nfft;++i)
                m_twiddles[i] = exp( Complex(0,i*phinc) );
        }
        m_inverse = inverse;
    }

    void factorize(int nfft)
    {
        if (m_stageRadix.size()==0 || m_stageRadix[0] * m_stageRemainder[0] != nfft)
        {
            m_stageRadix.resize(0);
            m_stageRemainder.resize(0);
            //factorize
            //start factoring out 4's, then 2's, then 3,5,7,9,...
            int n= nfft;
            int p=4;
            do {
                while (n % p) {
                    switch (p) {
                        case 4: p = 2; break;
                        case 2: p = 3; break;
                        default: p += 2; break;
                    }
                    if (p*p>n)
                        p=n;// impossible to have a factor > sqrt(n)
                }
                n /= p;
                m_stageRadix.push_back(p);
                m_stageRemainder.push_back(n);
            }while(n>1);
        }
        m_nfft = nfft;
    }

    void scale(Complex *dst,Scalar s) 
    {
        for (int k=0;k<m_nfft;++k)
            dst[k] *= s;
    }

    private:

    template <typename _Src>
    void work( int stage,Complex * xout, const _Src * xin, size_t fstride,size_t in_stride)
    {
      int p = m_stageRadix[stage];
      int m = m_stageRemainder[stage];
      Complex * Fout_beg = xout;
      Complex * Fout_end = xout + p*m;

      if (m>1) {
        do{
          // recursive call:
          // DFT of size m*p performed by doing
          // p instances of smaller DFTs of size m, 
          // each one takes a decimated version of the input
          work(stage+1, xout , xin, fstride*p,in_stride);
          xin += fstride*in_stride;
        }while( (xout += m) != Fout_end );
      }else{
          do{
              *xout = *xin;
              xin += fstride*in_stride;
          }while(++xout != Fout_end );
      }
      xout=Fout_beg;

      // recombine the p smaller DFTs 
      switch (p) {
        case 2: bfly2(xout,fstride,m); break;
        case 3: bfly3(xout,fstride,m); break;
        case 4: bfly4(xout,fstride,m); break;
        case 5: bfly5(xout,fstride,m); break;
        default: bfly_generic(xout,fstride,m,p); break;
      }
    }

    void bfly2( Complex * Fout, const size_t fstride, int m)
    {
      for (int k=0;k<m;++k) {
        Complex t = Fout[m+k] * m_twiddles[k*fstride];
        Fout[m+k] = Fout[k] - t;
        Fout[k] += t;
      }
    }

    void bfly4( Complex * Fout, const size_t fstride, const size_t m)
    {
      Complex scratch[6];
      int negative_if_inverse = m_inverse * -2 +1;
      for (size_t k=0;k<m;++k) {
        scratch[0] = Fout[k+m] * m_twiddles[k*fstride];
        scratch[1] = Fout[k+2*m] * m_twiddles[k*fstride*2];
        scratch[2] = Fout[k+3*m] * m_twiddles[k*fstride*3];
        scratch[5] = Fout[k] - scratch[1];

        Fout[k] += scratch[1];
        scratch[3] = scratch[0] + scratch[2];
        scratch[4] = scratch[0] - scratch[2];
        scratch[4] = Complex( scratch[4].imag()*negative_if_inverse , -scratch[4].real()* negative_if_inverse );

        Fout[k+2*m]  = Fout[k] - scratch[3];
        Fout[k] += scratch[3];
        Fout[k+m] = scratch[5] + scratch[4];
        Fout[k+3*m] = scratch[5] - scratch[4];
      }
    }

    void bfly3( Complex * Fout, const size_t fstride, const size_t m)
    {
      size_t k=m;
      const size_t m2 = 2*m;
      Complex *tw1,*tw2;
      Complex scratch[5];
      Complex epi3;
      epi3 = m_twiddles[fstride*m];

      tw1=tw2=&m_twiddles[0];

      do{
        scratch[1]=Fout[m] * *tw1;
        scratch[2]=Fout[m2] * *tw2;

        scratch[3]=scratch[1]+scratch[2];
        scratch[0]=scratch[1]-scratch[2];
        tw1 += fstride;
        tw2 += fstride*2;
        Fout[m] = Complex( Fout->real() - .5*scratch[3].real() , Fout->imag() - .5*scratch[3].imag() );
        scratch[0] *= epi3.imag();
        *Fout += scratch[3];
        Fout[m2] = Complex(  Fout[m].real() + scratch[0].imag() , Fout[m].imag() - scratch[0].real() );
        Fout[m] += Complex( -scratch[0].imag(),scratch[0].real() );
        ++Fout;
      }while(--k);
    }

    void bfly5( Complex * Fout, const size_t fstride, const size_t m)
    {
      Complex *Fout0,*Fout1,*Fout2,*Fout3,*Fout4;
      size_t u;
      Complex scratch[13];
      Complex * twiddles = &m_twiddles[0];
      Complex *tw;
      Complex ya,yb;
      ya = twiddles[fstride*m];
      yb = twiddles[fstride*2*m];

      Fout0=Fout;
      Fout1=Fout0+m;
      Fout2=Fout0+2*m;
      Fout3=Fout0+3*m;
      Fout4=Fout0+4*m;

      tw=twiddles;
      for ( u=0; u<m; ++u ) {
        scratch[0] = *Fout0;

        scratch[1]  = *Fout1 * tw[u*fstride];
        scratch[2]  = *Fout2 * tw[2*u*fstride];
        scratch[3]  = *Fout3 * tw[3*u*fstride];
        scratch[4]  = *Fout4 * tw[4*u*fstride];

        scratch[7] = scratch[1] + scratch[4];
        scratch[10] = scratch[1] - scratch[4];
        scratch[8] = scratch[2] + scratch[3];
        scratch[9] = scratch[2] - scratch[3];

        *Fout0 +=  scratch[7];
        *Fout0 +=  scratch[8];

        scratch[5] = scratch[0] + Complex(
            (scratch[7].real()*ya.real() ) + (scratch[8].real() *yb.real() ),
            (scratch[7].imag()*ya.real()) + (scratch[8].imag()*yb.real())
            );

        scratch[6] = Complex(
            (scratch[10].imag()*ya.imag()) + (scratch[9].imag()*yb.imag()),
            -(scratch[10].real()*ya.imag()) - (scratch[9].real()*yb.imag())
            );

        *Fout1 = scratch[5] - scratch[6];
        *Fout4 = scratch[5] + scratch[6];

        scratch[11] = scratch[0] +
          Complex(
              (scratch[7].real()*yb.real()) + (scratch[8].real()*ya.real()),
              (scratch[7].imag()*yb.real()) + (scratch[8].imag()*ya.real())
              );

        scratch[12] = Complex(
            -(scratch[10].imag()*yb.imag()) + (scratch[9].imag()*ya.imag()),
            (scratch[10].real()*yb.imag()) - (scratch[9].real()*ya.imag())
            );

        *Fout2=scratch[11]+scratch[12];
        *Fout3=scratch[11]-scratch[12];

        ++Fout0;++Fout1;++Fout2;++Fout3;++Fout4;
      }
    }

    /* perform the butterfly for one stage of a mixed radix FFT */
    void bfly_generic(
        Complex * Fout,
        const size_t fstride,
        int m,
        int p
        )
    {
      int u,k,q1,q;
      Complex * twiddles = &m_twiddles[0];
      Complex t;
      int Norig = m_nfft;
      Complex * scratchbuf = (Complex*)alloca(p*sizeof(Complex) );

      for ( u=0; u<m; ++u ) {
        k=u;
        for ( q1=0 ; q1<p ; ++q1 ) {
          scratchbuf[q1] = Fout[ k  ];
          k += m;
        }

        k=u;
        for ( q1=0 ; q1<p ; ++q1 ) {
          int twidx=0;
          Fout[ k ] = scratchbuf[0];
          for (q=1;q<p;++q ) {
            twidx += fstride * k;
            if (twidx>=Norig) twidx-=Norig;
            t=scratchbuf[q] * twiddles[twidx];
            Fout[ k ] += t;
          }
          k += m;
        }
      }
    }

    int m_nfft;
    bool m_inverse;
    std::vector<Complex> m_twiddles;
    std::vector<Complex> m_realTwiddles;
    std::vector<int> m_stageRadix;
    std::vector<int> m_stageRemainder;
  };
}
