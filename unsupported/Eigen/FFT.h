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

#ifndef EIGEN_FFT_H
#define EIGEN_FFT_H

// simple_fft_traits:  small, free, reasonably efficient default, derived from kissfft
#include "src/FFT/simple_fft_traits.h"
#define DEFAULT_FFT_TRAITS simple_fft_traits

// FFTW: faster, GPL-not LGPL, bigger code size
#ifdef FFTW_PATIENT  // definition of FFTW_PATIENT indicates the caller has included fftw3.h, we can use FFTW routines
// TODO 
// #include "src/FFT/fftw_traits.h"
// #define DEFAULT_FFT_TRAITS fftw_traits
#endif

// intel Math Kernel Library: fastest, commerical
#ifdef _MKL_DFTI_H_ // mkl_dfti.h has been included, we can use MKL FFT routines
// TODO 
// #include "src/FFT/imkl_traits.h"
// #define DEFAULT_FFT_TRAITS imkl_traits
#endif

namespace Eigen {

template <typename _Scalar,
         typename _Traits=DEFAULT_FFT_TRAITS<_Scalar> 
         >
class FFT
{
  public:
    typedef _Traits traits_type;
    typedef typename traits_type::Scalar Scalar;
    typedef typename traits_type::Complex Complex;

    FFT(const traits_type & traits=traits_type() ) :m_traits(traits) { }

    template <typename _Input>
    void fwd( Complex * dst, const _Input * src, int nfft)
    {
      m_traits.prepare(nfft,false,dst,src);
      m_traits.exec(dst,src);
      m_traits.postprocess(dst);
    }

    template <typename _Input>
    void fwd( std::vector<Complex> & dst, const std::vector<_Input> & src) 
    {
        dst.resize( src.size() );
        fwd( &dst[0],&src[0],src.size() );
    }

    template <typename _Output>
    void inv( _Output * dst, const Complex * src, int nfft)
    {
        m_traits.prepare(nfft,true,dst,src);
        m_traits.exec(dst,src);
        m_traits.postprocess(dst);
    }

    template <typename _Output>
    void inv( std::vector<_Output> & dst, const std::vector<Complex> & src) 
    {
        dst.resize( src.size() );
        inv( &dst[0],&src[0],src.size() );
    }

    // TODO: multi-dimensional FFTs
    // TODO: handle Eigen MatrixBase

    traits_type & traits() {return m_traits;}
  private:
    traits_type m_traits;
};
#undef DEFAULT_FFT_TRAITS
}
#endif
