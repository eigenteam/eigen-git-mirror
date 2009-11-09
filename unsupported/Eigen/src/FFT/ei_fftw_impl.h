// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. 
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



  // FFTW uses non-const arguments
  // so we must use ugly const_cast calls for all the args it uses
  //
  // This should be safe as long as 
  // 1. we use FFTW_ESTIMATE for all our planning
  //       see the FFTW docs section 4.3.2 "Planner Flags"
  // 2. fftw_complex is compatible with std::complex
  //    This assumes std::complex<T> layout is array of size 2 with real,imag
  template <typename T> 
  inline 
  T * ei_fftw_cast(const T* p) 
  { 
      return const_cast<T*>( p); 
  }

  inline 
  fftw_complex * ei_fftw_cast( const std::complex<double> * p) 
  {
      return const_cast<fftw_complex*>( reinterpret_cast<const fftw_complex*>(p) ); 
  }

  inline 
  fftwf_complex * ei_fftw_cast( const std::complex<float> * p) 
  { 
      return const_cast<fftwf_complex*>( reinterpret_cast<const fftwf_complex*>(p) ); 
  }

  inline 
  fftwl_complex * ei_fftw_cast( const std::complex<long double> * p) 
  { 
      return const_cast<fftwl_complex*>( reinterpret_cast<const fftwl_complex*>(p) ); 
  }

  template <typename T> 
  struct ei_fftw_plan {};

  template <> 
  struct ei_fftw_plan<float>
  {
      typedef float scalar_type;
      typedef fftwf_complex complex_type;
      fftwf_plan m_plan;
      ei_fftw_plan() :m_plan(NULL) {}
      ~ei_fftw_plan() {if (m_plan) fftwf_destroy_plan(m_plan);}

      inline
      void fwd(complex_type * dst,complex_type * src,int nfft) {
          if (m_plan==NULL) m_plan = fftwf_plan_dft_1d(nfft,src,dst, FFTW_FORWARD, FFTW_ESTIMATE);
          fftwf_execute_dft( m_plan, src,dst);
      }
      inline
      void inv(complex_type * dst,complex_type * src,int nfft) {
          if (m_plan==NULL) m_plan = fftwf_plan_dft_1d(nfft,src,dst, FFTW_BACKWARD , FFTW_ESTIMATE);
          fftwf_execute_dft( m_plan, src,dst);
      }
      inline
      void fwd(complex_type * dst,scalar_type * src,int nfft) {
          if (m_plan==NULL) m_plan = fftwf_plan_dft_r2c_1d(nfft,src,dst,FFTW_ESTIMATE);
          fftwf_execute_dft_r2c( m_plan,src,dst);
      }
      inline
      void inv(scalar_type * dst,complex_type * src,int nfft) {
          if (m_plan==NULL)
              m_plan = fftwf_plan_dft_c2r_1d(nfft,src,dst,FFTW_ESTIMATE);
          fftwf_execute_dft_c2r( m_plan, src,dst);
      }
  };
  template <> 
  struct ei_fftw_plan<double>
  {
      typedef double scalar_type;
      typedef fftw_complex complex_type;
      fftw_plan m_plan;
      ei_fftw_plan() :m_plan(NULL) {}
      ~ei_fftw_plan() {if (m_plan) fftw_destroy_plan(m_plan);}

      inline
      void fwd(complex_type * dst,complex_type * src,int nfft) {
          if (m_plan==NULL) m_plan = fftw_plan_dft_1d(nfft,src,dst, FFTW_FORWARD, FFTW_ESTIMATE);
          fftw_execute_dft( m_plan, src,dst);
      }
      inline
      void inv(complex_type * dst,complex_type * src,int nfft) {
          if (m_plan==NULL) m_plan = fftw_plan_dft_1d(nfft,src,dst, FFTW_BACKWARD , FFTW_ESTIMATE);
          fftw_execute_dft( m_plan, src,dst);
      }
      inline
      void fwd(complex_type * dst,scalar_type * src,int nfft) {
          if (m_plan==NULL) m_plan = fftw_plan_dft_r2c_1d(nfft,src,dst,FFTW_ESTIMATE);
          fftw_execute_dft_r2c( m_plan,src,dst);
      }
      inline
      void inv(scalar_type * dst,complex_type * src,int nfft) {
          if (m_plan==NULL)
              m_plan = fftw_plan_dft_c2r_1d(nfft,src,dst,FFTW_ESTIMATE);
          fftw_execute_dft_c2r( m_plan, src,dst);
      }
  };
  template <> 
  struct ei_fftw_plan<long double>
  {
      typedef long double scalar_type;
      typedef fftwl_complex complex_type;
      fftwl_plan m_plan;
      ei_fftw_plan() :m_plan(NULL) {}
      ~ei_fftw_plan() {if (m_plan) fftwl_destroy_plan(m_plan);}

      inline
      void fwd(complex_type * dst,complex_type * src,int nfft) {
          if (m_plan==NULL) m_plan = fftwl_plan_dft_1d(nfft,src,dst, FFTW_FORWARD, FFTW_ESTIMATE);
          fftwl_execute_dft( m_plan, src,dst);
      }
      inline
      void inv(complex_type * dst,complex_type * src,int nfft) {
          if (m_plan==NULL) m_plan = fftwl_plan_dft_1d(nfft,src,dst, FFTW_BACKWARD , FFTW_ESTIMATE);
          fftwl_execute_dft( m_plan, src,dst);
      }
      inline
      void fwd(complex_type * dst,scalar_type * src,int nfft) {
          if (m_plan==NULL) m_plan = fftwl_plan_dft_r2c_1d(nfft,src,dst,FFTW_ESTIMATE);
          fftwl_execute_dft_r2c( m_plan,src,dst);
      }
      inline
      void inv(scalar_type * dst,complex_type * src,int nfft) {
          if (m_plan==NULL)
              m_plan = fftwl_plan_dft_c2r_1d(nfft,src,dst,FFTW_ESTIMATE);
          fftwl_execute_dft_c2r( m_plan, src,dst);
      }
  };

  template <typename _Scalar>
  struct ei_fftw_impl
  {
      typedef _Scalar Scalar;
      typedef std::complex<Scalar> Complex;

      inline
      void clear() 
      {
        m_plans.clear();
      }

      // complex-to-complex forward FFT
      inline
      void fwd( Complex * dst,const Complex *src,int nfft)
      {
        get_plan(nfft,false,dst,src).fwd(ei_fftw_cast(dst), ei_fftw_cast(src),nfft );
      }

      // real-to-complex forward FFT
      inline
      void fwd( Complex * dst,const Scalar * src,int nfft) 
      {
          get_plan(nfft,false,dst,src).fwd(ei_fftw_cast(dst), ei_fftw_cast(src) ,nfft);
      }

      // inverse complex-to-complex
      inline
      void inv(Complex * dst,const Complex  *src,int nfft)
      {
        get_plan(nfft,true,dst,src).inv(ei_fftw_cast(dst), ei_fftw_cast(src),nfft );
      }

      // half-complex to scalar
      inline
      void inv( Scalar * dst,const Complex * src,int nfft) 
      {
        get_plan(nfft,true,dst,src).inv(ei_fftw_cast(dst), ei_fftw_cast(src),nfft );
      }

  protected:
      typedef ei_fftw_plan<Scalar> PlanData;
      typedef std::map<int,PlanData> PlanMap;

      PlanMap m_plans;

      inline
      PlanData & get_plan(int nfft,bool inverse,void * dst,const void * src)
      {
          bool inplace = (dst==src);
          bool aligned = ( (reinterpret_cast<size_t>(src)&15) | (reinterpret_cast<size_t>(dst)&15) ) == 0;
          int key = (nfft<<3 ) | (inverse<<2) | (inplace<<1) | aligned;
          return m_plans[key];
      }
  };
/* vim: set filetype=cpp et sw=2 ts=2 ai: */

