// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2009 Gael Guennebaud <g.gael@free.fr>
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

#ifndef EIGEN_GENERAL_MATRIX_MATRIX_H
#define EIGEN_GENERAL_MATRIX_MATRIX_H

template<typename Scalar, typename Packet, int PacketSize, int mr, int nr, typename Conj>
struct ei_gebp_kernel;

template<typename Scalar, int PacketSize, int nr>
struct ei_gemm_pack_rhs;

template<typename Scalar, int mr>
struct ei_gemm_pack_lhs;

template <int L2MemorySize,typename Scalar>
struct ei_L2_block_traits {
  enum {width = 8 * ei_meta_sqrt<L2MemorySize/(64*sizeof(Scalar))>::ret };
};

template<bool ConjLhs, bool ConjRhs> struct ei_conj_helper;

template<> struct ei_conj_helper<false,false>
{
  template<typename T>
  EIGEN_STRONG_INLINE T pmadd(const T& x, const T& y, const T& c) const { return  ei_pmadd(x,y,c); }
  template<typename T>
  EIGEN_STRONG_INLINE T pmul(const T& x, const T& y) const { return  ei_pmul(x,y); }
};

template<> struct ei_conj_helper<false,true>
{
  template<typename T> std::complex<T>
  pmadd(const std::complex<T>& x, const std::complex<T>& y, const std::complex<T>& c) const
  { return c + pmul(x,y); }

  template<typename T> std::complex<T> pmul(const std::complex<T>& x, const std::complex<T>& y) const
  { return std::complex<T>(ei_real(x)*ei_real(y) + ei_imag(x)*ei_imag(y), ei_imag(x)*ei_real(y) - ei_real(x)*ei_imag(y)); }
};

template<> struct ei_conj_helper<true,false>
{
  template<typename T> std::complex<T>
  pmadd(const std::complex<T>& x, const std::complex<T>& y, const std::complex<T>& c) const
  { return c + pmul(x,y); }

  template<typename T> std::complex<T> pmul(const std::complex<T>& x, const std::complex<T>& y) const
  { return std::complex<T>(ei_real(x)*ei_real(y) + ei_imag(x)*ei_imag(y), ei_real(x)*ei_imag(y) - ei_imag(x)*ei_real(y)); }
};

template<> struct ei_conj_helper<true,true>
{
  template<typename T> std::complex<T>
  pmadd(const std::complex<T>& x, const std::complex<T>& y, const std::complex<T>& c) const
  { return c + pmul(x,y); }

  template<typename T> std::complex<T> pmul(const std::complex<T>& x, const std::complex<T>& y) const
  { return std::complex<T>(ei_real(x)*ei_real(y) - ei_imag(x)*ei_imag(y), - ei_real(x)*ei_imag(y) - ei_imag(x)*ei_real(y)); }
};

#ifndef EIGEN_EXTERN_INSTANTIATIONS

/** \warning you should never call this function directly,
  * this is because the ConjugateLhs/ConjugateRhs have to
  * be flipped is resRowMajor==true */
template<typename Scalar, bool ConjugateLhs, bool ConjugateRhs>
static void ei_cache_friendly_product(
  int _rows, int _cols, int depth,
  bool _lhsRowMajor, const Scalar* _lhs, int _lhsStride,
  bool _rhsRowMajor, const Scalar* _rhs, int _rhsStride,
  bool resRowMajor, Scalar* res, int resStride,
  Scalar alpha)
{
  const Scalar* EIGEN_RESTRICT lhs;
  const Scalar* EIGEN_RESTRICT rhs;
  int lhsStride, rhsStride, rows, cols;
  bool lhsRowMajor;

  ei_conj_helper<ConjugateLhs,ConjugateRhs> cj;
  if (ConjugateRhs)
    alpha = ei_conj(alpha);
  bool hasAlpha = alpha != Scalar(1);

  if (resRowMajor)
  {
//     return ei_cache_friendly_product<Scalar,ConjugateRhs,ConjugateLhs>(_cols,_rows,depth,
//       !_rhsRowMajor, _rhs, _rhsStride,
//       !_lhsRowMajor, _lhs, _lhsStride,
//       false, res, resStride,
//       alpha);

    lhs = _rhs;
    rhs = _lhs;
    lhsStride = _rhsStride;
    rhsStride = _lhsStride;
    cols = _rows;
    rows = _cols;
    lhsRowMajor = !_rhsRowMajor;
    ei_assert(_lhsRowMajor);
  }
  else
  {
    lhs = _lhs;
    rhs = _rhs;
    lhsStride = _lhsStride;
    rhsStride = _rhsStride;
    rows = _rows;
    cols = _cols;
    lhsRowMajor = _lhsRowMajor;
    ei_assert(!_rhsRowMajor);
  }

  typedef typename ei_packet_traits<Scalar>::type PacketType;

  enum {
    PacketSize = sizeof(PacketType)/sizeof(Scalar),
    #if (defined __i386__)
    HalfRegisterCount = 4,
    #else
    HalfRegisterCount = 8,
    #endif

    // register block size along the N direction
    nr = HalfRegisterCount/2,

    // register block size along the M direction
    mr = 2 * PacketSize,

    // max cache block size along the K direction
    Max_kc = ei_L2_block_traits<EIGEN_TUNE_FOR_CPU_CACHE_SIZE,Scalar>::width,

    // max cache block size along the M direction
    Max_mc = 2*Max_kc
  };

  int kc = std::min<int>(Max_kc,depth);  // cache block size along the K direction
  int mc = std::min<int>(Max_mc,rows);   // cache block size along the M direction

  Scalar* blockA = ei_aligned_stack_new(Scalar, kc*mc);
  Scalar* blockB = ei_aligned_stack_new(Scalar, kc*cols*PacketSize);

  // number of columns which can be processed by packet of nr columns
  int packet_cols = (cols/nr)*nr;

  // GEMM_VAR1
  for(int k2=0; k2<depth; k2+=kc)
  {
    const int actual_kc = std::min(k2+kc,depth)-k2;

    // we have selected one row panel of rhs and one column panel of lhs
    // pack rhs's panel into a sequential chunk of memory
    // and expand each coeff to a constant packet for further reuse
    ei_gemm_pack_rhs<Scalar,PacketSize,nr>()(blockB, rhs, rhsStride, hasAlpha, alpha, actual_kc, packet_cols, k2, cols);

    // => GEPP_VAR1
    for(int i2=0; i2<rows; i2+=mc)
    {
      const int actual_mc = std::min(i2+mc,rows)-i2;

      ei_gemm_pack_lhs<Scalar,mr>()(blockA, lhs, lhsStride, lhsRowMajor, actual_kc, actual_mc, k2, i2);

      ei_gebp_kernel<Scalar, PacketType, PacketSize, mr, nr, ei_conj_helper<ConjugateLhs,ConjugateRhs> >()
        (res, resStride, blockA, blockB, actual_mc, actual_kc, packet_cols, i2, cols);
    }
  }

  ei_aligned_stack_delete(Scalar, blockA, kc*mc);
  ei_aligned_stack_delete(Scalar, blockB, kc*cols*PacketSize);
}

// optimized GEneral packed Block * packed Panel product kernel
template<typename Scalar, typename PacketType, int PacketSize, int mr, int nr, typename Conj>
struct ei_gebp_kernel
{
  void operator()(Scalar* res, int resStride, const Scalar* blockA, const Scalar* blockB, int actual_mc, int actual_kc, int packet_cols, int i2, int cols)
  {
    Conj cj;
    const int peeled_mc = (actual_mc/mr)*mr;
    // loops on each cache friendly block of the result/rhs
    for(int j2=0; j2<packet_cols; j2+=nr)
    {
      // loops on each register blocking of lhs/res
      for(int i=0; i<peeled_mc; i+=mr)
      {
        const Scalar* blA = &blockA[i*actual_kc];
        #ifdef EIGEN_VECTORIZE_SSE
        _mm_prefetch((const char*)(&blA[0]), _MM_HINT_T0);
        #endif

        // TODO move the res loads to the stores

        // gets res block as register
        PacketType C0, C1, C2, C3, C4, C5, C6, C7;
                  C0 = ei_ploadu(&res[(j2+0)*resStride + i2 + i]);
                  C1 = ei_ploadu(&res[(j2+1)*resStride + i2 + i]);
        if(nr==4) C2 = ei_ploadu(&res[(j2+2)*resStride + i2 + i]);
        if(nr==4) C3 = ei_ploadu(&res[(j2+3)*resStride + i2 + i]);
                  C4 = ei_ploadu(&res[(j2+0)*resStride + i2 + i + PacketSize]);
                  C5 = ei_ploadu(&res[(j2+1)*resStride + i2 + i + PacketSize]);
        if(nr==4) C6 = ei_ploadu(&res[(j2+2)*resStride + i2 + i + PacketSize]);
        if(nr==4) C7 = ei_ploadu(&res[(j2+3)*resStride + i2 + i + PacketSize]);

        // performs "inner" product
        // TODO let's check wether the flowing peeled loop could not be
        //      optimized via optimal prefetching from one loop to the other
        const Scalar* blB = &blockB[j2*actual_kc*PacketSize];
        const int peeled_kc = (actual_kc/4)*4;
        for(int k=0; k<peeled_kc; k+=4)
        {
          PacketType B0, B1, B2, B3, A0, A1;

                    A0 = ei_pload(&blA[0*PacketSize]);
                    A1 = ei_pload(&blA[1*PacketSize]);
                    B0 = ei_pload(&blB[0*PacketSize]);
                    B1 = ei_pload(&blB[1*PacketSize]);
                    C0 = cj.pmadd(A0, B0, C0);
          if(nr==4) B2 = ei_pload(&blB[2*PacketSize]);
                    C4 = cj.pmadd(A1, B0, C4);
          if(nr==4) B3 = ei_pload(&blB[3*PacketSize]);
                    B0 = ei_pload(&blB[(nr==4 ? 4 : 2)*PacketSize]);
                    C1 = cj.pmadd(A0, B1, C1);
                    C5 = cj.pmadd(A1, B1, C5);
                    B1 = ei_pload(&blB[(nr==4 ? 5 : 3)*PacketSize]);
          if(nr==4) C2 = cj.pmadd(A0, B2, C2);
          if(nr==4) C6 = cj.pmadd(A1, B2, C6);
          if(nr==4) B2 = ei_pload(&blB[6*PacketSize]);
          if(nr==4) C3 = cj.pmadd(A0, B3, C3);
                    A0 = ei_pload(&blA[2*PacketSize]);
          if(nr==4) C7 = cj.pmadd(A1, B3, C7);
                    A1 = ei_pload(&blA[3*PacketSize]);
          if(nr==4) B3 = ei_pload(&blB[7*PacketSize]);
                    C0 = cj.pmadd(A0, B0, C0);
                    C4 = cj.pmadd(A1, B0, C4);
                    B0 = ei_pload(&blB[(nr==4 ? 8 : 4)*PacketSize]);
                    C1 = cj.pmadd(A0, B1, C1);
                    C5 = cj.pmadd(A1, B1, C5);
                    B1 = ei_pload(&blB[(nr==4 ? 9 : 5)*PacketSize]);
          if(nr==4) C2 = cj.pmadd(A0, B2, C2);
          if(nr==4) C6 = cj.pmadd(A1, B2, C6);
          if(nr==4) B2 = ei_pload(&blB[10*PacketSize]);
          if(nr==4) C3 = cj.pmadd(A0, B3, C3);
                    A0 = ei_pload(&blA[4*PacketSize]);
          if(nr==4) C7 = cj.pmadd(A1, B3, C7);
                    A1 = ei_pload(&blA[5*PacketSize]);
          if(nr==4) B3 = ei_pload(&blB[11*PacketSize]);

                    C0 = cj.pmadd(A0, B0, C0);
                    C4 = cj.pmadd(A1, B0, C4);
                    B0 = ei_pload(&blB[(nr==4 ? 12 : 6)*PacketSize]);
                    C1 = cj.pmadd(A0, B1, C1);
                    C5 = cj.pmadd(A1, B1, C5);
                    B1 = ei_pload(&blB[(nr==4 ? 13 : 7)*PacketSize]);
          if(nr==4) C2 = cj.pmadd(A0, B2, C2);
          if(nr==4) C6 = cj.pmadd(A1, B2, C6);
          if(nr==4) B2 = ei_pload(&blB[14*PacketSize]);
          if(nr==4) C3 = cj.pmadd(A0, B3, C3);
                    A0 = ei_pload(&blA[6*PacketSize]);
          if(nr==4) C7 = cj.pmadd(A1, B3, C7);
                    A1 = ei_pload(&blA[7*PacketSize]);
          if(nr==4) B3 = ei_pload(&blB[15*PacketSize]);
                    C0 = cj.pmadd(A0, B0, C0);
                    C4 = cj.pmadd(A1, B0, C4);
                    C1 = cj.pmadd(A0, B1, C1);
                    C5 = cj.pmadd(A1, B1, C5);
          if(nr==4) C2 = cj.pmadd(A0, B2, C2);
          if(nr==4) C6 = cj.pmadd(A1, B2, C6);
          if(nr==4) C3 = cj.pmadd(A0, B3, C3);
          if(nr==4) C7 = cj.pmadd(A1, B3, C7);

          blB += 4*nr*PacketSize;
          blA += 4*mr;
        }
        // process remaining peeled loop
        for(int k=peeled_kc; k<actual_kc; k++)
        {
          PacketType B0, B1, B2, B3, A0, A1;

                    A0 = ei_pload(&blA[0*PacketSize]);
                    A1 = ei_pload(&blA[1*PacketSize]);
                    B0 = ei_pload(&blB[0*PacketSize]);
                    B1 = ei_pload(&blB[1*PacketSize]);
                    C0 = cj.pmadd(A0, B0, C0);
          if(nr==4) B2 = ei_pload(&blB[2*PacketSize]);
                    C4 = cj.pmadd(A1, B0, C4);
          if(nr==4) B3 = ei_pload(&blB[3*PacketSize]);
                    C1 = cj.pmadd(A0, B1, C1);
                    C5 = cj.pmadd(A1, B1, C5);
          if(nr==4) C2 = cj.pmadd(A0, B2, C2);
          if(nr==4) C6 = cj.pmadd(A1, B2, C6);
          if(nr==4) C3 = cj.pmadd(A0, B3, C3);
          if(nr==4) C7 = cj.pmadd(A1, B3, C7);

          blB += nr*PacketSize;
          blA += mr;
        }

                  ei_pstoreu(&res[(j2+0)*resStride + i2 + i], C0);
                  ei_pstoreu(&res[(j2+1)*resStride + i2 + i], C1);
        if(nr==4) ei_pstoreu(&res[(j2+2)*resStride + i2 + i], C2);
        if(nr==4) ei_pstoreu(&res[(j2+3)*resStride + i2 + i], C3);
                  ei_pstoreu(&res[(j2+0)*resStride + i2 + i + PacketSize], C4);
                  ei_pstoreu(&res[(j2+1)*resStride + i2 + i + PacketSize], C5);
        if(nr==4) ei_pstoreu(&res[(j2+2)*resStride + i2 + i + PacketSize], C6);
        if(nr==4) ei_pstoreu(&res[(j2+3)*resStride + i2 + i + PacketSize], C7);
      }
      for(int i=peeled_mc; i<actual_mc; i++)
      {
        const Scalar* blA = &blockA[i*actual_kc];
        #ifdef EIGEN_VECTORIZE_SSE
        _mm_prefetch((const char*)(&blA[0]), _MM_HINT_T0);
        #endif

        // gets a 1 x nr res block as registers
        Scalar C0(0), C1(0), C2(0), C3(0);
        const Scalar* blB = &blockB[j2*actual_kc*PacketSize];
        for(int k=0; k<actual_kc; k++)
        {
          Scalar B0, B1, B2, B3, A0;

                    A0 =  blA[k];
                    B0 =  blB[0*PacketSize];
                    B1 =  blB[1*PacketSize];
                    C0 = cj.pmadd(A0, B0, C0);
          if(nr==4) B2 =  blB[2*PacketSize];
          if(nr==4) B3 =  blB[3*PacketSize];
                    C1 = cj.pmadd(A0, B1, C1);
          if(nr==4) C2 = cj.pmadd(A0, B2, C2);
          if(nr==4) C3 = cj.pmadd(A0, B3, C3);

          blB += nr*PacketSize;
        }
        res[(j2+0)*resStride + i2 + i] += C0;
        res[(j2+1)*resStride + i2 + i] += C1;
        if(nr==4) res[(j2+2)*resStride + i2 + i] += C2;
        if(nr==4) res[(j2+3)*resStride + i2 + i] += C3;
      }
    }
    // remaining rhs/res columns (<nr)

    // process remaining rhs/res columns one at a time
    // => do the same but with nr==1
    for(int j2=packet_cols; j2<cols; j2++)
    {
      for(int i=0; i<peeled_mc; i+=mr)
      {
        const Scalar* blA = &blockA[i*actual_kc];
        #ifdef EIGEN_VECTORIZE_SSE
        _mm_prefetch((const char*)(&blA[0]), _MM_HINT_T0);
        #endif

        // TODO move the res loads to the stores

        // gets res block as register
        PacketType C0, C4;
        C0 = ei_ploadu(&res[(j2+0)*resStride + i2 + i]);
        C4 = ei_ploadu(&res[(j2+0)*resStride + i2 + i + PacketSize]);

        const Scalar* blB = &blockB[j2*actual_kc*PacketSize];
        for(int k=0; k<actual_kc; k++)
        {
          PacketType B0, A0, A1;

          A0 = ei_pload(&blA[0*PacketSize]);
          A1 = ei_pload(&blA[1*PacketSize]);
          B0 = ei_pload(&blB[0*PacketSize]);
          C0 = cj.pmadd(A0, B0, C0);
          C4 = cj.pmadd(A1, B0, C4);

          blB += PacketSize;
          blA += mr;
        }

        ei_pstoreu(&res[(j2+0)*resStride + i2 + i], C0);
        ei_pstoreu(&res[(j2+0)*resStride + i2 + i + PacketSize], C4);
      }
      for(int i=peeled_mc; i<actual_mc; i++)
      {
        const Scalar* blA = &blockA[i*actual_kc];
        #ifdef EIGEN_VECTORIZE_SSE
        _mm_prefetch((const char*)(&blA[0]), _MM_HINT_T0);
        #endif

        // gets a 1 x 1 res block as registers
        Scalar C0(0);
        const Scalar* blB = &blockB[j2*actual_kc*PacketSize];
        for(int k=0; k<actual_kc; k++)
          C0 = cj.pmadd(blA[k], blB[k*PacketSize], C0);
        res[(j2+0)*resStride + i2 + i] += C0;
      }
    }
  }
};

// pack a block of the lhs
template<typename Scalar, int mr>
struct ei_gemm_pack_lhs
{
  void operator()(Scalar* blockA, const Scalar* lhs, int lhsStride, bool lhsRowMajor, int actual_kc, int actual_mc, int k2, int i2)
  {
    int count = 0;
    const int peeled_mc = (actual_mc/mr)*mr;
    if (lhsRowMajor)
    {
      for(int i=0; i<peeled_mc; i+=mr)
        for(int k=0; k<actual_kc; k++)
          for(int w=0; w<mr; w++)
            blockA[count++] = lhs[(k2+k) + (i2+i+w)*lhsStride];
      for(int i=peeled_mc; i<actual_mc; i++)
      {
        const Scalar* llhs = &lhs[(k2) + (i2+i)*lhsStride];
        for(int k=0; k<actual_kc; k++)
          blockA[count++] = llhs[k];
      }
    }
    else
    {
      for(int i=0; i<peeled_mc; i+=mr)
        for(int k=0; k<actual_kc; k++)
          for(int w=0; w<mr; w++)
            blockA[count++] = lhs[(k2+k)*lhsStride + i2+i+w];
      for(int i=peeled_mc; i<actual_mc; i++)
        for(int k=0; k<actual_kc; k++)
          blockA[count++] = lhs[(k2+k)*lhsStride + i2+i];
    }
  }
};

// copy a complete panel of the rhs while expending each coefficient into a packet form
template<typename Scalar, int PacketSize, int nr>
struct ei_gemm_pack_rhs
{
  void operator()(Scalar* blockB, const Scalar* rhs, int rhsStride, bool hasAlpha, Scalar alpha, int actual_kc, int packet_cols, int k2, int cols)
  {
    int count = 0;
    for(int j2=0; j2<packet_cols; j2+=nr)
    {
      const Scalar* b0 = &rhs[(j2+0)*rhsStride + k2];
      const Scalar* b1 = &rhs[(j2+1)*rhsStride + k2];
      const Scalar* b2 = &rhs[(j2+2)*rhsStride + k2];
      const Scalar* b3 = &rhs[(j2+3)*rhsStride + k2];
      if (hasAlpha)
      {
        for(int k=0; k<actual_kc; k++)
        {
          ei_pstore(&blockB[count+0*PacketSize], ei_pset1(alpha*b0[k]));
          ei_pstore(&blockB[count+1*PacketSize], ei_pset1(alpha*b1[k]));
          if (nr==4)
          {
            ei_pstore(&blockB[count+2*PacketSize], ei_pset1(alpha*b2[k]));
            ei_pstore(&blockB[count+3*PacketSize], ei_pset1(alpha*b3[k]));
          }
          count += nr*PacketSize;
        }
      }
      else
      {
        for(int k=0; k<actual_kc; k++)
        {
          ei_pstore(&blockB[count+0*PacketSize], ei_pset1(b0[k]));
          ei_pstore(&blockB[count+1*PacketSize], ei_pset1(b1[k]));
          if (nr==4)
          {
            ei_pstore(&blockB[count+2*PacketSize], ei_pset1(b2[k]));
            ei_pstore(&blockB[count+3*PacketSize], ei_pset1(b3[k]));
          }
          count += nr*PacketSize;
        }
      }
    }
    // copy the remaining columns one at a time (nr==1)
    for(int j2=packet_cols; j2<cols; ++j2)
    {
      const Scalar* b0 = &rhs[(j2+0)*rhsStride + k2];
      if (hasAlpha)
      {
        for(int k=0; k<actual_kc; k++)
        {
          ei_pstore(&blockB[count], ei_pset1(alpha*b0[k]));
          count += PacketSize;
        }
      }
      else
      {
        for(int k=0; k<actual_kc; k++)
        {
          ei_pstore(&blockB[count], ei_pset1(b0[k]));
          count += PacketSize;
        }
      }
    }
  }
};

#endif // EIGEN_EXTERN_INSTANTIATIONS

#endif // EIGEN_GENERAL_MATRIX_MATRIX_H
