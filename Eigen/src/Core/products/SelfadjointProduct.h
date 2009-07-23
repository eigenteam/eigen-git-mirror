// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Gael Guennebaud <g.gael@free.fr>
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

#ifndef EIGEN_SELFADJOINT_PRODUCT_H
#define EIGEN_SELFADJOINT_PRODUCT_H

/**********************************************************************
* This file implement a self adjoint product: C += A A^T updating only
* an half of the selfadjoint matrix C.
* It corresponds to the level 3 SYRK Blas routine.
**********************************************************************/

// forward declarations (defined at the end of this file)
template<typename Scalar, int mr, int nr, typename Conj, int UpLo>
struct ei_sybb_kernel;

/* Optimized selfadjoint product (_SYRK) */
template <typename Scalar,
          int RhsStorageOrder,
          int ResStorageOrder, bool AAT, int UpLo>
struct ei_selfadjoint_product;

// as usual if the result is row major => we transpose the product
template <typename Scalar, int MatStorageOrder, bool AAT, int UpLo>
struct ei_selfadjoint_product<Scalar,MatStorageOrder, RowMajor, AAT, UpLo>
{
  static EIGEN_STRONG_INLINE void run(int size, int depth, const Scalar* mat, int matStride, Scalar* res, int resStride, Scalar alpha)
  {
    ei_selfadjoint_product<Scalar, MatStorageOrder, ColMajor, !AAT, UpLo==LowerTriangular?UpperTriangular:LowerTriangular>
      ::run(size, depth, mat, matStride, res, resStride, alpha);
  }
};

template <typename Scalar,
          int MatStorageOrder, bool AAT, int UpLo>
struct ei_selfadjoint_product<Scalar,MatStorageOrder, ColMajor, AAT, UpLo>
{

  static EIGEN_DONT_INLINE void run(
    int size, int depth,
    const Scalar* _mat, int matStride,
    Scalar* res,        int resStride,
    Scalar alpha)
  {
    ei_const_blas_data_mapper<Scalar, MatStorageOrder> mat(_mat,matStride);

    if(AAT)
      alpha = ei_conj(alpha);

    typedef ei_product_blocking_traits<Scalar> Blocking;

    int kc = std::min<int>(Blocking::Max_kc,depth);  // cache block size along the K direction
    int mc = std::min<int>(Blocking::Max_mc,size);  // cache block size along the M direction

    Scalar* blockA = ei_aligned_stack_new(Scalar, kc*mc);
    Scalar* blockB = ei_aligned_stack_new(Scalar, kc*size*Blocking::PacketSize);

    // number of columns which can be processed by packet of nr columns
    int packet_cols = (size/Blocking::nr)*Blocking::nr;

    // note that the actual rhs is the transpose/adjoint of mat
    typedef ei_conj_helper<NumTraits<Scalar>::IsComplex && !AAT, NumTraits<Scalar>::IsComplex && AAT> Conj;

    ei_gebp_kernel<Scalar, Blocking::mr, Blocking::nr, Conj> gebp_kernel;

    for(int k2=0; k2<depth; k2+=kc)
    {
      const int actual_kc = std::min(k2+kc,depth)-k2;

      // note that the actual rhs is the transpose/adjoint of mat
      ei_gemm_pack_rhs<Scalar,Blocking::nr,MatStorageOrder==RowMajor ? ColMajor : RowMajor>()
        (blockB, &mat(0,k2), matStride, alpha, actual_kc, packet_cols, size);

      for(int i2=0; i2<size; i2+=mc)
      {
        const int actual_mc = std::min(i2+mc,size)-i2;

        ei_gemm_pack_lhs<Scalar,Blocking::mr,MatStorageOrder, false>()
          (blockA, &mat(i2, k2), matStride, actual_kc, actual_mc);

        // the selected actual_mc * size panel of res is split into three different part:
        //  1 - before the diagonal => processed with gebp or skipped
        //  2 - the actual_mc x actual_mc symmetric block => processed with a special kernel
        //  3 - after the diagonal => processed with gebp or skipped
        if (UpLo==LowerTriangular)
          gebp_kernel(res, resStride, blockA, blockB, actual_mc, actual_kc, std::min(packet_cols,i2), i2, std::min(size,i2));

        ei_sybb_kernel<Scalar, Blocking::mr, Blocking::nr, Conj, UpLo>()
          (res+resStride*i2 + i2, resStride, blockA, blockB + actual_kc*Blocking::PacketSize*i2, actual_mc, actual_kc, std::min(actual_mc,std::max(packet_cols-i2,0)));

        if (UpLo==UpperTriangular)
        {
          int j2 = i2+actual_mc;
          gebp_kernel(res+resStride*j2, resStride, blockA, blockB+actual_kc*Blocking::PacketSize*j2, actual_mc, actual_kc,
                      std::max(0,packet_cols-j2), i2, std::max(0,size-j2));
        }
      }
    }
    ei_aligned_stack_delete(Scalar, blockA, kc*mc);
    ei_aligned_stack_delete(Scalar, blockB, kc*size*Blocking::PacketSize);
  }
};

// high level API

template<typename MatrixType, unsigned int UpLo>
template<typename DerivedU>
void SelfAdjointView<MatrixType,UpLo>
::rankKupdate(const MatrixBase<DerivedU>& u, Scalar alpha)
{
  typedef ei_blas_traits<DerivedU> UBlasTraits;
  typedef typename UBlasTraits::DirectLinearAccessType ActualUType;
  typedef typename ei_cleantype<ActualUType>::type _ActualUType;
  const ActualUType actualU = UBlasTraits::extract(u.derived());

  Scalar actualAlpha = alpha * UBlasTraits::extractScalarFactor(u.derived());

  enum { IsRowMajor = (ei_traits<MatrixType>::Flags&RowMajorBit)?1:0 };

  ei_selfadjoint_product<Scalar,
    _ActualUType::Flags&RowMajorBit ? RowMajor : ColMajor,
    ei_traits<MatrixType>::Flags&RowMajorBit ? RowMajor : ColMajor,
    !UBlasTraits::NeedToConjugate, UpLo>
    ::run(_expression().cols(), actualU.cols(), &actualU.coeff(0,0), actualU.stride(),
          const_cast<Scalar*>(_expression().data()), _expression().stride(), actualAlpha);
}


// optimized SYmmetric packed Block * packed Block product kernel
// this kernel is very similar to the gebp kernel: the only differences are
// the piece of code to avoid the writes off the diagonal
//  => TODO find a way to factorize the two kernels in a single one
template<typename Scalar, int mr, int nr, typename Conj, int UpLo>
struct ei_sybb_kernel
{
  void operator()(Scalar* res, int resStride, const Scalar* blockA, const Scalar* blockB, int actual_mc, int actual_kc, int packet_cols)
  {
    typedef typename ei_packet_traits<Scalar>::type PacketType;
    enum { PacketSize = ei_packet_traits<Scalar>::size };
    Conj cj;
    const int peeled_mc = (actual_mc/mr)*mr;
    // loops on each cache friendly block of the result/rhs
    for(int j2=0; j2<packet_cols; j2+=nr)
    {
      // here we selected a vertical mc x nr panel of the result that we'll
      // process normally until the end of the diagonal (or from the start if upper)
      //
      int start_i = UpLo==LowerTriangular ? (j2/mr)*mr : 0;
      int end_i   = UpLo==LowerTriangular ? actual_mc  : std::min(actual_mc,((j2+std::max(mr,nr))/mr)*mr);
      for(int i=start_i; i<std::min(peeled_mc,end_i); i+=mr)
      {
        const Scalar* blA = &blockA[i*actual_kc];
        #ifdef EIGEN_VECTORIZE_SSE
        _mm_prefetch((const char*)(&blA[0]), _MM_HINT_T0);
        #endif

        // TODO move the res loads to the stores

        // gets res block as register
        PacketType C0, C1, C2, C3, C4, C5, C6, C7;
                  C0 = ei_ploadu(&res[(j2+0)*resStride + i]);
                  C1 = ei_ploadu(&res[(j2+1)*resStride + i]);
        if(nr==4) C2 = ei_ploadu(&res[(j2+2)*resStride + i]);
        if(nr==4) C3 = ei_ploadu(&res[(j2+3)*resStride + i]);
                  C4 = ei_ploadu(&res[(j2+0)*resStride + i + PacketSize]);
                  C5 = ei_ploadu(&res[(j2+1)*resStride + i + PacketSize]);
        if(nr==4) C6 = ei_ploadu(&res[(j2+2)*resStride + i + PacketSize]);
        if(nr==4) C7 = ei_ploadu(&res[(j2+3)*resStride + i + PacketSize]);

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

        // let's check whether the mr x nr block overlap the diagonal,
        // is so then we have to carefully discard writes off the diagonal
        if(UpLo==LowerTriangular ? i>=j2+nr : i+mr<=j2)
        {
                    ei_pstoreu(&res[(j2+0)*resStride + i], C0);
                    ei_pstoreu(&res[(j2+1)*resStride + i], C1);
          if(nr==4) ei_pstoreu(&res[(j2+2)*resStride + i], C2);
          if(nr==4) ei_pstoreu(&res[(j2+3)*resStride + i], C3);
                    ei_pstoreu(&res[(j2+0)*resStride + i + PacketSize], C4);
                    ei_pstoreu(&res[(j2+1)*resStride + i + PacketSize], C5);
          if(nr==4) ei_pstoreu(&res[(j2+2)*resStride + i + PacketSize], C6);
          if(nr==4) ei_pstoreu(&res[(j2+3)*resStride + i + PacketSize], C7);
        }
        else
        {
          Scalar buf[mr*nr];
          // overlap => copy to a temporary mr x nr buffer and then triangular copy
                    ei_pstore(&buf[0*mr], C0);
                    ei_pstore(&buf[1*mr], C1);
          if(nr==4) ei_pstore(&buf[2*mr], C2);
          if(nr==4) ei_pstore(&buf[3*mr], C3);
                    ei_pstore(&buf[0*mr + PacketSize], C4);
                    ei_pstore(&buf[1*mr + PacketSize], C5);
          if(nr==4) ei_pstore(&buf[2*mr + PacketSize], C6);
          if(nr==4) ei_pstore(&buf[3*mr + PacketSize], C7);

          for(int j1=0; j1<nr; ++j1)
            for(int i1=0; i1<mr; ++i1)
            {
              if(UpLo==LowerTriangular ? i+i1 >= j2+j1 : i+i1 <= j2+j1)
                res[(j2+j1)*resStride + i+i1] = buf[i1 + j1 * mr];
            }
        }
      }
      for(int i=std::max(start_i,peeled_mc); i<std::min(end_i,actual_mc); i++)
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
        if(UpLo==LowerTriangular ? i>=j2+nr : i+mr<=j2) {
          res[(j2+0)*resStride + i] += C0;
          res[(j2+1)*resStride + i] += C1;
          if(nr==4) res[(j2+2)*resStride + i] += C2;
          if(nr==4) res[(j2+3)*resStride + i] += C3;
        }
        else
        {
                    if(UpLo==LowerTriangular ? i>=j2+0 : i<=j2+0) res[(j2+0)*resStride + i] += C0;
                    if(UpLo==LowerTriangular ? i>=j2+1 : i<=j2+1) res[(j2+1)*resStride + i] += C1;
          if(nr==4) if(UpLo==LowerTriangular ? i>=j2+2 : i<=j2+2) res[(j2+2)*resStride + i] += C2;
          if(nr==4) if(UpLo==LowerTriangular ? i>=j2+3 : i<=j2+3) res[(j2+3)*resStride + i] += C3;
        }
      }
    }

    // process remaining rhs/res columns one at a time
    // => do the same but with nr==1
    for(int j2=packet_cols; j2<actual_mc; j2++)
    {
      int start_i = UpLo==LowerTriangular ? (j2/mr)*mr : 0;
      int end_i   = UpLo==LowerTriangular ? actual_mc  : std::min(actual_mc,j2+1);
      for(int i=start_i; i<std::min(end_i,peeled_mc); i+=mr)
      {
        const Scalar* blA = &blockA[i*actual_kc];
        #ifdef EIGEN_VECTORIZE_SSE
        _mm_prefetch((const char*)(&blA[0]), _MM_HINT_T0);
        #endif

        // TODO move the res loads to the stores

        // gets res block as register
        PacketType C0, C4;
        C0 = ei_ploadu(&res[(j2+0)*resStride + i]);
        C4 = ei_ploadu(&res[(j2+0)*resStride + i + PacketSize]);

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

        if(UpLo==LowerTriangular ? i>=j2 : i<=j2)                       ei_pstoreu(&res[(j2+0)*resStride + i], C0);
        if(UpLo==LowerTriangular ? i+PacketSize>=j2 : i+PacketSize<=j2) ei_pstoreu(&res[(j2+0)*resStride + i + PacketSize], C4);
      }
      if(UpLo==LowerTriangular)
        start_i = j2;
      for(int i=std::max(start_i,peeled_mc); i<std::min(end_i,actual_mc); i++)
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
        res[(j2+0)*resStride + i] += C0;
      }
    }
  }
};

#endif // EIGEN_SELFADJOINT_PRODUCT_H
