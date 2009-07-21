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

#ifndef EIGEN_SELFADJOINT_MATRIX_MATRIX_H
#define EIGEN_SELFADJOINT_MATRIX_MATRIX_H

template<typename Scalar, int mr>
struct ei_symm_pack_lhs
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
      {
        for(int k=0; k<i; k++)
          for(int w=0; w<mr; w++)
            blockA[count++] = lhs[(k2+k)*lhsStride + i2+i+w];

        // symmetric copy
        int h = 0;
        for(int k=i; k<i+mr; k++)
        {
          // transposed copy
          for(int w=0; w<h; w++)
            blockA[count++] = lhs[(k2+k) + (i2+i+w)*lhsStride];
          for(int w=h; w<mr; w++)
            blockA[count++] = lhs[(k2+k)*lhsStride + i2+i+w];
          ++h;
        }

        // transposed copy
        for(int k=i+mr; k<actual_kc; k++)
          for(int w=0; w<mr; w++)
            blockA[count++] = lhs[(k2+k) + (i2+i+w)*lhsStride];
      }
      // do the same with mr==1
      for(int i=peeled_mc; i<actual_mc; i++)
      {
        for(int k=0; k<=i; k++)
          blockA[count++] = lhs[(k2+k)*lhsStride + i2+i];

        // transposed copy
        for(int k=i+1; k<actual_kc; k++)
          blockA[count++] = lhs[(k2+k) + (i2+i)*lhsStride];
      }
    }
  }
};

/* Optimized selfadjoint matrix * matrix (_SYMM) product built on top of
 * the general matrix matrix product.
 */
template<typename Scalar, int StorageOrder, int UpLo, bool ConjugateLhs, bool ConjugateRhs>
static EIGEN_DONT_INLINE void ei_product_selfadjoint_matrix(
  int size,
  const Scalar* lhs, int lhsStride,
  const Scalar* rhs, int rhsStride, bool rhsRowMajor, int cols,
  Scalar* res,       int resStride,
  Scalar alpha)
{
  typedef typename ei_packet_traits<Scalar>::type Packet;

  ei_conj_helper<ConjugateLhs,ConjugateRhs> cj;
  if (ConjugateRhs)
    alpha = ei_conj(alpha);
  bool hasAlpha = alpha != Scalar(1);

  typedef typename ei_packet_traits<Scalar>::type PacketType;

  const bool lhsRowMajor = (StorageOrder==RowMajor);

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

  int kc = std::min<int>(Max_kc,size);  // cache block size along the K direction
  int mc = std::min<int>(Max_mc,size);  // cache block size along the M direction

  Scalar* blockA = ei_aligned_stack_new(Scalar, kc*mc);
  Scalar* blockB = ei_aligned_stack_new(Scalar, kc*cols*PacketSize);

  // number of columns which can be processed by packet of nr columns
  int packet_cols = (cols/nr)*nr;

  for(int k2=0; k2<size; k2+=kc)
  {
    const int actual_kc = std::min(k2+kc,size)-k2;

    // we have selected one row panel of rhs and one column panel of lhs
    // pack rhs's panel into a sequential chunk of memory
    // and expand each coeff to a constant packet for further reuse
    ei_gemm_pack_rhs<Scalar,PacketSize,nr>()(blockB, rhs, rhsStride, hasAlpha, alpha, actual_kc, packet_cols, k2, cols);

    // the select lhs's panel has to be split in three different parts:
    //  1 - the transposed panel above the diagonal block => transposed packed copy
    //  2 - the diagonal block => special packed copy
    //  3 - the panel below the diagonal block => generic packed copy
    for(int i2=0; i2<k2; i2+=mc)
    {
      const int actual_mc = std::min(i2+mc,k2)-i2;
      // transposed packed copy
      ei_gemm_pack_lhs<Scalar,mr>()(blockA, lhs, lhsStride, !lhsRowMajor, actual_kc, actual_mc, k2, i2);

      ei_gebp_kernel<Scalar, PacketType, PacketSize, mr, nr, ei_conj_helper<ConjugateLhs,ConjugateRhs> >()
        (res, resStride, blockA, blockB, actual_mc, actual_kc, packet_cols, i2, cols);
    }
    // the block diagonal
    {
      const int actual_mc = std::min(k2+kc,size)-k2;
      // symmetric packed copy
      ei_symm_pack_lhs<Scalar,mr>()(blockA, lhs, lhsStride, lhsRowMajor, actual_kc, actual_mc, k2, k2);
      ei_gebp_kernel<Scalar, PacketType, PacketSize, mr, nr, ei_conj_helper<ConjugateLhs,ConjugateRhs> >()
        (res, resStride, blockA, blockB, actual_mc, actual_kc, packet_cols, k2, cols);
    }

    for(int i2=k2+kc; i2<size; i2+=mc)
    {
      const int actual_mc = std::min(i2+mc,size)-i2;
      ei_gemm_pack_lhs<Scalar,mr>()(blockA, lhs, lhsStride, lhsRowMajor, actual_kc, actual_mc, k2, i2);
      ei_gebp_kernel<Scalar, PacketType, PacketSize, mr, nr, ei_conj_helper<ConjugateLhs,ConjugateRhs> >()
        (res, resStride, blockA, blockB, actual_mc, actual_kc, packet_cols, i2, cols);
    }
  }

  ei_aligned_stack_delete(Scalar, blockA, kc*mc);
  ei_aligned_stack_delete(Scalar, blockB, kc*cols*PacketSize);
}


#endif // EIGEN_SELFADJOINT_MATRIX_MATRIX_H
