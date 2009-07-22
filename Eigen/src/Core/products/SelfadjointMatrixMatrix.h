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

// pack a selfadjoint block diagonal for use with the gebp_kernel
template<typename Scalar, int mr, int StorageOrder>
struct ei_symm_pack_lhs
{
  void operator()(Scalar* blockA, const Scalar* _lhs, int lhsStride, int actual_kc, int actual_mc)
  {
    ei_const_blas_data_mapper<Scalar, StorageOrder> lhs(_lhs,lhsStride);
    int count = 0;
    const int peeled_mc = (actual_mc/mr)*mr;
    for(int i=0; i<peeled_mc; i+=mr)
    {
      for(int k=0; k<i; k++)
        for(int w=0; w<mr; w++)
          blockA[count++] = lhs(i+w,k);
      // symmetric copy
      int h = 0;
      for(int k=i; k<i+mr; k++)
      {
        // transposed copy
        for(int w=0; w<h; w++)
          blockA[count++] = lhs(k, i+w);
        for(int w=h; w<mr; w++)
          blockA[count++] = lhs(i+w, k);
        ++h;
      }
      // transposed copy
      for(int k=i+mr; k<actual_kc; k++)
        for(int w=0; w<mr; w++)
          blockA[count++] = lhs(k, i+w);
    }
    // do the same with mr==1
    for(int i=peeled_mc; i<actual_mc; i++)
    {
      for(int k=0; k<=i; k++)
        blockA[count++] = lhs(i, k);
      // transposed copy
      for(int k=i+1; k<actual_kc; k++)
        blockA[count++] = lhs(k, i);
    }
  }
};

/* Optimized selfadjoint matrix * matrix (_SYMM) product built on top of
 * the general matrix matrix product.
 */
template<typename Scalar, int StorageOrder, int UpLo, bool ConjugateLhs, bool ConjugateRhs>
static EIGEN_DONT_INLINE void ei_product_selfadjoint_matrix(
  int size,
  const Scalar* _lhs, int lhsStride,
  const Scalar* _rhs, int rhsStride, bool rhsRowMajor, int cols,
  Scalar* res,       int resStride,
  Scalar alpha)
{
  typedef typename ei_packet_traits<Scalar>::type Packet;

  ei_const_blas_data_mapper<Scalar, StorageOrder> lhs(_lhs,lhsStride);
  ei_const_blas_data_mapper<Scalar, ColMajor> rhs(_rhs,rhsStride);

  if (ConjugateRhs)
    alpha = ei_conj(alpha);

  typedef typename ei_packet_traits<Scalar>::type PacketType;

  const bool lhsRowMajor = (StorageOrder==RowMajor);

  typedef ei_product_blocking_traits<Scalar> Blocking;

  int kc = std::min<int>(Blocking::Max_kc,size);  // cache block size along the K direction
  int mc = std::min<int>(Blocking::Max_mc,size);  // cache block size along the M direction

  Scalar* blockA = ei_aligned_stack_new(Scalar, kc*mc);
  Scalar* blockB = ei_aligned_stack_new(Scalar, kc*cols*Blocking::PacketSize);

  // number of columns which can be processed by packet of nr columns
  int packet_cols = (cols/Blocking::nr)*Blocking::nr;

  ei_gebp_kernel<Scalar, PacketType, Blocking::PacketSize,
                 Blocking::mr, Blocking::nr, ei_conj_helper<ConjugateLhs,ConjugateRhs> > gebp_kernel;

  for(int k2=0; k2<size; k2+=kc)
  {
    const int actual_kc = std::min(k2+kc,size)-k2;

    // we have selected one row panel of rhs and one column panel of lhs
    // pack rhs's panel into a sequential chunk of memory
    // and expand each coeff to a constant packet for further reuse
    ei_gemm_pack_rhs<Scalar,Blocking::PacketSize,Blocking::nr>()
      (blockB, &rhs(k2,0), rhsStride, alpha, actual_kc, packet_cols, cols);

    // the select lhs's panel has to be split in three different parts:
    //  1 - the transposed panel above the diagonal block => transposed packed copy
    //  2 - the diagonal block => special packed copy
    //  3 - the panel below the diagonal block => generic packed copy
    for(int i2=0; i2<k2; i2+=mc)
    {
      const int actual_mc = std::min(i2+mc,k2)-i2;
      // transposed packed copy
      ei_gemm_pack_lhs<Scalar,Blocking::mr,StorageOrder==RowMajor?ColMajor:RowMajor>()
        (blockA, &lhs(k2,i2), lhsStride, actual_kc, actual_mc);

      gebp_kernel(res, resStride, blockA, blockB, actual_mc, actual_kc, packet_cols, i2, cols);
    }
    // the block diagonal
    {
      const int actual_mc = std::min(k2+kc,size)-k2;
      // symmetric packed copy
      ei_symm_pack_lhs<Scalar,Blocking::mr,StorageOrder>()
        (blockA, &lhs(k2,k2), lhsStride, actual_kc, actual_mc);
      gebp_kernel(res, resStride, blockA, blockB, actual_mc, actual_kc, packet_cols, k2, cols);
    }

    for(int i2=k2+kc; i2<size; i2+=mc)
    {
      const int actual_mc = std::min(i2+mc,size)-i2;
      ei_gemm_pack_lhs<Scalar,Blocking::mr,StorageOrder>()
        (blockA, &lhs(i2,k2), lhsStride, actual_kc, actual_mc);
      gebp_kernel(res, resStride, blockA, blockB, actual_mc, actual_kc, packet_cols, i2, cols);
    }
  }

  ei_aligned_stack_delete(Scalar, blockA, kc*mc);
  ei_aligned_stack_delete(Scalar, blockB, kc*cols*Blocking::PacketSize);
}

#endif // EIGEN_SELFADJOINT_MATRIX_MATRIX_H
