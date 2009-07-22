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
    ei_const_blas_data_mapper<Scalar,StorageOrder> lhs(_lhs,lhsStride);
    int count = 0;
    const int peeled_mc = (actual_mc/mr)*mr;
    for(int i=0; i<peeled_mc; i+=mr)
    {
      // normal copy
      for(int k=0; k<i; k++)
        for(int w=0; w<mr; w++)
          blockA[count++] = lhs(i+w,k);           // normal
      // symmetric copy
      int h = 0;
      for(int k=i; k<i+mr; k++)
      {
        for(int w=0; w<h; w++)
          blockA[count++] = ei_conj(lhs(k, i+w)); // transposed
        for(int w=h; w<mr; w++)
          blockA[count++] = lhs(i+w, k);          // normal
        ++h;
      }
      // transposed copy
      for(int k=i+mr; k<actual_kc; k++)
        for(int w=0; w<mr; w++)
          blockA[count++] = ei_conj(lhs(k, i+w)); // transposed
    }

    // do the same with mr==1
    for(int i=peeled_mc; i<actual_mc; i++)
    {
      for(int k=0; k<=i; k++)
        blockA[count++] = lhs(i, k);              // normal
      for(int k=i+1; k<actual_kc; k++)
        blockA[count++] = ei_conj(lhs(k, i));     // transposed
    }
  }
};

template<typename Scalar, int nr, int StorageOrder>
struct ei_symm_pack_rhs
{
  enum { PacketSize = ei_packet_traits<Scalar>::size };
  void operator()(Scalar* blockB, const Scalar* _rhs, int rhsStride, Scalar alpha, int actual_kc, int packet_cols, int cols, int k2)
  {
    int end_k = k2 + actual_kc;
    int count = 0;
    ei_const_blas_data_mapper<Scalar,StorageOrder> rhs(_rhs,rhsStride);

    // first part: normal case
    for(int j2=0; j2<k2; j2+=nr)
    {
      for(int k=k2; k<end_k; k++)
      {
        ei_pstore(&blockB[count+0*PacketSize], ei_pset1(alpha*rhs(k,j2+0)));
        ei_pstore(&blockB[count+1*PacketSize], ei_pset1(alpha*rhs(k,j2+1)));
        if (nr==4)
        {
          ei_pstore(&blockB[count+2*PacketSize], ei_pset1(alpha*rhs(k,j2+2)));
          ei_pstore(&blockB[count+3*PacketSize], ei_pset1(alpha*rhs(k,j2+3)));
        }
        count += nr*PacketSize;
      }
    }

    // second part: diagonal block
    for(int j2=k2; j2<std::min(k2+actual_kc,packet_cols); j2+=nr)
    {
      // again we can split vertically in three different parts (transpose, symmetric, normal)
      // transpose
      for(int k=k2; k<j2; k++)
      {
        ei_pstore(&blockB[count+0*PacketSize], ei_pset1(alpha*ei_conj(rhs(j2+0,k))));
        ei_pstore(&blockB[count+1*PacketSize], ei_pset1(alpha*ei_conj(rhs(j2+1,k))));
        if (nr==4)
        {
          ei_pstore(&blockB[count+2*PacketSize], ei_pset1(alpha*ei_conj(rhs(j2+2,k))));
          ei_pstore(&blockB[count+3*PacketSize], ei_pset1(alpha*ei_conj(rhs(j2+3,k))));
        }
        count += nr*PacketSize;
      }
      // symmetric
      int h = 0;
      for(int k=j2; k<j2+nr; k++)
      {
        // normal
        for (int w=0 ; w<h; ++w)
          ei_pstore(&blockB[count+w*PacketSize], ei_pset1(alpha*rhs(k,j2+w)));
        // transpose
        for (int w=h ; w<nr; ++w)
          ei_pstore(&blockB[count+w*PacketSize], ei_pset1(alpha*ei_conj(rhs(j2+w,k))));
        count += nr*PacketSize;
        ++h;
      }
      // normal
      for(int k=j2+nr; k<end_k; k++)
      {
        ei_pstore(&blockB[count+0*PacketSize], ei_pset1(alpha*rhs(k,j2+0)));
        ei_pstore(&blockB[count+1*PacketSize], ei_pset1(alpha*rhs(k,j2+1)));
        if (nr==4)
        {
          ei_pstore(&blockB[count+2*PacketSize], ei_pset1(alpha*rhs(k,j2+2)));
          ei_pstore(&blockB[count+3*PacketSize], ei_pset1(alpha*rhs(k,j2+3)));
        }
        count += nr*PacketSize;
      }
    }

    // third part: transposed
    for(int j2=k2+actual_kc; j2<packet_cols; j2+=nr)
    {
      for(int k=k2; k<end_k; k++)
      {
        ei_pstore(&blockB[count+0*PacketSize], ei_pset1(alpha*ei_conj(rhs(j2+0,k))));
        ei_pstore(&blockB[count+1*PacketSize], ei_pset1(alpha*ei_conj(rhs(j2+1,k))));
        if (nr==4)
        {
          ei_pstore(&blockB[count+2*PacketSize], ei_pset1(alpha*ei_conj(rhs(j2+2,k))));
          ei_pstore(&blockB[count+3*PacketSize], ei_pset1(alpha*ei_conj(rhs(j2+3,k))));
        }
        count += nr*PacketSize;
      }
    }

    // copy the remaining columns one at a time (=> the same with nr==1)
    for(int j2=packet_cols; j2<cols; ++j2)
    {
      // transpose
      int half = std::min(end_k,j2);
      for(int k=k2; k<half; k++)
      {
        ei_pstore(&blockB[count], ei_pset1(alpha*ei_conj(rhs(j2,k))));
        count += PacketSize;
      }
      // normal
      for(int k=half; k<k2+actual_kc; k++)
      {
        ei_pstore(&blockB[count], ei_pset1(alpha*rhs(k,j2)));
        count += PacketSize;
      }
    }
  }
};

/* Optimized selfadjoint matrix * matrix (_SYMM) product built on top of
 * the general matrix matrix product.
 */
template <typename Scalar,
          int LhsStorageOrder, bool LhsSelfAdjoint, bool ConjugateLhs,
          int RhsStorageOrder, bool RhsSelfAdjoint, bool ConjugateRhs,
          int ResStorageOrder>
struct ei_product_selfadjoint_matrix;

template <typename Scalar,
          int LhsStorageOrder, bool LhsSelfAdjoint, bool ConjugateLhs,
          int RhsStorageOrder, bool RhsSelfAdjoint, bool ConjugateRhs>
struct ei_product_selfadjoint_matrix<Scalar,LhsStorageOrder,LhsSelfAdjoint,ConjugateLhs, RhsStorageOrder,RhsSelfAdjoint,ConjugateRhs,RowMajor>
{

  static EIGEN_STRONG_INLINE void run(
    int rows, int cols,
    const Scalar* lhs, int lhsStride,
    const Scalar* rhs, int rhsStride,
    Scalar* res,       int resStride,
    Scalar alpha)
  {
    ei_product_selfadjoint_matrix<Scalar,
      EIGEN_LOGICAL_XOR(RhsSelfAdjoint,RhsStorageOrder==RowMajor) ? ColMajor : RowMajor,
      RhsSelfAdjoint, NumTraits<Scalar>::IsComplex && EIGEN_LOGICAL_XOR(RhsSelfAdjoint,ConjugateRhs),
      EIGEN_LOGICAL_XOR(LhsSelfAdjoint,LhsStorageOrder==RowMajor) ? ColMajor : RowMajor,
      LhsSelfAdjoint, NumTraits<Scalar>::IsComplex && EIGEN_LOGICAL_XOR(LhsSelfAdjoint,ConjugateLhs),
      ColMajor>
      ::run(cols, rows,  rhs, rhsStride,  lhs, lhsStride,  res, resStride,  alpha);
  }
};

template <typename Scalar,
          int LhsStorageOrder, bool ConjugateLhs,
          int RhsStorageOrder, bool ConjugateRhs>
struct ei_product_selfadjoint_matrix<Scalar,LhsStorageOrder,true,ConjugateLhs, RhsStorageOrder,false,ConjugateRhs,ColMajor>
{

  static EIGEN_DONT_INLINE void run(
    int rows, int cols,
    const Scalar* _lhs, int lhsStride,
    const Scalar* _rhs, int rhsStride,
    Scalar* res,        int resStride,
    Scalar alpha)
  {
    int size = rows;

    ei_const_blas_data_mapper<Scalar, LhsStorageOrder> lhs(_lhs,lhsStride);
    ei_const_blas_data_mapper<Scalar, RhsStorageOrder> rhs(_rhs,rhsStride);

    if (ConjugateRhs)
      alpha = ei_conj(alpha);

    typedef ei_product_blocking_traits<Scalar> Blocking;

    int kc = std::min<int>(Blocking::Max_kc,size);  // cache block size along the K direction
    int mc = std::min<int>(Blocking::Max_mc,rows);  // cache block size along the M direction

    Scalar* blockA = ei_aligned_stack_new(Scalar, kc*mc);
    Scalar* blockB = ei_aligned_stack_new(Scalar, kc*cols*Blocking::PacketSize);

    // number of columns which can be processed by packet of nr columns
    int packet_cols = (cols/Blocking::nr)*Blocking::nr;

    ei_gebp_kernel<Scalar, Blocking::mr, Blocking::nr, ei_conj_helper<ConjugateLhs,ConjugateRhs> > gebp_kernel;

    for(int k2=0; k2<size; k2+=kc)
    {
      const int actual_kc = std::min(k2+kc,size)-k2;

      // we have selected one row panel of rhs and one column panel of lhs
      // pack rhs's panel into a sequential chunk of memory
      // and expand each coeff to a constant packet for further reuse
      ei_gemm_pack_rhs<Scalar,Blocking::nr,RhsStorageOrder>()
        (blockB, &rhs(k2,0), rhsStride, alpha, actual_kc, packet_cols, cols);

      // the select lhs's panel has to be split in three different parts:
      //  1 - the transposed panel above the diagonal block => transposed packed copy
      //  2 - the diagonal block => special packed copy
      //  3 - the panel below the diagonal block => generic packed copy
      for(int i2=0; i2<k2; i2+=mc)
      {
        const int actual_mc = std::min(i2+mc,k2)-i2;
        // transposed packed copy
        ei_gemm_pack_lhs<Scalar,Blocking::mr,LhsStorageOrder==RowMajor?ColMajor:RowMajor, true>()
          (blockA, &lhs(k2, i2), lhsStride, actual_kc, actual_mc);

        gebp_kernel(res, resStride, blockA, blockB, actual_mc, actual_kc, packet_cols, i2, cols);
      }
      // the block diagonal
      {
        const int actual_mc = std::min(k2+kc,size)-k2;
        // symmetric packed copy
        ei_symm_pack_lhs<Scalar,Blocking::mr,LhsStorageOrder>()
          (blockA, &lhs(k2,k2), lhsStride, actual_kc, actual_mc);

        gebp_kernel(res, resStride, blockA, blockB, actual_mc, actual_kc, packet_cols, k2, cols);
      }

      for(int i2=k2+kc; i2<size; i2+=mc)
      {
        const int actual_mc = std::min(i2+mc,size)-i2;
        ei_gemm_pack_lhs<Scalar,Blocking::mr,LhsStorageOrder,false>()
          (blockA, &lhs(i2, k2), lhsStride, actual_kc, actual_mc);

        gebp_kernel(res, resStride, blockA, blockB, actual_mc, actual_kc, packet_cols, i2, cols);
      }
    }

    ei_aligned_stack_delete(Scalar, blockA, kc*mc);
    ei_aligned_stack_delete(Scalar, blockB, kc*cols*Blocking::PacketSize);
  }
};

// matrix * selfadjoint product
template <typename Scalar,
          int LhsStorageOrder, bool ConjugateLhs,
          int RhsStorageOrder, bool ConjugateRhs>
struct ei_product_selfadjoint_matrix<Scalar,LhsStorageOrder,false,ConjugateLhs, RhsStorageOrder,true,ConjugateRhs,ColMajor>
{

  static EIGEN_DONT_INLINE void run(
    int rows, int cols,
    const Scalar* _lhs, int lhsStride,
    const Scalar* _rhs, int rhsStride,
    Scalar* res,        int resStride,
    Scalar alpha)
  {
    int size = cols;

    ei_const_blas_data_mapper<Scalar, LhsStorageOrder> lhs(_lhs,lhsStride);
    ei_const_blas_data_mapper<Scalar, RhsStorageOrder> rhs(_rhs,rhsStride);

    if (ConjugateRhs)
      alpha = ei_conj(alpha);

    typedef ei_product_blocking_traits<Scalar> Blocking;

    int kc = std::min<int>(Blocking::Max_kc,size);  // cache block size along the K direction
    int mc = std::min<int>(Blocking::Max_mc,rows);  // cache block size along the M direction

    Scalar* blockA = ei_aligned_stack_new(Scalar, kc*mc);
    Scalar* blockB = ei_aligned_stack_new(Scalar, kc*cols*Blocking::PacketSize);

    // number of columns which can be processed by packet of nr columns
    int packet_cols = (cols/Blocking::nr)*Blocking::nr;

    ei_gebp_kernel<Scalar, Blocking::mr, Blocking::nr, ei_conj_helper<ConjugateLhs,ConjugateRhs> > gebp_kernel;

    for(int k2=0; k2<size; k2+=kc)
    {
      const int actual_kc = std::min(k2+kc,size)-k2;

      ei_symm_pack_rhs<Scalar,Blocking::nr,RhsStorageOrder>()
        (blockB, _rhs, rhsStride, alpha, actual_kc, packet_cols, cols, k2);

      // => GEPP
      for(int i2=0; i2<rows; i2+=mc)
      {
        const int actual_mc = std::min(i2+mc,rows)-i2;
        ei_gemm_pack_lhs<Scalar,Blocking::mr,LhsStorageOrder>()
          (blockA, &lhs(i2, k2), lhsStride, actual_kc, actual_mc);

        gebp_kernel(res, resStride, blockA, blockB, actual_mc, actual_kc, packet_cols, i2, cols);
      }
    }

    ei_aligned_stack_delete(Scalar, blockA, kc*mc);
    ei_aligned_stack_delete(Scalar, blockB, kc*cols*Blocking::PacketSize);
  }
};

#endif // EIGEN_SELFADJOINT_MATRIX_MATRIX_H
