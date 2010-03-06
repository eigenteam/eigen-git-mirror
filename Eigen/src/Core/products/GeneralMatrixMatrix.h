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

#ifndef EIGEN_EXTERN_INSTANTIATIONS

/* Specialization for a row-major destination matrix => simple transposition of the product */
template<
  typename Scalar,
  int LhsStorageOrder, bool ConjugateLhs,
  int RhsStorageOrder, bool ConjugateRhs>
struct ei_general_matrix_matrix_product<Scalar,LhsStorageOrder,ConjugateLhs,RhsStorageOrder,ConjugateRhs,RowMajor>
{
  static EIGEN_STRONG_INLINE void run(
    int rows, int cols, int depth,
    const Scalar* lhs, int lhsStride,
    const Scalar* rhs, int rhsStride,
    Scalar* res, int resStride,
    Scalar alpha,
    GemmParallelInfo<Scalar>* info = 0)
  {
    // transpose the product such that the result is column major
    ei_general_matrix_matrix_product<Scalar,
      RhsStorageOrder==RowMajor ? ColMajor : RowMajor,
      ConjugateRhs,
      LhsStorageOrder==RowMajor ? ColMajor : RowMajor,
      ConjugateLhs,
      ColMajor>
    ::run(cols,rows,depth,rhs,rhsStride,lhs,lhsStride,res,resStride,alpha,info);
  }
};

/*  Specialization for a col-major destination matrix
 *    => Blocking algorithm following Goto's paper */
template<
  typename Scalar,
  int LhsStorageOrder, bool ConjugateLhs,
  int RhsStorageOrder, bool ConjugateRhs>
struct ei_general_matrix_matrix_product<Scalar,LhsStorageOrder,ConjugateLhs,RhsStorageOrder,ConjugateRhs,ColMajor>
{
static void run(int rows, int cols, int depth,
  const Scalar* _lhs, int lhsStride,
  const Scalar* _rhs, int rhsStride,
  Scalar* res, int resStride,
  Scalar alpha,
  GemmParallelInfo<Scalar>* info = 0)
{
  ei_const_blas_data_mapper<Scalar, LhsStorageOrder> lhs(_lhs,lhsStride);
  ei_const_blas_data_mapper<Scalar, RhsStorageOrder> rhs(_rhs,rhsStride);

  if (ConjugateRhs)
    alpha = ei_conj(alpha);

  typedef typename ei_packet_traits<Scalar>::type PacketType;
  typedef ei_product_blocking_traits<Scalar> Blocking;

  int kc = std::min<int>(Blocking::Max_kc,depth);  // cache block size along the K direction
  int mc = std::min<int>(Blocking::Max_mc,rows);   // cache block size along the M direction

  ei_gemm_pack_rhs<Scalar, Blocking::nr, RhsStorageOrder> pack_rhs;
  ei_gemm_pack_lhs<Scalar, Blocking::mr, LhsStorageOrder> pack_lhs;
  ei_gebp_kernel<Scalar, Blocking::mr, Blocking::nr, ei_conj_helper<ConjugateLhs,ConjugateRhs> > gebp;

#ifdef EIGEN_HAS_OPENMP
  if(info)
  {
    // this is the parallel version!
    int tid = omp_get_thread_num();
    int threads = omp_get_num_threads();

    Scalar* blockA = ei_aligned_stack_new(Scalar, kc*mc);
    std::size_t sizeW = kc*Blocking::PacketSize*Blocking::nr*8;
    Scalar* w = ei_aligned_stack_new(Scalar, sizeW);
    Scalar* blockB = (Scalar*)info[tid].blockB;

    // For each horizontal panel of the rhs, and corresponding panel of the lhs...
    // (==GEMM_VAR1)
    for(int k=0; k<depth; k+=kc)
    {
      const int actual_kc = std::min(k+kc,depth)-k; // => rows of B', and cols of the A'

      // In order to reduce the chance that a thread has to wait for the other,
      // let's start by packing A'.
      pack_lhs(blockA, &lhs(0,k), lhsStride, actual_kc, mc);

      // Pack B_k to B' in parallel fashion:
      // each thread packs the sub block B_k,j to B'_j where j is the thread id.

      // However, before copying to B'_j, we have to make sure that no other thread is still using it,
      // i.e., we test that info[tid].users equals 0.
      // Then, we set info[tid].users to the number of threads to mark that all other threads are going to use it.
      while(info[tid].users!=0) {}
      info[tid].users += threads;

      pack_rhs(blockB+info[tid].rhs_start*kc, &rhs(k,info[tid].rhs_start), rhsStride, alpha, actual_kc, info[tid].rhs_length);

      // Notify the other threads that the part B'_j is ready to go.
      info[tid].sync = k;

      // Computes C_i += A' * B' per B'_j
      for(int shift=0; shift<threads; ++shift)
      {
        int j = (tid+shift)%threads;

        // At this point we have to make sure that B'_j has been updated by the thread j,
        // we use testAndSetOrdered to mimic a volatile access.
        // However, no need to wait for the B' part which has been updated by the current thread!
        if(shift>0)
          while(info[j].sync!=k) {}

        gebp(res+info[j].rhs_start*resStride, resStride, blockA, blockB+info[j].rhs_start*kc, mc, actual_kc, info[j].rhs_length, -1,-1,0,0, w);
      }

      // Then keep going as usual with the remaining A'
      for(int i=mc; i<rows; i+=mc)
      {
        const int actual_mc = std::min(i+mc,rows)-i;

        // pack A_i,k to A'
        pack_lhs(blockA, &lhs(i,k), lhsStride, actual_kc, actual_mc);

        // C_i += A' * B'
        gebp(res+i, resStride, blockA, blockB, actual_mc, actual_kc, cols, -1,-1,0,0, w);
      }

      // Release all the sub blocks B'_j of B' for the current thread,
      // i.e., we simply decrement the number of users by 1
      for(int j=0; j<threads; ++j)
        #pragma omp atomic
        --(info[j].users);
    }

    ei_aligned_stack_delete(Scalar, blockA, kc*mc);
    ei_aligned_stack_delete(Scalar, w, sizeW);
  }
  else
#endif // EIGEN_HAS_OPENMP
  {
    (void)info; // info is not used
    // this is the sequential version!
    Scalar* blockA = ei_aligned_stack_new(Scalar, kc*mc);
    std::size_t sizeB = kc*Blocking::PacketSize*Blocking::nr + kc*cols;
    Scalar* allocatedBlockB = ei_aligned_stack_new(Scalar, sizeB);
    Scalar* blockB = allocatedBlockB + kc*Blocking::PacketSize*Blocking::nr;

    // For each horizontal panel of the rhs, and corresponding panel of the lhs...
    // (==GEMM_VAR1)
    for(int k2=0; k2<depth; k2+=kc)
    {
      const int actual_kc = std::min(k2+kc,depth)-k2;

      // OK, here we have selected one horizontal panel of rhs and one vertical panel of lhs.
      // => Pack rhs's panel into a sequential chunk of memory (L2 caching)
      // Note that this panel will be read as many times as the number of blocks in the lhs's
      // vertical panel which is, in practice, a very low number.
      pack_rhs(blockB, &rhs(k2,0), rhsStride, alpha, actual_kc, cols);


      // For each mc x kc block of the lhs's vertical panel...
      // (==GEPP_VAR1)
      for(int i2=0; i2<rows; i2+=mc)
      {
        const int actual_mc = std::min(i2+mc,rows)-i2;

        // We pack the lhs's block into a sequential chunk of memory (L1 caching)
        // Note that this block will be read a very high number of times, which is equal to the number of
        // micro vertical panel of the large rhs's panel (e.g., cols/4 times).
        pack_lhs(blockA, &lhs(i2,k2), lhsStride, actual_kc, actual_mc);

        // Everything is packed, we can now call the block * panel kernel:
        gebp(res+i2, resStride, blockA, blockB, actual_mc, actual_kc, cols);

      }
    }

    ei_aligned_stack_delete(Scalar, blockA, kc*mc);
    ei_aligned_stack_delete(Scalar, allocatedBlockB, sizeB);
  }
}

};

#endif // EIGEN_EXTERN_INSTANTIATIONS

/*********************************************************************************
*  Specialization of GeneralProduct<> for "large" GEMM, i.e.,
*  implementation of the high level wrapper to ei_general_matrix_matrix_product
**********************************************************************************/

template<typename Lhs, typename Rhs>
struct ei_traits<GeneralProduct<Lhs,Rhs,GemmProduct> >
 : ei_traits<ProductBase<GeneralProduct<Lhs,Rhs,GemmProduct>, Lhs, Rhs> >
{};

template<typename Scalar, typename Gemm, typename Lhs, typename Rhs, typename Dest>
struct ei_gemm_functor
{
  typedef typename Rhs::Scalar BlockBScalar;

  ei_gemm_functor(const Lhs& lhs, const Rhs& rhs, Dest& dest, Scalar actualAlpha)
    : m_lhs(lhs), m_rhs(rhs), m_dest(dest), m_actualAlpha(actualAlpha)
  {}

  void operator() (int row, int rows, int col=0, int cols=-1, GemmParallelInfo<BlockBScalar>* info=0) const
  {
    if(cols==-1)
      cols = m_rhs.cols();
    Gemm::run(rows, cols, m_lhs.cols(),
              (const Scalar*)&(m_lhs.const_cast_derived().coeffRef(row,0)), m_lhs.outerStride(),
              (const Scalar*)&(m_rhs.const_cast_derived().coeffRef(0,col)), m_rhs.outerStride(),
              (Scalar*)&(m_dest.coeffRef(row,col)), m_dest.outerStride(),
              m_actualAlpha,
              info);
  }


  int sharedBlockBSize() const
  {
    return std::min<int>(ei_product_blocking_traits<Scalar>::Max_kc,m_rhs.rows()) * m_rhs.cols();
  }

  protected:
    const Lhs& m_lhs;
    const Rhs& m_rhs;
    mutable Dest& m_dest;
    Scalar m_actualAlpha;
};

template<typename Lhs, typename Rhs>
class GeneralProduct<Lhs, Rhs, GemmProduct>
  : public ProductBase<GeneralProduct<Lhs,Rhs,GemmProduct>, Lhs, Rhs>
{
  public:
    EIGEN_PRODUCT_PUBLIC_INTERFACE(GeneralProduct)

    GeneralProduct(const Lhs& lhs, const Rhs& rhs) : Base(lhs,rhs)
    {
      EIGEN_STATIC_ASSERT((ei_is_same_type<typename Lhs::Scalar, typename Rhs::Scalar>::ret),
        YOU_MIXED_DIFFERENT_NUMERIC_TYPES__YOU_NEED_TO_USE_THE_CAST_METHOD_OF_MATRIXBASE_TO_CAST_NUMERIC_TYPES_EXPLICITLY)
    }

    template<typename Dest> void scaleAndAddTo(Dest& dst, Scalar alpha) const
    {
      ei_assert(dst.rows()==m_lhs.rows() && dst.cols()==m_rhs.cols());

      const ActualLhsType lhs = LhsBlasTraits::extract(m_lhs);
      const ActualRhsType rhs = RhsBlasTraits::extract(m_rhs);

      Scalar actualAlpha = alpha * LhsBlasTraits::extractScalarFactor(m_lhs)
                                 * RhsBlasTraits::extractScalarFactor(m_rhs);

      typedef ei_gemm_functor<
        Scalar,
        ei_general_matrix_matrix_product<
          Scalar,
          (_ActualLhsType::Flags&RowMajorBit) ? RowMajor : ColMajor, bool(LhsBlasTraits::NeedToConjugate),
          (_ActualRhsType::Flags&RowMajorBit) ? RowMajor : ColMajor, bool(RhsBlasTraits::NeedToConjugate),
          (Dest::Flags&RowMajorBit) ? RowMajor : ColMajor>,
        _ActualLhsType,
        _ActualRhsType,
        Dest> GemmFunctor;

      ei_parallelize_gemm<(Dest::MaxRowsAtCompileTime>32)>(GemmFunctor(lhs, rhs, dst, actualAlpha), this->rows(), this->cols());
    }
};

#endif // EIGEN_GENERAL_MATRIX_MATRIX_H
