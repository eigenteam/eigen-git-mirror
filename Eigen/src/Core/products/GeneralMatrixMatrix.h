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
    Scalar alpha)
  {
    // transpose the product such that the result is column major
    ei_general_matrix_matrix_product<Scalar,
      RhsStorageOrder==RowMajor ? ColMajor : RowMajor,
      ConjugateRhs,
      LhsStorageOrder==RowMajor ? ColMajor : RowMajor,
      ConjugateLhs,
      ColMajor>
    ::run(cols,rows,depth,rhs,rhsStride,lhs,lhsStride,res,resStride,alpha);
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
  Scalar alpha)
{
  ei_const_blas_data_mapper<Scalar, LhsStorageOrder> lhs(_lhs,lhsStride);
  ei_const_blas_data_mapper<Scalar, RhsStorageOrder> rhs(_rhs,rhsStride);

  if (ConjugateRhs)
    alpha = ei_conj(alpha);

  typedef typename ei_packet_traits<Scalar>::type PacketType;
  typedef ei_product_blocking_traits<Scalar> Blocking;

  int kc = std::min<int>(Blocking::Max_kc,depth);  // cache block size along the K direction
  int mc = std::min<int>(Blocking::Max_mc,rows);   // cache block size along the M direction

  Scalar* blockA = ei_aligned_stack_new(Scalar, kc*mc);
  Scalar* blockB = ei_aligned_stack_new(Scalar, kc*cols*Blocking::PacketSize);

  // For each horizontal panel of the rhs, and corresponding panel of the lhs...
  // (==GEMM_VAR1)
  for(int k2=0; k2<depth; k2+=kc)
  {
    const int actual_kc = std::min(k2+kc,depth)-k2;

    // OK, here we have selected one horizontal panel of rhs and one vertical panel of lhs.
    // => Pack rhs's panel into a sequential chunk of memory (L2 caching)
    // Note that this panel will be read as many times as the number of blocks in the lhs's
    // vertical panel which is, in practice, a very low number.
    ei_gemm_pack_rhs<Scalar, Blocking::nr, RhsStorageOrder>()(blockB, &rhs(k2,0), rhsStride, alpha, actual_kc, cols);

    // For each mc x kc block of the lhs's vertical panel...
    // (==GEPP_VAR1)
    for(int i2=0; i2<rows; i2+=mc)
    {
      const int actual_mc = std::min(i2+mc,rows)-i2;

      // We pack the lhs's block into a sequential chunk of memory (L1 caching)
      // Note that this block will be read a very high number of times, which is equal to the number of
      // micro vertical panel of the large rhs's panel (e.g., cols/4 times).
      ei_gemm_pack_lhs<Scalar, Blocking::mr, LhsStorageOrder>()(blockA, &lhs(i2,k2), lhsStride, actual_kc, actual_mc);

      // Everything is packed, we can now call the block * panel kernel:
      ei_gebp_kernel<Scalar, Blocking::mr, Blocking::nr, ei_conj_helper<ConjugateLhs,ConjugateRhs> >()
        (res+i2, resStride, blockA, blockB, actual_mc, actual_kc, cols);
    }
  }

  ei_aligned_stack_delete(Scalar, blockA, kc*mc);
  ei_aligned_stack_delete(Scalar, blockB, kc*cols*Blocking::PacketSize);
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

template<typename Lhs, typename Rhs>
class GeneralProduct<Lhs, Rhs, GemmProduct>
  : public ProductBase<GeneralProduct<Lhs,Rhs,GemmProduct>, Lhs, Rhs>
{
  public:
    EIGEN_PRODUCT_PUBLIC_INTERFACE(GeneralProduct)

    GeneralProduct(const Lhs& lhs, const Rhs& rhs) : Base(lhs,rhs) {}

    template<typename Dest> void scaleAndAddTo(Dest& dst, Scalar alpha) const
    {
      ei_assert(dst.rows()==m_lhs.rows() && dst.cols()==m_rhs.cols());

      const ActualLhsType lhs = LhsBlasTraits::extract(m_lhs);
      const ActualRhsType rhs = RhsBlasTraits::extract(m_rhs);

      Scalar actualAlpha = alpha * LhsBlasTraits::extractScalarFactor(m_lhs)
                                 * RhsBlasTraits::extractScalarFactor(m_rhs);

      ei_general_matrix_matrix_product<
        Scalar,
        (_ActualLhsType::Flags&RowMajorBit)?RowMajor:ColMajor, bool(LhsBlasTraits::NeedToConjugate),
        (_ActualRhsType::Flags&RowMajorBit)?RowMajor:ColMajor, bool(RhsBlasTraits::NeedToConjugate),
        (Dest::Flags&RowMajorBit)?RowMajor:ColMajor>
      ::run(
          this->rows(), this->cols(), lhs.cols(),
          (const Scalar*)&(lhs.const_cast_derived().coeffRef(0,0)), lhs.stride(),
          (const Scalar*)&(rhs.const_cast_derived().coeffRef(0,0)), rhs.stride(),
          (Scalar*)&(dst.coeffRef(0,0)), dst.stride(),
          actualAlpha);
    }
};

#endif // EIGEN_GENERAL_MATRIX_MATRIX_H
