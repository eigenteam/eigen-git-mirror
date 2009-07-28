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

#ifndef EIGEN_TRIANGULAR_MATRIX_MATRIX_H
#define EIGEN_TRIANGULAR_MATRIX_MATRIX_H

// template<typename Scalar, int mr, int StorageOrder, bool Conjugate, int Mode>
// struct ei_gemm_pack_lhs_triangular
// {
//   Matrix<Scalar,mr,mr,
//   void operator()(Scalar* blockA, const EIGEN_RESTRICT Scalar* _lhs, int lhsStride, int depth, int rows)
//   {
//     ei_conj_if<NumTraits<Scalar>::IsComplex && Conjugate> cj;
//     ei_const_blas_data_mapper<Scalar, StorageOrder> lhs(_lhs,lhsStride);
//     int count = 0;
//     const int peeled_mc = (rows/mr)*mr;
//     for(int i=0; i<peeled_mc; i+=mr)
//     {
//       for(int k=0; k<depth; k++)
//         for(int w=0; w<mr; w++)
//           blockA[count++] = cj(lhs(i+w, k));
//     }
//     for(int i=peeled_mc; i<rows; i++)
//     {
//       for(int k=0; k<depth; k++)
//         blockA[count++] = cj(lhs(i, k));
//     }
//   }
// };

/* Optimized selfadjoint matrix * matrix (_SYMM) product built on top of
 * the general matrix matrix product.
 */
template <typename Scalar,
          int Mode, bool LhsIsTriangular,
          int LhsStorageOrder, bool ConjugateLhs,
          int RhsStorageOrder, bool ConjugateRhs,
          int ResStorageOrder>
struct ei_product_triangular_matrix_matrix;

template <typename Scalar,
          int Mode, bool LhsIsTriangular,
          int LhsStorageOrder, bool ConjugateLhs,
          int RhsStorageOrder, bool ConjugateRhs>
struct ei_product_triangular_matrix_matrix<Scalar,Mode,LhsIsTriangular,
                                           LhsStorageOrder,ConjugateLhs,
                                           RhsStorageOrder,ConjugateRhs,RowMajor>
{
  static EIGEN_STRONG_INLINE void run(
    int size, int otherSize,
    const Scalar* lhs, int lhsStride,
    const Scalar* rhs, int rhsStride,
    Scalar* res,       int resStride,
    Scalar alpha)
  {
    ei_product_triangular_matrix_matrix<Scalar,
      (Mode&UnitDiagBit) | (Mode&UpperTriangular) ? LowerTriangular : UpperTriangular,
      (!LhsIsTriangular),
      RhsStorageOrder==RowMajor ? ColMajor : RowMajor,
      ConjugateRhs,
      LhsStorageOrder==RowMajor ? ColMajor : RowMajor,
      ConjugateLhs,
      ColMajor>
      ::run(size, otherSize, rhs, rhsStride, lhs, lhsStride, res, resStride, alpha);
  }
};

// implements col-major += alpha * op(triangular) * op(general)
template <typename Scalar, int Mode,
          int LhsStorageOrder, bool ConjugateLhs,
          int RhsStorageOrder, bool ConjugateRhs>
struct ei_product_triangular_matrix_matrix<Scalar,Mode,true,
                                           LhsStorageOrder,ConjugateLhs,
                                           RhsStorageOrder,ConjugateRhs,ColMajor>
{

  static EIGEN_DONT_INLINE void run(
    int size, int cols,
    const Scalar* _lhs, int lhsStride,
    const Scalar* _rhs, int rhsStride,
    Scalar* res,        int resStride,
    Scalar alpha)
  {
    int rows = size;
    
    ei_const_blas_data_mapper<Scalar, LhsStorageOrder> lhs(_lhs,lhsStride);
    ei_const_blas_data_mapper<Scalar, RhsStorageOrder> rhs(_rhs,rhsStride);

    if (ConjugateRhs)
      alpha = ei_conj(alpha);

    typedef ei_product_blocking_traits<Scalar> Blocking;
    enum {
      SmallPanelWidth   = EIGEN_ENUM_MAX(Blocking::mr,Blocking::nr),
      IsLowerTriangular = (Mode&LowerTriangular) == LowerTriangular
    };

    int kc = std::min<int>(Blocking::Max_kc/4,size); // cache block size along the K direction
    int mc = std::min<int>(Blocking::Max_mc,rows);   // cache block size along the M direction

    Scalar* blockA = ei_aligned_stack_new(Scalar, kc*mc);
    Scalar* blockB = ei_aligned_stack_new(Scalar, kc*cols*Blocking::PacketSize);

    Matrix<Scalar,SmallPanelWidth,SmallPanelWidth,LhsStorageOrder> triangularBuffer;
    triangularBuffer.setZero();
    triangularBuffer.diagonal().setOnes();

    ei_gebp_kernel<Scalar, Blocking::mr, Blocking::nr, ei_conj_helper<ConjugateLhs,ConjugateRhs> > gebp_kernel;
    ei_gemm_pack_lhs<Scalar,Blocking::mr,LhsStorageOrder> pack_lhs;
    ei_gemm_pack_rhs<Scalar,Blocking::nr,RhsStorageOrder> pack_rhs;

    for(int k2=IsLowerTriangular ? size : 0;
        IsLowerTriangular ? k2>0 : k2<size;
        IsLowerTriangular ? k2-=kc : k2+=kc)
    {
      const int actual_kc = std::min(IsLowerTriangular ? k2 : size-k2, kc);
      int actual_k2 = IsLowerTriangular ? k2-actual_kc : k2;

      pack_rhs(blockB, &rhs(actual_k2,0), rhsStride, alpha, actual_kc, cols);

      // the selected lhs's panel has to be split in three different parts:
      //  1 - the part which is above the diagonal block => skip it
      //  2 - the diagonal block => special kernel
      //  3 - the panel below the diagonal block => GEPP
      // the block diagonal
      {
        // for each small vertical panels of lhs
        for (int k1=0; k1<actual_kc; k1+=SmallPanelWidth)
        {
          int actualPanelWidth = std::min<int>(actual_kc-k1, SmallPanelWidth);
          int lengthTarget = IsLowerTriangular ? actual_kc-k1-actualPanelWidth : k1;
          int startBlock   = actual_k2+k1;
          int blockBOffset = k1;
          
          // => GEBP with the micro triangular block
          // The trick is to pack this micro block while filling the opposite triangular part with zeros.
          // To this end we do an extra triangular copy to small temporary buffer
          for (int k=0;k<actualPanelWidth;++k)
          {
            if (!(Mode&UnitDiagBit))
              triangularBuffer.coeffRef(k,k) = lhs(startBlock+k,startBlock+k);
            for (int i=IsLowerTriangular ? k+1 : 0; IsLowerTriangular ? i<actualPanelWidth : i<k; ++i)
              triangularBuffer.coeffRef(i,k) = lhs(startBlock+i,startBlock+k);
          }
          pack_lhs(blockA, triangularBuffer.data(), triangularBuffer.stride(), actualPanelWidth, actualPanelWidth);

          gebp_kernel(res+startBlock, resStride, blockA, blockB, actualPanelWidth, actualPanelWidth, cols,
                      actualPanelWidth, actual_kc, 0, blockBOffset*Blocking::PacketSize);

          // GEBP with remaining micro panel
          if (lengthTarget>0)
          {
            int startTarget  = IsLowerTriangular ? actual_k2+k1+actualPanelWidth : actual_k2;

            pack_lhs(blockA, &lhs(startTarget,startBlock), lhsStride, actualPanelWidth, lengthTarget);

            gebp_kernel(res+startTarget, resStride, blockA, blockB, lengthTarget, actualPanelWidth, cols,
                        actualPanelWidth, actual_kc, 0, blockBOffset*Blocking::PacketSize);
          }
        }
      }
      // the part below the diagonal => GEPP
      {
        int start = IsLowerTriangular ? k2 : 0;
        int end   = IsLowerTriangular ? size : actual_k2;
        for(int i2=start; i2<end; i2+=mc)
        {
          const int actual_mc = std::min(i2+mc,end)-i2;
          ei_gemm_pack_lhs<Scalar,Blocking::mr,LhsStorageOrder,false>()
            (blockA, &lhs(i2, actual_k2), lhsStride, actual_kc, actual_mc);

          gebp_kernel(res+i2, resStride, blockA, blockB, actual_mc, actual_kc, cols);
        }
      }
    }

    ei_aligned_stack_delete(Scalar, blockA, kc*mc);
    ei_aligned_stack_delete(Scalar, blockB, kc*cols*Blocking::PacketSize);
  }
};

// implements col-major += alpha * op(general) * op(triangular)
template <typename Scalar, int Mode,
          int LhsStorageOrder, bool ConjugateLhs,
          int RhsStorageOrder, bool ConjugateRhs>
struct ei_product_triangular_matrix_matrix<Scalar,Mode,false,
                                           LhsStorageOrder,ConjugateLhs,
                                           RhsStorageOrder,ConjugateRhs,ColMajor>
{

  static EIGEN_DONT_INLINE void run(
    int size, int rows,
    const Scalar* _lhs, int lhsStride,
    const Scalar* _rhs, int rhsStride,
    Scalar* res,        int resStride,
    Scalar alpha)
  {
    int cols = size;

    ei_const_blas_data_mapper<Scalar, LhsStorageOrder> lhs(_lhs,lhsStride);
    ei_const_blas_data_mapper<Scalar, RhsStorageOrder> rhs(_rhs,rhsStride);

    if (ConjugateRhs)
      alpha = ei_conj(alpha);

    typedef ei_product_blocking_traits<Scalar> Blocking;
    enum {
      SmallPanelWidth   = EIGEN_ENUM_MAX(Blocking::mr,Blocking::nr),
      IsLowerTriangular = (Mode&LowerTriangular) == LowerTriangular
    };

    int kc = std::min<int>(Blocking::Max_kc/4,size); // cache block size along the K direction
    int mc = std::min<int>(Blocking::Max_mc,rows);   // cache block size along the M direction

    Scalar* blockA = ei_aligned_stack_new(Scalar, kc*mc);
    Scalar* blockB = ei_aligned_stack_new(Scalar, kc*cols*Blocking::PacketSize);

    Matrix<Scalar,SmallPanelWidth,SmallPanelWidth,RhsStorageOrder> triangularBuffer;
    triangularBuffer.setZero();
    triangularBuffer.diagonal().setOnes();

    ei_gebp_kernel<Scalar, Blocking::mr, Blocking::nr, ei_conj_helper<ConjugateLhs,ConjugateRhs> > gebp_kernel;
    ei_gemm_pack_lhs<Scalar,Blocking::mr,LhsStorageOrder> pack_lhs;
    ei_gemm_pack_rhs<Scalar,Blocking::nr,RhsStorageOrder> pack_rhs;
    ei_gemm_pack_rhs<Scalar,Blocking::nr,RhsStorageOrder,true> pack_rhs_panel;

    for(int k2=IsLowerTriangular ? 0 : size;
        IsLowerTriangular ? k2<size  : k2>0;
        IsLowerTriangular ? k2+=kc   : k2-=kc)
    {
      const int actual_kc = std::min(IsLowerTriangular ? size-k2 : k2, kc);
      int actual_k2 = IsLowerTriangular ? k2 : k2-actual_kc;
      int rs = IsLowerTriangular ? actual_k2 : size - k2;
      Scalar* geb = blockB+actual_kc*actual_kc*Blocking::PacketSize;

      pack_rhs(geb, &rhs(actual_k2,IsLowerTriangular ? 0 : k2), rhsStride, alpha, actual_kc, rs);

      // pack the triangular part of the rhs padding the unrolled blocks with zeros
      {
        for (int j2=0; j2<actual_kc; j2+=SmallPanelWidth)
        {
          int actualPanelWidth = std::min<int>(actual_kc-j2, SmallPanelWidth);
          int actual_j2 = actual_k2 + j2;
          int panelOffset = IsLowerTriangular ? j2+actualPanelWidth : 0;
          int panelLength = IsLowerTriangular ? actual_kc-j2-actualPanelWidth : j2;
          // general part
          pack_rhs_panel(blockB+j2*actual_kc*Blocking::PacketSize,
                         &rhs(actual_k2+panelOffset, actual_j2), rhsStride, alpha,
                         panelLength, actualPanelWidth,
                         actual_kc, panelOffset);
          
          // append the triangular part via a temporary buffer
          for (int j=0;j<actualPanelWidth;++j)
          {
            if (!(Mode&UnitDiagBit))
              triangularBuffer.coeffRef(j,j) = rhs(actual_j2+j,actual_j2+j);
            for (int k=IsLowerTriangular ? j+1 : 0; IsLowerTriangular ? k<actualPanelWidth : k<j; ++k)
              triangularBuffer.coeffRef(k,j) = rhs(actual_j2+k,actual_j2+j);
          }

          pack_rhs_panel(blockB+j2*actual_kc*Blocking::PacketSize,
                         triangularBuffer.data(), triangularBuffer.stride(), alpha,
                         actualPanelWidth, actualPanelWidth,
                         actual_kc, j2);
        }
      }

      for (int i2=0; i2<rows; i2+=mc)
      {
        const int actual_mc = std::min(mc,rows-i2);
        pack_lhs(blockA, &lhs(i2, actual_k2), lhsStride, actual_kc, actual_mc);

        // triangular kernel
        {
          for (int j2=0; j2<actual_kc; j2+=SmallPanelWidth)
          {
            int actualPanelWidth = std::min<int>(actual_kc-j2, SmallPanelWidth);
            int panelLength = IsLowerTriangular ? actual_kc-j2 : j2+actualPanelWidth;
            int blockOffset = IsLowerTriangular ? j2 : 0;

            gebp_kernel(res+i2+(actual_k2+j2)*resStride, resStride,
                        blockA, blockB+j2*actual_kc*Blocking::PacketSize,
                        actual_mc, panelLength, actualPanelWidth,
                        actual_kc, actual_kc,  // strides
                        blockOffset, blockOffset*Blocking::PacketSize);// offsets
          }
        }
        gebp_kernel(res+i2+(IsLowerTriangular ? 0 : k2)*resStride, resStride,
                    blockA, geb, actual_mc, actual_kc, rs);
      }
    }

    ei_aligned_stack_delete(Scalar, blockA, kc*mc);
    ei_aligned_stack_delete(Scalar, blockB, kc*cols*Blocking::PacketSize);
  }
};

/***************************************************************************
* Wrapper to ei_product_triangular_matrix_matrix
***************************************************************************/

template<int Mode, bool LhsIsTriangular, typename Lhs, typename Rhs>
struct ei_triangular_product_returntype<Mode,LhsIsTriangular,Lhs,false,Rhs,false>
  : public ReturnByValue<ei_triangular_product_returntype<Mode,LhsIsTriangular,Lhs,false,Rhs,false>,
                         Matrix<typename ei_traits<Rhs>::Scalar,
                                Lhs::RowsAtCompileTime,Rhs::ColsAtCompileTime> >
{
  ei_triangular_product_returntype(const Lhs& lhs, const Rhs& rhs)
    : m_lhs(lhs), m_rhs(rhs)
  {}

  inline int rows() const { return m_lhs.rows(); }
  inline int cols() const { return m_lhs.cols(); }

  typedef typename Lhs::Scalar Scalar;

  typedef typename Lhs::Nested LhsNested;
  typedef typename ei_cleantype<LhsNested>::type _LhsNested;
  typedef ei_blas_traits<_LhsNested> LhsBlasTraits;
  typedef typename LhsBlasTraits::DirectLinearAccessType ActualLhsType;
  typedef typename ei_cleantype<ActualLhsType>::type _ActualLhsType;

  typedef typename Rhs::Nested RhsNested;
  typedef typename ei_cleantype<RhsNested>::type _RhsNested;
  typedef ei_blas_traits<_RhsNested> RhsBlasTraits;
  typedef typename RhsBlasTraits::DirectLinearAccessType ActualRhsType;
  typedef typename ei_cleantype<ActualRhsType>::type _ActualRhsType;

  template<typename Dest> inline void _addTo(Dest& dst) const
  { evalTo(dst,1); }
  template<typename Dest> inline void _subTo(Dest& dst) const
  { evalTo(dst,-1); }

  template<typename Dest> void evalTo(Dest& dst) const
  {
    dst.resize(m_lhs.rows(), m_rhs.cols());
    dst.setZero();
    evalTo(dst,1);
  }

  template<typename Dest> void evalTo(Dest& dst, Scalar alpha) const
  {
    const ActualLhsType lhs = LhsBlasTraits::extract(m_lhs);
    const ActualRhsType rhs = RhsBlasTraits::extract(m_rhs);

    Scalar actualAlpha = alpha * LhsBlasTraits::extractScalarFactor(m_lhs)
                               * RhsBlasTraits::extractScalarFactor(m_rhs);

    ei_product_triangular_matrix_matrix<Scalar,
      Mode, LhsIsTriangular,
      (ei_traits<_ActualLhsType>::Flags&RowMajorBit) ? RowMajor : ColMajor, LhsBlasTraits::NeedToConjugate,
      (ei_traits<_ActualRhsType>::Flags&RowMajorBit) ? RowMajor : ColMajor, RhsBlasTraits::NeedToConjugate,
      (ei_traits<Dest          >::Flags&RowMajorBit) ? RowMajor : ColMajor>
      ::run(
        lhs.rows(), LhsIsTriangular ? rhs.cols() : lhs.rows(),           // sizes
        &lhs.coeff(0,0),    lhs.stride(), // lhs info
        &rhs.coeff(0,0),    rhs.stride(), // rhs info
        &dst.coeffRef(0,0), dst.stride(), // result info
        actualAlpha                       // alpha
      );
  }

  const LhsNested m_lhs;
  const RhsNested m_rhs;
};

#endif // EIGEN_TRIANGULAR_MATRIX_MATRIX_H
