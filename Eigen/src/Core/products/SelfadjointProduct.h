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
    ei_selfadjoint_product<Scalar, MatStorageOrder, ColMajor, !AAT, UpLo==Lower?Upper:Lower>
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

    int kc = std::min<int>(Blocking::Max_kc,depth); // cache block size along the K direction
    int mc = std::min<int>(Blocking::Max_mc,size);  // cache block size along the M direction

    Scalar* blockA = ei_aligned_stack_new(Scalar, kc*mc);
    std::size_t sizeB = kc*Blocking::PacketSize*Blocking::nr + kc*size;
    Scalar* allocatedBlockB = ei_aligned_stack_new(Scalar, sizeB);
    Scalar* blockB = allocatedBlockB + kc*Blocking::PacketSize*Blocking::nr;
    
    // note that the actual rhs is the transpose/adjoint of mat
    typedef ei_conj_helper<NumTraits<Scalar>::IsComplex && !AAT, NumTraits<Scalar>::IsComplex && AAT> Conj;

    ei_gebp_kernel<Scalar, Blocking::mr, Blocking::nr, Conj> gebp_kernel;
    ei_gemm_pack_rhs<Scalar,Blocking::nr,MatStorageOrder==RowMajor ? ColMajor : RowMajor> pack_rhs;
    ei_gemm_pack_lhs<Scalar,Blocking::mr,MatStorageOrder, false> pack_lhs;
    ei_sybb_kernel<Scalar, Blocking::mr, Blocking::nr, Conj, UpLo> sybb;

    for(int k2=0; k2<depth; k2+=kc)
    {
      const int actual_kc = std::min(k2+kc,depth)-k2;

      // note that the actual rhs is the transpose/adjoint of mat
      pack_rhs(blockB, &mat(0,k2), matStride, alpha, actual_kc, size);

      for(int i2=0; i2<size; i2+=mc)
      {
        const int actual_mc = std::min(i2+mc,size)-i2;

        pack_lhs(blockA, &mat(i2, k2), matStride, actual_kc, actual_mc);

        // the selected actual_mc * size panel of res is split into three different part:
        //  1 - before the diagonal => processed with gebp or skipped
        //  2 - the actual_mc x actual_mc symmetric block => processed with a special kernel
        //  3 - after the diagonal => processed with gebp or skipped
        if (UpLo==Lower)
          gebp_kernel(res+i2, resStride, blockA, blockB, actual_mc, actual_kc, std::min(size,i2),
                      -1, -1, 0, 0, allocatedBlockB);

        sybb(res+resStride*i2 + i2, resStride, blockA, blockB + actual_kc*i2, actual_mc, actual_kc, allocatedBlockB);

        if (UpLo==Upper)
        {
          int j2 = i2+actual_mc;
          gebp_kernel(res+resStride*j2+i2, resStride, blockA, blockB+actual_kc*j2, actual_mc, actual_kc, std::max(0,size-j2),
                      -1, -1, 0, 0, allocatedBlockB);
        }
      }
    }
    ei_aligned_stack_delete(Scalar, blockA, kc*mc);
    ei_aligned_stack_delete(Scalar, allocatedBlockB, sizeB);
  }
};

// high level API

template<typename MatrixType, unsigned int UpLo>
template<typename DerivedU>
SelfAdjointView<MatrixType,UpLo>& SelfAdjointView<MatrixType,UpLo>
::rankUpdate(const MatrixBase<DerivedU>& u, Scalar alpha)
{
  typedef ei_blas_traits<DerivedU> UBlasTraits;
  typedef typename UBlasTraits::DirectLinearAccessType ActualUType;
  typedef typename ei_cleantype<ActualUType>::type _ActualUType;
  const ActualUType actualU = UBlasTraits::extract(u.derived());

  Scalar actualAlpha = alpha * UBlasTraits::extractScalarFactor(u.derived());

  enum { IsRowMajor = (ei_traits<MatrixType>::Flags&RowMajorBit) ? 1 : 0 };

  ei_selfadjoint_product<Scalar,
    _ActualUType::Flags&RowMajorBit ? RowMajor : ColMajor,
    ei_traits<MatrixType>::Flags&RowMajorBit ? RowMajor : ColMajor,
    !UBlasTraits::NeedToConjugate, UpLo>
    ::run(_expression().cols(), actualU.cols(), &actualU.coeff(0,0), actualU.stride(),
          const_cast<Scalar*>(_expression().data()), _expression().stride(), actualAlpha);

  return *this;
}


// Optimized SYmmetric packed Block * packed Block product kernel.
// This kernel is built on top of the gebp kernel:
// - the current selfadjoint block (res) is processed per panel of actual_mc x BlockSize
//   where BlockSize is set to the minimal value allowing gebp to be as fast as possible
// - then, as usual, each panel is split into three parts along the diagonal,
//   the sub blocks above and below the diagonal are processed as usual,
//   while the selfadjoint block overlapping the diagonal is evaluated into a
//   small temporary buffer which is then accumulated into the result using a
//   triangular traversal.
template<typename Scalar, int mr, int nr, typename Conj, int UpLo>
struct ei_sybb_kernel
{
  enum {
    PacketSize = ei_packet_traits<Scalar>::size,
    BlockSize  = EIGEN_ENUM_MAX(mr,nr)
  };
  void operator()(Scalar* res, int resStride, const Scalar* blockA, const Scalar* blockB, int size, int depth, Scalar* workspace)
  {
    ei_gebp_kernel<Scalar, mr, nr, Conj> gebp_kernel;
    Matrix<Scalar,BlockSize,BlockSize,ColMajor> buffer;

    // let's process the block per panel of actual_mc x BlockSize,
    // again, each is split into three parts, etc.
    for (int j=0; j<size; j+=BlockSize)
    {
      int actualBlockSize = std::min<int>(BlockSize,size - j);
      const Scalar* actual_b = blockB+j*depth;

      if(UpLo==Upper)
        gebp_kernel(res+j*resStride, resStride, blockA, actual_b, j, depth, actualBlockSize);

      // selfadjoint micro block
      {
        int i = j;
        buffer.setZero();
        // 1 - apply the kernel on the temporary buffer
        gebp_kernel(buffer.data(), BlockSize, blockA+depth*i, actual_b, actualBlockSize, depth, actualBlockSize,
                    -1, -1, 0, 0, workspace);
        // 2 - triangular accumulation
        for(int j1=0; j1<actualBlockSize; ++j1)
        {
          Scalar* r = res + (j+j1)*resStride + i;
          for(int i1=UpLo==Lower ? j1 : 0;
              UpLo==Lower ? i1<actualBlockSize : i1<=j1; ++i1)
            r[i1] += buffer(i1,j1);
        }
      }

      if(UpLo==Lower)
      {
        int i = j+actualBlockSize;
        gebp_kernel(res+j*resStride+i, resStride, blockA+depth*i, actual_b, size-i, depth, actualBlockSize,
                    -1, -1, 0, 0, workspace);
      }
    }
  }
};

#endif // EIGEN_SELFADJOINT_PRODUCT_H
