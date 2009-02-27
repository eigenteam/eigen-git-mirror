// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
// Copyright (C) 2008 Gael Guennebaud <g.gael@free.fr>
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

template <int L2MemorySize,typename Scalar>
struct ei_L2_block_traits {
  enum {width = 8 * ei_meta_sqrt<L2MemorySize/(64*sizeof(Scalar))>::ret };
};

#ifndef EIGEN_EXTERN_INSTANTIATIONS

template<typename Scalar>
static void ei_cache_friendly_product(
  int _rows, int _cols, int depth,
  bool _lhsRowMajor, const Scalar* _lhs, int _lhsStride,
  bool _rhsRowMajor, const Scalar* _rhs, int _rhsStride,
  bool resRowMajor, Scalar* res, int resStride)
{
  const Scalar* EIGEN_RESTRICT lhs;
  const Scalar* EIGEN_RESTRICT rhs;
  int lhsStride, rhsStride, rows, cols;
  bool lhsRowMajor;

  if (resRowMajor)
  {
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
    // i386 architecture provides only 8 xmm registers,
    // so let's reduce the max number of rows processed at once.
    MaxBlockRows = 4,
    MaxBlockRows_ClampingMask = 0xFFFFFC,
    #else
    MaxBlockRows = 8,
    MaxBlockRows_ClampingMask = 0xFFFFF8,
    #endif
    // maximal size of the blocks fitted in L2 cache
    MaxL2BlockSize = ei_L2_block_traits<EIGEN_TUNE_FOR_CPU_CACHE_SIZE,Scalar>::width
  };

  const bool resIsAligned = (PacketSize==1) || (((resStride%PacketSize) == 0) && (size_t(res)%16==0));

  const int remainingSize = depth % PacketSize;
  const int size = depth - remainingSize; // third dimension of the product clamped to packet boundaries
  const int l2BlockRows = MaxL2BlockSize > rows ? rows : MaxL2BlockSize;
  const int l2BlockCols = MaxL2BlockSize > cols ? cols : MaxL2BlockSize;
  const int l2BlockSize = MaxL2BlockSize > size ? size : MaxL2BlockSize;
  const int l2BlockSizeAligned = (1 + std::max(l2BlockSize,l2BlockCols)/PacketSize)*PacketSize;
  const bool needRhsCopy = (PacketSize>1) && ((rhsStride%PacketSize!=0) || (size_t(rhs)%16!=0));
  Scalar* EIGEN_RESTRICT block = 0;
  const int allocBlockSize = l2BlockRows*size;
  block = ei_aligned_stack_new(Scalar, allocBlockSize);
  Scalar* EIGEN_RESTRICT rhsCopy
    = ei_aligned_stack_new(Scalar, l2BlockSizeAligned*l2BlockSizeAligned);

#ifndef EIGEN_USE_NEW_PRODUCT
  // loops on each L2 cache friendly blocks of the result
  for(int l2i=0; l2i<rows; l2i+=l2BlockRows)
  {
    const int l2blockRowEnd = std::min(l2i+l2BlockRows, rows);
    const int l2blockRowEndBW = l2blockRowEnd & MaxBlockRows_ClampingMask;    // end of the rows aligned to bw
    const int l2blockRemainingRows = l2blockRowEnd - l2blockRowEndBW;         // number of remaining rows
    //const int l2blockRowEndBWPlusOne = l2blockRowEndBW + (l2blockRemainingRows?0:MaxBlockRows);

    // build a cache friendly blocky matrix
    int count = 0;

    // copy l2blocksize rows of m_lhs to blocks of ps x bw
    for(int l2k=0; l2k<size; l2k+=l2BlockSize)
    {
      const int l2blockSizeEnd = std::min(l2k+l2BlockSize, size);

      for (int i = l2i; i<l2blockRowEndBW/*PlusOne*/; i+=MaxBlockRows)
      {
        // TODO merge the "if l2blockRemainingRows" using something like:
        // const int blockRows = std::min(i+MaxBlockRows, rows) - i;

        for (int k=l2k; k<l2blockSizeEnd; k+=PacketSize)
        {
          // TODO write these loops using meta unrolling
          // negligible for large matrices but useful for small ones
          if (lhsRowMajor)
          {
            for (int w=0; w<MaxBlockRows; ++w)
              for (int s=0; s<PacketSize; ++s)
                block[count++] = lhs[(i+w)*lhsStride + (k+s)];
          }
          else
          {
            for (int w=0; w<MaxBlockRows; ++w)
              for (int s=0; s<PacketSize; ++s)
                block[count++] = lhs[(i+w) + (k+s)*lhsStride];
          }
        }
      }
      if (l2blockRemainingRows>0)
      {
        for (int k=l2k; k<l2blockSizeEnd; k+=PacketSize)
        {
          if (lhsRowMajor)
          {
            for (int w=0; w<l2blockRemainingRows; ++w)
              for (int s=0; s<PacketSize; ++s)
                block[count++] = lhs[(l2blockRowEndBW+w)*lhsStride + (k+s)];
          }
          else
          {
            for (int w=0; w<l2blockRemainingRows; ++w)
              for (int s=0; s<PacketSize; ++s)
                block[count++] = lhs[(l2blockRowEndBW+w) + (k+s)*lhsStride];
          }
        }
      }
    }

    for(int l2j=0; l2j<cols; l2j+=l2BlockCols)
    {
      int l2blockColEnd = std::min(l2j+l2BlockCols, cols);

      for(int l2k=0; l2k<size; l2k+=l2BlockSize)
      {
        // acumulate bw rows of lhs time a single column of rhs to a bw x 1 block of res
        int l2blockSizeEnd = std::min(l2k+l2BlockSize, size);

        // if not aligned, copy the rhs block
        if (needRhsCopy)
          for(int l1j=l2j; l1j<l2blockColEnd; l1j+=1)
          {
            ei_internal_assert(l2BlockSizeAligned*(l1j-l2j)+(l2blockSizeEnd-l2k) < l2BlockSizeAligned*l2BlockSizeAligned);
            memcpy(rhsCopy+l2BlockSizeAligned*(l1j-l2j),&(rhs[l1j*rhsStride+l2k]),(l2blockSizeEnd-l2k)*sizeof(Scalar));
          }

        // for each bw x 1 result's block
        for(int l1i=l2i; l1i<l2blockRowEndBW; l1i+=MaxBlockRows)
        {
          int offsetblock = l2k * (l2blockRowEnd-l2i) + (l1i-l2i)*(l2blockSizeEnd-l2k) - l2k*MaxBlockRows;
          const Scalar* EIGEN_RESTRICT localB = &block[offsetblock];

          for(int l1j=l2j; l1j<l2blockColEnd; l1j+=1)
          {
            const Scalar* EIGEN_RESTRICT rhsColumn;
            if (needRhsCopy)
              rhsColumn = &(rhsCopy[l2BlockSizeAligned*(l1j-l2j)-l2k]);
            else
              rhsColumn = &(rhs[l1j*rhsStride]);

            PacketType dst[MaxBlockRows];
            dst[3] = dst[2] = dst[1] = dst[0] = ei_pset1(Scalar(0.));
            if (MaxBlockRows==8)
              dst[7] = dst[6] = dst[5] = dst[4] = dst[0];

            PacketType tmp;

            for(int k=l2k; k<l2blockSizeEnd; k+=PacketSize)
            {
              tmp = ei_ploadu(&rhsColumn[k]);
              PacketType A0, A1, A2, A3, A4, A5;
              A0 = ei_pload(localB + k*MaxBlockRows);
              A1 = ei_pload(localB + k*MaxBlockRows+1*PacketSize);
              A2 = ei_pload(localB + k*MaxBlockRows+2*PacketSize);
              A3 = ei_pload(localB + k*MaxBlockRows+3*PacketSize);
              if (MaxBlockRows==8) A4 = ei_pload(localB + k*MaxBlockRows+4*PacketSize);
              if (MaxBlockRows==8) A5 = ei_pload(localB + k*MaxBlockRows+5*PacketSize);
              dst[0] = ei_pmadd(tmp, A0, dst[0]);
              if (MaxBlockRows==8) A0 = ei_pload(localB + k*MaxBlockRows+6*PacketSize);
              dst[1] = ei_pmadd(tmp, A1, dst[1]);
              if (MaxBlockRows==8) A1 = ei_pload(localB + k*MaxBlockRows+7*PacketSize);
              dst[2] = ei_pmadd(tmp, A2, dst[2]);
              dst[3] = ei_pmadd(tmp, A3, dst[3]);
              if (MaxBlockRows==8)
              {
                dst[4] = ei_pmadd(tmp, A4, dst[4]);
                dst[5] = ei_pmadd(tmp, A5, dst[5]);
                dst[6] = ei_pmadd(tmp, A0, dst[6]);
                dst[7] = ei_pmadd(tmp, A1, dst[7]);
              }
            }

            Scalar* EIGEN_RESTRICT localRes = &(res[l1i + l1j*resStride]);

            if (PacketSize>1 && resIsAligned)
            {
              // the result is aligned: let's do packet reduction
              ei_pstore(&(localRes[0]), ei_padd(ei_pload(&(localRes[0])), ei_preduxp(&dst[0])));
              if (PacketSize==2)
                ei_pstore(&(localRes[2]), ei_padd(ei_pload(&(localRes[2])), ei_preduxp(&(dst[2]))));
              if (MaxBlockRows==8)
              {
                ei_pstore(&(localRes[4]), ei_padd(ei_pload(&(localRes[4])), ei_preduxp(&(dst[4]))));
                if (PacketSize==2)
                  ei_pstore(&(localRes[6]), ei_padd(ei_pload(&(localRes[6])), ei_preduxp(&(dst[6]))));
              }
            }
            else
            {
              // not aligned => per coeff packet reduction
              localRes[0] += ei_predux(dst[0]);
              localRes[1] += ei_predux(dst[1]);
              localRes[2] += ei_predux(dst[2]);
              localRes[3] += ei_predux(dst[3]);
              if (MaxBlockRows==8)
              {
                localRes[4] += ei_predux(dst[4]);
                localRes[5] += ei_predux(dst[5]);
                localRes[6] += ei_predux(dst[6]);
                localRes[7] += ei_predux(dst[7]);
              }
            }
          }
        }
        if (l2blockRemainingRows>0)
        {
          int offsetblock = l2k * (l2blockRowEnd-l2i) + (l2blockRowEndBW-l2i)*(l2blockSizeEnd-l2k) - l2k*l2blockRemainingRows;
          const Scalar* localB = &block[offsetblock];

          for(int l1j=l2j; l1j<l2blockColEnd; l1j+=1)
          {
            const Scalar* EIGEN_RESTRICT rhsColumn;
            if (needRhsCopy)
              rhsColumn = &(rhsCopy[l2BlockSizeAligned*(l1j-l2j)-l2k]);
            else
              rhsColumn = &(rhs[l1j*rhsStride]);

            PacketType dst[MaxBlockRows];
            dst[3] = dst[2] = dst[1] = dst[0] = ei_pset1(Scalar(0.));
            if (MaxBlockRows==8)
              dst[7] = dst[6] = dst[5] = dst[4] = dst[0];

            // let's declare a few other temporary registers
            PacketType tmp;

            for(int k=l2k; k<l2blockSizeEnd; k+=PacketSize)
            {
              tmp = ei_pload(&rhsColumn[k]);

                                           dst[0] = ei_pmadd(tmp, ei_pload(&(localB[k*l2blockRemainingRows             ])), dst[0]);
              if (l2blockRemainingRows>=2) dst[1] = ei_pmadd(tmp, ei_pload(&(localB[k*l2blockRemainingRows+  PacketSize])), dst[1]);
              if (l2blockRemainingRows>=3) dst[2] = ei_pmadd(tmp, ei_pload(&(localB[k*l2blockRemainingRows+2*PacketSize])), dst[2]);
              if (l2blockRemainingRows>=4) dst[3] = ei_pmadd(tmp, ei_pload(&(localB[k*l2blockRemainingRows+3*PacketSize])), dst[3]);
              if (MaxBlockRows==8)
              {
                if (l2blockRemainingRows>=5) dst[4] = ei_pmadd(tmp, ei_pload(&(localB[k*l2blockRemainingRows+4*PacketSize])), dst[4]);
                if (l2blockRemainingRows>=6) dst[5] = ei_pmadd(tmp, ei_pload(&(localB[k*l2blockRemainingRows+5*PacketSize])), dst[5]);
                if (l2blockRemainingRows>=7) dst[6] = ei_pmadd(tmp, ei_pload(&(localB[k*l2blockRemainingRows+6*PacketSize])), dst[6]);
                if (l2blockRemainingRows>=8) dst[7] = ei_pmadd(tmp, ei_pload(&(localB[k*l2blockRemainingRows+7*PacketSize])), dst[7]);
              }
            }

            Scalar* EIGEN_RESTRICT localRes = &(res[l2blockRowEndBW + l1j*resStride]);

            // process the remaining rows once at a time
                                         localRes[0] += ei_predux(dst[0]);
            if (l2blockRemainingRows>=2) localRes[1] += ei_predux(dst[1]);
            if (l2blockRemainingRows>=3) localRes[2] += ei_predux(dst[2]);
            if (l2blockRemainingRows>=4) localRes[3] += ei_predux(dst[3]);
            if (MaxBlockRows==8)
            {
              if (l2blockRemainingRows>=5) localRes[4] += ei_predux(dst[4]);
              if (l2blockRemainingRows>=6) localRes[5] += ei_predux(dst[5]);
              if (l2blockRemainingRows>=7) localRes[6] += ei_predux(dst[6]);
              if (l2blockRemainingRows>=8) localRes[7] += ei_predux(dst[7]);
            }

          }
        }
      }
    }
  }
  if (PacketSize>1 && remainingSize)
  {
    if (lhsRowMajor)
    {
      for (int j=0; j<cols; ++j)
        for (int i=0; i<rows; ++i)
        {
          Scalar tmp = lhs[i*lhsStride+size] * rhs[j*rhsStride+size];
          // FIXME this loop get vectorized by the compiler !
          for (int k=1; k<remainingSize; ++k)
            tmp += lhs[i*lhsStride+size+k] * rhs[j*rhsStride+size+k];
          res[i+j*resStride] += tmp;
        }
    }
    else
    {
      for (int j=0; j<cols; ++j)
        for (int i=0; i<rows; ++i)
        {
          Scalar tmp = lhs[i+size*lhsStride] * rhs[j*rhsStride+size];
          for (int k=1; k<remainingSize; ++k)
            tmp += lhs[i+(size+k)*lhsStride] * rhs[j*rhsStride+size+k];
          res[i+j*resStride] += tmp;
        }
    }
  }

#else
  // loops on each L2 cache friendly blocks of the result

  for(int l2j=0; l2j<cols; l2j+=l2BlockCols)
  {
    for(int l2i=0; l2i<rows; l2i+=l2BlockRows)
    {
      // We have selected a block of lhs
      // Packs this block into 'block'
      int count = 0;
      for(int j=0; j<l2BlockCols; j+=MaxBlockRows)
      {
        for(int i=0; i<l2BlockRows; i+=2*PacketSize)
          for (int w=0; w<MaxBlockRows; ++w)
            for (int y=0; y<2*PacketSize; ++y)
              block[count++] = lhs[(j+l2j+w)*rows + l2i+i+ y];
      }

      // loops on each L2 cache firendly block of the result/rhs
      for(int l2k=0; l2k<cols; l2k+=l2BlockCols)
      {
        for(int i=0; i<l2BlockRows; i+=MaxBlockRows)
        {
          for(int j=0; j<l2BlockCols; ++j)
          {
            PacketType A0, A1, A2, A3, A4, A5, A6, A7;

            // Load the packets from rhs and reorder them

            // Here we need some vector reordering
            // Right now its hardcoded to packets of 4 elements
            const Scalar* lrhs = &rhs[(j+l2k)*rows+(i+l2j)];
            A0 = ei_pset1(lrhs[0]);
            A1 = ei_pset1(lrhs[1]);
            A2 = ei_pset1(lrhs[2]);
            A3 = ei_pset1(lrhs[3]);
            if (MaxBlockRows==8)
            {
              A4 = ei_pset1(lrhs[4]);
              A5 = ei_pset1(lrhs[5]);
              A6 = ei_pset1(lrhs[6]);
              A7 = ei_pset1(lrhs[7]);
            }

            Scalar * lb = &block[l2BlockRows * i];
            for(int k=0; k<l2BlockRows; k+=2*PacketSize)
            {
              PacketType R0, R1, L0, L1, T0, T1;
              asm("#begin sgemm");

              // We perform "cross products" of vectors to avoid
              // reductions (horizontal ops) afterwards
              T0 = ei_pload(&res[(j+l2k)*rows+l2i+k]);
              T1 = ei_pload(&res[(j+l2k)*rows+l2i+k+PacketSize]);
              // uncomment to remove res cache miss
//               T0 = ei_pload(&res[k]);
//               T1 = ei_pload(&res[k+PacketSize]);

              R0 = ei_pload(&lb[0*PacketSize]);
              L0 = ei_pload(&lb[1*PacketSize]);
              R1 = ei_pload(&lb[2*PacketSize]);
              L1 = ei_pload(&lb[3*PacketSize]);
              T0 = ei_pmadd(R0, A0, T0);
              T1 = ei_pmadd(L0, A0, T1);
              R0 = ei_pload(&lb[4*PacketSize]);
              L0 = ei_pload(&lb[5*PacketSize]);
              T0 = ei_pmadd(R1, A1, T0);
              T1 = ei_pmadd(L1, A1, T1);
              R1 = ei_pload(&lb[6*PacketSize]);
              L1 = ei_pload(&lb[7*PacketSize]);
              T0 = ei_pmadd(R0, A2, T0);
              T1 = ei_pmadd(L0, A2, T1);
              if(MaxBlockRows==8)
              {
                R0 = ei_pload(&lb[8*PacketSize]);
                L0 = ei_pload(&lb[9*PacketSize]);
              }
              T0 = ei_pmadd(R1, A3, T0);
              T1 = ei_pmadd(L1, A3, T1);
              if(MaxBlockRows==8)
              {
                R1 = ei_pload(&lb[10*PacketSize]);
                L1 = ei_pload(&lb[11*PacketSize]);
                T0 = ei_pmadd(R0, A4, T0);
                T1 = ei_pmadd(L0, A4, T1);
                R0 = ei_pload(&lb[12*PacketSize]);
                L0 = ei_pload(&lb[13*PacketSize]);
                T0 = ei_pmadd(R1, A5, T0);
                T1 = ei_pmadd(L1, A5, T1);
                R1 = ei_pload(&lb[14*PacketSize]);
                L1 = ei_pload(&lb[15*PacketSize]);
                T0 = ei_pmadd(R0, A6, T0);
                T1 = ei_pmadd(L0, A6, T1);
                T0 = ei_pmadd(R1, A7, T0);
                T1 = ei_pmadd(L1, A7, T1);
              }
              lb += MaxBlockRows*2*PacketSize;

              ei_pstore(&res[(j+l2k)*rows+l2i+k], T0);
              ei_pstore(&res[(j+l2k)*rows+l2i+k+PacketSize], T1);
              // uncomment to remove res cache miss
//               ei_pstore(&res[k], T0);
//               ei_pstore(&res[k+PacketSize], T1);
              asm("#end sgemm");
            }
          }
        }
      }
    }
  }
#endif

  ei_aligned_stack_delete(Scalar, block, allocBlockSize);
  ei_aligned_stack_delete(Scalar, rhsCopy, l2BlockSizeAligned*l2BlockSizeAligned);
}

#endif // EIGEN_EXTERN_INSTANTIATIONS

#endif // EIGEN_GENERAL_MATRIX_MATRIX_H
