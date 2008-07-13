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

#ifndef EIGEN_CACHE_FRIENDLY_PRODUCT_H
#define EIGEN_CACHE_FRIENDLY_PRODUCT_H

#ifndef EIGEN_EXTERN_INSTANTIATIONS

template<typename Scalar>
static void ei_cache_friendly_product(
  int _rows, int _cols, int depth,
  bool _lhsRowMajor, const Scalar* _lhs, int _lhsStride,
  bool _rhsRowMajor, const Scalar* _rhs, int _rhsStride,
  bool resRowMajor, Scalar* res, int resStride)
{
  const Scalar* __restrict__ lhs;
  const Scalar* __restrict__ rhs;
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
    MaxL2BlockSize = EIGEN_TUNE_FOR_L2_CACHE_SIZE / sizeof(Scalar)
  };

  const bool resIsAligned = (PacketSize==1) || (((resStride%PacketSize) == 0) && (size_t(res)%16==0));

  const int remainingSize = depth % PacketSize;
  const int size = depth - remainingSize; // third dimension of the product clamped to packet boundaries
  const int l2BlockRows = MaxL2BlockSize > rows ? rows : MaxL2BlockSize;
  const int l2BlockCols = MaxL2BlockSize > cols ? cols : MaxL2BlockSize;
  const int l2BlockSize = MaxL2BlockSize > size ? size : MaxL2BlockSize;
  Scalar* __restrict__ block = 0;
  const int allocBlockSize = sizeof(Scalar)*l2BlockRows*size;
  if (allocBlockSize>16000000)
    block = (Scalar*)malloc(allocBlockSize);
  else
    block = (Scalar*)alloca(allocBlockSize);
  Scalar* __restrict__ rhsCopy = (Scalar*)alloca(sizeof(Scalar)*l2BlockSize);

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
    asm("#eigen begin buildblocks");
    for(int l2k=0; l2k<size; l2k+=l2BlockSize)
    {
      const int l2blockSizeEnd = std::min(l2k+l2BlockSize, size);

      for (int i = l2i; i<l2blockRowEndBW/*PlusOne*/; i+=MaxBlockRows)
      {
        // TODO merge the if l2blockRemainingRows
//         const int blockRows = std::min(i+MaxBlockRows, rows) - i;

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
    asm("#eigen end buildblocks");

    for(int l2j=0; l2j<cols; l2j+=l2BlockCols)
    {
      int l2blockColEnd = std::min(l2j+l2BlockCols, cols);

      for(int l2k=0; l2k<size; l2k+=l2BlockSize)
      {
        // acumulate bw rows of lhs time a single column of rhs to a bw x 1 block of res
        int l2blockSizeEnd = std::min(l2k+l2BlockSize, size);

        // for each bw x 1 result's block
        for(int l1i=l2i; l1i<l2blockRowEndBW; l1i+=MaxBlockRows)
        {
          for(int l1j=l2j; l1j<l2blockColEnd; l1j+=1)
          {
            int offsetblock = l2k * (l2blockRowEnd-l2i) + (l1i-l2i)*(l2blockSizeEnd-l2k) - l2k*MaxBlockRows;
            const Scalar* __restrict__ localB = &block[offsetblock];

            const Scalar* __restrict__ rhsColumn = &(rhs[l1j*rhsStride]);

            // copy unaligned rhs data
            // YES it seems to be faster to copy some part of rhs multiple times
            // to aligned memory rather than using unligned load.
            // Moreover this avoids a "if" in the most nested loop :)
            if (PacketSize>1 && size_t(rhsColumn)%16)
            {
              int count = 0;
              for (int k = l2k; k<l2blockSizeEnd; ++k)
              {
                rhsCopy[count++] = rhsColumn[k];
              }
              rhsColumn = &(rhsCopy[-l2k]);
            }

            PacketType dst[MaxBlockRows];
            dst[0] = ei_pset1(Scalar(0.));
            dst[1] = dst[0];
            dst[2] = dst[0];
            dst[3] = dst[0];
            if (MaxBlockRows==8)
            {
              dst[4] = dst[0];
              dst[5] = dst[0];
              dst[6] = dst[0];
              dst[7] = dst[0];
            }

            PacketType tmp;

            asm("#eigen begincore");
            for(int k=l2k; k<l2blockSizeEnd; k+=PacketSize)
            {
              tmp = ei_pload(&rhsColumn[k]);

              dst[0] = ei_pmadd(tmp, ei_pload(&(localB[k*MaxBlockRows             ])), dst[0]);
              dst[1] = ei_pmadd(tmp, ei_pload(&(localB[k*MaxBlockRows+  PacketSize])), dst[1]);
              dst[2] = ei_pmadd(tmp, ei_pload(&(localB[k*MaxBlockRows+2*PacketSize])), dst[2]);
              dst[3] = ei_pmadd(tmp, ei_pload(&(localB[k*MaxBlockRows+3*PacketSize])), dst[3]);
              if (MaxBlockRows==8)
              {
                dst[4] = ei_pmadd(tmp, ei_pload(&(localB[k*MaxBlockRows+4*PacketSize])), dst[4]);
                dst[5] = ei_pmadd(tmp, ei_pload(&(localB[k*MaxBlockRows+5*PacketSize])), dst[5]);
                dst[6] = ei_pmadd(tmp, ei_pload(&(localB[k*MaxBlockRows+6*PacketSize])), dst[6]);
                dst[7] = ei_pmadd(tmp, ei_pload(&(localB[k*MaxBlockRows+7*PacketSize])), dst[7]);
              }
            }

            Scalar* __restrict__ localRes = &(res[l1i + l1j*resStride]);

            if (PacketSize>1 && resIsAligned)
            {
              ei_pstore(&(localRes[0]), ei_padd(ei_pload(&(localRes[0])), ei_preduxp(dst)));
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
            asm("#eigen endcore");
          }
        }
        if (l2blockRemainingRows>0)
        {
          int offsetblock = l2k * (l2blockRowEnd-l2i) + (l2blockRowEndBW-l2i)*(l2blockSizeEnd-l2k) - l2k*l2blockRemainingRows;
          const Scalar* localB = &block[offsetblock];

          asm("#eigen begin dynkernel");
          for(int l1j=l2j; l1j<l2blockColEnd; l1j+=1)
          {
            const Scalar* __restrict__ rhsColumn = &(rhs[l1j*rhsStride]);

            // copy unaligned rhs data
            if (PacketSize>1 && size_t(rhsColumn)%16)
            {
              int count = 0;
              for (int k = l2k; k<l2blockSizeEnd; ++k)
              {
                rhsCopy[count++] = rhsColumn[k];
              }
              rhsColumn = &(rhsCopy[-l2k]);
            }

            PacketType dst[MaxBlockRows];
            dst[0] = ei_pset1(Scalar(0.));
            dst[1] = dst[0];
            dst[2] = dst[0];
            dst[3] = dst[0];
            if (MaxBlockRows>4)
            {
              dst[4] = dst[0];
              dst[5] = dst[0];
              dst[6] = dst[0];
              dst[7] = dst[0];
            }

            // let's declare a few other temporary registers
            PacketType tmp;

            for(int k=l2k; k<l2blockSizeEnd; k+=PacketSize)
            {
              tmp = ei_pload(&rhsColumn[k]);

                                           dst[0] = ei_pmadd(tmp, ei_pload(&(localB[k*l2blockRemainingRows             ])), dst[0]);
              if (l2blockRemainingRows>=2) dst[1] = ei_pmadd(tmp, ei_pload(&(localB[k*l2blockRemainingRows+  PacketSize])), dst[1]);
              if (l2blockRemainingRows>=3) dst[2] = ei_pmadd(tmp, ei_pload(&(localB[k*l2blockRemainingRows+2*PacketSize])), dst[2]);
              if (l2blockRemainingRows>=4) dst[3] = ei_pmadd(tmp, ei_pload(&(localB[k*l2blockRemainingRows+3*PacketSize])), dst[3]);
              if (MaxBlockRows>4)
              {
                if (l2blockRemainingRows>=5) dst[4] = ei_pmadd(tmp, ei_pload(&(localB[k*l2blockRemainingRows+4*PacketSize])), dst[4]);
                if (l2blockRemainingRows>=6) dst[5] = ei_pmadd(tmp, ei_pload(&(localB[k*l2blockRemainingRows+5*PacketSize])), dst[5]);
                if (l2blockRemainingRows>=7) dst[6] = ei_pmadd(tmp, ei_pload(&(localB[k*l2blockRemainingRows+6*PacketSize])), dst[6]);
                if (l2blockRemainingRows>=8) dst[7] = ei_pmadd(tmp, ei_pload(&(localB[k*l2blockRemainingRows+7*PacketSize])), dst[7]);
              }
            }

            Scalar* __restrict__ localRes = &(res[l2blockRowEndBW + l1j*resStride]);

            // process the remaining rows once at a time
                                         localRes[0] += ei_predux(dst[0]);
            if (l2blockRemainingRows>=2) localRes[1] += ei_predux(dst[1]);
            if (l2blockRemainingRows>=3) localRes[2] += ei_predux(dst[2]);
            if (l2blockRemainingRows>=4) localRes[3] += ei_predux(dst[3]);
            if (MaxBlockRows>4)
            {
              if (l2blockRemainingRows>=5) localRes[4] += ei_predux(dst[4]);
              if (l2blockRemainingRows>=6) localRes[5] += ei_predux(dst[5]);
              if (l2blockRemainingRows>=7) localRes[6] += ei_predux(dst[6]);
              if (l2blockRemainingRows>=8) localRes[7] += ei_predux(dst[7]);
            }

            asm("#eigen end dynkernel");
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

  if (allocBlockSize>16000000)
    free(block);
}

#endif // EIGEN_EXTERN_INSTANTIATIONS

/* Optimized col-major matrix * vector product:
 * This algorithm processes 4 columns at onces that allows to both reduce
 * the number of load/stores of the result by a factor 4 and to reduce
 * the instruction dependency. Moreover, we know that all bands have the
 * same alignment pattern.
 * TODO: since rhs gets evaluated only once, no need to evaluate it
 */
template<typename Scalar, typename RhsType>
EIGEN_DONT_INLINE static void ei_cache_friendly_product_colmajor_times_vector(
  int size,
  const Scalar* lhs, int lhsStride,
  const RhsType& rhs,
  Scalar* res)
{
  #ifdef _EIGEN_ACCUMULATE_PACKETS
  #error _EIGEN_ACCUMULATE_PACKETS has already been defined
  #endif

  #define _EIGEN_ACCUMULATE_PACKETS(A0,A13,A2,OFFSET) \
    ei_pstore(&res[j OFFSET], \
      ei_padd(ei_pload(&res[j OFFSET]), \
        ei_padd( \
          ei_padd(ei_pmul(ptmp0,ei_pload ## A0(&lhs[j OFFSET +iN0])),ei_pmul(ptmp1,ei_pload ## A13(&lhs[j OFFSET +iN1]))), \
          ei_padd(ei_pmul(ptmp2,ei_pload ## A2(&lhs[j OFFSET +iN2])),ei_pmul(ptmp3,ei_pload ## A13(&lhs[j OFFSET +iN3]))) )))

  asm("#begin matrix_vector_product");
  typedef typename ei_packet_traits<Scalar>::type Packet;
  const int PacketSize = sizeof(Packet)/sizeof(Scalar);

  enum { AllAligned, EvenAligned, FirstAligned, NoneAligned };
  const int columnsAtOnce = 4;
  const int peels = 2;
  const int PacketAlignedMask = PacketSize-1;
  const int PeelAlignedMask = PacketSize*peels-1;
  const bool Vectorized = sizeof(Packet) != sizeof(Scalar);

  // How many coeffs of the result do we have to skip to be aligned.
  // Here we assume data are at least aligned on the base scalar type that is mandatory anyway.
  const int alignedStart = Vectorized
                         ? std::min<int>( (PacketSize - ((size_t(res)/sizeof(Scalar)) & PacketAlignedMask)) & PacketAlignedMask, size)
                         : 0;
  const int alignedSize = alignedStart + ((size-alignedStart) & ~PacketAlignedMask);
  const int peeledSize  = peels>1 ? alignedStart + ((alignedSize-alignedStart) & ~PeelAlignedMask) : 0;

  const int alignmentStep = lhsStride % PacketSize;
  int alignmentPattern = alignmentStep==0 ? AllAligned
                       : alignmentStep==2 ? EvenAligned
                       : FirstAligned;

  // find how many columns do we have to skip to be aligned with the result (if possible)
  int skipColumns=0;
  for (; skipColumns<PacketSize && alignedStart != alignmentStep*skipColumns; ++skipColumns)
  {}
  if (skipColumns==PacketSize)
  {
    // nothing can be aligned, no need to skip any column
    alignmentPattern = NoneAligned;
    skipColumns = 0;
  }
  else
  {
    skipColumns = std::min(skipColumns,rhs.size());
    // note that the skiped columns are processed later.
  }

  int columnBound = (rhs.size()/columnsAtOnce)*columnsAtOnce;
  for (int i=skipColumns; i<columnBound; i+=columnsAtOnce)
  {
    Scalar tmp0  = rhs[i], tmp1 = rhs[i+1], tmp2 = rhs[i+2], tmp3 = rhs[i+3];
    Packet ptmp0 = ei_pset1(tmp0), ptmp1 = ei_pset1(tmp1), ptmp2 = ei_pset1(tmp2), ptmp3 = ei_pset1(tmp3);
    int iN0 = i*lhsStride, iN1 = (i+1)*lhsStride, iN2 = (i+2)*lhsStride, iN3 = (i+3)*lhsStride;

    // process initial unaligned coeffs
    for (int j=0; j<alignedStart; j++)
      res[j] += tmp0 * lhs[j+iN0] + tmp1 * lhs[j+iN1] + tmp2 * lhs[j+iN2] + tmp3 * lhs[j+iN3];

    if (alignedSize>alignedStart)
    {
      switch(alignmentPattern)
      {
        case AllAligned:
          for (int j = alignedStart; j<alignedSize; j+=PacketSize)
            _EIGEN_ACCUMULATE_PACKETS(,,,);
          break;
        case EvenAligned:
          for (int j = alignedStart; j<alignedSize; j+=PacketSize)
            _EIGEN_ACCUMULATE_PACKETS(,u,,);
          break;
        case FirstAligned:
          if (peels>1)
            for (int j = alignedStart; j<peeledSize; j+=peels*PacketSize)
            {
              _EIGEN_ACCUMULATE_PACKETS(,u,u,);
              _EIGEN_ACCUMULATE_PACKETS(,u,u,+PacketSize);
              if (peels>2) _EIGEN_ACCUMULATE_PACKETS(,u,u,+2*PacketSize);
              if (peels>3) _EIGEN_ACCUMULATE_PACKETS(,u,u,+3*PacketSize);
            }
          for (int j = peeledSize; j<alignedSize; j+=PacketSize)
            _EIGEN_ACCUMULATE_PACKETS(,u,u,);
          break;
        default:
          for (int j = alignedStart; j<alignedSize; j+=PacketSize)
            _EIGEN_ACCUMULATE_PACKETS(u,u,u,);
          break;
      }
    }

    // process remaining coeffs
    for (int j=alignedSize; j<size; j++)
      res[j] += tmp0 * lhs[j+iN0] + tmp1 * lhs[j+iN1] + tmp2 * lhs[j+iN2] + tmp3 * lhs[j+iN3];
  }

  // process remaining first and last columns (at most columnsAtOnce-1)
  int end = rhs.size();
  int start = columnBound;
  do
  {
    for (int i=columnBound; i<end; i++)
    {
      Scalar tmp0 = rhs[i];
      Packet ptmp0 = ei_pset1(tmp0);
      int iN0 = i*lhsStride;
      // process first unaligned result's coeffs
      for (int j=0; j<alignedStart; j++)
        res[j] += tmp0 * lhs[j+iN0];

      // process aligned result's coeffs
      if ((iN0 % PacketSize) == 0)
        for (int j = alignedStart;j<alignedSize;j+=PacketSize)
          ei_pstore(&res[j], ei_padd(ei_pmul(ptmp0,ei_pload(&lhs[j+iN0])),ei_pload(&res[j])));
      else
        for (int j = alignedStart;j<alignedSize;j+=PacketSize)
          ei_pstore(&res[j], ei_padd(ei_pmul(ptmp0,ei_ploadu(&lhs[j+iN0])),ei_pload(&res[j])));

      // process remaining scalars
      for (int j=alignedSize; j<size; j++)
        res[j] += tmp0 * lhs[j+iN0];
    }
    if (skipColumns)
    {
      start = 0;
      end = skipColumns;
      skipColumns = 0;
    }
    else
      break;
  } while(true);
  asm("#end matrix_vector_product");

  #undef _EIGEN_ACCUMULATE_PACKETS
}

template<typename Scalar, typename ResType>
EIGEN_DONT_INLINE static void ei_cache_friendly_product_rowmajor_times_vector(
  const Scalar* lhs, int lhsStride,
  const Scalar* rhs, int rhsSize,
  ResType& res)
{
  #ifdef _EIGEN_ACCUMULATE_PACKETS
  #error _EIGEN_ACCUMULATE_PACKETS has already been defined
  #endif

  #define _EIGEN_ACCUMULATE_PACKETS(A0,A13,A2,OFFSET) {\
    Packet b = ei_pload(&rhs[j]); \
    ptmp0 = ei_padd(ptmp0, ei_pmul(b, ei_pload##A0 (&lhs[j+iN0]))); \
    ptmp1 = ei_padd(ptmp1, ei_pmul(b, ei_pload##A13(&lhs[j+iN1]))); \
    ptmp2 = ei_padd(ptmp2, ei_pmul(b, ei_pload##A2 (&lhs[j+iN2]))); \
    ptmp3 = ei_padd(ptmp3, ei_pmul(b, ei_pload##A13(&lhs[j+iN3]))); }

  asm("#begin matrix_vector_product");
  typedef typename ei_packet_traits<Scalar>::type Packet;
  const int PacketSize = sizeof(Packet)/sizeof(Scalar);

  enum { AllAligned, EvenAligned, FirstAligned, NoneAligned };
  const int rowsAtOnce = 4;
  const int peels = 2;
  const int PacketAlignedMask = PacketSize-1;
  const int PeelAlignedMask = PacketSize*peels-1;
  const bool Vectorized = sizeof(Packet) != sizeof(Scalar);
  const int size = rhsSize;

  // How many coeffs of the result do we have to skip to be aligned.
  // Here we assume data are at least aligned on the base scalar type that is mandatory anyway.
  const int alignedStart = Vectorized
                         ? std::min<int>( (PacketSize - ((size_t(rhs)/sizeof(Scalar)) & PacketAlignedMask)) & PacketAlignedMask, size)
                         : 0;
  const int alignedSize = alignedStart + ((size-alignedStart) & ~PacketAlignedMask);
  const int peeledSize  = peels>1 ? alignedStart + ((alignedSize-alignedStart) & ~PeelAlignedMask) : 0;

  const int alignmentStep = lhsStride % PacketSize;
  int alignmentPattern = alignmentStep==0 ? AllAligned
                       : alignmentStep==2 ? EvenAligned
                       : FirstAligned;

  // find how many rows do we have to skip to be aligned with rhs (if possible)
  int skipRows=0;
  for (; skipRows<PacketSize && alignedStart != alignmentStep*skipRows; ++skipRows)
  {}
  if (skipRows==PacketSize)
  {
    // nothing can be aligned, no need to skip any column
    alignmentPattern = NoneAligned;
    skipRows = 0;
  }
  else
  {
    skipRows = std::min(skipRows,res.size());
    // note that the skiped columns are processed later.
  }

  int rowBound = (res.size()/rowsAtOnce)*rowsAtOnce;
  for (int i=skipRows; i<rowBound; i+=rowsAtOnce)
  {
    Scalar tmp0 = Scalar(0), tmp1 = Scalar(0), tmp2 = Scalar(0), tmp3 = Scalar(0);
    Packet ptmp0 = ei_pset1(Scalar(0)), ptmp1 = ei_pset1(Scalar(0)), ptmp2 = ei_pset1(Scalar(0)), ptmp3 = ei_pset1(Scalar(0));
    int iN0 = i*lhsStride, iN1 = (i+1)*lhsStride, iN2 = (i+2)*lhsStride, iN3 = (i+3)*lhsStride;

    // process initial unaligned coeffs
    for (int j=0; j<alignedStart; j++)
    {
      Scalar b = rhs[j];
      tmp0 += b * lhs[j+iN0]; tmp1 += b * lhs[j+iN1]; tmp2 += b * lhs[j+iN2]; tmp3 += b * lhs[j+iN3];
    }

    if (alignedSize>alignedStart)
    {
      switch(alignmentPattern)
      {
        case AllAligned:
          for (int j = alignedStart; j<alignedSize; j+=PacketSize)
            _EIGEN_ACCUMULATE_PACKETS(,,,);
          break;
        case EvenAligned:
          for (int j = alignedStart; j<alignedSize; j+=PacketSize)
            _EIGEN_ACCUMULATE_PACKETS(,u,,);
          break;
        case FirstAligned:
          for (int j = alignedStart; j<alignedSize; j+=PacketSize)
            _EIGEN_ACCUMULATE_PACKETS(,u,u,);
          break;
        default:
          for (int j = alignedStart; j<alignedSize; j+=PacketSize)
            _EIGEN_ACCUMULATE_PACKETS(u,u,u,);
          break;
      }
      tmp0 += ei_predux(ptmp0);
      tmp1 += ei_predux(ptmp1);
      tmp2 += ei_predux(ptmp2);
      tmp3 += ei_predux(ptmp3);
    }

    // process remaining coeffs
    for (int j=alignedSize; j<size; j++)
    {
      Scalar b = rhs[j];
      tmp0 += b * lhs[j+iN0]; tmp1 += b * lhs[j+iN1]; tmp2 += b * lhs[j+iN2]; tmp3 += b * lhs[j+iN3];
    }
    res[i] = tmp0; res[i+1] = tmp1; res[i+2] = tmp2; res[i+3] = tmp3;
  }

  // process remaining first and last rows (at most columnsAtOnce-1)
  int end = res.size();
  int start = rowBound;
  do
  {
    for (int i=rowBound; i<end; i++)
    {
      Scalar tmp0 = Scalar(0);
      Packet ptmp0 = ei_pset1(tmp0);
      int iN0 = i*lhsStride;
      // process first unaligned result's coeffs
      for (int j=0; j<alignedStart; j++)
        tmp0 += rhs[j] * lhs[j+iN0];

      if (alignedSize>alignedStart)
      {
        // process aligned rhs coeffs
        if (iN0 % PacketSize==0)
          for (int j = alignedStart;j<alignedSize;j+=PacketSize)
            ptmp0 = ei_padd(ptmp0, ei_pmul(ei_pload(&rhs[j]), ei_pload(&lhs[j+iN0])));
        else
          for (int j = alignedStart;j<alignedSize;j+=PacketSize)
            ptmp0 = ei_padd(ptmp0, ei_pmul(ei_pload(&rhs[j]), ei_ploadu(&lhs[j+iN0])));
        tmp0 += ei_predux(ptmp0);
      }

      // process remaining scalars
      for (int j=alignedSize; j<size; j++)
        tmp0 += rhs[j] * lhs[j+iN0];
      res[i] += tmp0;
    }
    if (skipRows)
    {
      start = 0;
      end = skipRows;
      skipRows = 0;
    }
    else
      break;
  } while(true);
  asm("#end matrix_vector_product");

  #undef _EIGEN_ACCUMULATE_PACKETS
}

#endif // EIGEN_CACHE_FRIENDLY_PRODUCT_H
