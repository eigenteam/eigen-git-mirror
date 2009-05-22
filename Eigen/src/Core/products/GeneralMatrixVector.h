// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
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

#ifndef EIGEN_GENERAL_MATRIX_VECTOR_H
#define EIGEN_GENERAL_MATRIX_VECTOR_H

/* Optimized col-major matrix * vector product:
 * This algorithm processes 4 columns at onces that allows to both reduce
 * the number of load/stores of the result by a factor 4 and to reduce
 * the instruction dependency. Moreover, we know that all bands have the
 * same alignment pattern.
 * TODO: since rhs gets evaluated only once, no need to evaluate it
 */
template<typename Scalar, typename RhsType>
static EIGEN_DONT_INLINE void ei_cache_friendly_product_colmajor_times_vector(
  int size,
  const Scalar* lhs, int lhsStride,
  const RhsType& rhs,
  Scalar* res)
{
  #ifdef _EIGEN_ACCUMULATE_PACKETS
  #error _EIGEN_ACCUMULATE_PACKETS has already been defined
  #endif
  #define _EIGEN_ACCUMULATE_PACKETS(A0,A13,A2) \
    ei_pstore(&res[j], \
      ei_padd(ei_pload(&res[j]), \
        ei_padd( \
          ei_padd(ei_pmul(ptmp0,EIGEN_CAT(ei_ploa , A0)(&lhs0[j])), \
                  ei_pmul(ptmp1,EIGEN_CAT(ei_ploa , A13)(&lhs1[j]))), \
          ei_padd(ei_pmul(ptmp2,EIGEN_CAT(ei_ploa , A2)(&lhs2[j])), \
                  ei_pmul(ptmp3,EIGEN_CAT(ei_ploa , A13)(&lhs3[j]))) )))

  typedef typename ei_packet_traits<Scalar>::type Packet;
  const int PacketSize = sizeof(Packet)/sizeof(Scalar);

  enum { AllAligned = 0, EvenAligned, FirstAligned, NoneAligned };
  const int columnsAtOnce = 4;
  const int peels = 2;
  const int PacketAlignedMask = PacketSize-1;
  const int PeelAlignedMask = PacketSize*peels-1;

  // How many coeffs of the result do we have to skip to be aligned.
  // Here we assume data are at least aligned on the base scalar type that is mandatory anyway.
  const int alignedStart = ei_alignmentOffset(res,size);
  const int alignedSize = PacketSize>1 ? alignedStart + ((size-alignedStart) & ~PacketAlignedMask) : 0;
  const int peeledSize  = peels>1 ? alignedStart + ((alignedSize-alignedStart) & ~PeelAlignedMask) : alignedStart;

  const int alignmentStep = PacketSize>1 ? (PacketSize - lhsStride % PacketSize) & PacketAlignedMask : 0;
  int alignmentPattern = alignmentStep==0 ? AllAligned
                       : alignmentStep==(PacketSize/2) ? EvenAligned
                       : FirstAligned;

  // we cannot assume the first element is aligned because of sub-matrices
  const int lhsAlignmentOffset = ei_alignmentOffset(lhs,size);

  // find how many columns do we have to skip to be aligned with the result (if possible)
  int skipColumns = 0;
  if (PacketSize>1)
  {
    ei_internal_assert(size_t(lhs+lhsAlignmentOffset)%sizeof(Packet)==0 || size<PacketSize);

    while (skipColumns<PacketSize &&
           alignedStart != ((lhsAlignmentOffset + alignmentStep*skipColumns)%PacketSize))
      ++skipColumns;
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

    ei_internal_assert((alignmentPattern==NoneAligned) || (size_t(lhs+alignedStart+lhsStride*skipColumns)%sizeof(Packet))==0);
  }

  int offset1 = (FirstAligned && alignmentStep==1?3:1);
  int offset3 = (FirstAligned && alignmentStep==1?1:3);

  int columnBound = ((rhs.size()-skipColumns)/columnsAtOnce)*columnsAtOnce + skipColumns;
  for (int i=skipColumns; i<columnBound; i+=columnsAtOnce)
  {
    Packet ptmp0 = ei_pset1(rhs[i]),   ptmp1 = ei_pset1(rhs[i+offset1]),
           ptmp2 = ei_pset1(rhs[i+2]), ptmp3 = ei_pset1(rhs[i+offset3]);

    // this helps a lot generating better binary code
    const Scalar *lhs0 = lhs + i*lhsStride, *lhs1 = lhs + (i+offset1)*lhsStride,
                 *lhs2 = lhs + (i+2)*lhsStride, *lhs3 = lhs + (i+offset3)*lhsStride;

    if (PacketSize>1)
    {
      /* explicit vectorization */
      // process initial unaligned coeffs
      for (int j=0; j<alignedStart; ++j)
        res[j] += ei_pfirst(ptmp0)*lhs0[j] + ei_pfirst(ptmp1)*lhs1[j] + ei_pfirst(ptmp2)*lhs2[j] + ei_pfirst(ptmp3)*lhs3[j];

      if (alignedSize>alignedStart)
      {
        switch(alignmentPattern)
        {
          case AllAligned:
            for (int j = alignedStart; j<alignedSize; j+=PacketSize)
              _EIGEN_ACCUMULATE_PACKETS(d,d,d);
            break;
          case EvenAligned:
            for (int j = alignedStart; j<alignedSize; j+=PacketSize)
              _EIGEN_ACCUMULATE_PACKETS(d,du,d);
            break;
          case FirstAligned:
            if(peels>1)
            {
              Packet A00, A01, A02, A03, A10, A11, A12, A13;

              A01 = ei_pload(&lhs1[alignedStart-1]);
              A02 = ei_pload(&lhs2[alignedStart-2]);
              A03 = ei_pload(&lhs3[alignedStart-3]);

              for (int j = alignedStart; j<peeledSize; j+=peels*PacketSize)
              {
                A11 = ei_pload(&lhs1[j-1+PacketSize]);  ei_palign<1>(A01,A11);
                A12 = ei_pload(&lhs2[j-2+PacketSize]);  ei_palign<2>(A02,A12);
                A13 = ei_pload(&lhs3[j-3+PacketSize]);  ei_palign<3>(A03,A13);

                A00 = ei_pload (&lhs0[j]);
                A10 = ei_pload (&lhs0[j+PacketSize]);
                A00 = ei_pmadd(ptmp0, A00, ei_pload(&res[j]));
                A10 = ei_pmadd(ptmp0, A10, ei_pload(&res[j+PacketSize]));

                A00 = ei_pmadd(ptmp1, A01, A00);
                A01 = ei_pload(&lhs1[j-1+2*PacketSize]);  ei_palign<1>(A11,A01);
                A00 = ei_pmadd(ptmp2, A02, A00);
                A02 = ei_pload(&lhs2[j-2+2*PacketSize]);  ei_palign<2>(A12,A02);
                A00 = ei_pmadd(ptmp3, A03, A00);
                ei_pstore(&res[j],A00);
                A03 = ei_pload(&lhs3[j-3+2*PacketSize]);  ei_palign<3>(A13,A03);
                A10 = ei_pmadd(ptmp1, A11, A10);
                A10 = ei_pmadd(ptmp2, A12, A10);
                A10 = ei_pmadd(ptmp3, A13, A10);
                ei_pstore(&res[j+PacketSize],A10);
              }
            }
            for (int j = peeledSize; j<alignedSize; j+=PacketSize)
              _EIGEN_ACCUMULATE_PACKETS(d,du,du);
            break;
          default:
            for (int j = alignedStart; j<alignedSize; j+=PacketSize)
              _EIGEN_ACCUMULATE_PACKETS(du,du,du);
            break;
        }
      }
    } // end explicit vectorization

    /* process remaining coeffs (or all if there is no explicit vectorization) */
    for (int j=alignedSize; j<size; ++j)
	  res[j] += ei_pfirst(ptmp0)*lhs0[j] + ei_pfirst(ptmp1)*lhs1[j] + ei_pfirst(ptmp2)*lhs2[j] + ei_pfirst(ptmp3)*lhs3[j];
  }

  // process remaining first and last columns (at most columnsAtOnce-1)
  int end = rhs.size();
  int start = columnBound;
  do
  {
    for (int i=start; i<end; ++i)
    {
      Packet ptmp0 = ei_pset1(rhs[i]);
      const Scalar* lhs0 = lhs + i*lhsStride;

      if (PacketSize>1)
      {
        /* explicit vectorization */
        // process first unaligned result's coeffs
        for (int j=0; j<alignedStart; ++j)
          res[j] += ei_pfirst(ptmp0) * lhs0[j];

        // process aligned result's coeffs
        if ((size_t(lhs0+alignedStart)%sizeof(Packet))==0)
          for (int j = alignedStart;j<alignedSize;j+=PacketSize)
            ei_pstore(&res[j], ei_pmadd(ptmp0,ei_pload(&lhs0[j]),ei_pload(&res[j])));
        else
          for (int j = alignedStart;j<alignedSize;j+=PacketSize)
            ei_pstore(&res[j], ei_pmadd(ptmp0,ei_ploadu(&lhs0[j]),ei_pload(&res[j])));
      }

      // process remaining scalars (or all if no explicit vectorization)
      for (int j=alignedSize; j<size; ++j)
        res[j] += ei_pfirst(ptmp0) * lhs0[j];
    }
    if (skipColumns)
    {
      start = 0;
      end = skipColumns;
      skipColumns = 0;
    }
    else
      break;
  } while(PacketSize>1);
  #undef _EIGEN_ACCUMULATE_PACKETS
}

// TODO add peeling to mask unaligned load/stores
template<typename Scalar, typename ResType>
static EIGEN_DONT_INLINE void ei_cache_friendly_product_rowmajor_times_vector(
  const Scalar* lhs, int lhsStride,
  const Scalar* rhs, int rhsSize,
  ResType& res)
{
  #ifdef _EIGEN_ACCUMULATE_PACKETS
  #error _EIGEN_ACCUMULATE_PACKETS has already been defined
  #endif

  #define _EIGEN_ACCUMULATE_PACKETS(A0,A13,A2) {\
    Packet b = ei_pload(&rhs[j]); \
    ptmp0 = ei_pmadd(b, EIGEN_CAT(ei_ploa,A0) (&lhs0[j]), ptmp0); \
    ptmp1 = ei_pmadd(b, EIGEN_CAT(ei_ploa,A13)(&lhs1[j]), ptmp1); \
    ptmp2 = ei_pmadd(b, EIGEN_CAT(ei_ploa,A2) (&lhs2[j]), ptmp2); \
    ptmp3 = ei_pmadd(b, EIGEN_CAT(ei_ploa,A13)(&lhs3[j]), ptmp3); }

  typedef typename ei_packet_traits<Scalar>::type Packet;
  const int PacketSize = sizeof(Packet)/sizeof(Scalar);

  enum { AllAligned=0, EvenAligned=1, FirstAligned=2, NoneAligned=3 };
  const int rowsAtOnce = 4;
  const int peels = 2;
  const int PacketAlignedMask = PacketSize-1;
  const int PeelAlignedMask = PacketSize*peels-1;
  const int size = rhsSize;

  // How many coeffs of the result do we have to skip to be aligned.
  // Here we assume data are at least aligned on the base scalar type that is mandatory anyway.
  const int alignedStart = ei_alignmentOffset(rhs, size);
  const int alignedSize = PacketSize>1 ? alignedStart + ((size-alignedStart) & ~PacketAlignedMask) : 0;
  const int peeledSize  = peels>1 ? alignedStart + ((alignedSize-alignedStart) & ~PeelAlignedMask) : alignedStart;

  const int alignmentStep = PacketSize>1 ? (PacketSize - lhsStride % PacketSize) & PacketAlignedMask : 0;
  int alignmentPattern = alignmentStep==0 ? AllAligned
                       : alignmentStep==(PacketSize/2) ? EvenAligned
                       : FirstAligned;

  // we cannot assume the first element is aligned because of sub-matrices
  const int lhsAlignmentOffset = ei_alignmentOffset(lhs,size);

  // find how many rows do we have to skip to be aligned with rhs (if possible)
  int skipRows = 0;
  if (PacketSize>1)
  {
    ei_internal_assert(size_t(lhs+lhsAlignmentOffset)%sizeof(Packet)==0  || size<PacketSize);

    while (skipRows<PacketSize &&
           alignedStart != ((lhsAlignmentOffset + alignmentStep*skipRows)%PacketSize))
      ++skipRows;
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
    ei_internal_assert((alignmentPattern==NoneAligned) || PacketSize==1
      || (size_t(lhs+alignedStart+lhsStride*skipRows)%sizeof(Packet))==0);
  }

  int offset1 = (FirstAligned && alignmentStep==1?3:1);
  int offset3 = (FirstAligned && alignmentStep==1?1:3);

  int rowBound = ((res.size()-skipRows)/rowsAtOnce)*rowsAtOnce + skipRows;
  for (int i=skipRows; i<rowBound; i+=rowsAtOnce)
  {
    Scalar tmp0 = Scalar(0), tmp1 = Scalar(0), tmp2 = Scalar(0), tmp3 = Scalar(0);

    // this helps the compiler generating good binary code
    const Scalar *lhs0 = lhs + i*lhsStride,     *lhs1 = lhs + (i+offset1)*lhsStride,
                 *lhs2 = lhs + (i+2)*lhsStride, *lhs3 = lhs + (i+offset3)*lhsStride;

    if (PacketSize>1)
    {
      /* explicit vectorization */
      Packet ptmp0 = ei_pset1(Scalar(0)), ptmp1 = ei_pset1(Scalar(0)), ptmp2 = ei_pset1(Scalar(0)), ptmp3 = ei_pset1(Scalar(0));

      // process initial unaligned coeffs
      // FIXME this loop get vectorized by the compiler !
      for (int j=0; j<alignedStart; ++j)
      {
        Scalar b = rhs[j];
        tmp0 += b*lhs0[j]; tmp1 += b*lhs1[j]; tmp2 += b*lhs2[j]; tmp3 += b*lhs3[j];
      }

      if (alignedSize>alignedStart)
      {
        switch(alignmentPattern)
        {
          case AllAligned:
            for (int j = alignedStart; j<alignedSize; j+=PacketSize)
              _EIGEN_ACCUMULATE_PACKETS(d,d,d);
            break;
          case EvenAligned:
            for (int j = alignedStart; j<alignedSize; j+=PacketSize)
              _EIGEN_ACCUMULATE_PACKETS(d,du,d);
            break;
          case FirstAligned:
            if (peels>1)
            {
              /* Here we proccess 4 rows with with two peeled iterations to hide
               * tghe overhead of unaligned loads. Moreover unaligned loads are handled
               * using special shift/move operations between the two aligned packets
               * overlaping the desired unaligned packet. This is *much* more efficient
               * than basic unaligned loads.
               */
              Packet A01, A02, A03, b, A11, A12, A13;
              A01 = ei_pload(&lhs1[alignedStart-1]);
              A02 = ei_pload(&lhs2[alignedStart-2]);
              A03 = ei_pload(&lhs3[alignedStart-3]);

              for (int j = alignedStart; j<peeledSize; j+=peels*PacketSize)
              {
                b = ei_pload(&rhs[j]);
                A11 = ei_pload(&lhs1[j-1+PacketSize]);  ei_palign<1>(A01,A11);
                A12 = ei_pload(&lhs2[j-2+PacketSize]);  ei_palign<2>(A02,A12);
                A13 = ei_pload(&lhs3[j-3+PacketSize]);  ei_palign<3>(A03,A13);

                ptmp0 = ei_pmadd(b, ei_pload (&lhs0[j]), ptmp0);
                ptmp1 = ei_pmadd(b, A01, ptmp1);
                A01 = ei_pload(&lhs1[j-1+2*PacketSize]);  ei_palign<1>(A11,A01);
                ptmp2 = ei_pmadd(b, A02, ptmp2);
                A02 = ei_pload(&lhs2[j-2+2*PacketSize]);  ei_palign<2>(A12,A02);
                ptmp3 = ei_pmadd(b, A03, ptmp3);
                A03 = ei_pload(&lhs3[j-3+2*PacketSize]);  ei_palign<3>(A13,A03);

                b = ei_pload(&rhs[j+PacketSize]);
                ptmp0 = ei_pmadd(b, ei_pload (&lhs0[j+PacketSize]), ptmp0);
                ptmp1 = ei_pmadd(b, A11, ptmp1);
                ptmp2 = ei_pmadd(b, A12, ptmp2);
                ptmp3 = ei_pmadd(b, A13, ptmp3);
              }
            }
            for (int j = peeledSize; j<alignedSize; j+=PacketSize)
              _EIGEN_ACCUMULATE_PACKETS(d,du,du);
            break;
          default:
            for (int j = alignedStart; j<alignedSize; j+=PacketSize)
              _EIGEN_ACCUMULATE_PACKETS(du,du,du);
            break;
        }
        tmp0 += ei_predux(ptmp0);
        tmp1 += ei_predux(ptmp1);
        tmp2 += ei_predux(ptmp2);
        tmp3 += ei_predux(ptmp3);
      }
    } // end explicit vectorization

    // process remaining coeffs (or all if no explicit vectorization)
    // FIXME this loop get vectorized by the compiler !
    for (int j=alignedSize; j<size; ++j)
    {
      Scalar b = rhs[j];
      tmp0 += b*lhs0[j]; tmp1 += b*lhs1[j]; tmp2 += b*lhs2[j]; tmp3 += b*lhs3[j];
    }
    res[i] += tmp0; res[i+offset1] += tmp1; res[i+2] += tmp2; res[i+offset3] += tmp3;
  }

  // process remaining first and last rows (at most columnsAtOnce-1)
  int end = res.size();
  int start = rowBound;
  do
  {
    for (int i=start; i<end; ++i)
    {
      Scalar tmp0 = Scalar(0);
      Packet ptmp0 = ei_pset1(tmp0);
      const Scalar* lhs0 = lhs + i*lhsStride;
      // process first unaligned result's coeffs
      // FIXME this loop get vectorized by the compiler !
      for (int j=0; j<alignedStart; ++j)
        tmp0 += rhs[j] * lhs0[j];

      if (alignedSize>alignedStart)
      {
        // process aligned rhs coeffs
        if ((size_t(lhs0+alignedStart)%sizeof(Packet))==0)
          for (int j = alignedStart;j<alignedSize;j+=PacketSize)
            ptmp0 = ei_pmadd(ei_pload(&rhs[j]), ei_pload(&lhs0[j]), ptmp0);
        else
          for (int j = alignedStart;j<alignedSize;j+=PacketSize)
            ptmp0 = ei_pmadd(ei_pload(&rhs[j]), ei_ploadu(&lhs0[j]), ptmp0);
        tmp0 += ei_predux(ptmp0);
      }

      // process remaining scalars
      // FIXME this loop get vectorized by the compiler !
      for (int j=alignedSize; j<size; ++j)
        tmp0 += rhs[j] * lhs0[j];
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
  } while(PacketSize>1);

  #undef _EIGEN_ACCUMULATE_PACKETS
}

#endif // EIGEN_GENERAL_MATRIX_VECTOR_H
