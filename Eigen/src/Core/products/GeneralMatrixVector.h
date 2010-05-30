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

#ifndef EIGEN_GENERAL_MATRIX_VECTOR_H
#define EIGEN_GENERAL_MATRIX_VECTOR_H

/* Optimized col-major matrix * vector product:
 * This algorithm processes 4 columns at onces that allows to both reduce
 * the number of load/stores of the result by a factor 4 and to reduce
 * the instruction dependency. Moreover, we know that all bands have the
 * same alignment pattern.
 * TODO: since rhs gets evaluated only once, no need to evaluate it
 */
template<bool ConjugateLhs, bool ConjugateRhs, typename Scalar, typename Index, typename RhsType>
static EIGEN_DONT_INLINE
void ei_cache_friendly_product_colmajor_times_vector(
  Index size,
  const Scalar* lhs, Index lhsStride,
  const RhsType& rhs,
  Scalar* res,
  Scalar alpha)
{
  #ifdef _EIGEN_ACCUMULATE_PACKETS
  #error _EIGEN_ACCUMULATE_PACKETS has already been defined
  #endif
  #define _EIGEN_ACCUMULATE_PACKETS(A0,A13,A2) \
    ei_pstore(&res[j], \
      ei_padd(ei_pload(&res[j]), \
        ei_padd( \
          ei_padd(cj.pmul(EIGEN_CAT(ei_ploa , A0)(&lhs0[j]),    ptmp0), \
                  cj.pmul(EIGEN_CAT(ei_ploa , A13)(&lhs1[j]),   ptmp1)), \
          ei_padd(cj.pmul(EIGEN_CAT(ei_ploa , A2)(&lhs2[j]),    ptmp2), \
                  cj.pmul(EIGEN_CAT(ei_ploa , A13)(&lhs3[j]),   ptmp3)) )))

  ei_conj_helper<ConjugateLhs,ConjugateRhs> cj;
  if(ConjugateRhs)
    alpha = ei_conj(alpha);

  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef typename ei_packet_traits<Scalar>::type Packet;
  const Index PacketSize = sizeof(Packet)/sizeof(Scalar);

  enum { AllAligned = 0, EvenAligned, FirstAligned, NoneAligned };
  const Index columnsAtOnce = 4;
  const Index peels = 2;
  const Index PacketAlignedMask = PacketSize-1;
  const Index PeelAlignedMask = PacketSize*peels-1;

  // How many coeffs of the result do we have to skip to be aligned.
  // Here we assume data are at least aligned on the base scalar type.
  Index alignedStart = ei_first_aligned(res,size);
  Index alignedSize = PacketSize>1 ? alignedStart + ((size-alignedStart) & ~PacketAlignedMask) : 0;
  const Index peeledSize  = peels>1 ? alignedStart + ((alignedSize-alignedStart) & ~PeelAlignedMask) : alignedStart;

  const Index alignmentStep = PacketSize>1 ? (PacketSize - lhsStride % PacketSize) & PacketAlignedMask : 0;
  Index alignmentPattern = alignmentStep==0 ? AllAligned
                       : alignmentStep==(PacketSize/2) ? EvenAligned
                       : FirstAligned;

  // we cannot assume the first element is aligned because of sub-matrices
  const Index lhsAlignmentOffset = ei_first_aligned(lhs,size);

  // find how many columns do we have to skip to be aligned with the result (if possible)
  Index skipColumns = 0;
  // if the data cannot be aligned (TODO add some compile time tests when possible, e.g. for floats)
  if( (size_t(lhs)%sizeof(RealScalar)) || (size_t(res)%sizeof(RealScalar)) )
  {
    alignedSize = 0;
    alignedStart = 0;
  }
  else if (PacketSize>1)
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

    ei_internal_assert(  (alignmentPattern==NoneAligned)
                      || (skipColumns + columnsAtOnce >= rhs.size())
                      || PacketSize > size
                      || (size_t(lhs+alignedStart+lhsStride*skipColumns)%sizeof(Packet))==0);
  }

  Index offset1 = (FirstAligned && alignmentStep==1?3:1);
  Index offset3 = (FirstAligned && alignmentStep==1?1:3);

  Index columnBound = ((rhs.size()-skipColumns)/columnsAtOnce)*columnsAtOnce + skipColumns;
  for (Index i=skipColumns; i<columnBound; i+=columnsAtOnce)
  {
    Packet ptmp0 = ei_pset1(alpha*rhs[i]),   ptmp1 = ei_pset1(alpha*rhs[i+offset1]),
           ptmp2 = ei_pset1(alpha*rhs[i+2]), ptmp3 = ei_pset1(alpha*rhs[i+offset3]);

    // this helps a lot generating better binary code
    const Scalar *lhs0 = lhs + i*lhsStride,     *lhs1 = lhs + (i+offset1)*lhsStride,
                 *lhs2 = lhs + (i+2)*lhsStride, *lhs3 = lhs + (i+offset3)*lhsStride;

    if (PacketSize>1)
    {
      /* explicit vectorization */
      // process initial unaligned coeffs
      for (Index j=0; j<alignedStart; ++j)
      {
        res[j] = cj.pmadd(lhs0[j], ei_pfirst(ptmp0), res[j]);
        res[j] = cj.pmadd(lhs1[j], ei_pfirst(ptmp1), res[j]);
        res[j] = cj.pmadd(lhs2[j], ei_pfirst(ptmp2), res[j]);
        res[j] = cj.pmadd(lhs3[j], ei_pfirst(ptmp3), res[j]);
      }

      if (alignedSize>alignedStart)
      {
        switch(alignmentPattern)
        {
          case AllAligned:
            for (Index j = alignedStart; j<alignedSize; j+=PacketSize)
              _EIGEN_ACCUMULATE_PACKETS(d,d,d);
            break;
          case EvenAligned:
            for (Index j = alignedStart; j<alignedSize; j+=PacketSize)
              _EIGEN_ACCUMULATE_PACKETS(d,du,d);
            break;
          case FirstAligned:
            if(peels>1)
            {
              Packet A00, A01, A02, A03, A10, A11, A12, A13;

              A01 = ei_pload(&lhs1[alignedStart-1]);
              A02 = ei_pload(&lhs2[alignedStart-2]);
              A03 = ei_pload(&lhs3[alignedStart-3]);

              for (Index j = alignedStart; j<peeledSize; j+=peels*PacketSize)
              {
                A11 = ei_pload(&lhs1[j-1+PacketSize]);  ei_palign<1>(A01,A11);
                A12 = ei_pload(&lhs2[j-2+PacketSize]);  ei_palign<2>(A02,A12);
                A13 = ei_pload(&lhs3[j-3+PacketSize]);  ei_palign<3>(A03,A13);

                A00 = ei_pload (&lhs0[j]);
                A10 = ei_pload (&lhs0[j+PacketSize]);
                A00 = cj.pmadd(A00, ptmp0, ei_pload(&res[j]));
                A10 = cj.pmadd(A10, ptmp0, ei_pload(&res[j+PacketSize]));

                A00 = cj.pmadd(A01, ptmp1, A00);
                A01 = ei_pload(&lhs1[j-1+2*PacketSize]);  ei_palign<1>(A11,A01);
                A00 = cj.pmadd(A02, ptmp2, A00);
                A02 = ei_pload(&lhs2[j-2+2*PacketSize]);  ei_palign<2>(A12,A02);
                A00 = cj.pmadd(A03, ptmp3, A00);
                ei_pstore(&res[j],A00);
                A03 = ei_pload(&lhs3[j-3+2*PacketSize]);  ei_palign<3>(A13,A03);
                A10 = cj.pmadd(A11, ptmp1, A10);
                A10 = cj.pmadd(A12, ptmp2, A10);
                A10 = cj.pmadd(A13, ptmp3, A10);
                ei_pstore(&res[j+PacketSize],A10);
              }
            }
            for (Index j = peeledSize; j<alignedSize; j+=PacketSize)
              _EIGEN_ACCUMULATE_PACKETS(d,du,du);
            break;
          default:
            for (Index j = alignedStart; j<alignedSize; j+=PacketSize)
              _EIGEN_ACCUMULATE_PACKETS(du,du,du);
            break;
        }
      }
    } // end explicit vectorization

    /* process remaining coeffs (or all if there is no explicit vectorization) */
    for (Index j=alignedSize; j<size; ++j)
    {
      res[j] = cj.pmadd(lhs0[j], ei_pfirst(ptmp0), res[j]);
      res[j] = cj.pmadd(lhs1[j], ei_pfirst(ptmp1), res[j]);
      res[j] = cj.pmadd(lhs2[j], ei_pfirst(ptmp2), res[j]);
      res[j] = cj.pmadd(lhs3[j], ei_pfirst(ptmp3), res[j]);
    }
  }

  // process remaining first and last columns (at most columnsAtOnce-1)
  Index end = rhs.size();
  Index start = columnBound;
  do
  {
    for (Index i=start; i<end; ++i)
    {
      Packet ptmp0 = ei_pset1(alpha*rhs[i]);
      const Scalar* lhs0 = lhs + i*lhsStride;

      if (PacketSize>1)
      {
        /* explicit vectorization */
        // process first unaligned result's coeffs
        for (Index j=0; j<alignedStart; ++j)
          res[j] += cj.pmul(lhs0[j], ei_pfirst(ptmp0));

        // process aligned result's coeffs
        if ((size_t(lhs0+alignedStart)%sizeof(Packet))==0)
          for (Index j = alignedStart;j<alignedSize;j+=PacketSize)
            ei_pstore(&res[j], cj.pmadd(ei_pload(&lhs0[j]), ptmp0, ei_pload(&res[j])));
        else
          for (Index j = alignedStart;j<alignedSize;j+=PacketSize)
            ei_pstore(&res[j], cj.pmadd(ei_ploadu(&lhs0[j]), ptmp0, ei_pload(&res[j])));
      }

      // process remaining scalars (or all if no explicit vectorization)
      for (Index j=alignedSize; j<size; ++j)
        res[j] += cj.pmul(lhs0[j], ei_pfirst(ptmp0));
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
template<bool ConjugateLhs, bool ConjugateRhs, typename Scalar, typename Index, typename ResType>
static EIGEN_DONT_INLINE void ei_cache_friendly_product_rowmajor_times_vector(
  const Scalar* lhs, Index lhsStride,
  const Scalar* rhs, Index rhsSize,
  ResType& res,
  Scalar alpha)
{
  #ifdef _EIGEN_ACCUMULATE_PACKETS
  #error _EIGEN_ACCUMULATE_PACKETS has already been defined
  #endif

  #define _EIGEN_ACCUMULATE_PACKETS(A0,A13,A2) {\
    Packet b = ei_pload(&rhs[j]); \
    ptmp0 = cj.pmadd(EIGEN_CAT(ei_ploa,A0) (&lhs0[j]), b, ptmp0); \
    ptmp1 = cj.pmadd(EIGEN_CAT(ei_ploa,A13)(&lhs1[j]), b, ptmp1); \
    ptmp2 = cj.pmadd(EIGEN_CAT(ei_ploa,A2) (&lhs2[j]), b, ptmp2); \
    ptmp3 = cj.pmadd(EIGEN_CAT(ei_ploa,A13)(&lhs3[j]), b, ptmp3); }

  ei_conj_helper<ConjugateLhs,ConjugateRhs> cj;

  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef typename ei_packet_traits<Scalar>::type Packet;
  const Index PacketSize = sizeof(Packet)/sizeof(Scalar);

  enum { AllAligned=0, EvenAligned=1, FirstAligned=2, NoneAligned=3 };
  const Index rowsAtOnce = 4;
  const Index peels = 2;
  const Index PacketAlignedMask = PacketSize-1;
  const Index PeelAlignedMask = PacketSize*peels-1;
  const Index size = rhsSize;

  // How many coeffs of the result do we have to skip to be aligned.
  // Here we assume data are at least aligned on the base scalar type
  // if that's not the case then vectorization is discarded, see below.
  Index alignedStart = ei_first_aligned(rhs, size);
  Index alignedSize = PacketSize>1 ? alignedStart + ((size-alignedStart) & ~PacketAlignedMask) : 0;
  const Index peeledSize  = peels>1 ? alignedStart + ((alignedSize-alignedStart) & ~PeelAlignedMask) : alignedStart;

  const Index alignmentStep = PacketSize>1 ? (PacketSize - lhsStride % PacketSize) & PacketAlignedMask : 0;
  Index alignmentPattern = alignmentStep==0 ? AllAligned
                       : alignmentStep==(PacketSize/2) ? EvenAligned
                       : FirstAligned;

  // we cannot assume the first element is aligned because of sub-matrices
  const Index lhsAlignmentOffset = ei_first_aligned(lhs,size);

  // find how many rows do we have to skip to be aligned with rhs (if possible)
  Index skipRows = 0;
  // if the data cannot be aligned (TODO add some compile time tests when possible, e.g. for floats)
  if( (size_t(lhs)%sizeof(RealScalar)) || (size_t(rhs)%sizeof(RealScalar)) )
  {
    alignedSize = 0;
    alignedStart = 0;
  }
  else if (PacketSize>1)
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
      skipRows = std::min(skipRows,Index(res.size()));
      // note that the skiped columns are processed later.
    }
    ei_internal_assert(  alignmentPattern==NoneAligned
                      || PacketSize==1
                      || (skipRows + rowsAtOnce >= res.size())
                      || PacketSize > rhsSize
                      || (size_t(lhs+alignedStart+lhsStride*skipRows)%sizeof(Packet))==0);
  }

  Index offset1 = (FirstAligned && alignmentStep==1?3:1);
  Index offset3 = (FirstAligned && alignmentStep==1?1:3);

  Index rowBound = ((res.size()-skipRows)/rowsAtOnce)*rowsAtOnce + skipRows;
  for (Index i=skipRows; i<rowBound; i+=rowsAtOnce)
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
      for (Index j=0; j<alignedStart; ++j)
      {
        Scalar b = rhs[j];
        tmp0 += cj.pmul(lhs0[j],b); tmp1 += cj.pmul(lhs1[j],b);
        tmp2 += cj.pmul(lhs2[j],b); tmp3 += cj.pmul(lhs3[j],b);
      }

      if (alignedSize>alignedStart)
      {
        switch(alignmentPattern)
        {
          case AllAligned:
            for (Index j = alignedStart; j<alignedSize; j+=PacketSize)
              _EIGEN_ACCUMULATE_PACKETS(d,d,d);
            break;
          case EvenAligned:
            for (Index j = alignedStart; j<alignedSize; j+=PacketSize)
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

              for (Index j = alignedStart; j<peeledSize; j+=peels*PacketSize)
              {
                b = ei_pload(&rhs[j]);
                A11 = ei_pload(&lhs1[j-1+PacketSize]);  ei_palign<1>(A01,A11);
                A12 = ei_pload(&lhs2[j-2+PacketSize]);  ei_palign<2>(A02,A12);
                A13 = ei_pload(&lhs3[j-3+PacketSize]);  ei_palign<3>(A03,A13);

                ptmp0 = cj.pmadd(ei_pload (&lhs0[j]), b, ptmp0);
                ptmp1 = cj.pmadd(A01, b, ptmp1);
                A01 = ei_pload(&lhs1[j-1+2*PacketSize]);  ei_palign<1>(A11,A01);
                ptmp2 = cj.pmadd(A02, b, ptmp2);
                A02 = ei_pload(&lhs2[j-2+2*PacketSize]);  ei_palign<2>(A12,A02);
                ptmp3 = cj.pmadd(A03, b, ptmp3);
                A03 = ei_pload(&lhs3[j-3+2*PacketSize]);  ei_palign<3>(A13,A03);

                b = ei_pload(&rhs[j+PacketSize]);
                ptmp0 = cj.pmadd(ei_pload (&lhs0[j+PacketSize]), b, ptmp0);
                ptmp1 = cj.pmadd(A11, b, ptmp1);
                ptmp2 = cj.pmadd(A12, b, ptmp2);
                ptmp3 = cj.pmadd(A13, b, ptmp3);
              }
            }
            for (Index j = peeledSize; j<alignedSize; j+=PacketSize)
              _EIGEN_ACCUMULATE_PACKETS(d,du,du);
            break;
          default:
            for (Index j = alignedStart; j<alignedSize; j+=PacketSize)
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
    for (Index j=alignedSize; j<size; ++j)
    {
      Scalar b = rhs[j];
      tmp0 += cj.pmul(lhs0[j],b); tmp1 += cj.pmul(lhs1[j],b);
      tmp2 += cj.pmul(lhs2[j],b); tmp3 += cj.pmul(lhs3[j],b);
    }
    res[i] += alpha*tmp0; res[i+offset1] += alpha*tmp1; res[i+2] += alpha*tmp2; res[i+offset3] += alpha*tmp3;
  }

  // process remaining first and last rows (at most columnsAtOnce-1)
  Index end = res.size();
  Index start = rowBound;
  do
  {
    for (Index i=start; i<end; ++i)
    {
      Scalar tmp0 = Scalar(0);
      Packet ptmp0 = ei_pset1(tmp0);
      const Scalar* lhs0 = lhs + i*lhsStride;
      // process first unaligned result's coeffs
      // FIXME this loop get vectorized by the compiler !
      for (Index j=0; j<alignedStart; ++j)
        tmp0 += cj.pmul(lhs0[j], rhs[j]);

      if (alignedSize>alignedStart)
      {
        // process aligned rhs coeffs
        if ((size_t(lhs0+alignedStart)%sizeof(Packet))==0)
          for (Index j = alignedStart;j<alignedSize;j+=PacketSize)
            ptmp0 = cj.pmadd(ei_pload(&lhs0[j]), ei_pload(&rhs[j]), ptmp0);
        else
          for (Index j = alignedStart;j<alignedSize;j+=PacketSize)
            ptmp0 = cj.pmadd(ei_ploadu(&lhs0[j]), ei_pload(&rhs[j]), ptmp0);
        tmp0 += ei_predux(ptmp0);
      }

      // process remaining scalars
      // FIXME this loop get vectorized by the compiler !
      for (Index j=alignedSize; j<size; ++j)
        tmp0 += cj.pmul(lhs0[j], rhs[j]);
      res[i] += alpha*tmp0;
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
