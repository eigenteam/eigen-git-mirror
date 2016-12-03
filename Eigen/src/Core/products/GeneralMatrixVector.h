// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2016 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_GENERAL_MATRIX_VECTOR_H
#define EIGEN_GENERAL_MATRIX_VECTOR_H

namespace Eigen {

namespace internal {

/* Optimized col-major matrix * vector product:
 * This algorithm processes the matrix per vertical panels,
 * which are then processed horizontaly per chunck of 8*PacketSize x 1 vertical segments.
 *
 * Mixing type logic: C += alpha * A * B
 *  |  A  |  B  |alpha| comments
 *  |real |cplx |cplx | no vectorization
 *  |real |cplx |real | alpha is converted to a cplx when calling the run function, no vectorization
 *  |cplx |real |cplx | invalid, the caller has to do tmp: = A * B; C += alpha*tmp
 *  |cplx |real |real | optimal case, vectorization possible via real-cplx mul
 *
 * The same reasoning apply for the transposed case.
 */
template<typename Index, typename LhsScalar, typename LhsMapper, bool ConjugateLhs, typename RhsScalar, typename RhsMapper, bool ConjugateRhs, int Version>
struct general_matrix_vector_product<Index,LhsScalar,LhsMapper,ColMajor,ConjugateLhs,RhsScalar,RhsMapper,ConjugateRhs,Version>
{
  typedef typename ScalarBinaryOpTraits<LhsScalar, RhsScalar>::ReturnType ResScalar;

enum {
  Vectorizable = packet_traits<LhsScalar>::Vectorizable && packet_traits<RhsScalar>::Vectorizable
              && int(packet_traits<LhsScalar>::size)==int(packet_traits<RhsScalar>::size),
  LhsPacketSize = Vectorizable ? packet_traits<LhsScalar>::size : 1,
  RhsPacketSize = Vectorizable ? packet_traits<RhsScalar>::size : 1,
  ResPacketSize = Vectorizable ? packet_traits<ResScalar>::size : 1
};

typedef typename packet_traits<LhsScalar>::type  _LhsPacket;
typedef typename packet_traits<RhsScalar>::type  _RhsPacket;
typedef typename packet_traits<ResScalar>::type  _ResPacket;

typedef typename conditional<Vectorizable,_LhsPacket,LhsScalar>::type LhsPacket;
typedef typename conditional<Vectorizable,_RhsPacket,RhsScalar>::type RhsPacket;
typedef typename conditional<Vectorizable,_ResPacket,ResScalar>::type ResPacket;

EIGEN_DONT_INLINE static void run(
  Index rows, Index cols,
  const LhsMapper& lhs,
  const RhsMapper& rhs,
        ResScalar* res, Index resIncr,
  RhsScalar alpha);
};

template<typename Index, typename LhsScalar, typename LhsMapper, bool ConjugateLhs, typename RhsScalar, typename RhsMapper, bool ConjugateRhs, int Version>
EIGEN_DONT_INLINE void general_matrix_vector_product<Index,LhsScalar,LhsMapper,ColMajor,ConjugateLhs,RhsScalar,RhsMapper,ConjugateRhs,Version>::run(
  Index rows, Index cols,
  const LhsMapper& alhs,
  const RhsMapper& rhs,
        ResScalar* res, Index resIncr,
  RhsScalar alpha)
{
  EIGEN_UNUSED_VARIABLE(resIncr);
  eigen_internal_assert(resIncr==1);

  // The following copy tells the compiler that lhs's attributes are not modified outside this function
  // This helps GCC to generate propoer code.
  LhsMapper lhs(alhs);

  conj_helper<LhsScalar,RhsScalar,ConjugateLhs,ConjugateRhs> cj;
  conj_helper<LhsPacket,RhsPacket,ConjugateLhs,ConjugateRhs> pcj;
  const Index lhsStride = lhs.stride();
  // TODO: for padded aligned inputs, we could enable aligned reads
  enum { LhsAlignment = Unaligned };

  const Index n8 = rows-8*ResPacketSize+1;
  const Index n4 = rows-4*ResPacketSize+1;
  const Index n3 = rows-3*ResPacketSize+1;
  const Index n2 = rows-2*ResPacketSize+1;
  const Index n1 = rows-1*ResPacketSize+1;

  // TODO: improve the following heuristic:
  const Index block_cols = cols<128 ? cols : (lhsStride*sizeof(LhsScalar)<32000?16:4);
  ResPacket palpha = pset1<ResPacket>(alpha);

  for(Index j2=0; j2<cols; j2+=block_cols)
  {
    Index jend = numext::mini(j2+block_cols,cols);
    Index i=0;
    for(; i<n8; i+=ResPacketSize*8)
    {
      ResPacket c0 = pset1<ResPacket>(ResScalar(0)),
                c1 = pset1<ResPacket>(ResScalar(0)),
                c2 = pset1<ResPacket>(ResScalar(0)),
                c3 = pset1<ResPacket>(ResScalar(0)),
                c4 = pset1<ResPacket>(ResScalar(0)),
                c5 = pset1<ResPacket>(ResScalar(0)),
                c6 = pset1<ResPacket>(ResScalar(0)),
                c7 = pset1<ResPacket>(ResScalar(0));

      for(Index j=j2; j<jend; j+=1)
      {
        RhsPacket b0 = pset1<RhsPacket>(rhs(j,0));
        c0 = pcj.pmadd(lhs.template load<LhsPacket,LhsAlignment>(i+LhsPacketSize*0,j),b0,c0);
        c1 = pcj.pmadd(lhs.template load<LhsPacket,LhsAlignment>(i+LhsPacketSize*1,j),b0,c1);
        c2 = pcj.pmadd(lhs.template load<LhsPacket,LhsAlignment>(i+LhsPacketSize*2,j),b0,c2);
        c3 = pcj.pmadd(lhs.template load<LhsPacket,LhsAlignment>(i+LhsPacketSize*3,j),b0,c3);
        c4 = pcj.pmadd(lhs.template load<LhsPacket,LhsAlignment>(i+LhsPacketSize*4,j),b0,c4);
        c5 = pcj.pmadd(lhs.template load<LhsPacket,LhsAlignment>(i+LhsPacketSize*5,j),b0,c5);
        c6 = pcj.pmadd(lhs.template load<LhsPacket,LhsAlignment>(i+LhsPacketSize*6,j),b0,c6);
        c7 = pcj.pmadd(lhs.template load<LhsPacket,LhsAlignment>(i+LhsPacketSize*7,j),b0,c7);
      }
      pstoreu(res+i+ResPacketSize*0, pmadd(c0,palpha,ploadu<ResPacket>(res+i+ResPacketSize*0)));
      pstoreu(res+i+ResPacketSize*1, pmadd(c1,palpha,ploadu<ResPacket>(res+i+ResPacketSize*1)));
      pstoreu(res+i+ResPacketSize*2, pmadd(c2,palpha,ploadu<ResPacket>(res+i+ResPacketSize*2)));
      pstoreu(res+i+ResPacketSize*3, pmadd(c3,palpha,ploadu<ResPacket>(res+i+ResPacketSize*3)));
      pstoreu(res+i+ResPacketSize*4, pmadd(c4,palpha,ploadu<ResPacket>(res+i+ResPacketSize*4)));
      pstoreu(res+i+ResPacketSize*5, pmadd(c5,palpha,ploadu<ResPacket>(res+i+ResPacketSize*5)));
      pstoreu(res+i+ResPacketSize*6, pmadd(c6,palpha,ploadu<ResPacket>(res+i+ResPacketSize*6)));
      pstoreu(res+i+ResPacketSize*7, pmadd(c7,palpha,ploadu<ResPacket>(res+i+ResPacketSize*7)));
    }
    if(i<n4)
    {
      ResPacket c0 = pset1<ResPacket>(ResScalar(0)),
                c1 = pset1<ResPacket>(ResScalar(0)),
                c2 = pset1<ResPacket>(ResScalar(0)),
                c3 = pset1<ResPacket>(ResScalar(0));

      for(Index j=j2; j<jend; j+=1)
      {
        RhsPacket b0 = pset1<RhsPacket>(rhs(j,0));
        c0 = pcj.pmadd(lhs.template load<LhsPacket,LhsAlignment>(i+LhsPacketSize*0,j),b0,c0);
        c1 = pcj.pmadd(lhs.template load<LhsPacket,LhsAlignment>(i+LhsPacketSize*1,j),b0,c1);
        c2 = pcj.pmadd(lhs.template load<LhsPacket,LhsAlignment>(i+LhsPacketSize*2,j),b0,c2);
        c3 = pcj.pmadd(lhs.template load<LhsPacket,LhsAlignment>(i+LhsPacketSize*3,j),b0,c3);
      }
      pstoreu(res+i+ResPacketSize*0, pmadd(c0,palpha,ploadu<ResPacket>(res+i+ResPacketSize*0)));
      pstoreu(res+i+ResPacketSize*1, pmadd(c1,palpha,ploadu<ResPacket>(res+i+ResPacketSize*1)));
      pstoreu(res+i+ResPacketSize*2, pmadd(c2,palpha,ploadu<ResPacket>(res+i+ResPacketSize*2)));
      pstoreu(res+i+ResPacketSize*3, pmadd(c3,palpha,ploadu<ResPacket>(res+i+ResPacketSize*3)));

      i+=ResPacketSize*4;
    }
    if(i<n3)
    {
      ResPacket c0 = pset1<ResPacket>(ResScalar(0)),
                c1 = pset1<ResPacket>(ResScalar(0)),
                c2 = pset1<ResPacket>(ResScalar(0));

      for(Index j=j2; j<jend; j+=1)
      {
        RhsPacket b0 = pset1<RhsPacket>(rhs(j,0));
        c0 = pcj.pmadd(lhs.template load<LhsPacket,LhsAlignment>(i+LhsPacketSize*0,j),b0,c0);
        c1 = pcj.pmadd(lhs.template load<LhsPacket,LhsAlignment>(i+LhsPacketSize*1,j),b0,c1);
        c2 = pcj.pmadd(lhs.template load<LhsPacket,LhsAlignment>(i+LhsPacketSize*2,j),b0,c2);
      }
      pstoreu(res+i+ResPacketSize*0, pmadd(c0,palpha,ploadu<ResPacket>(res+i+ResPacketSize*0)));
      pstoreu(res+i+ResPacketSize*1, pmadd(c1,palpha,ploadu<ResPacket>(res+i+ResPacketSize*1)));
      pstoreu(res+i+ResPacketSize*2, pmadd(c2,palpha,ploadu<ResPacket>(res+i+ResPacketSize*2)));

      i+=ResPacketSize*3;
    }
    if(i<n2)
    {
      ResPacket c0 = pset1<ResPacket>(ResScalar(0)),
                c1 = pset1<ResPacket>(ResScalar(0));

      for(Index j=j2; j<jend; j+=1)
      {
        RhsPacket b0 = pset1<RhsPacket>(rhs(j,0));
        c0 = pcj.pmadd(lhs.template load<LhsPacket,LhsAlignment>(i+LhsPacketSize*0,j),b0,c0);
        c1 = pcj.pmadd(lhs.template load<LhsPacket,LhsAlignment>(i+LhsPacketSize*1,j),b0,c1);
      }
      pstoreu(res+i+ResPacketSize*0, pmadd(c0,palpha,ploadu<ResPacket>(res+i+ResPacketSize*0)));
      pstoreu(res+i+ResPacketSize*1, pmadd(c1,palpha,ploadu<ResPacket>(res+i+ResPacketSize*1)));
      i+=ResPacketSize*2;
    }
    if(i<n1)
    {
      ResPacket c0 = pset1<ResPacket>(ResScalar(0));
      for(Index j=j2; j<jend; j+=1)
      {
        RhsPacket b0 = pset1<RhsPacket>(rhs(j,0));
        c0 = pcj.pmadd(lhs.template load<LhsPacket,LhsAlignment>(i+0,j),b0,c0);
      }
      pstoreu(res+i+ResPacketSize*0, pmadd(c0,palpha,ploadu<ResPacket>(res+i+ResPacketSize*0)));
      i+=ResPacketSize;
    }
    for(;i<rows;++i)
    {
      ResScalar c0(0);
      for(Index j=j2; j<jend; j+=1)
        c0 += cj.pmul(lhs(i,j), rhs(j,0));
      res[i] += alpha*c0;
    }
  }
}

/* Optimized row-major matrix * vector product:
 * This algorithm processes 4 rows at onces that allows to both reduce
 * the number of load/stores of the result by a factor 4 and to reduce
 * the instruction dependency. Moreover, we know that all bands have the
 * same alignment pattern.
 *
 * Mixing type logic:
 *  - alpha is always a complex (or converted to a complex)
 *  - no vectorization
 */
template<typename Index, typename LhsScalar, typename LhsMapper, bool ConjugateLhs, typename RhsScalar, typename RhsMapper, bool ConjugateRhs, int Version>
struct general_matrix_vector_product<Index,LhsScalar,LhsMapper,RowMajor,ConjugateLhs,RhsScalar,RhsMapper,ConjugateRhs,Version>
{
typedef typename ScalarBinaryOpTraits<LhsScalar, RhsScalar>::ReturnType ResScalar;

enum {
  Vectorizable = packet_traits<LhsScalar>::Vectorizable && packet_traits<RhsScalar>::Vectorizable
              && int(packet_traits<LhsScalar>::size)==int(packet_traits<RhsScalar>::size),
  LhsPacketSize = Vectorizable ? packet_traits<LhsScalar>::size : 1,
  RhsPacketSize = Vectorizable ? packet_traits<RhsScalar>::size : 1,
  ResPacketSize = Vectorizable ? packet_traits<ResScalar>::size : 1
};

typedef typename packet_traits<LhsScalar>::type  _LhsPacket;
typedef typename packet_traits<RhsScalar>::type  _RhsPacket;
typedef typename packet_traits<ResScalar>::type  _ResPacket;

typedef typename conditional<Vectorizable,_LhsPacket,LhsScalar>::type LhsPacket;
typedef typename conditional<Vectorizable,_RhsPacket,RhsScalar>::type RhsPacket;
typedef typename conditional<Vectorizable,_ResPacket,ResScalar>::type ResPacket;

EIGEN_DONT_INLINE static void run(
  Index rows, Index cols,
  const LhsMapper& lhs,
  const RhsMapper& rhs,
        ResScalar* res, Index resIncr,
  ResScalar alpha);
};

template<typename Index, typename LhsScalar, typename LhsMapper, bool ConjugateLhs, typename RhsScalar, typename RhsMapper, bool ConjugateRhs, int Version>
EIGEN_DONT_INLINE void general_matrix_vector_product<Index,LhsScalar,LhsMapper,RowMajor,ConjugateLhs,RhsScalar,RhsMapper,ConjugateRhs,Version>::run(
  Index rows, Index cols,
  const LhsMapper& lhs,
  const RhsMapper& rhs,
  ResScalar* res, Index resIncr,
  ResScalar alpha)
{
  eigen_internal_assert(rhs.stride()==1);

  #ifdef _EIGEN_ACCUMULATE_PACKETS
  #error _EIGEN_ACCUMULATE_PACKETS has already been defined
  #endif

  #define _EIGEN_ACCUMULATE_PACKETS(Alignment0,Alignment13,Alignment2) {\
    RhsPacket b = rhs.getVectorMapper(j, 0).template load<RhsPacket, Aligned>(0);  \
    ptmp0 = pcj.pmadd(lhs0.template load<LhsPacket, Alignment0>(j), b, ptmp0); \
    ptmp1 = pcj.pmadd(lhs1.template load<LhsPacket, Alignment13>(j), b, ptmp1); \
    ptmp2 = pcj.pmadd(lhs2.template load<LhsPacket, Alignment2>(j), b, ptmp2); \
    ptmp3 = pcj.pmadd(lhs3.template load<LhsPacket, Alignment13>(j), b, ptmp3); }

  conj_helper<LhsScalar,RhsScalar,ConjugateLhs,ConjugateRhs> cj;
  conj_helper<LhsPacket,RhsPacket,ConjugateLhs,ConjugateRhs> pcj;

  typedef typename LhsMapper::VectorMapper LhsScalars;

  enum { AllAligned=0, EvenAligned=1, FirstAligned=2, NoneAligned=3 };
  const Index rowsAtOnce = 4;
  const Index peels = 2;
  const Index RhsPacketAlignedMask = RhsPacketSize-1;
  const Index LhsPacketAlignedMask = LhsPacketSize-1;
  const Index depth = cols;
  const Index lhsStride = lhs.stride();

  // How many coeffs of the result do we have to skip to be aligned.
  // Here we assume data are at least aligned on the base scalar type
  // if that's not the case then vectorization is discarded, see below.
  Index alignedStart = rhs.firstAligned(depth);
  Index alignedSize = RhsPacketSize>1 ? alignedStart + ((depth-alignedStart) & ~RhsPacketAlignedMask) : 0;
  const Index peeledSize = alignedSize - RhsPacketSize*peels - RhsPacketSize + 1;

  const Index alignmentStep = LhsPacketSize>1 ? (LhsPacketSize - lhsStride % LhsPacketSize) & LhsPacketAlignedMask : 0;
  Index alignmentPattern = alignmentStep==0 ? AllAligned
                           : alignmentStep==(LhsPacketSize/2) ? EvenAligned
                           : FirstAligned;

  // we cannot assume the first element is aligned because of sub-matrices
  const Index lhsAlignmentOffset = lhs.firstAligned(depth);
  const Index rhsAlignmentOffset = rhs.firstAligned(rows);

  // find how many rows do we have to skip to be aligned with rhs (if possible)
  Index skipRows = 0;
  // if the data cannot be aligned (TODO add some compile time tests when possible, e.g. for floats)
  if( (sizeof(LhsScalar)!=sizeof(RhsScalar)) ||
      (lhsAlignmentOffset < 0) || (lhsAlignmentOffset == depth) ||
      (rhsAlignmentOffset < 0) || (rhsAlignmentOffset == rows) )
  {
    alignedSize = 0;
    alignedStart = 0;
    alignmentPattern = NoneAligned;
  }
  else if(LhsPacketSize > 4)
  {
    // TODO: extend the code to support aligned loads whenever possible when LhsPacketSize > 4.
    alignmentPattern = NoneAligned;
  }
  else if (LhsPacketSize>1)
  {
  //    eigen_internal_assert(size_t(firstLhs+lhsAlignmentOffset)%sizeof(LhsPacket)==0  || depth<LhsPacketSize);

    while (skipRows<LhsPacketSize &&
           alignedStart != ((lhsAlignmentOffset + alignmentStep*skipRows)%LhsPacketSize))
      ++skipRows;
    if (skipRows==LhsPacketSize)
    {
      // nothing can be aligned, no need to skip any column
      alignmentPattern = NoneAligned;
      skipRows = 0;
    }
    else
    {
      skipRows = (std::min)(skipRows,Index(rows));
      // note that the skiped columns are processed later.
    }
    /*    eigen_internal_assert(  alignmentPattern==NoneAligned
                      || LhsPacketSize==1
                      || (skipRows + rowsAtOnce >= rows)
                      || LhsPacketSize > depth
                      || (size_t(firstLhs+alignedStart+lhsStride*skipRows)%sizeof(LhsPacket))==0);*/
  }
  else if(Vectorizable)
  {
    alignedStart = 0;
    alignedSize = depth;
    alignmentPattern = AllAligned;
  }

  const Index offset1 = (FirstAligned && alignmentStep==1)?3:1;
  const Index offset3 = (FirstAligned && alignmentStep==1)?1:3;

  Index rowBound = ((rows-skipRows)/rowsAtOnce)*rowsAtOnce + skipRows;
  for (Index i=skipRows; i<rowBound; i+=rowsAtOnce)
  {
    // FIXME: what is the purpose of this EIGEN_ALIGN_DEFAULT ??
    EIGEN_ALIGN_MAX ResScalar tmp0 = ResScalar(0);
    ResScalar tmp1 = ResScalar(0), tmp2 = ResScalar(0), tmp3 = ResScalar(0);

    // this helps the compiler generating good binary code
    const LhsScalars lhs0 = lhs.getVectorMapper(i+0, 0),    lhs1 = lhs.getVectorMapper(i+offset1, 0),
                     lhs2 = lhs.getVectorMapper(i+2, 0),    lhs3 = lhs.getVectorMapper(i+offset3, 0);

    if (Vectorizable)
    {
      /* explicit vectorization */
      ResPacket ptmp0 = pset1<ResPacket>(ResScalar(0)), ptmp1 = pset1<ResPacket>(ResScalar(0)),
                ptmp2 = pset1<ResPacket>(ResScalar(0)), ptmp3 = pset1<ResPacket>(ResScalar(0));

      // process initial unaligned coeffs
      // FIXME this loop get vectorized by the compiler !
      for (Index j=0; j<alignedStart; ++j)
      {
        RhsScalar b = rhs(j, 0);
        tmp0 += cj.pmul(lhs0(j),b); tmp1 += cj.pmul(lhs1(j),b);
        tmp2 += cj.pmul(lhs2(j),b); tmp3 += cj.pmul(lhs3(j),b);
      }

      if (alignedSize>alignedStart)
      {
        switch(alignmentPattern)
        {
          case AllAligned:
            for (Index j = alignedStart; j<alignedSize; j+=RhsPacketSize)
              _EIGEN_ACCUMULATE_PACKETS(Aligned,Aligned,Aligned);
            break;
          case EvenAligned:
            for (Index j = alignedStart; j<alignedSize; j+=RhsPacketSize)
              _EIGEN_ACCUMULATE_PACKETS(Aligned,Unaligned,Aligned);
            break;
          case FirstAligned:
          {
            Index j = alignedStart;
            if (peels>1)
            {
              /* Here we proccess 4 rows with with two peeled iterations to hide
               * the overhead of unaligned loads. Moreover unaligned loads are handled
               * using special shift/move operations between the two aligned packets
               * overlaping the desired unaligned packet. This is *much* more efficient
               * than basic unaligned loads.
               */
              LhsPacket A01, A02, A03, A11, A12, A13;
              A01 = lhs1.template load<LhsPacket, Aligned>(alignedStart-1);
              A02 = lhs2.template load<LhsPacket, Aligned>(alignedStart-2);
              A03 = lhs3.template load<LhsPacket, Aligned>(alignedStart-3);

              for (; j<peeledSize; j+=peels*RhsPacketSize)
              {
                RhsPacket b = rhs.getVectorMapper(j, 0).template load<RhsPacket, Aligned>(0);
                A11 = lhs1.template load<LhsPacket, Aligned>(j-1+LhsPacketSize);  palign<1>(A01,A11);
                A12 = lhs2.template load<LhsPacket, Aligned>(j-2+LhsPacketSize);  palign<2>(A02,A12);
                A13 = lhs3.template load<LhsPacket, Aligned>(j-3+LhsPacketSize);  palign<3>(A03,A13);

                ptmp0 = pcj.pmadd(lhs0.template load<LhsPacket, Aligned>(j), b, ptmp0);
                ptmp1 = pcj.pmadd(A01, b, ptmp1);
                A01 = lhs1.template load<LhsPacket, Aligned>(j-1+2*LhsPacketSize);  palign<1>(A11,A01);
                ptmp2 = pcj.pmadd(A02, b, ptmp2);
                A02 = lhs2.template load<LhsPacket, Aligned>(j-2+2*LhsPacketSize);  palign<2>(A12,A02);
                ptmp3 = pcj.pmadd(A03, b, ptmp3);
                A03 = lhs3.template load<LhsPacket, Aligned>(j-3+2*LhsPacketSize);  palign<3>(A13,A03);

                b = rhs.getVectorMapper(j+RhsPacketSize, 0).template load<RhsPacket, Aligned>(0);
                ptmp0 = pcj.pmadd(lhs0.template load<LhsPacket, Aligned>(j+LhsPacketSize), b, ptmp0);
                ptmp1 = pcj.pmadd(A11, b, ptmp1);
                ptmp2 = pcj.pmadd(A12, b, ptmp2);
                ptmp3 = pcj.pmadd(A13, b, ptmp3);
              }
            }
            for (; j<alignedSize; j+=RhsPacketSize)
              _EIGEN_ACCUMULATE_PACKETS(Aligned,Unaligned,Unaligned);
            break;
          }
          default:
            for (Index j = alignedStart; j<alignedSize; j+=RhsPacketSize)
              _EIGEN_ACCUMULATE_PACKETS(Unaligned,Unaligned,Unaligned);
            break;
        }
        tmp0 += predux(ptmp0);
        tmp1 += predux(ptmp1);
        tmp2 += predux(ptmp2);
        tmp3 += predux(ptmp3);
      }
    } // end explicit vectorization

    // process remaining coeffs (or all if no explicit vectorization)
    // FIXME this loop get vectorized by the compiler !
    for (Index j=alignedSize; j<depth; ++j)
    {
      RhsScalar b = rhs(j, 0);
      tmp0 += cj.pmul(lhs0(j),b); tmp1 += cj.pmul(lhs1(j),b);
      tmp2 += cj.pmul(lhs2(j),b); tmp3 += cj.pmul(lhs3(j),b);
    }
    res[i*resIncr]            += alpha*tmp0;
    res[(i+offset1)*resIncr]  += alpha*tmp1;
    res[(i+2)*resIncr]        += alpha*tmp2;
    res[(i+offset3)*resIncr]  += alpha*tmp3;
  }

  // process remaining first and last rows (at most columnsAtOnce-1)
  Index end = rows;
  Index start = rowBound;
  do
  {
    for (Index i=start; i<end; ++i)
    {
      EIGEN_ALIGN_MAX ResScalar tmp0 = ResScalar(0);
      ResPacket ptmp0 = pset1<ResPacket>(tmp0);
      const LhsScalars lhs0 = lhs.getVectorMapper(i, 0);
      // process first unaligned result's coeffs
      // FIXME this loop get vectorized by the compiler !
      for (Index j=0; j<alignedStart; ++j)
        tmp0 += cj.pmul(lhs0(j), rhs(j, 0));

      if (alignedSize>alignedStart)
      {
        // process aligned rhs coeffs
        if (lhs0.template aligned<LhsPacket>(alignedStart))
          for (Index j = alignedStart;j<alignedSize;j+=RhsPacketSize)
            ptmp0 = pcj.pmadd(lhs0.template load<LhsPacket, Aligned>(j), rhs.getVectorMapper(j, 0).template load<RhsPacket, Aligned>(0), ptmp0);
        else
          for (Index j = alignedStart;j<alignedSize;j+=RhsPacketSize)
            ptmp0 = pcj.pmadd(lhs0.template load<LhsPacket, Unaligned>(j), rhs.getVectorMapper(j, 0).template load<RhsPacket, Aligned>(0), ptmp0);
        tmp0 += predux(ptmp0);
      }

      // process remaining scalars
      // FIXME this loop get vectorized by the compiler !
      for (Index j=alignedSize; j<depth; ++j)
        tmp0 += cj.pmul(lhs0(j), rhs(j, 0));
      res[i*resIncr] += alpha*tmp0;
    }
    if (skipRows)
    {
      start = 0;
      end = skipRows;
      skipRows = 0;
    }
    else
      break;
  } while(Vectorizable);

  #undef _EIGEN_ACCUMULATE_PACKETS
}

} // end namespace internal

} // end namespace Eigen

#endif // EIGEN_GENERAL_MATRIX_VECTOR_H
