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

#ifndef EIGEN_GENERAL_BLOCK_PANEL_H
#define EIGEN_GENERAL_BLOCK_PANEL_H

/** \internal */
inline void ei_manage_caching_sizes(Action action, std::ptrdiff_t* l1=0, std::ptrdiff_t* l2=0)
{
  static std::ptrdiff_t m_l1CacheSize = 0;
  static std::ptrdiff_t m_l2CacheSize = 0;
  if(m_l1CacheSize==0)
  {
    m_l1CacheSize = ei_queryL1CacheSize();
    m_l2CacheSize = ei_queryTopLevelCacheSize();

    if(m_l1CacheSize<=0) m_l1CacheSize = 8 * 1024;
    if(m_l2CacheSize<=0) m_l2CacheSize = 1 * 1024 * 1024;
  }

  if(action==SetAction)
  {
    // set the cpu cache size and cache all block sizes from a global cache size in byte
    ei_internal_assert(l1!=0 && l2!=0);
    m_l1CacheSize = *l1;
    m_l2CacheSize = *l2;
  }
  else if(action==GetAction)
  {
    ei_internal_assert(l1!=0 && l2!=0);
    *l1 = m_l1CacheSize;
    *l2 = m_l2CacheSize;
  }
  else
  {
    ei_internal_assert(false);
  }
}

/** \returns the currently set level 1 cpu cache size (in bytes) used to estimate the ideal blocking size parameters.
  * \sa setCpuCacheSize */
inline std::ptrdiff_t l1CacheSize()
{
  std::ptrdiff_t l1, l2;
  ei_manage_caching_sizes(GetAction, &l1, &l2);
  return l1;
}

/** \returns the currently set level 2 cpu cache size (in bytes) used to estimate the ideal blocking size parameters.
  * \sa setCpuCacheSize */
inline std::ptrdiff_t l2CacheSize()
{
  std::ptrdiff_t l1, l2;
  ei_manage_caching_sizes(GetAction, &l1, &l2);
  return l2;
}

/** Set the cpu L1 and L2 cache sizes (in bytes).
  * These values are use to adjust the size of the blocks
  * for the algorithms working per blocks.
  *
  * \sa computeProductBlockingSizes */
inline void setCpuCacheSizes(std::ptrdiff_t l1, std::ptrdiff_t l2)
{
  ei_manage_caching_sizes(SetAction, &l1, &l2);
}

/** \brief Computes the blocking parameters for a m x k times k x n matrix product
  *
  * \param[in,out] k Input: the third dimension of the product. Output: the blocking size along the same dimension.
  * \param[in,out] m Input: the number of rows of the left hand side. Output: the blocking size along the same dimension.
  * \param[in,out] n Input: the number of columns of the right hand side. Output: the blocking size along the same dimension.
  *
  * Given a m x k times k x n matrix product of scalar types \c LhsScalar and \c RhsScalar,
  * this function computes the blocking size parameters along the respective dimensions
  * for matrix products and related algorithms. The blocking sizes depends on various
  * parameters:
  * - the L1 and L2 cache sizes,
  * - the register level blocking sizes defined by ei_product_blocking_traits,
  * - the number of scalars that fit into a packet (when vectorization is enabled).
  *
  * \sa setCpuCacheSizes */
template<typename LhsScalar, typename RhsScalar, int KcFactor>
void computeProductBlockingSizes(std::ptrdiff_t& k, std::ptrdiff_t& m, std::ptrdiff_t& n)
{
  // Explanations:
  // Let's recall the product algorithms form kc x nc horizontal panels B' on the rhs and
  // mc x kc blocks A' on the lhs. A' has to fit into L2 cache. Moreover, B' is processed
  // per kc x nr vertical small panels where nr is the blocking size along the n dimension
  // at the register level. For vectorization purpose, these small vertical panels are unpacked,
  // i.e., each coefficient is replicated to fit a packet. This small vertical panel has to
  // stay in L1 cache.
  std::ptrdiff_t l1, l2;

  enum {
    kdiv = KcFactor * 2 * ei_product_blocking_traits<RhsScalar>::nr
         * ei_packet_traits<RhsScalar>::size * sizeof(RhsScalar),
    mr = ei_product_blocking_traits<LhsScalar>::mr,
    mr_mask = (0xffffffff/mr)*mr
  };

  ei_manage_caching_sizes(GetAction, &l1, &l2);
  k = std::min<std::ptrdiff_t>(k, l1/kdiv);
  std::ptrdiff_t _m = l2/(4 * sizeof(LhsScalar) * k);
  if(_m<m) m = _m & mr_mask;
  n = n;
}

template<typename LhsScalar, typename RhsScalar>
inline void computeProductBlockingSizes(std::ptrdiff_t& k, std::ptrdiff_t& m, std::ptrdiff_t& n)
{
  computeProductBlockingSizes<LhsScalar,RhsScalar,1>(k, m, n);
}

#ifdef EIGEN_HAS_FUSE_CJMADD
  #define CJMADD(A,B,C,T)  C = cj.pmadd(A,B,C);
#else
  #define CJMADD(A,B,C,T)  T = B; T = cj.pmul(A,T); C = ei_padd(C,T);
#endif

// optimized GEneral packed Block * packed Panel product kernel
template<typename Scalar, typename Index, int mr, int nr, typename Conj>
struct ei_gebp_kernel
{
  void operator()(Scalar* res, Index resStride, const Scalar* blockA, const Scalar* blockB, Index rows, Index depth, Index cols,
                  Index strideA=-1, Index strideB=-1, Index offsetA=0, Index offsetB=0, Scalar* unpackedB = 0)
  {
    typedef typename ei_packet_traits<Scalar>::type PacketType;
    enum { PacketSize = ei_packet_traits<Scalar>::size };
    if(strideA==-1) strideA = depth;
    if(strideB==-1) strideB = depth;
    Conj cj;
    Index packet_cols = (cols/nr) * nr;
    const Index peeled_mc = (rows/mr)*mr;
    const Index peeled_mc2 = peeled_mc + (rows-peeled_mc >= PacketSize ? PacketSize : 0);
    const Index peeled_kc = (depth/4)*4;

    if(unpackedB==0)
      unpackedB = const_cast<Scalar*>(blockB - strideB * nr * PacketSize);

    // loops on each micro vertical panel of rhs (depth x nr)
    for(Index j2=0; j2<packet_cols; j2+=nr)
    {
      // unpack B
      {
        const Scalar* blB = &blockB[j2*strideB+offsetB*nr];
        Index n = depth*nr;
        for(Index k=0; k<n; k++)
          ei_pstore(&unpackedB[k*PacketSize], ei_pset1(blB[k]));
        /*Scalar* dest = unpackedB;
        for(Index k=0; k<n; k+=4*PacketSize)
        {
          #ifdef EIGEN_VECTORIZE_SSE
          const Index S = 128;
          const Index G = 16;
          _mm_prefetch((const char*)(&blB[S/2+0]), _MM_HINT_T0);
          _mm_prefetch((const char*)(&dest[S+0*G]), _MM_HINT_T0);
          _mm_prefetch((const char*)(&dest[S+1*G]), _MM_HINT_T0);
          _mm_prefetch((const char*)(&dest[S+2*G]), _MM_HINT_T0);
          _mm_prefetch((const char*)(&dest[S+3*G]), _MM_HINT_T0);
          #endif

          PacketType C0[PacketSize], C1[PacketSize], C2[PacketSize], C3[PacketSize];
          C0[0] = ei_pload(blB+0*PacketSize);
          C1[0] = ei_pload(blB+1*PacketSize);
          C2[0] = ei_pload(blB+2*PacketSize);
          C3[0] = ei_pload(blB+3*PacketSize);

          ei_punpackp(C0);
          ei_punpackp(C1);
          ei_punpackp(C2);
          ei_punpackp(C3);

          ei_pstore(dest+ 0*PacketSize, C0[0]);
          ei_pstore(dest+ 1*PacketSize, C0[1]);
          ei_pstore(dest+ 2*PacketSize, C0[2]);
          ei_pstore(dest+ 3*PacketSize, C0[3]);

          ei_pstore(dest+ 4*PacketSize, C1[0]);
          ei_pstore(dest+ 5*PacketSize, C1[1]);
          ei_pstore(dest+ 6*PacketSize, C1[2]);
          ei_pstore(dest+ 7*PacketSize, C1[3]);

          ei_pstore(dest+ 8*PacketSize, C2[0]);
          ei_pstore(dest+ 9*PacketSize, C2[1]);
          ei_pstore(dest+10*PacketSize, C2[2]);
          ei_pstore(dest+11*PacketSize, C2[3]);

          ei_pstore(dest+12*PacketSize, C3[0]);
          ei_pstore(dest+13*PacketSize, C3[1]);
          ei_pstore(dest+14*PacketSize, C3[2]);
          ei_pstore(dest+15*PacketSize, C3[3]);

          blB += 4*PacketSize;
          dest += 16*PacketSize;
        }*/
      }
      // loops on each micro horizontal panel of lhs (mr x depth)
      // => we select a mr x nr micro block of res which is entirely
      //    stored into mr/packet_size x nr registers.
      for(Index i=0; i<peeled_mc; i+=mr)
      {
        const Scalar* blA = &blockA[i*strideA+offsetA*mr];
        ei_prefetch(&blA[0]);

        // TODO move the res loads to the stores

        // gets res block as register
        PacketType C0, C1, C2, C3, C4, C5, C6, C7;
                  C0 = ei_pset1(Scalar(0));
                  C1 = ei_pset1(Scalar(0));
        if(nr==4) C2 = ei_pset1(Scalar(0));
        if(nr==4) C3 = ei_pset1(Scalar(0));
                  C4 = ei_pset1(Scalar(0));
                  C5 = ei_pset1(Scalar(0));
        if(nr==4) C6 = ei_pset1(Scalar(0));
        if(nr==4) C7 = ei_pset1(Scalar(0));

        Scalar* r0 = &res[(j2+0)*resStride + i];
        Scalar* r1 = r0 + resStride;
        Scalar* r2 = r1 + resStride;
        Scalar* r3 = r2 + resStride;

        ei_prefetch(r0+16);
        ei_prefetch(r1+16);
        ei_prefetch(r2+16);
        ei_prefetch(r3+16);

        // performs "inner" product
        // TODO let's check wether the folowing peeled loop could not be
        //      optimized via optimal prefetching from one loop to the other
        const Scalar* blB = unpackedB;
        for(Index k=0; k<peeled_kc; k+=4)
        {
          if(nr==2)
          {
            PacketType B0, T0, A0, A1;

            A0 = ei_pload(&blA[0*PacketSize]);
            A1 = ei_pload(&blA[1*PacketSize]);
            B0 = ei_pload(&blB[0*PacketSize]);
            CJMADD(A0,B0,C0,T0);
            CJMADD(A1,B0,C4,B0);
            B0 = ei_pload(&blB[1*PacketSize]);
            CJMADD(A0,B0,C1,T0);
            CJMADD(A1,B0,C5,B0);

            A0 = ei_pload(&blA[2*PacketSize]);
            A1 = ei_pload(&blA[3*PacketSize]);
            B0 = ei_pload(&blB[2*PacketSize]);
            CJMADD(A0,B0,C0,T0);
            CJMADD(A1,B0,C4,B0);
            B0 = ei_pload(&blB[3*PacketSize]);
            CJMADD(A0,B0,C1,T0);
            CJMADD(A1,B0,C5,B0);

            A0 = ei_pload(&blA[4*PacketSize]);
            A1 = ei_pload(&blA[5*PacketSize]);
            B0 = ei_pload(&blB[4*PacketSize]);
            CJMADD(A0,B0,C0,T0);
            CJMADD(A1,B0,C4,B0);
            B0 = ei_pload(&blB[5*PacketSize]);
            CJMADD(A0,B0,C1,T0);
            CJMADD(A1,B0,C5,B0);

            A0 = ei_pload(&blA[6*PacketSize]);
            A1 = ei_pload(&blA[7*PacketSize]);
            B0 = ei_pload(&blB[6*PacketSize]);
            CJMADD(A0,B0,C0,T0);
            CJMADD(A1,B0,C4,B0);
            B0 = ei_pload(&blB[7*PacketSize]);
            CJMADD(A0,B0,C1,T0);
            CJMADD(A1,B0,C5,B0);
          }
          else
          {
            PacketType B0, B1, B2, B3, A0, A1;
            PacketType T0;

            A0 = ei_pload(&blA[0*PacketSize]);
            A1 = ei_pload(&blA[1*PacketSize]);
            B0 = ei_pload(&blB[0*PacketSize]);
            B1 = ei_pload(&blB[1*PacketSize]);

            CJMADD(A0,B0,C0,T0);
            B2 = ei_pload(&blB[2*PacketSize]);
            CJMADD(A1,B0,C4,B0);
            B3 = ei_pload(&blB[3*PacketSize]);
            B0 = ei_pload(&blB[4*PacketSize]);
            CJMADD(A0,B1,C1,T0);
            CJMADD(A1,B1,C5,B1);
            B1 = ei_pload(&blB[5*PacketSize]);
            CJMADD(A0,B2,C2,T0);
            CJMADD(A1,B2,C6,B2);
            B2 = ei_pload(&blB[6*PacketSize]);
            CJMADD(A0,B3,C3,T0);
            A0 = ei_pload(&blA[2*PacketSize]);
            CJMADD(A1,B3,C7,B3);
            A1 = ei_pload(&blA[3*PacketSize]);
            B3 = ei_pload(&blB[7*PacketSize]);
            CJMADD(A0,B0,C0,T0);
            CJMADD(A1,B0,C4,B0);
            B0 = ei_pload(&blB[8*PacketSize]);
            CJMADD(A0,B1,C1,T0);
            CJMADD(A1,B1,C5,B1);
            B1 = ei_pload(&blB[9*PacketSize]);
            CJMADD(A0,B2,C2,T0);
            CJMADD(A1,B2,C6,B2);
            B2 = ei_pload(&blB[10*PacketSize]);
            CJMADD(A0,B3,C3,T0);
            A0 = ei_pload(&blA[4*PacketSize]);
            CJMADD(A1,B3,C7,B3);
            A1 = ei_pload(&blA[5*PacketSize]);
            B3 = ei_pload(&blB[11*PacketSize]);

            CJMADD(A0,B0,C0,T0);
            CJMADD(A1,B0,C4,B0);
            B0 = ei_pload(&blB[12*PacketSize]);
            CJMADD(A0,B1,C1,T0);
            CJMADD(A1,B1,C5,B1);
            B1 = ei_pload(&blB[13*PacketSize]);
            CJMADD(A0,B2,C2,T0);
            CJMADD(A1,B2,C6,B2);
            B2 = ei_pload(&blB[14*PacketSize]);
            CJMADD(A0,B3,C3,T0);
            A0 = ei_pload(&blA[6*PacketSize]);
            CJMADD(A1,B3,C7,B3);
            A1 = ei_pload(&blA[7*PacketSize]);
            B3 = ei_pload(&blB[15*PacketSize]);
            CJMADD(A0,B0,C0,T0);
            CJMADD(A1,B0,C4,B0);
            CJMADD(A0,B1,C1,T0);
            CJMADD(A1,B1,C5,B1);
            CJMADD(A0,B2,C2,T0);
            CJMADD(A1,B2,C6,B2);
            CJMADD(A0,B3,C3,T0);
            CJMADD(A1,B3,C7,B3);
          }

          blB += 4*nr*PacketSize;
          blA += 4*mr;
        }
        // process remaining peeled loop
        for(Index k=peeled_kc; k<depth; k++)
        {
          if(nr==2)
          {
            PacketType B0, T0, A0, A1;

            A0 = ei_pload(&blA[0*PacketSize]);
            A1 = ei_pload(&blA[1*PacketSize]);
            B0 = ei_pload(&blB[0*PacketSize]);
            CJMADD(A0,B0,C0,T0);
            CJMADD(A1,B0,C4,B0);
            B0 = ei_pload(&blB[1*PacketSize]);
            CJMADD(A0,B0,C1,T0);
            CJMADD(A1,B0,C5,B0);
          }
          else
          {
            PacketType B0, B1, B2, B3, A0, A1, T0;

            A0 = ei_pload(&blA[0*PacketSize]);
            A1 = ei_pload(&blA[1*PacketSize]);
            B0 = ei_pload(&blB[0*PacketSize]);
            B1 = ei_pload(&blB[1*PacketSize]);

            CJMADD(A0,B0,C0,T0);
            B2 = ei_pload(&blB[2*PacketSize]);
            CJMADD(A1,B0,C4,B0);
            B3 = ei_pload(&blB[3*PacketSize]);
            CJMADD(A0,B1,C1,T0);
            CJMADD(A1,B1,C5,B1);
            CJMADD(A0,B2,C2,T0);
            CJMADD(A1,B2,C6,B2);
            CJMADD(A0,B3,C3,T0);
            CJMADD(A1,B3,C7,B3);
          }

          blB += nr*PacketSize;
          blA += mr;
        }

        PacketType R0, R1, R2, R3, R4, R5, R6, R7;

                  R0 = ei_ploadu(r0);
                  R1 = ei_ploadu(r1);
        if(nr==4) R2 = ei_ploadu(r2);
        if(nr==4) R3 = ei_ploadu(r3);
                  R4 = ei_ploadu(r0 + PacketSize);
                  R5 = ei_ploadu(r1 + PacketSize);
        if(nr==4) R6 = ei_ploadu(r2 + PacketSize);
        if(nr==4) R7 = ei_ploadu(r3 + PacketSize);

                  C0 = ei_padd(R0, C0);
                  C1 = ei_padd(R1, C1);
        if(nr==4) C2 = ei_padd(R2, C2);
        if(nr==4) C3 = ei_padd(R3, C3);
                  C4 = ei_padd(R4, C4);
                  C5 = ei_padd(R5, C5);
        if(nr==4) C6 = ei_padd(R6, C6);
        if(nr==4) C7 = ei_padd(R7, C7);

                  ei_pstoreu(r0, C0);
                  ei_pstoreu(r1, C1);
        if(nr==4) ei_pstoreu(r2, C2);
        if(nr==4) ei_pstoreu(r3, C3);
                  ei_pstoreu(r0 + PacketSize, C4);
                  ei_pstoreu(r1 + PacketSize, C5);
        if(nr==4) ei_pstoreu(r2 + PacketSize, C6);
        if(nr==4) ei_pstoreu(r3 + PacketSize, C7);
      }
      if(rows-peeled_mc>=PacketSize)
      {
        Index i = peeled_mc;
        const Scalar* blA = &blockA[i*strideA+offsetA*PacketSize];
        ei_prefetch(&blA[0]);

        // gets res block as register
        PacketType C0, C1, C2, C3;
                  C0 = ei_ploadu(&res[(j2+0)*resStride + i]);
                  C1 = ei_ploadu(&res[(j2+1)*resStride + i]);
        if(nr==4) C2 = ei_ploadu(&res[(j2+2)*resStride + i]);
        if(nr==4) C3 = ei_ploadu(&res[(j2+3)*resStride + i]);

        // performs "inner" product
        const Scalar* blB = unpackedB;
        for(Index k=0; k<peeled_kc; k+=4)
        {
          if(nr==2)
          {
            PacketType B0, B1, A0;

            A0 = ei_pload(&blA[0*PacketSize]);
            B0 = ei_pload(&blB[0*PacketSize]);
            B1 = ei_pload(&blB[1*PacketSize]);
            CJMADD(A0,B0,C0,B0);
            B0 = ei_pload(&blB[2*PacketSize]);
            CJMADD(A0,B1,C1,B1);
            A0 = ei_pload(&blA[1*PacketSize]);
            B1 = ei_pload(&blB[3*PacketSize]);
            CJMADD(A0,B0,C0,B0);
            B0 = ei_pload(&blB[4*PacketSize]);
            CJMADD(A0,B1,C1,B1);
            A0 = ei_pload(&blA[2*PacketSize]);
            B1 = ei_pload(&blB[5*PacketSize]);
            CJMADD(A0,B0,C0,B0);
            B0 = ei_pload(&blB[6*PacketSize]);
            CJMADD(A0,B1,C1,B1);
            A0 = ei_pload(&blA[3*PacketSize]);
            B1 = ei_pload(&blB[7*PacketSize]);
            CJMADD(A0,B0,C0,B0);
            CJMADD(A0,B1,C1,B1);
          }
          else
          {
            PacketType B0, B1, B2, B3, A0;

            A0 = ei_pload(&blA[0*PacketSize]);
            B0 = ei_pload(&blB[0*PacketSize]);
            B1 = ei_pload(&blB[1*PacketSize]);

            CJMADD(A0,B0,C0,B0);
            B2 = ei_pload(&blB[2*PacketSize]);
            B3 = ei_pload(&blB[3*PacketSize]);
            B0 = ei_pload(&blB[4*PacketSize]);
            CJMADD(A0,B1,C1,B1);
            B1 = ei_pload(&blB[5*PacketSize]);
            CJMADD(A0,B2,C2,B2);
            B2 = ei_pload(&blB[6*PacketSize]);
            CJMADD(A0,B3,C3,B3);
            A0 = ei_pload(&blA[1*PacketSize]);
            B3 = ei_pload(&blB[7*PacketSize]);
            CJMADD(A0,B0,C0,B0);
            B0 = ei_pload(&blB[8*PacketSize]);
            CJMADD(A0,B1,C1,B1);
            B1 = ei_pload(&blB[9*PacketSize]);
            CJMADD(A0,B2,C2,B2);
            B2 = ei_pload(&blB[10*PacketSize]);
            CJMADD(A0,B3,C3,B3);
            A0 = ei_pload(&blA[2*PacketSize]);
            B3 = ei_pload(&blB[11*PacketSize]);

            CJMADD(A0,B0,C0,B0);
            B0 = ei_pload(&blB[12*PacketSize]);
            CJMADD(A0,B1,C1,B1);
            B1 = ei_pload(&blB[13*PacketSize]);
            CJMADD(A0,B2,C2,B2);
            B2 = ei_pload(&blB[14*PacketSize]);
            CJMADD(A0,B3,C3,B3);

            A0 = ei_pload(&blA[3*PacketSize]);
            B3 = ei_pload(&blB[15*PacketSize]);
            CJMADD(A0,B0,C0,B0);
            CJMADD(A0,B1,C1,B1);
            CJMADD(A0,B2,C2,B2);
            CJMADD(A0,B3,C3,B3);
          }

          blB += 4*nr*PacketSize;
          blA += 4*PacketSize;
        }
        // process remaining peeled loop
        for(Index k=peeled_kc; k<depth; k++)
        {
          if(nr==2)
          {
            PacketType B0, T0, A0;

            A0 = ei_pload(&blA[0*PacketSize]);
            B0 = ei_pload(&blB[0*PacketSize]);
            CJMADD(A0,B0,C0,T0);
            B0 = ei_pload(&blB[1*PacketSize]);
            CJMADD(A0,B0,C1,T0);
          }
          else
          {
            PacketType B0, B1, B2, B3, A0;
            PacketType T0, T1;

            A0 = ei_pload(&blA[0*PacketSize]);
            B0 = ei_pload(&blB[0*PacketSize]);
            B1 = ei_pload(&blB[1*PacketSize]);
            B2 = ei_pload(&blB[2*PacketSize]);
            B3 = ei_pload(&blB[3*PacketSize]);

            CJMADD(A0,B0,C0,T0);
            CJMADD(A0,B1,C1,T1);
            CJMADD(A0,B2,C2,T0);
            CJMADD(A0,B3,C3,T1);
          }

          blB += nr*PacketSize;
          blA += PacketSize;
        }

                  ei_pstoreu(&res[(j2+0)*resStride + i], C0);
                  ei_pstoreu(&res[(j2+1)*resStride + i], C1);
        if(nr==4) ei_pstoreu(&res[(j2+2)*resStride + i], C2);
        if(nr==4) ei_pstoreu(&res[(j2+3)*resStride + i], C3);
      }
      for(Index i=peeled_mc2; i<rows; i++)
      {
        const Scalar* blA = &blockA[i*strideA+offsetA];
        ei_prefetch(&blA[0]);

        // gets a 1 x nr res block as registers
        Scalar C0(0), C1(0), C2(0), C3(0);
        // TODO directly use blockB ???
        const Scalar* blB = unpackedB;//&blockB[j2*strideB+offsetB*nr];
        for(Index k=0; k<depth; k++)
        {
          if(nr==2)
          {
            Scalar B0, T0, A0;

            A0 = blA[k];
            B0 = blB[0*PacketSize];
            CJMADD(A0,B0,C0,T0);
            B0 = blB[1*PacketSize];
            CJMADD(A0,B0,C1,T0);
          }
          else
          {
            Scalar B0, B1, B2, B3, A0;
            Scalar T0, T1;

            A0 = blA[k];
            B0 = blB[0*PacketSize];
            B1 = blB[1*PacketSize];
            B2 = blB[2*PacketSize];
            B3 = blB[3*PacketSize];

            CJMADD(A0,B0,C0,T0);
            CJMADD(A0,B1,C1,T1);
            CJMADD(A0,B2,C2,T0);
            CJMADD(A0,B3,C3,T1);
          }

          blB += nr*PacketSize;
        }
        res[(j2+0)*resStride + i] += C0;
        res[(j2+1)*resStride + i] += C1;
        if(nr==4) res[(j2+2)*resStride + i] += C2;
        if(nr==4) res[(j2+3)*resStride + i] += C3;
      }
    }

    // process remaining rhs/res columns one at a time
    // => do the same but with nr==1
    for(Index j2=packet_cols; j2<cols; j2++)
    {
      // unpack B
      {
        const Scalar* blB = &blockB[j2*strideB+offsetB];
        for(Index k=0; k<depth; k++)
          ei_pstore(&unpackedB[k*PacketSize], ei_pset1(blB[k]));
      }

      for(Index i=0; i<peeled_mc; i+=mr)
      {
        const Scalar* blA = &blockA[i*strideA+offsetA*mr];
        ei_prefetch(&blA[0]);

        // TODO move the res loads to the stores

        // get res block as registers
        PacketType C0, C4;
        C0 = ei_ploadu(&res[(j2+0)*resStride + i]);
        C4 = ei_ploadu(&res[(j2+0)*resStride + i + PacketSize]);

        const Scalar* blB = unpackedB;
        for(Index k=0; k<depth; k++)
        {
          PacketType B0, A0, A1, T0, T1;

          A0 = ei_pload(&blA[0*PacketSize]);
          A1 = ei_pload(&blA[1*PacketSize]);
          B0 = ei_pload(&blB[0*PacketSize]);
          CJMADD(A0,B0,C0,T0);
          CJMADD(A1,B0,C4,T1);

          blB += PacketSize;
          blA += mr;
        }

        ei_pstoreu(&res[(j2+0)*resStride + i], C0);
        ei_pstoreu(&res[(j2+0)*resStride + i + PacketSize], C4);
      }
      if(rows-peeled_mc>=PacketSize)
      {
        Index i = peeled_mc;
        const Scalar* blA = &blockA[i*strideA+offsetA*PacketSize];
        ei_prefetch(&blA[0]);

        PacketType C0 = ei_ploadu(&res[(j2+0)*resStride + i]);

        const Scalar* blB = unpackedB;
        for(Index k=0; k<depth; k++)
        {
          C0 = cj.pmadd(ei_pload(blA), ei_pload(blB), C0);
          blB += PacketSize;
          blA += PacketSize;
        }

        ei_pstoreu(&res[(j2+0)*resStride + i], C0);
      }
      for(Index i=peeled_mc2; i<rows; i++)
      {
        const Scalar* blA = &blockA[i*strideA+offsetA];
        ei_prefetch(&blA[0]);

        // gets a 1 x 1 res block as registers
        Scalar C0(0);
        // FIXME directly use blockB ??
        const Scalar* blB = unpackedB;
        for(Index k=0; k<depth; k++)
          C0 = cj.pmadd(blA[k], blB[k*PacketSize], C0);
        res[(j2+0)*resStride + i] += C0;
      }
    }
  }
};

#undef CJMADD

// pack a block of the lhs
// The travesal is as follow (mr==4):
//   0  4  8 12 ...
//   1  5  9 13 ...
//   2  6 10 14 ...
//   3  7 11 15 ...
//
//  16 20 24 28 ...
//  17 21 25 29 ...
//  18 22 26 30 ...
//  19 23 27 31 ...
//
//  32 33 34 35 ...
//  36 36 38 39 ...
template<typename Scalar, typename Index, int mr, int StorageOrder, bool Conjugate, bool PanelMode>
struct ei_gemm_pack_lhs
{
  void operator()(Scalar* blockA, const Scalar* EIGEN_RESTRICT _lhs, Index lhsStride, Index depth, Index rows,
                  Index stride=0, Index offset=0)
  {
    enum { PacketSize = ei_packet_traits<Scalar>::size };
    ei_assert(((!PanelMode) && stride==0 && offset==0) || (PanelMode && stride>=depth && offset<=stride));
    ei_conj_if<NumTraits<Scalar>::IsComplex && Conjugate> cj;
    ei_const_blas_data_mapper<Scalar, Index, StorageOrder> lhs(_lhs,lhsStride);
    Index count = 0;
    Index peeled_mc = (rows/mr)*mr;
    for(Index i=0; i<peeled_mc; i+=mr)
    {
      if(PanelMode) count += mr * offset;
      for(Index k=0; k<depth; k++)
        for(Index w=0; w<mr; w++)
          blockA[count++] = cj(lhs(i+w, k));
      if(PanelMode) count += mr * (stride-offset-depth);
    }
    if(rows-peeled_mc>=PacketSize)
    {
      if(PanelMode) count += PacketSize*offset;
      for(Index k=0; k<depth; k++)
        for(Index w=0; w<PacketSize; w++)
          blockA[count++] = cj(lhs(peeled_mc+w, k));
      if(PanelMode) count += PacketSize * (stride-offset-depth);
      peeled_mc += PacketSize;
    }
    for(Index i=peeled_mc; i<rows; i++)
    {
      if(PanelMode) count += offset;
      for(Index k=0; k<depth; k++)
        blockA[count++] = cj(lhs(i, k));
      if(PanelMode) count += (stride-offset-depth);
    }
  }
};

// copy a complete panel of the rhs
// this version is optimized for column major matrices
// The traversal order is as follow: (nr==4):
//  0  1  2  3   12 13 14 15   24 27
//  4  5  6  7   16 17 18 19   25 28
//  8  9 10 11   20 21 22 23   26 29
//  .  .  .  .    .  .  .  .    .  .
template<typename Scalar, typename Index, int nr, bool PanelMode>
struct ei_gemm_pack_rhs<Scalar, Index, nr, ColMajor, PanelMode>
{
  typedef typename ei_packet_traits<Scalar>::type Packet;
  enum { PacketSize = ei_packet_traits<Scalar>::size };
  void operator()(Scalar* blockB, const Scalar* rhs, Index rhsStride, Scalar alpha, Index depth, Index cols,
                  Index stride=0, Index offset=0)
  {
    ei_assert(((!PanelMode) && stride==0 && offset==0) || (PanelMode && stride>=depth && offset<=stride));
    bool hasAlpha = alpha != Scalar(1);
    Index packet_cols = (cols/nr) * nr;
    Index count = 0;
    for(Index j2=0; j2<packet_cols; j2+=nr)
    {
      // skip what we have before
      if(PanelMode) count += nr * offset;
      const Scalar* b0 = &rhs[(j2+0)*rhsStride];
      const Scalar* b1 = &rhs[(j2+1)*rhsStride];
      const Scalar* b2 = &rhs[(j2+2)*rhsStride];
      const Scalar* b3 = &rhs[(j2+3)*rhsStride];
      if (hasAlpha)
        for(Index k=0; k<depth; k++)
        {
                    blockB[count+0] = alpha*b0[k];
                    blockB[count+1] = alpha*b1[k];
          if(nr==4) blockB[count+2] = alpha*b2[k];
          if(nr==4) blockB[count+3] = alpha*b3[k];
          count += nr;
        }
      else
        for(Index k=0; k<depth; k++)
        {
                    blockB[count+0] = b0[k];
                    blockB[count+1] = b1[k];
          if(nr==4) blockB[count+2] = b2[k];
          if(nr==4) blockB[count+3] = b3[k];
          count += nr;
        }
      // skip what we have after
      if(PanelMode) count += nr * (stride-offset-depth);
    }

    // copy the remaining columns one at a time (nr==1)
    for(Index j2=packet_cols; j2<cols; ++j2)
    {
      if(PanelMode) count += offset;
      const Scalar* b0 = &rhs[(j2+0)*rhsStride];
      if (hasAlpha)
        for(Index k=0; k<depth; k++)
        {
          blockB[count] = alpha*b0[k];
          count += 1;
        }
      else
        for(Index k=0; k<depth; k++)
        {
          blockB[count] = b0[k];
          count += 1;
        }
      if(PanelMode) count += (stride-offset-depth);
    }
  }
};

// this version is optimized for row major matrices
template<typename Scalar, typename Index, int nr, bool PanelMode>
struct ei_gemm_pack_rhs<Scalar, Index, nr, RowMajor, PanelMode>
{
  enum { PacketSize = ei_packet_traits<Scalar>::size };
  void operator()(Scalar* blockB, const Scalar* rhs, Index rhsStride, Scalar alpha, Index depth, Index cols,
                  Index stride=0, Index offset=0)
  {
    ei_assert(((!PanelMode) && stride==0 && offset==0) || (PanelMode && stride>=depth && offset<=stride));
    bool hasAlpha = alpha != Scalar(1);
    Index packet_cols = (cols/nr) * nr;
    Index count = 0;
    for(Index j2=0; j2<packet_cols; j2+=nr)
    {
      // skip what we have before
      if(PanelMode) count += nr * offset;
      if (hasAlpha)
      {
        for(Index k=0; k<depth; k++)
        {
          const Scalar* b0 = &rhs[k*rhsStride + j2];
                    blockB[count+0] = alpha*b0[0];
                    blockB[count+1] = alpha*b0[1];
          if(nr==4) blockB[count+2] = alpha*b0[2];
          if(nr==4) blockB[count+3] = alpha*b0[3];
          count += nr;
        }
      }
      else
      {
        for(Index k=0; k<depth; k++)
        {
          const Scalar* b0 = &rhs[k*rhsStride + j2];
                    blockB[count+0] = b0[0];
                    blockB[count+1] = b0[1];
          if(nr==4) blockB[count+2] = b0[2];
          if(nr==4) blockB[count+3] = b0[3];
          count += nr;
        }
      }
      // skip what we have after
      if(PanelMode) count += nr * (stride-offset-depth);
    }
    // copy the remaining columns one at a time (nr==1)
    for(Index j2=packet_cols; j2<cols; ++j2)
    {
      if(PanelMode) count += offset;
      const Scalar* b0 = &rhs[j2];
      for(Index k=0; k<depth; k++)
      {
        blockB[count] = alpha*b0[k*rhsStride];
        count += 1;
      }
      if(PanelMode) count += stride-offset-depth;
    }
  }
};

#endif // EIGEN_GENERAL_BLOCK_PANEL_H
