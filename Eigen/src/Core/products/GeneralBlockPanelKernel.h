// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2009 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_GENERAL_BLOCK_PANEL_H
#define EIGEN_GENERAL_BLOCK_PANEL_H

#ifdef USE_IACA
#include "iacaMarks.h"
#else
#define IACA_START
#define IACA_END
#endif

namespace Eigen { 
  
namespace internal {

template<typename _LhsScalar, typename _RhsScalar, bool _ConjLhs=false, bool _ConjRhs=false>
class gebp_traits;


/** \internal \returns b if a<=0, and returns a otherwise. */
inline std::ptrdiff_t manage_caching_sizes_helper(std::ptrdiff_t a, std::ptrdiff_t b)
{
  return a<=0 ? b : a;
}

/** \internal */
inline void manage_caching_sizes(Action action, std::ptrdiff_t* l1=0, std::ptrdiff_t* l2=0)
{
  static std::ptrdiff_t m_l1CacheSize = 0;
  static std::ptrdiff_t m_l2CacheSize = 0;
  if(m_l2CacheSize==0)
  {
    m_l1CacheSize = manage_caching_sizes_helper(queryL1CacheSize(),8 * 1024);
    m_l2CacheSize = manage_caching_sizes_helper(queryTopLevelCacheSize(),1*1024*1024);
  }
  
  if(action==SetAction)
  {
    // set the cpu cache size and cache all block sizes from a global cache size in byte
    eigen_internal_assert(l1!=0 && l2!=0);
    m_l1CacheSize = *l1;
    m_l2CacheSize = *l2;
  }
  else if(action==GetAction)
  {
    eigen_internal_assert(l1!=0 && l2!=0);
    *l1 = m_l1CacheSize;
    *l2 = m_l2CacheSize;
  }
  else
  {
    eigen_internal_assert(false);
  }
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
  * - the register level blocking sizes defined by gebp_traits,
  * - the number of scalars that fit into a packet (when vectorization is enabled).
  *
  * \sa setCpuCacheSizes */
template<typename LhsScalar, typename RhsScalar, int KcFactor, typename SizeType>
void computeProductBlockingSizes(SizeType& k, SizeType& m, SizeType& n)
{
  EIGEN_UNUSED_VARIABLE(n);
  // Explanations:
  // Let's recall the product algorithms form kc x nc horizontal panels B' on the rhs and
  // mc x kc blocks A' on the lhs. A' has to fit into L2 cache. Moreover, B' is processed
  // per kc x nr vertical small panels where nr is the blocking size along the n dimension
  // at the register level. For vectorization purpose, these small vertical panels are unpacked,
  // e.g., each coefficient is replicated to fit a packet. This small vertical panel has to
  // stay in L1 cache.
  std::ptrdiff_t l1, l2;

  typedef gebp_traits<LhsScalar,RhsScalar> Traits;
  enum {
    kdiv = KcFactor * 2 * Traits::nr
         * Traits::RhsProgress * sizeof(RhsScalar),
    mr = gebp_traits<LhsScalar,RhsScalar>::mr,
    mr_mask = (0xffffffff/mr)*mr
  };

  manage_caching_sizes(GetAction, &l1, &l2);

//   k = std::min<SizeType>(k, l1/kdiv);
//   SizeType _m = k>0 ? l2/(4 * sizeof(LhsScalar) * k) : 0;
//   if(_m<m) m = _m & mr_mask;
  
  // In unit tests we do not want to use extra large matrices,
  // so we reduce the block size to check the blocking strategy is not flawed
#ifndef EIGEN_DEBUG_SMALL_PRODUCT_BLOCKS
//   k = std::min<SizeType>(k,240);
//   n = std::min<SizeType>(n,3840/sizeof(RhsScalar));
//   m = std::min<SizeType>(m,3840/sizeof(RhsScalar));
  
  k = std::min<SizeType>(k,sizeof(LhsScalar)<=4 ? 360 : 240);
  n = std::min<SizeType>(n,3840/sizeof(RhsScalar));
  m = std::min<SizeType>(m,3840/sizeof(RhsScalar));
#else
  k = std::min<SizeType>(k,24);
  n = std::min<SizeType>(n,384/sizeof(RhsScalar));
  m = std::min<SizeType>(m,384/sizeof(RhsScalar));
#endif
}

template<typename LhsScalar, typename RhsScalar, typename SizeType>
inline void computeProductBlockingSizes(SizeType& k, SizeType& m, SizeType& n)
{
  computeProductBlockingSizes<LhsScalar,RhsScalar,1>(k, m, n);
}

#ifdef EIGEN_HAS_FUSE_CJMADD
  #define MADD(CJ,A,B,C,T)  C = CJ.pmadd(A,B,C);
#else

  // FIXME (a bit overkill maybe ?)

  template<typename CJ, typename A, typename B, typename C, typename T> struct gebp_madd_selector {
    EIGEN_ALWAYS_INLINE static void run(const CJ& cj, A& a, B& b, C& c, T& /*t*/)
    {
      c = cj.pmadd(a,b,c);
    }
  };

  template<typename CJ, typename T> struct gebp_madd_selector<CJ,T,T,T,T> {
    EIGEN_ALWAYS_INLINE static void run(const CJ& cj, T& a, T& b, T& c, T& t)
    {
      t = b; t = cj.pmul(a,t); c = padd(c,t);
    }
  };

  template<typename CJ, typename A, typename B, typename C, typename T>
  EIGEN_STRONG_INLINE void gebp_madd(const CJ& cj, A& a, B& b, C& c, T& t)
  {
    gebp_madd_selector<CJ,A,B,C,T>::run(cj,a,b,c,t);
  }

  #define MADD(CJ,A,B,C,T)  gebp_madd(CJ,A,B,C,T);
//   #define MADD(CJ,A,B,C,T)  T = B; T = CJ.pmul(A,T); C = padd(C,T);
#endif

/* Vectorization logic
 *  real*real: unpack rhs to constant packets, ...
 * 
 *  cd*cd : unpack rhs to (b_r,b_r), (b_i,b_i), mul to get (a_r b_r,a_i b_r) (a_r b_i,a_i b_i),
 *          storing each res packet into two packets (2x2),
 *          at the end combine them: swap the second and addsub them 
 *  cf*cf : same but with 2x4 blocks
 *  cplx*real : unpack rhs to constant packets, ...
 *  real*cplx : load lhs as (a0,a0,a1,a1), and mul as usual
 */
template<typename _LhsScalar, typename _RhsScalar, bool _ConjLhs, bool _ConjRhs>
class gebp_traits
{
public:
  typedef _LhsScalar LhsScalar;
  typedef _RhsScalar RhsScalar;
  typedef typename scalar_product_traits<LhsScalar, RhsScalar>::ReturnType ResScalar;

  enum {
    ConjLhs = _ConjLhs,
    ConjRhs = _ConjRhs,
    Vectorizable = packet_traits<LhsScalar>::Vectorizable && packet_traits<RhsScalar>::Vectorizable,
    LhsPacketSize = Vectorizable ? packet_traits<LhsScalar>::size : 1,
    RhsPacketSize = Vectorizable ? packet_traits<RhsScalar>::size : 1,
    ResPacketSize = Vectorizable ? packet_traits<ResScalar>::size : 1,
    
    NumberOfRegisters = EIGEN_ARCH_DEFAULT_NUMBER_OF_REGISTERS,

    // register block size along the N direction must be 1 or 4
    nr = 4,

    // register block size along the M direction (currently, this one cannot be modified)
#if defined(EIGEN_HAS_FUSED_MADD) && !defined(EIGEN_VECTORIZE_ALTIVEC)
    // we assume 16 registers
    mr = 3*LhsPacketSize,
#else
    mr = (EIGEN_PLAIN_ENUM_MIN(16,NumberOfRegisters)/2/nr)*LhsPacketSize,
#endif
    
    LhsProgress = LhsPacketSize,
    RhsProgress = 1
  };

  typedef typename packet_traits<LhsScalar>::type  _LhsPacket;
  typedef typename packet_traits<RhsScalar>::type  _RhsPacket;
  typedef typename packet_traits<ResScalar>::type  _ResPacket;

  typedef typename conditional<Vectorizable,_LhsPacket,LhsScalar>::type LhsPacket;
  typedef typename conditional<Vectorizable,_RhsPacket,RhsScalar>::type RhsPacket;
  typedef typename conditional<Vectorizable,_ResPacket,ResScalar>::type ResPacket;

  typedef ResPacket AccPacket;
  
  EIGEN_STRONG_INLINE void initAcc(AccPacket& p)
  {
    p = pset1<ResPacket>(ResScalar(0));
  }
  
  EIGEN_STRONG_INLINE void broadcastRhs(const RhsScalar* b, RhsPacket& b0, RhsPacket& b1, RhsPacket& b2, RhsPacket& b3)
  {
    pbroadcast4(b, b0, b1, b2, b3);
  }
  
//   EIGEN_STRONG_INLINE void broadcastRhs(const RhsScalar* b, RhsPacket& b0, RhsPacket& b1)
//   {
//     pbroadcast2(b, b0, b1);
//   }
  
  template<typename RhsPacketType>
  EIGEN_STRONG_INLINE void loadRhs(const RhsScalar* b, RhsPacketType& dest) const
  {
    dest = pset1<RhsPacketType>(*b);
  }
  
  EIGEN_STRONG_INLINE void loadRhsQuad(const RhsScalar* b, RhsPacket& dest) const
  {
    dest = ploadquad<RhsPacket>(b);
  }

  template<typename LhsPacketType>
  EIGEN_STRONG_INLINE void loadLhs(const LhsScalar* a, LhsPacketType& dest) const
  {
    dest = pload<LhsPacketType>(a);
  }

  template<typename LhsPacketType>
  EIGEN_STRONG_INLINE void loadLhsUnaligned(const LhsScalar* a, LhsPacketType& dest) const
  {
    dest = ploadu<LhsPacketType>(a);
  }

  template<typename LhsPacketType, typename RhsPacketType, typename AccPacketType>
  EIGEN_STRONG_INLINE void madd(const LhsPacketType& a, const RhsPacketType& b, AccPacketType& c, AccPacketType& tmp) const
  {
    // It would be a lot cleaner to call pmadd all the time. Unfortunately if we
    // let gcc allocate the register in which to store the result of the pmul
    // (in the case where there is no FMA) gcc fails to figure out how to avoid
    // spilling register.
#ifdef EIGEN_HAS_FUSED_MADD
    EIGEN_UNUSED_VARIABLE(tmp);
    c = pmadd(a,b,c);
#else
    tmp = b; tmp = pmul(a,tmp); c = padd(c,tmp);
#endif
  }

  EIGEN_STRONG_INLINE void acc(const AccPacket& c, const ResPacket& alpha, ResPacket& r) const
  {
    r = pmadd(c,alpha,r);
  }
  
  template<typename ResPacketHalf>
  EIGEN_STRONG_INLINE void acc(const ResPacketHalf& c, const ResPacketHalf& alpha, ResPacketHalf& r) const
  {
    r = pmadd(c,alpha,r);
  }

protected:
//   conj_helper<LhsScalar,RhsScalar,ConjLhs,ConjRhs> cj;
//   conj_helper<LhsPacket,RhsPacket,ConjLhs,ConjRhs> pcj;
};

template<typename RealScalar, bool _ConjLhs>
class gebp_traits<std::complex<RealScalar>, RealScalar, _ConjLhs, false>
{
public:
  typedef std::complex<RealScalar> LhsScalar;
  typedef RealScalar RhsScalar;
  typedef typename scalar_product_traits<LhsScalar, RhsScalar>::ReturnType ResScalar;

  enum {
    ConjLhs = _ConjLhs,
    ConjRhs = false,
    Vectorizable = packet_traits<LhsScalar>::Vectorizable && packet_traits<RhsScalar>::Vectorizable,
    LhsPacketSize = Vectorizable ? packet_traits<LhsScalar>::size : 1,
    RhsPacketSize = Vectorizable ? packet_traits<RhsScalar>::size : 1,
    ResPacketSize = Vectorizable ? packet_traits<ResScalar>::size : 1,
    
    NumberOfRegisters = EIGEN_ARCH_DEFAULT_NUMBER_OF_REGISTERS,
    nr = 4,
#if defined(EIGEN_HAS_FUSED_MADD) && !defined(EIGEN_VECTORIZE_ALTIVEC)
    // we assume 16 registers
    mr = 3*LhsPacketSize,
#else
    mr = (EIGEN_PLAIN_ENUM_MIN(16,NumberOfRegisters)/2/nr)*LhsPacketSize,
#endif

    LhsProgress = LhsPacketSize,
    RhsProgress = 1
  };

  typedef typename packet_traits<LhsScalar>::type  _LhsPacket;
  typedef typename packet_traits<RhsScalar>::type  _RhsPacket;
  typedef typename packet_traits<ResScalar>::type  _ResPacket;

  typedef typename conditional<Vectorizable,_LhsPacket,LhsScalar>::type LhsPacket;
  typedef typename conditional<Vectorizable,_RhsPacket,RhsScalar>::type RhsPacket;
  typedef typename conditional<Vectorizable,_ResPacket,ResScalar>::type ResPacket;

  typedef ResPacket AccPacket;

  EIGEN_STRONG_INLINE void initAcc(AccPacket& p)
  {
    p = pset1<ResPacket>(ResScalar(0));
  }

  EIGEN_STRONG_INLINE void loadRhs(const RhsScalar* b, RhsPacket& dest) const
  {
    dest = pset1<RhsPacket>(*b);
  }
  
  EIGEN_STRONG_INLINE void loadRhsQuad(const RhsScalar* b, RhsPacket& dest) const
  {
    dest = pset1<RhsPacket>(*b);
  }

  EIGEN_STRONG_INLINE void loadLhs(const LhsScalar* a, LhsPacket& dest) const
  {
    dest = pload<LhsPacket>(a);
  }

  EIGEN_STRONG_INLINE void loadLhsUnaligned(const LhsScalar* a, LhsPacket& dest) const
  {
    dest = ploadu<LhsPacket>(a);
  }

  EIGEN_STRONG_INLINE void broadcastRhs(const RhsScalar* b, RhsPacket& b0, RhsPacket& b1, RhsPacket& b2, RhsPacket& b3)
  {
    pbroadcast4(b, b0, b1, b2, b3);
  }
  
//   EIGEN_STRONG_INLINE void broadcastRhs(const RhsScalar* b, RhsPacket& b0, RhsPacket& b1)
//   {
//     pbroadcast2(b, b0, b1);
//   }

  EIGEN_STRONG_INLINE void madd(const LhsPacket& a, const RhsPacket& b, AccPacket& c, RhsPacket& tmp) const
  {
    madd_impl(a, b, c, tmp, typename conditional<Vectorizable,true_type,false_type>::type());
  }

  EIGEN_STRONG_INLINE void madd_impl(const LhsPacket& a, const RhsPacket& b, AccPacket& c, RhsPacket& tmp, const true_type&) const
  {
#ifdef EIGEN_HAS_FUSED_MADD
    EIGEN_UNUSED_VARIABLE(tmp);
    c.v = pmadd(a.v,b,c.v);
#else
    tmp = b; tmp = pmul(a.v,tmp); c.v = padd(c.v,tmp);
#endif
  }

  EIGEN_STRONG_INLINE void madd_impl(const LhsScalar& a, const RhsScalar& b, ResScalar& c, RhsScalar& /*tmp*/, const false_type&) const
  {
    c += a * b;
  }

  EIGEN_STRONG_INLINE void acc(const AccPacket& c, const ResPacket& alpha, ResPacket& r) const
  {
    r = cj.pmadd(c,alpha,r);
  }

protected:
  conj_helper<ResPacket,ResPacket,ConjLhs,false> cj;
};

template<typename Packet>
struct DoublePacket
{
  Packet first;
  Packet second;
};

template<typename Packet>
DoublePacket<Packet> padd(const DoublePacket<Packet> &a, const DoublePacket<Packet> &b)
{
  DoublePacket<Packet> res;
  res.first  = padd(a.first, b.first);
  res.second = padd(a.second,b.second);
  return res;
}

template<typename Packet>
const DoublePacket<Packet>& predux4(const DoublePacket<Packet> &a)
{
  return a;
}

template<typename Packet> struct unpacket_traits<DoublePacket<Packet> > { typedef DoublePacket<Packet> half; };
// template<typename Packet>
// DoublePacket<Packet> pmadd(const DoublePacket<Packet> &a, const DoublePacket<Packet> &b)
// {
//   DoublePacket<Packet> res;
//   res.first  = padd(a.first, b.first);
//   res.second = padd(a.second,b.second);
//   return res;
// }

template<typename RealScalar, bool _ConjLhs, bool _ConjRhs>
class gebp_traits<std::complex<RealScalar>, std::complex<RealScalar>, _ConjLhs, _ConjRhs >
{
public:
  typedef std::complex<RealScalar>  Scalar;
  typedef std::complex<RealScalar>  LhsScalar;
  typedef std::complex<RealScalar>  RhsScalar;
  typedef std::complex<RealScalar>  ResScalar;
  
  enum {
    ConjLhs = _ConjLhs,
    ConjRhs = _ConjRhs,
    Vectorizable = packet_traits<RealScalar>::Vectorizable
                && packet_traits<Scalar>::Vectorizable,
    RealPacketSize  = Vectorizable ? packet_traits<RealScalar>::size : 1,
    ResPacketSize   = Vectorizable ? packet_traits<ResScalar>::size : 1,
    LhsPacketSize = Vectorizable ? packet_traits<LhsScalar>::size : 1,
    RhsPacketSize = Vectorizable ? packet_traits<RhsScalar>::size : 1,

    // FIXME: should depend on NumberOfRegisters
    nr = 4,
    mr = ResPacketSize,

    LhsProgress = ResPacketSize,
    RhsProgress = 1
  };
  
  typedef typename packet_traits<RealScalar>::type RealPacket;
  typedef typename packet_traits<Scalar>::type     ScalarPacket;
  typedef DoublePacket<RealPacket> DoublePacketType;

  typedef typename conditional<Vectorizable,RealPacket,  Scalar>::type LhsPacket;
  typedef typename conditional<Vectorizable,DoublePacketType,Scalar>::type RhsPacket;
  typedef typename conditional<Vectorizable,ScalarPacket,Scalar>::type ResPacket;
  typedef typename conditional<Vectorizable,DoublePacketType,Scalar>::type AccPacket;
  
  EIGEN_STRONG_INLINE void initAcc(Scalar& p) { p = Scalar(0); }

  EIGEN_STRONG_INLINE void initAcc(DoublePacketType& p)
  {
    p.first   = pset1<RealPacket>(RealScalar(0));
    p.second  = pset1<RealPacket>(RealScalar(0));
  }

  // Scalar path
  EIGEN_STRONG_INLINE void loadRhs(const RhsScalar* b, ResPacket& dest) const
  {
    dest = pset1<ResPacket>(*b);
  }

  // Vectorized path
  EIGEN_STRONG_INLINE void loadRhs(const RhsScalar* b, DoublePacketType& dest) const
  {
    dest.first  = pset1<RealPacket>(real(*b));
    dest.second = pset1<RealPacket>(imag(*b));
  }
  
  EIGEN_STRONG_INLINE void loadRhsQuad(const RhsScalar* b, ResPacket& dest) const
  {
    loadRhs(b,dest);
  }
  EIGEN_STRONG_INLINE void loadRhsQuad(const RhsScalar* b, DoublePacketType& dest) const
  {
    eigen_internal_assert(unpacket_traits<ScalarPacket>::size<=4);
    loadRhs(b,dest);
  }
  
  EIGEN_STRONG_INLINE void broadcastRhs(const RhsScalar* b, RhsPacket& b0, RhsPacket& b1, RhsPacket& b2, RhsPacket& b3)
  {
    // FIXME not sure that's the best way to implement it!
    loadRhs(b+0, b0);
    loadRhs(b+1, b1);
    loadRhs(b+2, b2);
    loadRhs(b+3, b3);
  }
  
  // Vectorized path
  EIGEN_STRONG_INLINE void broadcastRhs(const RhsScalar* b, DoublePacketType& b0, DoublePacketType& b1)
  {
    // FIXME not sure that's the best way to implement it!
    loadRhs(b+0, b0);
    loadRhs(b+1, b1);
  }
  
  // Scalar path
  EIGEN_STRONG_INLINE void broadcastRhs(const RhsScalar* b, RhsScalar& b0, RhsScalar& b1)
  {
    // FIXME not sure that's the best way to implement it!
    loadRhs(b+0, b0);
    loadRhs(b+1, b1);
  }

  // nothing special here
  EIGEN_STRONG_INLINE void loadLhs(const LhsScalar* a, LhsPacket& dest) const
  {
    dest = pload<LhsPacket>((const typename unpacket_traits<LhsPacket>::type*)(a));
  }

  EIGEN_STRONG_INLINE void loadLhsUnaligned(const LhsScalar* a, LhsPacket& dest) const
  {
    dest = ploadu<LhsPacket>((const typename unpacket_traits<LhsPacket>::type*)(a));
  }

  EIGEN_STRONG_INLINE void madd(const LhsPacket& a, const RhsPacket& b, DoublePacketType& c, RhsPacket& /*tmp*/) const
  {
    c.first   = padd(pmul(a,b.first), c.first);
    c.second  = padd(pmul(a,b.second),c.second);
  }

  EIGEN_STRONG_INLINE void madd(const LhsPacket& a, const RhsPacket& b, ResPacket& c, RhsPacket& /*tmp*/) const
  {
    c = cj.pmadd(a,b,c);
  }
  
  EIGEN_STRONG_INLINE void acc(const Scalar& c, const Scalar& alpha, Scalar& r) const { r += alpha * c; }
  
  EIGEN_STRONG_INLINE void acc(const DoublePacketType& c, const ResPacket& alpha, ResPacket& r) const
  {
    // assemble c
    ResPacket tmp;
    if((!ConjLhs)&&(!ConjRhs))
    {
      tmp = pcplxflip(pconj(ResPacket(c.second)));
      tmp = padd(ResPacket(c.first),tmp);
    }
    else if((!ConjLhs)&&(ConjRhs))
    {
      tmp = pconj(pcplxflip(ResPacket(c.second)));
      tmp = padd(ResPacket(c.first),tmp);
    }
    else if((ConjLhs)&&(!ConjRhs))
    {
      tmp = pcplxflip(ResPacket(c.second));
      tmp = padd(pconj(ResPacket(c.first)),tmp);
    }
    else if((ConjLhs)&&(ConjRhs))
    {
      tmp = pcplxflip(ResPacket(c.second));
      tmp = psub(pconj(ResPacket(c.first)),tmp);
    }
    
    r = pmadd(tmp,alpha,r);
  }

protected:
  conj_helper<LhsScalar,RhsScalar,ConjLhs,ConjRhs> cj;
};

template<typename RealScalar, bool _ConjRhs>
class gebp_traits<RealScalar, std::complex<RealScalar>, false, _ConjRhs >
{
public:
  typedef std::complex<RealScalar>  Scalar;
  typedef RealScalar  LhsScalar;
  typedef Scalar      RhsScalar;
  typedef Scalar      ResScalar;

  enum {
    ConjLhs = false,
    ConjRhs = _ConjRhs,
    Vectorizable = packet_traits<RealScalar>::Vectorizable
                && packet_traits<Scalar>::Vectorizable,
    LhsPacketSize = Vectorizable ? packet_traits<LhsScalar>::size : 1,
    RhsPacketSize = Vectorizable ? packet_traits<RhsScalar>::size : 1,
    ResPacketSize = Vectorizable ? packet_traits<ResScalar>::size : 1,
    
    NumberOfRegisters = EIGEN_ARCH_DEFAULT_NUMBER_OF_REGISTERS,
    // FIXME: should depend on NumberOfRegisters
    nr = 4,
    mr = (EIGEN_PLAIN_ENUM_MIN(16,NumberOfRegisters)/2/nr)*ResPacketSize,

    LhsProgress = ResPacketSize,
    RhsProgress = 1
  };

  typedef typename packet_traits<LhsScalar>::type  _LhsPacket;
  typedef typename packet_traits<RhsScalar>::type  _RhsPacket;
  typedef typename packet_traits<ResScalar>::type  _ResPacket;

  typedef typename conditional<Vectorizable,_LhsPacket,LhsScalar>::type LhsPacket;
  typedef typename conditional<Vectorizable,_RhsPacket,RhsScalar>::type RhsPacket;
  typedef typename conditional<Vectorizable,_ResPacket,ResScalar>::type ResPacket;

  typedef ResPacket AccPacket;

  EIGEN_STRONG_INLINE void initAcc(AccPacket& p)
  {
    p = pset1<ResPacket>(ResScalar(0));
  }

  EIGEN_STRONG_INLINE void loadRhs(const RhsScalar* b, RhsPacket& dest) const
  {
    dest = pset1<RhsPacket>(*b);
  }
  
  void broadcastRhs(const RhsScalar* b, RhsPacket& b0, RhsPacket& b1, RhsPacket& b2, RhsPacket& b3)
  {
    pbroadcast4(b, b0, b1, b2, b3);
  }
  
//   EIGEN_STRONG_INLINE void broadcastRhs(const RhsScalar* b, RhsPacket& b0, RhsPacket& b1)
//   {
//     // FIXME not sure that's the best way to implement it!
//     b0 = pload1<RhsPacket>(b+0);
//     b1 = pload1<RhsPacket>(b+1);
//   }

  EIGEN_STRONG_INLINE void loadLhs(const LhsScalar* a, LhsPacket& dest) const
  {
    dest = ploaddup<LhsPacket>(a);
  }
  
  EIGEN_STRONG_INLINE void loadRhsQuad(const RhsScalar* b, RhsPacket& dest) const
  {
    eigen_internal_assert(unpacket_traits<RhsPacket>::size<=4);
    loadRhs(b,dest);
  }

  EIGEN_STRONG_INLINE void loadLhsUnaligned(const LhsScalar* a, LhsPacket& dest) const
  {
    dest = ploaddup<LhsPacket>(a);
  }

  EIGEN_STRONG_INLINE void madd(const LhsPacket& a, const RhsPacket& b, AccPacket& c, RhsPacket& tmp) const
  {
    madd_impl(a, b, c, tmp, typename conditional<Vectorizable,true_type,false_type>::type());
  }

  EIGEN_STRONG_INLINE void madd_impl(const LhsPacket& a, const RhsPacket& b, AccPacket& c, RhsPacket& tmp, const true_type&) const
  {
#ifdef EIGEN_HAS_FUSED_MADD
    EIGEN_UNUSED_VARIABLE(tmp);
    c.v = pmadd(a,b.v,c.v);
#else
    tmp = b; tmp.v = pmul(a,tmp.v); c = padd(c,tmp);
#endif
    
  }

  EIGEN_STRONG_INLINE void madd_impl(const LhsScalar& a, const RhsScalar& b, ResScalar& c, RhsScalar& /*tmp*/, const false_type&) const
  {
    c += a * b;
  }

  EIGEN_STRONG_INLINE void acc(const AccPacket& c, const ResPacket& alpha, ResPacket& r) const
  {
    r = cj.pmadd(alpha,c,r);
  }

protected:
  conj_helper<ResPacket,ResPacket,false,ConjRhs> cj;
};

/* optimized GEneral packed Block * packed Panel product kernel
 *
 * Mixing type logic: C += A * B
 *  |  A  |  B  | comments
 *  |real |cplx | no vectorization yet, would require to pack A with duplication
 *  |cplx |real | easy vectorization
 */
template<typename LhsScalar, typename RhsScalar, typename Index, int mr, int nr, bool ConjugateLhs, bool ConjugateRhs>
struct gebp_kernel
{
  typedef gebp_traits<LhsScalar,RhsScalar,ConjugateLhs,ConjugateRhs> Traits;
  typedef typename Traits::ResScalar ResScalar;
  typedef typename Traits::LhsPacket LhsPacket;
  typedef typename Traits::RhsPacket RhsPacket;
  typedef typename Traits::ResPacket ResPacket;
  typedef typename Traits::AccPacket AccPacket;
  
  typedef gebp_traits<RhsScalar,LhsScalar,ConjugateRhs,ConjugateLhs> SwappedTraits;
  typedef typename SwappedTraits::ResScalar SResScalar;
  typedef typename SwappedTraits::LhsPacket SLhsPacket;
  typedef typename SwappedTraits::RhsPacket SRhsPacket;
  typedef typename SwappedTraits::ResPacket SResPacket;
  typedef typename SwappedTraits::AccPacket SAccPacket;
            

  enum {
    Vectorizable  = Traits::Vectorizable,
    LhsProgress   = Traits::LhsProgress,
    RhsProgress   = Traits::RhsProgress,
    ResPacketSize = Traits::ResPacketSize
  };

  EIGEN_DONT_INLINE
  void operator()(ResScalar* res, Index resStride, const LhsScalar* blockA, const RhsScalar* blockB, Index rows, Index depth, Index cols, ResScalar alpha,
                  Index strideA=-1, Index strideB=-1, Index offsetA=0, Index offsetB=0);
};

template<typename LhsScalar, typename RhsScalar, typename Index, int mr, int nr, bool ConjugateLhs, bool ConjugateRhs>
EIGEN_DONT_INLINE
void gebp_kernel<LhsScalar,RhsScalar,Index,mr,nr,ConjugateLhs,ConjugateRhs>
  ::operator()(ResScalar* res, Index resStride, const LhsScalar* blockA, const RhsScalar* blockB, Index rows, Index depth, Index cols, ResScalar alpha,
               Index strideA, Index strideB, Index offsetA, Index offsetB)
  {
    Traits traits;
    SwappedTraits straits;
    
    if(strideA==-1) strideA = depth;
    if(strideB==-1) strideB = depth;
    conj_helper<LhsScalar,RhsScalar,ConjugateLhs,ConjugateRhs> cj;
    Index packet_cols4 = nr>=4 ? (cols/4) * 4 : 0;
    const Index peeled_mc3 = mr>=3*Traits::LhsProgress ? (rows/(3*LhsProgress))*(3*LhsProgress) : 0;
    const Index peeled_mc2 = mr>=2*Traits::LhsProgress ? peeled_mc3+((rows-peeled_mc3)/(2*LhsProgress))*(2*LhsProgress) : 0;
    const Index peeled_mc1 = mr>=1*Traits::LhsProgress ? (rows/(1*LhsProgress))*(1*LhsProgress) : 0;
    enum { pk = 8 }; // NOTE Such a large peeling factor is important for large matrices (~ +5% when >1000 on Haswell)
    const Index peeled_kc  = depth & ~(pk-1);
    const Index prefetch_res_offset = 32/sizeof(ResScalar);    
//     const Index depth2     = depth & ~1;
    
    //---------- Process 3 * LhsProgress rows at once ----------
    // This corresponds to 3*LhsProgress x nr register blocks.
    // Usually, make sense only with FMA
    if(mr>=3*Traits::LhsProgress)
    {
      // loops on each largest micro horizontal panel of lhs (3*Traits::LhsProgress x depth)
      for(Index i=0; i<peeled_mc3; i+=3*Traits::LhsProgress)
      {
        // loops on each largest micro vertical panel of rhs (depth * nr)
        for(Index j2=0; j2<packet_cols4; j2+=nr)
        {
          // We select a 3*Traits::LhsProgress x nr micro block of res which is entirely
          // stored into 3 x nr registers.
          
          const LhsScalar* blA = &blockA[i*strideA+offsetA*(3*Traits::LhsProgress)];
          prefetch(&blA[0]);

          // gets res block as register
          AccPacket C0, C1, C2,  C3,
                    C4, C5, C6,  C7,
                    C8, C9, C10, C11;
          traits.initAcc(C0);  traits.initAcc(C1);  traits.initAcc(C2);  traits.initAcc(C3);
          traits.initAcc(C4);  traits.initAcc(C5);  traits.initAcc(C6);  traits.initAcc(C7);
          traits.initAcc(C8);  traits.initAcc(C9);  traits.initAcc(C10); traits.initAcc(C11);

          ResScalar* r0 = &res[(j2+0)*resStride + i];
          ResScalar* r1 = &res[(j2+1)*resStride + i];
          ResScalar* r2 = &res[(j2+2)*resStride + i];
          ResScalar* r3 = &res[(j2+3)*resStride + i];
          
          internal::prefetch(r0);
          internal::prefetch(r1);
          internal::prefetch(r2);
          internal::prefetch(r3);

          // performs "inner" products
          const RhsScalar* blB = &blockB[j2*strideB+offsetB*nr];
          prefetch(&blB[0]);
          LhsPacket A0, A1;
          
          for(Index k=0; k<peeled_kc; k+=pk)
          {
            EIGEN_ASM_COMMENT("begin gegp micro kernel 3p x 4");
            RhsPacket B_0, T0;
            LhsPacket A2;

#define EIGEN_GEBGP_ONESTEP(K) \
            internal::prefetch(blA+(3*K+16)*LhsProgress); \
            traits.loadLhs(&blA[(0+3*K)*LhsProgress], A0);  \
            traits.loadLhs(&blA[(1+3*K)*LhsProgress], A1);  \
            traits.loadLhs(&blA[(2+3*K)*LhsProgress], A2);  \
            traits.loadRhs(&blB[(0+4*K)*RhsProgress], B_0); \
            traits.madd(A0, B_0, C0, T0); \
            traits.madd(A1, B_0, C4, T0); \
            traits.madd(A2, B_0, C8, B_0); \
            traits.loadRhs(&blB[1+4*K*RhsProgress], B_0); \
            traits.madd(A0, B_0, C1, T0); \
            traits.madd(A1, B_0, C5, T0); \
            traits.madd(A2, B_0, C9, B_0); \
            traits.loadRhs(&blB[2+4*K*RhsProgress], B_0); \
            traits.madd(A0, B_0, C2,  T0); \
            traits.madd(A1, B_0, C6,  T0); \
            traits.madd(A2, B_0, C10, B_0); \
            traits.loadRhs(&blB[3+4*K*RhsProgress], B_0); \
            traits.madd(A0, B_0, C3 , T0); \
            traits.madd(A1, B_0, C7,  T0); \
            traits.madd(A2, B_0, C11, B_0)
        
            internal::prefetch(blB+(48+0));
            EIGEN_GEBGP_ONESTEP(0);
            EIGEN_GEBGP_ONESTEP(1);
            EIGEN_GEBGP_ONESTEP(2);
            EIGEN_GEBGP_ONESTEP(3);
            internal::prefetch(blB+(48+16));
            EIGEN_GEBGP_ONESTEP(4);
            EIGEN_GEBGP_ONESTEP(5);
            EIGEN_GEBGP_ONESTEP(6);
            EIGEN_GEBGP_ONESTEP(7);

            blB += pk*4*RhsProgress;
            blA += pk*3*Traits::LhsProgress;
            IACA_END
          }
          // process remaining peeled loop
          for(Index k=peeled_kc; k<depth; k++)
          {
            RhsPacket B_0, T0;
            LhsPacket A2;
            EIGEN_GEBGP_ONESTEP(0);
            blB += 4*RhsProgress;
            blA += 3*Traits::LhsProgress;
          }
  #undef EIGEN_GEBGP_ONESTEP
  
          ResPacket R0, R1, R2;
          ResPacket alphav = pset1<ResPacket>(alpha);
          
          R0 = ploadu<ResPacket>(r0+0*Traits::ResPacketSize);
          R1 = ploadu<ResPacket>(r0+1*Traits::ResPacketSize);
          R2 = ploadu<ResPacket>(r0+2*Traits::ResPacketSize);
          traits.acc(C0, alphav, R0);
          traits.acc(C4, alphav, R1);
          traits.acc(C8, alphav, R2);
          pstoreu(r0+0*Traits::ResPacketSize, R0);
          pstoreu(r0+1*Traits::ResPacketSize, R1);
          pstoreu(r0+2*Traits::ResPacketSize, R2);
          
          R0 = ploadu<ResPacket>(r1+0*Traits::ResPacketSize);
          R1 = ploadu<ResPacket>(r1+1*Traits::ResPacketSize);
          R2 = ploadu<ResPacket>(r1+2*Traits::ResPacketSize);
          traits.acc(C1, alphav, R0);
          traits.acc(C5, alphav, R1);
          traits.acc(C9, alphav, R2);
          pstoreu(r1+0*Traits::ResPacketSize, R0);
          pstoreu(r1+1*Traits::ResPacketSize, R1);
          pstoreu(r1+2*Traits::ResPacketSize, R2);
          
          R0 = ploadu<ResPacket>(r2+0*Traits::ResPacketSize);
          R1 = ploadu<ResPacket>(r2+1*Traits::ResPacketSize);
          R2 = ploadu<ResPacket>(r2+2*Traits::ResPacketSize);
          traits.acc(C2, alphav, R0);
          traits.acc(C6, alphav, R1);
          traits.acc(C10, alphav, R2);
          pstoreu(r2+0*Traits::ResPacketSize, R0);
          pstoreu(r2+1*Traits::ResPacketSize, R1);
          pstoreu(r2+2*Traits::ResPacketSize, R2);
          
          R0 = ploadu<ResPacket>(r3+0*Traits::ResPacketSize);
          R1 = ploadu<ResPacket>(r3+1*Traits::ResPacketSize);
          R2 = ploadu<ResPacket>(r3+2*Traits::ResPacketSize);
          traits.acc(C3, alphav, R0);
          traits.acc(C7, alphav, R1);
          traits.acc(C11, alphav, R2);
          pstoreu(r3+0*Traits::ResPacketSize, R0);
          pstoreu(r3+1*Traits::ResPacketSize, R1);
          pstoreu(r3+2*Traits::ResPacketSize, R2);
        }
        
        // Deal with remaining columns of the rhs
        for(Index j2=packet_cols4; j2<cols; j2++)
        {
          // One column at a time
          const LhsScalar* blA = &blockA[i*strideA+offsetA*(3*Traits::LhsProgress)];
          prefetch(&blA[0]);

          // gets res block as register
          AccPacket C0, C4, C8;
          traits.initAcc(C0);
          traits.initAcc(C4);
          traits.initAcc(C8);

          ResScalar* r0 = &res[(j2+0)*resStride + i];

          // performs "inner" products
          const RhsScalar* blB = &blockB[j2*strideB+offsetB];
          LhsPacket A0, A1, A2;
          
          for(Index k=0; k<peeled_kc; k+=pk)
          {
            EIGEN_ASM_COMMENT("begin gegp micro kernel 3p x 1");
            RhsPacket B_0;
#define EIGEN_GEBGP_ONESTEP(K) \
            traits.loadLhs(&blA[(0+3*K)*LhsProgress], A0);  \
            traits.loadLhs(&blA[(1+3*K)*LhsProgress], A1);  \
            traits.loadLhs(&blA[(2+3*K)*LhsProgress], A2);  \
            traits.loadRhs(&blB[(0+K)*RhsProgress], B_0);   \
            traits.madd(A0, B_0, C0, B_0); \
            traits.madd(A1, B_0, C4, B_0); \
            traits.madd(A2, B_0, C8, B_0)
        
            EIGEN_GEBGP_ONESTEP(0);
            EIGEN_GEBGP_ONESTEP(1);
            EIGEN_GEBGP_ONESTEP(2);
            EIGEN_GEBGP_ONESTEP(3);
            EIGEN_GEBGP_ONESTEP(4);
            EIGEN_GEBGP_ONESTEP(5);
            EIGEN_GEBGP_ONESTEP(6);
            EIGEN_GEBGP_ONESTEP(7);

            blB += pk*RhsProgress;
            blA += pk*3*Traits::LhsProgress;
          }

          // process remaining peeled loop
          for(Index k=peeled_kc; k<depth; k++)
          {
            RhsPacket B_0;
            EIGEN_GEBGP_ONESTEP(0);
            blB += RhsProgress;
            blA += 3*Traits::LhsProgress;
          }
#undef EIGEN_GEBGP_ONESTEP
          ResPacket R0, R1, R2;
          ResPacket alphav = pset1<ResPacket>(alpha);

          R0 = ploadu<ResPacket>(r0+0*Traits::ResPacketSize);
          R1 = ploadu<ResPacket>(r0+1*Traits::ResPacketSize);
          R2 = ploadu<ResPacket>(r0+2*Traits::ResPacketSize);
          traits.acc(C0, alphav, R0);
          traits.acc(C4, alphav, R1);
          traits.acc(C8 , alphav, R2);
          pstoreu(r0+0*Traits::ResPacketSize, R0);
          pstoreu(r0+1*Traits::ResPacketSize, R1);
          pstoreu(r0+2*Traits::ResPacketSize, R2);
        }
      }
    }
      
    //---------- Process 2 * LhsProgress rows at once ----------
    if(mr>=2*Traits::LhsProgress)
    {
      // loops on each largest micro horizontal panel of lhs (2*LhsProgress x depth)
      for(Index i=peeled_mc3; i<peeled_mc2; i+=2*LhsProgress)
      {
        // loops on each largest micro vertical panel of rhs (depth * nr)
        for(Index j2=0; j2<packet_cols4; j2+=nr)
        {
          // We select a 2*Traits::LhsProgress x nr micro block of res which is entirely
          // stored into 2 x nr registers.
          
          const LhsScalar* blA = &blockA[i*strideA+offsetA*(2*Traits::LhsProgress)];
          prefetch(&blA[0]);

          // gets res block as register
          AccPacket C0, C1, C2, C3,
                    C4, C5, C6, C7;
          traits.initAcc(C0); traits.initAcc(C1); traits.initAcc(C2); traits.initAcc(C3);
          traits.initAcc(C4); traits.initAcc(C5); traits.initAcc(C6); traits.initAcc(C7);

          ResScalar* r0 = &res[(j2+0)*resStride + i];
          ResScalar* r1 = &res[(j2+1)*resStride + i];
          ResScalar* r2 = &res[(j2+2)*resStride + i];
          ResScalar* r3 = &res[(j2+3)*resStride + i];
          
          internal::prefetch(r0+prefetch_res_offset);
          internal::prefetch(r1+prefetch_res_offset);
          internal::prefetch(r2+prefetch_res_offset);
          internal::prefetch(r3+prefetch_res_offset);

          // performs "inner" products
          const RhsScalar* blB = &blockB[j2*strideB+offsetB*nr];
          prefetch(&blB[0]);
          LhsPacket A0, A1;

          for(Index k=0; k<peeled_kc; k+=pk)
          {
            EIGEN_ASM_COMMENT("begin gegp micro kernel 2pX4");
            RhsPacket B_0, B1, B2, B3, T0;

   #define EIGEN_GEBGP_ONESTEP(K) \
            traits.loadLhs(&blA[(0+2*K)*LhsProgress], A0);                    \
            traits.loadLhs(&blA[(1+2*K)*LhsProgress], A1);                    \
            traits.broadcastRhs(&blB[(0+4*K)*RhsProgress], B_0, B1, B2, B3);  \
            traits.madd(A0, B_0, C0, T0);                                     \
            traits.madd(A1, B_0, C4, B_0);                                    \
            traits.madd(A0, B1,  C1, T0);                                     \
            traits.madd(A1, B1,  C5, B1);                                     \
            traits.madd(A0, B2,  C2, T0);                                     \
            traits.madd(A1, B2,  C6, B2);                                     \
            traits.madd(A0, B3,  C3, T0);                                     \
            traits.madd(A1, B3,  C7, B3)
            
            internal::prefetch(blB+(48+0));
            EIGEN_GEBGP_ONESTEP(0);
            EIGEN_GEBGP_ONESTEP(1);
            EIGEN_GEBGP_ONESTEP(2);
            EIGEN_GEBGP_ONESTEP(3);
            internal::prefetch(blB+(48+16));
            EIGEN_GEBGP_ONESTEP(4);
            EIGEN_GEBGP_ONESTEP(5);
            EIGEN_GEBGP_ONESTEP(6);
            EIGEN_GEBGP_ONESTEP(7);

            blB += pk*4*RhsProgress;
            blA += pk*(2*Traits::LhsProgress);
          }
          // process remaining peeled loop
          for(Index k=peeled_kc; k<depth; k++)
          {
            RhsPacket B_0, B1, B2, B3, T0;
            EIGEN_GEBGP_ONESTEP(0);
            blB += 4*RhsProgress;
            blA += 2*Traits::LhsProgress;
          }
#undef EIGEN_GEBGP_ONESTEP
  
          ResPacket R0, R1, R2, R3;
          ResPacket alphav = pset1<ResPacket>(alpha);
          
          R0 = ploadu<ResPacket>(r0+0*Traits::ResPacketSize);
          R1 = ploadu<ResPacket>(r0+1*Traits::ResPacketSize);
          R2 = ploadu<ResPacket>(r1+0*Traits::ResPacketSize);
          R3 = ploadu<ResPacket>(r1+1*Traits::ResPacketSize);
          traits.acc(C0, alphav, R0);
          traits.acc(C4, alphav, R1);
          traits.acc(C1, alphav, R2);
          traits.acc(C5, alphav, R3);
          pstoreu(r0+0*Traits::ResPacketSize, R0);
          pstoreu(r0+1*Traits::ResPacketSize, R1);
          pstoreu(r1+0*Traits::ResPacketSize, R2);
          pstoreu(r1+1*Traits::ResPacketSize, R3);
          
          R0 = ploadu<ResPacket>(r2+0*Traits::ResPacketSize);
          R1 = ploadu<ResPacket>(r2+1*Traits::ResPacketSize);
          R2 = ploadu<ResPacket>(r3+0*Traits::ResPacketSize);
          R3 = ploadu<ResPacket>(r3+1*Traits::ResPacketSize);
          traits.acc(C2,  alphav, R0);
          traits.acc(C6,  alphav, R1);
          traits.acc(C3,  alphav, R2);
          traits.acc(C7,  alphav, R3);
          pstoreu(r2+0*Traits::ResPacketSize, R0);
          pstoreu(r2+1*Traits::ResPacketSize, R1);
          pstoreu(r3+0*Traits::ResPacketSize, R2);
          pstoreu(r3+1*Traits::ResPacketSize, R3);
        }
        
        // Deal with remaining columns of the rhs
        for(Index j2=packet_cols4; j2<cols; j2++)
        {
          // One column at a time
          const LhsScalar* blA = &blockA[i*strideA+offsetA*(2*Traits::LhsProgress)];
          prefetch(&blA[0]);

          // gets res block as register
          AccPacket C0, C4;
          traits.initAcc(C0);
          traits.initAcc(C4);

          ResScalar* r0 = &res[(j2+0)*resStride + i];
          internal::prefetch(r0+prefetch_res_offset);

          // performs "inner" products
          const RhsScalar* blB = &blockB[j2*strideB+offsetB];
          LhsPacket A0, A1;

          for(Index k=0; k<peeled_kc; k+=pk)
          {
            EIGEN_ASM_COMMENT("begin gegp micro kernel 2p x 1");
            RhsPacket B_0, B1;
        
#define EIGEN_GEBGP_ONESTEP(K) \
            traits.loadLhs(&blA[(0+2*K)*LhsProgress], A0);  \
            traits.loadLhs(&blA[(1+2*K)*LhsProgress], A1);  \
            traits.loadRhs(&blB[(0+K)*RhsProgress], B_0);   \
            traits.madd(A0, B_0, C0, B1);                   \
            traits.madd(A1, B_0, C4, B_0)
        
            EIGEN_GEBGP_ONESTEP(0);
            EIGEN_GEBGP_ONESTEP(1);
            EIGEN_GEBGP_ONESTEP(2);
            EIGEN_GEBGP_ONESTEP(3);
            EIGEN_GEBGP_ONESTEP(4);
            EIGEN_GEBGP_ONESTEP(5);
            EIGEN_GEBGP_ONESTEP(6);
            EIGEN_GEBGP_ONESTEP(7);

            blB += pk*RhsProgress;
            blA += pk*2*Traits::LhsProgress;
          }

          // process remaining peeled loop
          for(Index k=peeled_kc; k<depth; k++)
          {
            RhsPacket B_0, B1;
            EIGEN_GEBGP_ONESTEP(0);
            blB += RhsProgress;
            blA += 2*Traits::LhsProgress;
          }
#undef EIGEN_GEBGP_ONESTEP
          ResPacket R0, R1;
          ResPacket alphav = pset1<ResPacket>(alpha);

          R0 = ploadu<ResPacket>(r0+0*Traits::ResPacketSize);
          R1 = ploadu<ResPacket>(r0+1*Traits::ResPacketSize);
          traits.acc(C0, alphav, R0);
          traits.acc(C4, alphav, R1);
          pstoreu(r0+0*Traits::ResPacketSize, R0);
          pstoreu(r0+1*Traits::ResPacketSize, R1);
        }
      }
    }
    //---------- Process 1 * LhsProgress rows at once ----------
    if(mr>=1*Traits::LhsProgress)
    {
      // loops on each largest micro horizontal panel of lhs (1*LhsProgress x depth)
      for(Index i=peeled_mc2; i<peeled_mc1; i+=1*LhsProgress)
      {
        // loops on each largest micro vertical panel of rhs (depth * nr)
        for(Index j2=0; j2<packet_cols4; j2+=nr)
        {
          // We select a 1*Traits::LhsProgress x nr micro block of res which is entirely
          // stored into 1 x nr registers.
          
          const LhsScalar* blA = &blockA[i*strideA+offsetA*(1*Traits::LhsProgress)];
          prefetch(&blA[0]);

          // gets res block as register
          AccPacket C0, C1, C2, C3;
          traits.initAcc(C0);
          traits.initAcc(C1);
          traits.initAcc(C2);
          traits.initAcc(C3);

          ResScalar* r0 = &res[(j2+0)*resStride + i];
          ResScalar* r1 = &res[(j2+1)*resStride + i];
          ResScalar* r2 = &res[(j2+2)*resStride + i];
          ResScalar* r3 = &res[(j2+3)*resStride + i];
          
          internal::prefetch(r0+prefetch_res_offset);
          internal::prefetch(r1+prefetch_res_offset);
          internal::prefetch(r2+prefetch_res_offset);
          internal::prefetch(r3+prefetch_res_offset);

          // performs "inner" products
          const RhsScalar* blB = &blockB[j2*strideB+offsetB*nr];
          prefetch(&blB[0]);
          LhsPacket A0;

          for(Index k=0; k<peeled_kc; k+=pk)
          {
            EIGEN_ASM_COMMENT("begin gegp micro kernel 1pX4");
            RhsPacket B_0, B1, B2, B3;
               
#define EIGEN_GEBGP_ONESTEP(K) \
            traits.loadLhs(&blA[(0+1*K)*LhsProgress], A0);                    \
            traits.broadcastRhs(&blB[(0+4*K)*RhsProgress], B_0, B1, B2, B3);  \
            traits.madd(A0, B_0, C0, B_0);                                    \
            traits.madd(A0, B1,  C1, B1);                                     \
            traits.madd(A0, B2,  C2, B2);                                     \
            traits.madd(A0, B3,  C3, B3);
            
            internal::prefetch(blB+(48+0));
            EIGEN_GEBGP_ONESTEP(0);
            EIGEN_GEBGP_ONESTEP(1);
            EIGEN_GEBGP_ONESTEP(2);
            EIGEN_GEBGP_ONESTEP(3);
            internal::prefetch(blB+(48+16));
            EIGEN_GEBGP_ONESTEP(4);
            EIGEN_GEBGP_ONESTEP(5);
            EIGEN_GEBGP_ONESTEP(6);
            EIGEN_GEBGP_ONESTEP(7);

            blB += pk*4*RhsProgress;
            blA += pk*1*LhsProgress;
          }
          // process remaining peeled loop
          for(Index k=peeled_kc; k<depth; k++)
          {
            RhsPacket B_0, B1, B2, B3;
            EIGEN_GEBGP_ONESTEP(0);
            blB += 4*RhsProgress;
            blA += 1*LhsProgress;
          }
#undef EIGEN_GEBGP_ONESTEP
  
          ResPacket R0, R1;
          ResPacket alphav = pset1<ResPacket>(alpha);
          
          R0 = ploadu<ResPacket>(r0+0*Traits::ResPacketSize);
          R1 = ploadu<ResPacket>(r1+0*Traits::ResPacketSize);
          traits.acc(C0, alphav, R0);
          traits.acc(C1,  alphav, R1);
          pstoreu(r0+0*Traits::ResPacketSize, R0);
          pstoreu(r1+0*Traits::ResPacketSize, R1);
          
          R0 = ploadu<ResPacket>(r2+0*Traits::ResPacketSize);
          R1 = ploadu<ResPacket>(r3+0*Traits::ResPacketSize);
          traits.acc(C2,  alphav, R0);
          traits.acc(C3,  alphav, R1);
          pstoreu(r2+0*Traits::ResPacketSize, R0);
          pstoreu(r3+0*Traits::ResPacketSize, R1);
        }
        
        // Deal with remaining columns of the rhs
        for(Index j2=packet_cols4; j2<cols; j2++)
        {
          // One column at a time
          const LhsScalar* blA = &blockA[i*strideA+offsetA*(1*Traits::LhsProgress)];
          prefetch(&blA[0]);

          // gets res block as register
          AccPacket C0;
          traits.initAcc(C0);

          ResScalar* r0 = &res[(j2+0)*resStride + i];

          // performs "inner" products
          const RhsScalar* blB = &blockB[j2*strideB+offsetB];
          LhsPacket A0;

          for(Index k=0; k<peeled_kc; k+=pk)
          {
            EIGEN_ASM_COMMENT("begin gegp micro kernel 2p x 1");
            RhsPacket B_0;
        
#define EIGEN_GEBGP_ONESTEP(K) \
            traits.loadLhs(&blA[(0+1*K)*LhsProgress], A0);  \
            traits.loadRhs(&blB[(0+K)*RhsProgress], B_0);   \
            traits.madd(A0, B_0, C0, B_0); \
        
            EIGEN_GEBGP_ONESTEP(0);
            EIGEN_GEBGP_ONESTEP(1);
            EIGEN_GEBGP_ONESTEP(2);
            EIGEN_GEBGP_ONESTEP(3);
            EIGEN_GEBGP_ONESTEP(4);
            EIGEN_GEBGP_ONESTEP(5);
            EIGEN_GEBGP_ONESTEP(6);
            EIGEN_GEBGP_ONESTEP(7);

            blB += pk*RhsProgress;
            blA += pk*1*Traits::LhsProgress;
          }

          // process remaining peeled loop
          for(Index k=peeled_kc; k<depth; k++)
          {
            RhsPacket B_0;
            EIGEN_GEBGP_ONESTEP(0);
            blB += RhsProgress;
            blA += 1*Traits::LhsProgress;
          }
#undef EIGEN_GEBGP_ONESTEP
          ResPacket R0;
          ResPacket alphav = pset1<ResPacket>(alpha);
          R0 = ploadu<ResPacket>(r0+0*Traits::ResPacketSize);
          traits.acc(C0, alphav, R0);
          pstoreu(r0+0*Traits::ResPacketSize, R0);
        }
      }
    }
    //---------- Process remaining rows, 1 at once ----------
    {
      // loop on each panel of the rhs
      for(Index j2=0; j2<packet_cols4; j2+=nr)
      {
        // loop on each row of the lhs (1*LhsProgress x depth)
        for(Index i=peeled_mc1; i<rows; i+=1)
        {
          const LhsScalar* blA = &blockA[i*strideA+offsetA];
          prefetch(&blA[0]);
          const RhsScalar* blB = &blockB[j2*strideB+offsetB*nr];
          
          if( (SwappedTraits::LhsProgress % 4)==0 )
          {
            // NOTE The following piece of code wont work for 512 bit registers
            SAccPacket C0, C1, C2, C3;
            straits.initAcc(C0);
            straits.initAcc(C1);
            straits.initAcc(C2);
            straits.initAcc(C3);
            
            const Index spk   = (std::max)(1,SwappedTraits::LhsProgress/4);
            const Index endk  = (depth/spk)*spk;
            const Index endk4 = (depth/(spk*4))*(spk*4);
            
            Index k=0;
            for(; k<endk4; k+=4*spk)
            {
              SLhsPacket A0,A1;
              SRhsPacket B_0,B_1;
              
              straits.loadLhsUnaligned(blB+0*SwappedTraits::LhsProgress, A0);
              straits.loadLhsUnaligned(blB+1*SwappedTraits::LhsProgress, A1);
              
              
              straits.loadRhsQuad(blA+0*spk, B_0);
              straits.loadRhsQuad(blA+1*spk, B_1);
              straits.madd(A0,B_0,C0,B_0);
              straits.madd(A1,B_1,C1,B_1);
              
              
              straits.loadLhsUnaligned(blB+2*SwappedTraits::LhsProgress, A0);
              straits.loadLhsUnaligned(blB+3*SwappedTraits::LhsProgress, A1);
              straits.loadRhsQuad(blA+2*spk, B_0);
              straits.loadRhsQuad(blA+3*spk, B_1);
              straits.madd(A0,B_0,C2,B_0);
              straits.madd(A1,B_1,C3,B_1);
              
              blB += 4*SwappedTraits::LhsProgress;
              blA += 4*spk;
            }
            C0 = padd(padd(C0,C1),padd(C2,C3));
            for(; k<endk; k+=spk)
            {
              SLhsPacket A0;
              SRhsPacket B_0;
              
              straits.loadLhsUnaligned(blB, A0);
              straits.loadRhsQuad(blA, B_0);
              straits.madd(A0,B_0,C0,B_0);
              
              blB += SwappedTraits::LhsProgress;
              blA += spk;
            }
            if(SwappedTraits::LhsProgress==8)
            {
              // Special case where we have to first reduce the accumulation register C0
              typedef typename conditional<SwappedTraits::LhsProgress==8,typename unpacket_traits<SResPacket>::half,SResPacket>::type SResPacketHalf;
              typedef typename conditional<SwappedTraits::LhsProgress==8,typename unpacket_traits<SLhsPacket>::half,SLhsPacket>::type SLhsPacketHalf;
              typedef typename conditional<SwappedTraits::LhsProgress==8,typename unpacket_traits<SLhsPacket>::half,SRhsPacket>::type SRhsPacketHalf;
              typedef typename conditional<SwappedTraits::LhsProgress==8,typename unpacket_traits<SAccPacket>::half,SAccPacket>::type SAccPacketHalf;
              
              SResPacketHalf R = pgather<SResScalar, SResPacketHalf>(&res[j2*resStride + i], resStride);
              SResPacketHalf alphav = pset1<SResPacketHalf>(alpha);
             
              if(depth-endk>0)
              {
                // We have to handle the last row of the rhs which corresponds to a half-packet
                SLhsPacketHalf a0;
                SRhsPacketHalf b0;
                straits.loadLhsUnaligned(blB, a0);
                straits.loadRhs(blA, b0);
                SAccPacketHalf c0 = predux4(C0);
                straits.madd(a0,b0,c0,b0);
                straits.acc(c0, alphav, R);
              }
              else
              {
                straits.acc(predux4(C0), alphav, R);
              }
              pscatter(&res[j2*resStride + i], R, resStride);
            }
            else
            {
              SResPacket R = pgather<SResScalar, SResPacket>(&res[j2*resStride + i], resStride);
              SResPacket alphav = pset1<SResPacket>(alpha);
              straits.acc(C0, alphav, R);
              pscatter(&res[j2*resStride + i], R, resStride);
            }
          }
          else // scalar path
          {
            // get a 1 x 4 res block as registers
            ResScalar C0(0), C1(0), C2(0), C3(0);

            for(Index k=0; k<depth; k++)
            {
              LhsScalar A0;
              RhsScalar B_0, B_1;
              
              A0 = blA[k];
              
              B_0 = blB[0];
              B_1 = blB[1];
              MADD(cj,A0,B_0,C0,  B_0);
              MADD(cj,A0,B_1,C1,  B_1);
              
              B_0 = blB[2];
              B_1 = blB[3];
              MADD(cj,A0,B_0,C2,  B_0);
              MADD(cj,A0,B_1,C3,  B_1);
              
              blB += 4;
            }
            res[(j2+0)*resStride + i] += alpha*C0;
            res[(j2+1)*resStride + i] += alpha*C1;
            res[(j2+2)*resStride + i] += alpha*C2;
            res[(j2+3)*resStride + i] += alpha*C3;
          }
        }
      }
      // remaining columns
      for(Index j2=packet_cols4; j2<cols; j2++)
      {
        // loop on each row of the lhs (1*LhsProgress x depth)
        for(Index i=peeled_mc1; i<rows; i+=1)
        {
          const LhsScalar* blA = &blockA[i*strideA+offsetA];
          prefetch(&blA[0]);
          // gets a 1 x 1 res block as registers
          ResScalar C0(0);
          const RhsScalar* blB = &blockB[j2*strideB+offsetB];
          for(Index k=0; k<depth; k++)
          {
            LhsScalar A0 = blA[k];
            RhsScalar B_0 = blB[k];
            MADD(cj, A0, B_0, C0, B_0);
          }
          res[(j2+0)*resStride + i] += alpha*C0;
        }
      }
    }
  }


#undef CJMADD

// pack a block of the lhs
// The traversal is as follow (mr==4):
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
template<typename Scalar, typename Index, int Pack1, int Pack2, bool Conjugate, bool PanelMode>
struct gemm_pack_lhs<Scalar, Index, Pack1, Pack2, ColMajor, Conjugate, PanelMode>
{
  EIGEN_DONT_INLINE void operator()(Scalar* blockA, const Scalar* EIGEN_RESTRICT _lhs, Index lhsStride, Index depth, Index rows, Index stride=0, Index offset=0);
};

template<typename Scalar, typename Index, int Pack1, int Pack2, bool Conjugate, bool PanelMode>
EIGEN_DONT_INLINE void gemm_pack_lhs<Scalar, Index, Pack1, Pack2, ColMajor, Conjugate, PanelMode>
  ::operator()(Scalar* blockA, const Scalar* EIGEN_RESTRICT _lhs, Index lhsStride, Index depth, Index rows, Index stride, Index offset)
{
  typedef typename packet_traits<Scalar>::type Packet;
  enum { PacketSize = packet_traits<Scalar>::size };

  EIGEN_ASM_COMMENT("EIGEN PRODUCT PACK LHS");
  EIGEN_UNUSED_VARIABLE(stride);
  EIGEN_UNUSED_VARIABLE(offset);
  eigen_assert(((!PanelMode) && stride==0 && offset==0) || (PanelMode && stride>=depth && offset<=stride));
  eigen_assert( ((Pack1%PacketSize)==0 && Pack1<=4*PacketSize) || (Pack1<=4) );
  conj_if<NumTraits<Scalar>::IsComplex && Conjugate> cj;
  const_blas_data_mapper<Scalar, Index, ColMajor> lhs(_lhs,lhsStride);
  Index count = 0;
  
  const Index peeled_mc3 = Pack1>=3*PacketSize ? (rows/(3*PacketSize))*(3*PacketSize) : 0;
  const Index peeled_mc2 = Pack1>=2*PacketSize ? peeled_mc3+((rows-peeled_mc3)/(2*PacketSize))*(2*PacketSize) : 0;
  const Index peeled_mc1 = Pack1>=1*PacketSize ? (rows/(1*PacketSize))*(1*PacketSize) : 0;
  const Index peeled_mc0 = Pack2>=1*PacketSize ? peeled_mc1
                         : Pack2>1             ? (rows/Pack2)*Pack2 : 0;
  
  Index i=0;
  
  // Pack 3 packets
  if(Pack1>=3*PacketSize)
  {
    for(; i<peeled_mc3; i+=3*PacketSize)
    {
      if(PanelMode) count += (3*PacketSize) * offset;
      
      for(Index k=0; k<depth; k++)
      {
        Packet A, B, C;
        A = ploadu<Packet>(&lhs(i+0*PacketSize, k));
        B = ploadu<Packet>(&lhs(i+1*PacketSize, k));
        C = ploadu<Packet>(&lhs(i+2*PacketSize, k));
        pstore(blockA+count, cj.pconj(A)); count+=PacketSize;
        pstore(blockA+count, cj.pconj(B)); count+=PacketSize;
        pstore(blockA+count, cj.pconj(C)); count+=PacketSize;
      }
      if(PanelMode) count += (3*PacketSize) * (stride-offset-depth);
    }
  }
  // Pack 2 packets
  if(Pack1>=2*PacketSize)
  {
    for(; i<peeled_mc2; i+=2*PacketSize)
    {
      if(PanelMode) count += (2*PacketSize) * offset;
      
      for(Index k=0; k<depth; k++)
      {
        Packet A, B;
        A = ploadu<Packet>(&lhs(i+0*PacketSize, k));
        B = ploadu<Packet>(&lhs(i+1*PacketSize, k));
        pstore(blockA+count, cj.pconj(A)); count+=PacketSize;
        pstore(blockA+count, cj.pconj(B)); count+=PacketSize;
      }
      if(PanelMode) count += (2*PacketSize) * (stride-offset-depth);
    }
  }
  // Pack 1 packets
  if(Pack1>=1*PacketSize)
  {
    for(; i<peeled_mc1; i+=1*PacketSize)
    {
      if(PanelMode) count += (1*PacketSize) * offset;
      
      for(Index k=0; k<depth; k++)
      {
        Packet A;
        A = ploadu<Packet>(&lhs(i+0*PacketSize, k));
        pstore(blockA+count, cj.pconj(A));
        count+=PacketSize;
      }
      if(PanelMode) count += (1*PacketSize) * (stride-offset-depth);
    }
  }
  // Pack scalars
  if(Pack2<PacketSize && Pack2>1)
  {
    for(; i<peeled_mc0; i+=Pack2)
    {
      if(PanelMode) count += Pack2 * offset;
      
      for(Index k=0; k<depth; k++)
        for(Index w=0; w<Pack2; w++)
          blockA[count++] = cj(lhs(i+w, k));
        
      if(PanelMode) count += Pack2 * (stride-offset-depth);
    }
  }
  for(; i<rows; i++)
  {
    if(PanelMode) count += offset;
    for(Index k=0; k<depth; k++)
      blockA[count++] = cj(lhs(i, k));
    if(PanelMode) count += (stride-offset-depth);
  }
}

template<typename Scalar, typename Index, int Pack1, int Pack2, bool Conjugate, bool PanelMode>
struct gemm_pack_lhs<Scalar, Index, Pack1, Pack2, RowMajor, Conjugate, PanelMode>
{
  EIGEN_DONT_INLINE void operator()(Scalar* blockA, const Scalar* EIGEN_RESTRICT _lhs, Index lhsStride, Index depth, Index rows, Index stride=0, Index offset=0);
};

template<typename Scalar, typename Index, int Pack1, int Pack2, bool Conjugate, bool PanelMode>
EIGEN_DONT_INLINE void gemm_pack_lhs<Scalar, Index, Pack1, Pack2, RowMajor, Conjugate, PanelMode>
  ::operator()(Scalar* blockA, const Scalar* EIGEN_RESTRICT _lhs, Index lhsStride, Index depth, Index rows, Index stride, Index offset)
{
  typedef typename packet_traits<Scalar>::type Packet;
  enum { PacketSize = packet_traits<Scalar>::size };

  EIGEN_ASM_COMMENT("EIGEN PRODUCT PACK LHS");
  EIGEN_UNUSED_VARIABLE(stride);
  EIGEN_UNUSED_VARIABLE(offset);
  eigen_assert(((!PanelMode) && stride==0 && offset==0) || (PanelMode && stride>=depth && offset<=stride));
  conj_if<NumTraits<Scalar>::IsComplex && Conjugate> cj;
  const_blas_data_mapper<Scalar, Index, RowMajor> lhs(_lhs,lhsStride);
  Index count = 0;
  
//   const Index peeled_mc3 = Pack1>=3*PacketSize ? (rows/(3*PacketSize))*(3*PacketSize) : 0;
//   const Index peeled_mc2 = Pack1>=2*PacketSize ? peeled_mc3+((rows-peeled_mc3)/(2*PacketSize))*(2*PacketSize) : 0;
//   const Index peeled_mc1 = Pack1>=1*PacketSize ? (rows/(1*PacketSize))*(1*PacketSize) : 0;
  
  int pack = Pack1;
  Index i = 0;
  while(pack>0)
  {
    Index remaining_rows = rows-i;
    Index peeled_mc = i+(remaining_rows/pack)*pack;
    for(; i<peeled_mc; i+=pack)
    {
      if(PanelMode) count += pack * offset;

      const Index peeled_k = (depth/PacketSize)*PacketSize;
      Index k=0;
      if(pack>=PacketSize)
      {
        for(; k<peeled_k; k+=PacketSize)
        {
          for (Index m = 0; m < pack; m += PacketSize)
          {
            PacketBlock<Packet> kernel;
            for (int p = 0; p < PacketSize; ++p) kernel.packet[p] = ploadu<Packet>(&lhs(i+p+m, k));
            ptranspose(kernel);
            for (int p = 0; p < PacketSize; ++p) pstore(blockA+count+m+(pack)*p, cj.pconj(kernel.packet[p]));
          }
          count += PacketSize*pack;
        }
      }
      for(; k<depth; k++)
      {
        Index w=0;
        for(; w<pack-3; w+=4)
        {
          Scalar a(cj(lhs(i+w+0, k))),
                 b(cj(lhs(i+w+1, k))),
                 c(cj(lhs(i+w+2, k))),
                 d(cj(lhs(i+w+3, k)));
          blockA[count++] = a;
          blockA[count++] = b;
          blockA[count++] = c;
          blockA[count++] = d;
        }
        if(pack%4)
          for(;w<pack;++w)
            blockA[count++] = cj(lhs(i+w, k));
      }
      
      if(PanelMode) count += pack * (stride-offset-depth);
    }
    
    pack -= PacketSize;
    if(pack<Pack2 && (pack+PacketSize)!=Pack2)
      pack = Pack2;
  }
  
  for(; i<rows; i++)
  {
    if(PanelMode) count += offset;
    for(Index k=0; k<depth; k++)
      blockA[count++] = cj(lhs(i, k));
    if(PanelMode) count += (stride-offset-depth);
  }
}

// copy a complete panel of the rhs
// this version is optimized for column major matrices
// The traversal order is as follow: (nr==4):
//  0  1  2  3   12 13 14 15   24 27
//  4  5  6  7   16 17 18 19   25 28
//  8  9 10 11   20 21 22 23   26 29
//  .  .  .  .    .  .  .  .    .  .
template<typename Scalar, typename Index, int nr, bool Conjugate, bool PanelMode>
struct gemm_pack_rhs<Scalar, Index, nr, ColMajor, Conjugate, PanelMode>
{
  typedef typename packet_traits<Scalar>::type Packet;
  enum { PacketSize = packet_traits<Scalar>::size };
  EIGEN_DONT_INLINE void operator()(Scalar* blockB, const Scalar* rhs, Index rhsStride, Index depth, Index cols, Index stride=0, Index offset=0);
};

template<typename Scalar, typename Index, int nr, bool Conjugate, bool PanelMode>
EIGEN_DONT_INLINE void gemm_pack_rhs<Scalar, Index, nr, ColMajor, Conjugate, PanelMode>
  ::operator()(Scalar* blockB, const Scalar* rhs, Index rhsStride, Index depth, Index cols, Index stride, Index offset)
{
  EIGEN_ASM_COMMENT("EIGEN PRODUCT PACK RHS COLMAJOR");
  EIGEN_UNUSED_VARIABLE(stride);
  EIGEN_UNUSED_VARIABLE(offset);
  eigen_assert(((!PanelMode) && stride==0 && offset==0) || (PanelMode && stride>=depth && offset<=stride));
  conj_if<NumTraits<Scalar>::IsComplex && Conjugate> cj;
  Index packet_cols8 = nr>=8 ? (cols/8) * 8 : 0;
  Index packet_cols4 = nr>=4 ? (cols/4) * 4 : 0;
  Index count = 0;
  const Index peeled_k = (depth/PacketSize)*PacketSize;
//   if(nr>=8)
//   {
//     for(Index j2=0; j2<packet_cols8; j2+=8)
//     {
//       // skip what we have before
//       if(PanelMode) count += 8 * offset;
//       const Scalar* b0 = &rhs[(j2+0)*rhsStride];
//       const Scalar* b1 = &rhs[(j2+1)*rhsStride];
//       const Scalar* b2 = &rhs[(j2+2)*rhsStride];
//       const Scalar* b3 = &rhs[(j2+3)*rhsStride];
//       const Scalar* b4 = &rhs[(j2+4)*rhsStride];
//       const Scalar* b5 = &rhs[(j2+5)*rhsStride];
//       const Scalar* b6 = &rhs[(j2+6)*rhsStride];
//       const Scalar* b7 = &rhs[(j2+7)*rhsStride];
//       Index k=0;
//       if(PacketSize==8) // TODO enbale vectorized transposition for PacketSize==4
//       {
//         for(; k<peeled_k; k+=PacketSize) {
//           PacketBlock<Packet> kernel;
//           for (int p = 0; p < PacketSize; ++p) {
//             kernel.packet[p] = ploadu<Packet>(&rhs[(j2+p)*rhsStride+k]);
//           }
//           ptranspose(kernel);
//           for (int p = 0; p < PacketSize; ++p) {
//             pstoreu(blockB+count, cj.pconj(kernel.packet[p]));
//             count+=PacketSize;
//           }
//         }
//       }
//       for(; k<depth; k++)
//       {
//         blockB[count+0] = cj(b0[k]);
//         blockB[count+1] = cj(b1[k]);
//         blockB[count+2] = cj(b2[k]);
//         blockB[count+3] = cj(b3[k]);
//         blockB[count+4] = cj(b4[k]);
//         blockB[count+5] = cj(b5[k]);
//         blockB[count+6] = cj(b6[k]);
//         blockB[count+7] = cj(b7[k]);
//         count += 8;
//       }
//       // skip what we have after
//       if(PanelMode) count += 8 * (stride-offset-depth);
//     }
//   }
  
  if(nr>=4)
  {
    for(Index j2=packet_cols8; j2<packet_cols4; j2+=4)
    {
      // skip what we have before
      if(PanelMode) count += 4 * offset;
      const Scalar* b0 = &rhs[(j2+0)*rhsStride];
      const Scalar* b1 = &rhs[(j2+1)*rhsStride];
      const Scalar* b2 = &rhs[(j2+2)*rhsStride];
      const Scalar* b3 = &rhs[(j2+3)*rhsStride];
      
      Index k=0;
      if((PacketSize%4)==0) // TODO enbale vectorized transposition for PacketSize==2 ??
      {
        for(; k<peeled_k; k+=PacketSize) {
          PacketBlock<Packet,(PacketSize%4)==0?4:PacketSize> kernel;
          kernel.packet[0] = ploadu<Packet>(&b0[k]);
          kernel.packet[1] = ploadu<Packet>(&b1[k]);
          kernel.packet[2] = ploadu<Packet>(&b2[k]);
          kernel.packet[3] = ploadu<Packet>(&b3[k]);
          ptranspose(kernel);
          pstoreu(blockB+count+0*PacketSize, cj.pconj(kernel.packet[0]));
          pstoreu(blockB+count+1*PacketSize, cj.pconj(kernel.packet[1]));
          pstoreu(blockB+count+2*PacketSize, cj.pconj(kernel.packet[2]));
          pstoreu(blockB+count+3*PacketSize, cj.pconj(kernel.packet[3]));
          count+=4*PacketSize;
        }
      }
      for(; k<depth; k++)
      {
        blockB[count+0] = cj(b0[k]);
        blockB[count+1] = cj(b1[k]);
        blockB[count+2] = cj(b2[k]);
        blockB[count+3] = cj(b3[k]);
        count += 4;
      }
      // skip what we have after
      if(PanelMode) count += 4 * (stride-offset-depth);
    }
  }

  // copy the remaining columns one at a time (nr==1)
  for(Index j2=packet_cols4; j2<cols; ++j2)
  {
    if(PanelMode) count += offset;
    const Scalar* b0 = &rhs[(j2+0)*rhsStride];
    for(Index k=0; k<depth; k++)
    {
      blockB[count] = cj(b0[k]);
      count += 1;
    }
    if(PanelMode) count += (stride-offset-depth);
  }
}

// this version is optimized for row major matrices
template<typename Scalar, typename Index, int nr, bool Conjugate, bool PanelMode>
struct gemm_pack_rhs<Scalar, Index, nr, RowMajor, Conjugate, PanelMode>
{
  typedef typename packet_traits<Scalar>::type Packet;
  enum { PacketSize = packet_traits<Scalar>::size };
  EIGEN_DONT_INLINE void operator()(Scalar* blockB, const Scalar* rhs, Index rhsStride, Index depth, Index cols, Index stride=0, Index offset=0);
};

template<typename Scalar, typename Index, int nr, bool Conjugate, bool PanelMode>
EIGEN_DONT_INLINE void gemm_pack_rhs<Scalar, Index, nr, RowMajor, Conjugate, PanelMode>
  ::operator()(Scalar* blockB, const Scalar* rhs, Index rhsStride, Index depth, Index cols, Index stride, Index offset)
{
  EIGEN_ASM_COMMENT("EIGEN PRODUCT PACK RHS ROWMAJOR");
  EIGEN_UNUSED_VARIABLE(stride);
  EIGEN_UNUSED_VARIABLE(offset);
  eigen_assert(((!PanelMode) && stride==0 && offset==0) || (PanelMode && stride>=depth && offset<=stride));
  conj_if<NumTraits<Scalar>::IsComplex && Conjugate> cj;
  Index packet_cols8 = nr>=8 ? (cols/8) * 8 : 0;
  Index packet_cols4 = nr>=4 ? (cols/4) * 4 : 0;
  Index count = 0;
  
//   if(nr>=8)
//   {
//     for(Index j2=0; j2<packet_cols8; j2+=8)
//     {
//       // skip what we have before
//       if(PanelMode) count += 8 * offset;
//       for(Index k=0; k<depth; k++)
//       {
//         if (PacketSize==8) {
//           Packet A = ploadu<Packet>(&rhs[k*rhsStride + j2]);
//           pstoreu(blockB+count, cj.pconj(A));
//         } else if (PacketSize==4) {
//           Packet A = ploadu<Packet>(&rhs[k*rhsStride + j2]);
//           Packet B = ploadu<Packet>(&rhs[k*rhsStride + j2 + PacketSize]);
//           pstoreu(blockB+count, cj.pconj(A));
//           pstoreu(blockB+count+PacketSize, cj.pconj(B));
//         } else {
//           const Scalar* b0 = &rhs[k*rhsStride + j2];
//           blockB[count+0] = cj(b0[0]);
//           blockB[count+1] = cj(b0[1]);
//           blockB[count+2] = cj(b0[2]);
//           blockB[count+3] = cj(b0[3]);
//           blockB[count+4] = cj(b0[4]);
//           blockB[count+5] = cj(b0[5]);
//           blockB[count+6] = cj(b0[6]);
//           blockB[count+7] = cj(b0[7]);
//         }
//         count += 8;
//       }
//       // skip what we have after
//       if(PanelMode) count += 8 * (stride-offset-depth);
//     }
//   }
  if(nr>=4)
  {
    for(Index j2=packet_cols8; j2<packet_cols4; j2+=4)
    {
      // skip what we have before
      if(PanelMode) count += 4 * offset;
      for(Index k=0; k<depth; k++)
      {
        if (PacketSize==4) {
          Packet A = ploadu<Packet>(&rhs[k*rhsStride + j2]);
          pstoreu(blockB+count, cj.pconj(A));
          count += PacketSize;
        } else {
          const Scalar* b0 = &rhs[k*rhsStride + j2];
          blockB[count+0] = cj(b0[0]);
          blockB[count+1] = cj(b0[1]);
          blockB[count+2] = cj(b0[2]);
          blockB[count+3] = cj(b0[3]);
          count += 4;
        }
      }
      // skip what we have after
      if(PanelMode) count += 4 * (stride-offset-depth);
    }
  }
  // copy the remaining columns one at a time (nr==1)
  for(Index j2=packet_cols4; j2<cols; ++j2)
  {
    if(PanelMode) count += offset;
    const Scalar* b0 = &rhs[j2];
    for(Index k=0; k<depth; k++)
    {
      blockB[count] = cj(b0[k*rhsStride]);
      count += 1;
    }
    if(PanelMode) count += stride-offset-depth;
  }
}

} // end namespace internal

/** \returns the currently set level 1 cpu cache size (in bytes) used to estimate the ideal blocking size parameters.
  * \sa setCpuCacheSize */
inline std::ptrdiff_t l1CacheSize()
{
  std::ptrdiff_t l1, l2;
  internal::manage_caching_sizes(GetAction, &l1, &l2);
  return l1;
}

/** \returns the currently set level 2 cpu cache size (in bytes) used to estimate the ideal blocking size parameters.
  * \sa setCpuCacheSize */
inline std::ptrdiff_t l2CacheSize()
{
  std::ptrdiff_t l1, l2;
  internal::manage_caching_sizes(GetAction, &l1, &l2);
  return l2;
}

/** Set the cpu L1 and L2 cache sizes (in bytes).
  * These values are use to adjust the size of the blocks
  * for the algorithms working per blocks.
  *
  * \sa computeProductBlockingSizes */
inline void setCpuCacheSizes(std::ptrdiff_t l1, std::ptrdiff_t l2)
{
  internal::manage_caching_sizes(SetAction, &l1, &l2);
}

} // end namespace Eigen

#endif // EIGEN_GENERAL_BLOCK_PANEL_H
