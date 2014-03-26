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
  k = std::min<SizeType>(k, l1/kdiv);
  SizeType _m = k>0 ? l2/(4 * sizeof(LhsScalar) * k) : 0;
  if(_m<m) m = _m & mr_mask;
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

    // register block size along the N direction (must be either 4 or 8)
    nr = NumberOfRegisters/2,

    // register block size along the M direction (currently, this one cannot be modified)
    mr = LhsPacketSize,
    
    WorkSpaceFactor = nr * RhsPacketSize,

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
  
  EIGEN_STRONG_INLINE void broadcastRhs(const RhsScalar* b, RhsPacket& b0, RhsPacket& b1)
  {
    pbroadcast2(b, b0, b1);
  }

  EIGEN_STRONG_INLINE void loadRhs(const RhsScalar* b, RhsPacket& dest) const
  {
    dest = pset1<RhsPacket>(*b);
  }

  EIGEN_STRONG_INLINE void loadLhs(const LhsScalar* a, LhsPacket& dest) const
  {
    dest = pload<LhsPacket>(a);
  }

  EIGEN_STRONG_INLINE void madd(const LhsPacket& a, const RhsPacket& b, AccPacket& c, AccPacket& tmp) const
  {
    // It would be a lot cleaner to call pmadd all the time. Unfortunately if we
    // let gcc allocate the register in which to store the result of the pmul
    // (in the case where there is no FMA) gcc fails to figure out how to avoid
    // spilling register.
#ifdef EIGEN_VECTORIZE_FMA
    c = pmadd(a,b,c);
#else
    tmp = b; tmp = pmul(a,tmp); c = padd(c,tmp);
#endif
  }

  EIGEN_STRONG_INLINE void acc(const AccPacket& c, const ResPacket& alpha, ResPacket& r) const
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
    nr = NumberOfRegisters/2,
    mr = LhsPacketSize,
    WorkSpaceFactor = nr*RhsPacketSize,

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

  EIGEN_STRONG_INLINE void loadLhs(const LhsScalar* a, LhsPacket& dest) const
  {
    dest = pload<LhsPacket>(a);
  }
  
  EIGEN_STRONG_INLINE void broadcastRhs(const RhsScalar* b, RhsPacket& b0, RhsPacket& b1, RhsPacket& b2, RhsPacket& b3)
  {
    pbroadcast4(b, b0, b1, b2, b3);
  }
  
  EIGEN_STRONG_INLINE void broadcastRhs(const RhsScalar* b, RhsPacket& b0, RhsPacket& b1)
  {
    pbroadcast2(b, b0, b1);
  }

  EIGEN_STRONG_INLINE void madd(const LhsPacket& a, const RhsPacket& b, AccPacket& c, RhsPacket& tmp) const
  {
    madd_impl(a, b, c, tmp, typename conditional<Vectorizable,true_type,false_type>::type());
  }

  EIGEN_STRONG_INLINE void madd_impl(const LhsPacket& a, const RhsPacket& b, AccPacket& c, RhsPacket& tmp, const true_type&) const
  {
#ifdef EIGEN_VECTORIZE_FMA
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
    
    // FIXME: should depend on NumberOfRegisters
    nr = 4,
    mr = ResPacketSize,
    WorkSpaceFactor = Vectorizable ? 2*nr*RealPacketSize : nr,

    LhsProgress = ResPacketSize,
    RhsProgress = 1
  };
  
  typedef typename packet_traits<RealScalar>::type RealPacket;
  typedef typename packet_traits<Scalar>::type     ScalarPacket;
  struct DoublePacket
  {
    RealPacket first;
    RealPacket second;
  };

  typedef typename conditional<Vectorizable,RealPacket,  Scalar>::type LhsPacket;
  typedef typename conditional<Vectorizable,DoublePacket,Scalar>::type RhsPacket;
  typedef typename conditional<Vectorizable,ScalarPacket,Scalar>::type ResPacket;
  typedef typename conditional<Vectorizable,DoublePacket,Scalar>::type AccPacket;
  
  EIGEN_STRONG_INLINE void initAcc(Scalar& p) { p = Scalar(0); }

  EIGEN_STRONG_INLINE void initAcc(DoublePacket& p)
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
  EIGEN_STRONG_INLINE void loadRhs(const RhsScalar* b, DoublePacket& dest) const
  {
    dest.first  = pset1<RealPacket>(real(*b));
    dest.second = pset1<RealPacket>(imag(*b));
  }
  
  // linking error if instantiated without being optimized out:
  void broadcastRhs(const RhsScalar* b, RhsPacket& b0, RhsPacket& b1, RhsPacket& b2, RhsPacket& b3);
  
  // Vectorized path
  EIGEN_STRONG_INLINE void broadcastRhs(const RhsScalar* b, DoublePacket& b0, DoublePacket& b1)
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

  EIGEN_STRONG_INLINE void madd(const LhsPacket& a, const RhsPacket& b, DoublePacket& c, RhsPacket& /*tmp*/) const
  {
    c.first   = padd(pmul(a,b.first), c.first);
    c.second  = padd(pmul(a,b.second),c.second);
  }

  EIGEN_STRONG_INLINE void madd(const LhsPacket& a, const RhsPacket& b, ResPacket& c, RhsPacket& /*tmp*/) const
  {
    c = cj.pmadd(a,b,c);
  }
  
  EIGEN_STRONG_INLINE void acc(const Scalar& c, const Scalar& alpha, Scalar& r) const { r += alpha * c; }
  
  EIGEN_STRONG_INLINE void acc(const DoublePacket& c, const ResPacket& alpha, ResPacket& r) const
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
    mr = ResPacketSize,
    WorkSpaceFactor = nr*RhsPacketSize,

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
  
  // linking error if instantiated without being optimized out:
  void broadcastRhs(const RhsScalar* b, RhsPacket& b0, RhsPacket& b1, RhsPacket& b2, RhsPacket& b3);
  
  EIGEN_STRONG_INLINE void broadcastRhs(const RhsScalar* b, RhsPacket& b0, RhsPacket& b1)
  {
    // FIXME not sure that's the best way to implement it!
    b0 = pload1<RhsPacket>(b+0);
    b1 = pload1<RhsPacket>(b+1);
  }

  EIGEN_STRONG_INLINE void loadLhs(const LhsScalar* a, LhsPacket& dest) const
  {
    dest = ploaddup<LhsPacket>(a);
  }

  EIGEN_STRONG_INLINE void madd(const LhsPacket& a, const RhsPacket& b, AccPacket& c, RhsPacket& tmp) const
  {
    madd_impl(a, b, c, tmp, typename conditional<Vectorizable,true_type,false_type>::type());
  }

  EIGEN_STRONG_INLINE void madd_impl(const LhsPacket& a, const RhsPacket& b, AccPacket& c, RhsPacket& tmp, const true_type&) const
  {
#ifdef EIGEN_VECTORIZE_FMA
    c = pmadd(a,b,c);
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
    
    if(strideA==-1) strideA = depth;
    if(strideB==-1) strideB = depth;
    conj_helper<LhsScalar,RhsScalar,ConjugateLhs,ConjugateRhs> cj;
    Index packet_cols = (cols/nr) * nr;
    // Here we assume that mr==LhsProgress
    const Index peeled_mc = (rows/mr)*mr;
    const Index peeled_kc = (depth/4)*4;

    // loops on each micro vertical panel of rhs (depth x nr)
    for(Index j2=0; j2<packet_cols; j2+=nr)
    {
      // loops on each largest micro horizontal panel of lhs (mr x depth)
      // => we select a mr x nr micro block of res which is entirely
      //    stored into mr/packet_size x nr registers.
      for(Index i=0; i<peeled_mc; i+=mr)
      {
        const LhsScalar* blA = &blockA[i*strideA+offsetA*mr];
        // prefetch(&blA[0]);

        // gets res block as register
        AccPacket C0, C1, C2, C3, C4, C5, C6, C7;
                  traits.initAcc(C0);
                  traits.initAcc(C1);
                  traits.initAcc(C2);
                  traits.initAcc(C3);
        if(nr==8) traits.initAcc(C4);
        if(nr==8) traits.initAcc(C5);
        if(nr==8) traits.initAcc(C6);
        if(nr==8) traits.initAcc(C7);

        ResScalar* r0 = &res[(j2+0)*resStride + i];

        // performs "inner" products
        const RhsScalar* blB = &blockB[j2*strideB+offsetB*nr];
        LhsPacket A0, A1;
        // uncomment for register prefetching
        // traits.loadLhs(blA, A0);
        for(Index k=0; k<peeled_kc; k+=4)
        {
          if(nr==4)
          {
            EIGEN_ASM_COMMENT("begin gegp micro kernel 1p x 4");
            
            RhsPacket B_0, B1;
            
#define EIGEN_GEBGP_ONESTEP4(K) \
            traits.loadLhs(&blA[K*LhsProgress], A0);  \
            traits.broadcastRhs(&blB[0+4*K*RhsProgress], B_0, B1); \
            traits.madd(A0, B_0,C0, B_0); \
            traits.madd(A0, B1, C1, B1); \
            traits.broadcastRhs(&blB[2+4*K*RhsProgress], B_0, B1); \
            traits.madd(A0, B_0,C2, B_0); \
            traits.madd(A0, B1, C3, B1)
        
            EIGEN_GEBGP_ONESTEP4(0);
            EIGEN_GEBGP_ONESTEP4(1);
            EIGEN_GEBGP_ONESTEP4(2);
            EIGEN_GEBGP_ONESTEP4(3);
          }
          else // nr==8
          {
            EIGEN_ASM_COMMENT("begin gegp micro kernel 1p x 8");
            RhsPacket B_0, B1, B2, B3;
            
            // The following version is faster on some architures
            // but sometimes leads to segfaults because it might read one packet outside the bounds
            // To test it, you also need to uncomment the initialization of A0 above and the copy of A1 to A0 below.
#if 0
#define EIGEN_GEBGP_ONESTEP8(K,L,M) \
            traits.loadLhs(&blA[(K+1)*LhsProgress], L);  \
            traits.broadcastRhs(&blB[0+8*K*RhsProgress], B_0, B1, B2, B3); \
            traits.madd(M, B_0,C0, B_0); \
            traits.madd(M, B1, C1, B1); \
            traits.madd(M, B2, C2, B2); \
            traits.madd(M, B3, C3, B3); \
            traits.broadcastRhs(&blB[4+8*K*RhsProgress], B_0, B1, B2, B3); \
            traits.madd(M, B_0,C4, B_0); \
            traits.madd(M, B1, C5, B1); \
            traits.madd(M, B2, C6, B2); \
            traits.madd(M, B3, C7, B3)
#endif

#define EIGEN_GEBGP_ONESTEP8(K,L,M) \
            traits.loadLhs(&blA[K*LhsProgress], A0);  \
            traits.broadcastRhs(&blB[0+8*K*RhsProgress], B_0, B1, B2, B3); \
            traits.madd(A0, B_0,C0, B_0); \
            traits.madd(A0, B1, C1, B1); \
            traits.madd(A0, B2, C2, B2); \
            traits.madd(A0, B3, C3, B3); \
            traits.broadcastRhs(&blB[4+8*K*RhsProgress], B_0, B1, B2, B3); \
            traits.madd(A0, B_0,C4, B_0); \
            traits.madd(A0, B1, C5, B1); \
            traits.madd(A0, B2, C6, B2); \
            traits.madd(A0, B3, C7, B3)
        
            EIGEN_GEBGP_ONESTEP8(0,A1,A0);
            EIGEN_GEBGP_ONESTEP8(1,A0,A1);
            EIGEN_GEBGP_ONESTEP8(2,A1,A0);
            EIGEN_GEBGP_ONESTEP8(3,A0,A1);
          }

          blB += 4*nr*RhsProgress;
          blA += 4*mr;
        }
        // process remaining peeled loop
        for(Index k=peeled_kc; k<depth; k++)
        {
          if(nr==4)
          {
            RhsPacket B_0, B1;
            EIGEN_GEBGP_ONESTEP4(0);
          }
          else // nr == 8
          {
            RhsPacket B_0, B1, B2, B3;
            EIGEN_GEBGP_ONESTEP8(0,A1,A0);
            // uncomment for register prefetching
            // A0 = A1;
          }

          blB += nr*RhsProgress;
          blA += mr;
        }
#undef EIGEN_GEBGP_ONESTEP4
#undef EIGEN_GEBGP_ONESTEP8

        if(nr==8)
        {
          ResPacket R0, R1, R2, R3, R4, R5, R6;
          ResPacket alphav = pset1<ResPacket>(alpha);

          R0 = ploadu<ResPacket>(r0+0*resStride);
          R1 = ploadu<ResPacket>(r0+1*resStride);
          R2 = ploadu<ResPacket>(r0+2*resStride);
          R3 = ploadu<ResPacket>(r0+3*resStride);
          R4 = ploadu<ResPacket>(r0+4*resStride);
          R5 = ploadu<ResPacket>(r0+5*resStride);
          R6 = ploadu<ResPacket>(r0+6*resStride);
          traits.acc(C0, alphav, R0);
          pstoreu(r0+0*resStride, R0);
          R0 = ploadu<ResPacket>(r0+7*resStride);

          traits.acc(C1, alphav, R1);
          traits.acc(C2, alphav, R2);
          traits.acc(C3, alphav, R3);
          traits.acc(C4, alphav, R4);
          traits.acc(C5, alphav, R5);
          traits.acc(C6, alphav, R6);
          traits.acc(C7, alphav, R0);
          
          pstoreu(r0+1*resStride, R1);
          pstoreu(r0+2*resStride, R2);
          pstoreu(r0+3*resStride, R3);
          pstoreu(r0+4*resStride, R4);
          pstoreu(r0+5*resStride, R5);
          pstoreu(r0+6*resStride, R6);
          pstoreu(r0+7*resStride, R0);
        }
        else // nr==4
        {
          ResPacket R0, R1, R2;
          ResPacket alphav = pset1<ResPacket>(alpha);

          R0 = ploadu<ResPacket>(r0+0*resStride);
          R1 = ploadu<ResPacket>(r0+1*resStride);
          R2 = ploadu<ResPacket>(r0+2*resStride);
          traits.acc(C0, alphav, R0);
          pstoreu(r0+0*resStride, R0);
          R0 = ploadu<ResPacket>(r0+3*resStride);

          traits.acc(C1, alphav, R1);
          traits.acc(C2, alphav, R2);
          traits.acc(C3, alphav, R0);
          
          pstoreu(r0+1*resStride, R1);
          pstoreu(r0+2*resStride, R2);
          pstoreu(r0+3*resStride, R0);
        }
        
      }
      
      for(Index i=peeled_mc; i<rows; i++)
      {
        const LhsScalar* blA = &blockA[i*strideA+offsetA];
        prefetch(&blA[0]);

        // gets a 1 x nr res block as registers
        ResScalar C0(0), C1(0), C2(0), C3(0), C4(0), C5(0), C6(0), C7(0);
        // FIXME directly use blockB ???
        const RhsScalar* blB = &blockB[j2*strideB+offsetB*nr];
        // TODO peel this loop
        for(Index k=0; k<depth; k++)
        {
          if(nr==4)
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
          }
          else // nr==8
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
            
            B_0 = blB[4];
            B_1 = blB[5];
            MADD(cj,A0,B_0,C4,  B_0);
            MADD(cj,A0,B_1,C5,  B_1);
            
            B_0 = blB[6];
            B_1 = blB[7];
            MADD(cj,A0,B_0,C6,  B_0);
            MADD(cj,A0,B_1,C7,  B_1);
          }

          blB += nr;
        }
                  res[(j2+0)*resStride + i] += alpha*C0;
                  res[(j2+1)*resStride + i] += alpha*C1;
                  res[(j2+2)*resStride + i] += alpha*C2;
                  res[(j2+3)*resStride + i] += alpha*C3;
        if(nr==8) res[(j2+4)*resStride + i] += alpha*C4;
        if(nr==8) res[(j2+5)*resStride + i] += alpha*C5;
        if(nr==8) res[(j2+6)*resStride + i] += alpha*C6;
        if(nr==8) res[(j2+7)*resStride + i] += alpha*C7;
      }
    }
    
    // process remaining rhs/res columns one at a time
    // => do the same but with nr==1
    for(Index j2=packet_cols; j2<cols; j2++)
    {
      for(Index i=0; i<peeled_mc; i+=mr)
      {
        const LhsScalar* blA = &blockA[i*strideA+offsetA*mr];
        prefetch(&blA[0]);

        // TODO move the res loads to the stores

        // get res block as registers
        AccPacket C0;
        traits.initAcc(C0);

        const RhsScalar* blB = &blockB[j2*strideB+offsetB];
        for(Index k=0; k<depth; k++)
        {
          LhsPacket A0;
          RhsPacket B_0;
          RhsPacket T0;

          traits.loadLhs(&blA[0*LhsProgress], A0);
          traits.loadRhs(&blB[0*RhsProgress], B_0);
          traits.madd(A0,B_0,C0,T0);

          blB += RhsProgress;
          blA += LhsProgress;
        }
        ResPacket R0;
        ResPacket alphav = pset1<ResPacket>(alpha);
        ResScalar* r0 = &res[(j2+0)*resStride + i];
        R0 = ploadu<ResPacket>(r0);
        traits.acc(C0, alphav, R0);
        pstoreu(r0, R0);
      }
      for(Index i=peeled_mc; i<rows; i++)
      {
        const LhsScalar* blA = &blockA[i*strideA+offsetA];
        prefetch(&blA[0]);

        // gets a 1 x 1 res block as registers
        ResScalar C0(0);
        // FIXME directly use blockB ??
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
template<typename Scalar, typename Index, int Pack1, int Pack2, int StorageOrder, bool Conjugate, bool PanelMode>
struct gemm_pack_lhs
{
  EIGEN_DONT_INLINE void operator()(Scalar* blockA, const Scalar* EIGEN_RESTRICT _lhs, Index lhsStride, Index depth, Index rows, Index stride=0, Index offset=0);
};

template<typename Scalar, typename Index, int Pack1, int Pack2, int StorageOrder, bool Conjugate, bool PanelMode>
EIGEN_DONT_INLINE void gemm_pack_lhs<Scalar, Index, Pack1, Pack2, StorageOrder, Conjugate, PanelMode>
  ::operator()(Scalar* blockA, const Scalar* EIGEN_RESTRICT _lhs, Index lhsStride, Index depth, Index rows, Index stride, Index offset)
{
  typedef typename packet_traits<Scalar>::type Packet;
  enum { PacketSize = packet_traits<Scalar>::size };

  EIGEN_ASM_COMMENT("EIGEN PRODUCT PACK LHS");
  EIGEN_UNUSED_VARIABLE(stride);
  EIGEN_UNUSED_VARIABLE(offset);
  eigen_assert(((!PanelMode) && stride==0 && offset==0) || (PanelMode && stride>=depth && offset<=stride));
  eigen_assert( (StorageOrder==RowMajor) || ((Pack1%PacketSize)==0 && Pack1<=4*PacketSize) || (Pack1<=4) );
  conj_if<NumTraits<Scalar>::IsComplex && Conjugate> cj;
  const_blas_data_mapper<Scalar, Index, StorageOrder> lhs(_lhs,lhsStride);
  Index count = 0;
  Index peeled_mc = (rows/Pack1)*Pack1;
  for(Index i=0; i<peeled_mc; i+=Pack1)
  {
    if(PanelMode) count += Pack1 * offset;

    if(StorageOrder==ColMajor)
    {
      for(Index k=0; k<depth; k++)
      {
        if((Pack1%PacketSize)==0)
        {
          Packet A, B, C, D;
          if(Pack1>=1*PacketSize) A = ploadu<Packet>(&lhs(i+0*PacketSize, k));
          if(Pack1>=2*PacketSize) B = ploadu<Packet>(&lhs(i+1*PacketSize, k));
          if(Pack1>=3*PacketSize) C = ploadu<Packet>(&lhs(i+2*PacketSize, k));
          if(Pack1>=4*PacketSize) D = ploadu<Packet>(&lhs(i+3*PacketSize, k));
          if(Pack1>=1*PacketSize) { pstore(blockA+count, cj.pconj(A)); count+=PacketSize; }
          if(Pack1>=2*PacketSize) { pstore(blockA+count, cj.pconj(B)); count+=PacketSize; }
          if(Pack1>=3*PacketSize) { pstore(blockA+count, cj.pconj(C)); count+=PacketSize; }
          if(Pack1>=4*PacketSize) { pstore(blockA+count, cj.pconj(D)); count+=PacketSize; }
        }
        else
        {
          if(Pack1>=1) blockA[count++] = cj(lhs(i+0, k));
          if(Pack1>=2) blockA[count++] = cj(lhs(i+1, k));
          if(Pack1>=3) blockA[count++] = cj(lhs(i+2, k));
          if(Pack1>=4) blockA[count++] = cj(lhs(i+3, k));
        }
      }
    }
    else
    {
      for(Index k=0; k<depth; k++)
      {
        // TODO add a vectorized transpose here
        Index w=0;
        for(; w<Pack1-3; w+=4)
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
        if(Pack1%4)
          for(;w<Pack1;++w)
            blockA[count++] = cj(lhs(i+w, k));
      }
    }
    if(PanelMode) count += Pack1 * (stride-offset-depth);
  }
  if(rows-peeled_mc>=Pack2)
  {
    if(PanelMode) count += Pack2*offset;
    for(Index k=0; k<depth; k++)
      for(Index w=0; w<Pack2; w++)
        blockA[count++] = cj(lhs(peeled_mc+w, k));
    if(PanelMode) count += Pack2 * (stride-offset-depth);
    peeled_mc += Pack2;
  }
  for(Index i=peeled_mc; i<rows; i++)
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
    const Scalar* b4 = &rhs[(j2+4)*rhsStride];
    const Scalar* b5 = &rhs[(j2+5)*rhsStride];
    const Scalar* b6 = &rhs[(j2+6)*rhsStride];
    const Scalar* b7 = &rhs[(j2+7)*rhsStride];
    for(Index k=0; k<depth; k++)
    {
                blockB[count+0] = cj(b0[k]);
                blockB[count+1] = cj(b1[k]);
      if(nr>=4) blockB[count+2] = cj(b2[k]);
      if(nr>=4) blockB[count+3] = cj(b3[k]);
      if(nr>=8) blockB[count+4] = cj(b4[k]);
      if(nr>=8) blockB[count+5] = cj(b5[k]);
      if(nr>=8) blockB[count+6] = cj(b6[k]);
      if(nr>=8) blockB[count+7] = cj(b7[k]);
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
  Index packet_cols = (cols/nr) * nr;
  Index count = 0;
  for(Index j2=0; j2<packet_cols; j2+=nr)
  {
    // skip what we have before
    if(PanelMode) count += nr * offset;
    for(Index k=0; k<depth; k++)
    {
      if (nr == PacketSize) {
        Packet A = ploadu<Packet>(&rhs[k*rhsStride + j2]);
        pstoreu(blockB+count, cj.pconj(A));
        count += PacketSize;
      } else {
        const Scalar* b0 = &rhs[k*rhsStride + j2];
                  blockB[count+0] = cj(b0[0]);
                  blockB[count+1] = cj(b0[1]);
        if(nr>=4) blockB[count+2] = cj(b0[2]);
        if(nr>=4) blockB[count+3] = cj(b0[3]);
        if(nr>=8) blockB[count+4] = cj(b0[4]);
        if(nr>=8) blockB[count+5] = cj(b0[5]);
        if(nr>=8) blockB[count+6] = cj(b0[6]);
        if(nr>=8) blockB[count+7] = cj(b0[7]);
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
