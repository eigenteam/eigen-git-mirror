// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2009 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2006-2008 Benoit Jacob <jacob.benoit.1@gmail.com>
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

#include "main.h"

// using namespace Eigen;

template<typename T> T ei_negate(const T& x) { return -x; }

template<typename Scalar> bool isApproxAbs(const Scalar& a, const Scalar& b, const typename NumTraits<Scalar>::Real& refvalue)
{
  return ei_isMuchSmallerThan(a-b, refvalue);
}

template<typename Scalar> bool areApproxAbs(const Scalar* a, const Scalar* b, int size, const typename NumTraits<Scalar>::Real& refvalue)
{
  for (int i=0; i<size; ++i)
  {
    if (!isApproxAbs(a[i],b[i],refvalue))
    {
      std::cout << "a[" << i << "]: " << a[i] << " != b[" << i << "]: " << b[i] << std::endl;
      return false;
    }
  }
  return true;
}

template<typename Scalar> bool areApprox(const Scalar* a, const Scalar* b, int size)
{
  for (int i=0; i<size; ++i)
  {
    if (!ei_isApprox(a[i],b[i]))
    {
      std::cout << "a[" << i << "]: " << a[i] << " != b[" << i << "]: " << b[i] << std::endl;
      return false;
    }
  }
  return true;
}


#define CHECK_CWISE2(REFOP, POP) { \
  for (int i=0; i<PacketSize; ++i) \
    ref[i] = REFOP(data1[i], data1[i+PacketSize]); \
  ei_pstore(data2, POP(ei_pload(data1), ei_pload(data1+PacketSize))); \
  VERIFY(areApprox(ref, data2, PacketSize) && #POP); \
}

#define CHECK_CWISE1(REFOP, POP) { \
  for (int i=0; i<PacketSize; ++i) \
    ref[i] = REFOP(data1[i]); \
  ei_pstore(data2, POP(ei_pload(data1))); \
  VERIFY(areApprox(ref, data2, PacketSize) && #POP); \
}

template<bool Cond,typename Packet>
struct packet_helper
{
  template<typename T>
  inline Packet load(const T* from) const { return ei_pload(from); }

  template<typename T>
  inline void store(T* to, const Packet& x) const { ei_pstore(to,x); }
};

template<typename Packet>
struct packet_helper<false,Packet>
{
  template<typename T>
  inline T load(const T* from) const { return *from; }

  template<typename T>
  inline void store(T* to, const T& x) const { *to = x; }
};

#define CHECK_CWISE1_IF(COND, REFOP, POP) if(COND) { \
  packet_helper<COND,Packet> h; \
  for (int i=0; i<PacketSize; ++i) \
    ref[i] = REFOP(data1[i]); \
  h.store(data2, POP(h.load(data1))); \
  VERIFY(areApprox(ref, data2, PacketSize) && #POP); \
}

#define REF_ADD(a,b) ((a)+(b))
#define REF_SUB(a,b) ((a)-(b))
#define REF_MUL(a,b) ((a)*(b))
#define REF_DIV(a,b) ((a)/(b))

template<typename Scalar> void packetmath()
{
  typedef typename ei_packet_traits<Scalar>::type Packet;
  const int PacketSize = ei_packet_traits<Scalar>::size;
  typedef typename NumTraits<Scalar>::Real RealScalar;

  const int size = PacketSize*4;
  EIGEN_ALIGN16 Scalar data1[ei_packet_traits<Scalar>::size*4];
  EIGEN_ALIGN16 Scalar data2[ei_packet_traits<Scalar>::size*4];
  EIGEN_ALIGN16 Packet packets[PacketSize*2];
  EIGEN_ALIGN16 Scalar ref[ei_packet_traits<Scalar>::size*4];
  RealScalar refvalue = 0;
  for (int i=0; i<size; ++i)
  {
    data1[i] = ei_random<Scalar>();
    data2[i] = ei_random<Scalar>();
    refvalue = std::max(refvalue,ei_abs(data1[i]));
  }

  ei_pstore(data2, ei_pload(data1));
  VERIFY(areApprox(data1, data2, PacketSize) && "aligned load/store");

  for (int offset=0; offset<PacketSize; ++offset)
  {
    ei_pstore(data2, ei_ploadu(data1+offset));
    VERIFY(areApprox(data1+offset, data2, PacketSize) && "ei_ploadu");
  }

  for (int offset=0; offset<PacketSize; ++offset)
  {
    ei_pstoreu(data2+offset, ei_pload(data1));
    VERIFY(areApprox(data1, data2+offset, PacketSize) && "ei_pstoreu");
  }

  for (int offset=0; offset<PacketSize; ++offset)
  {
    packets[0] = ei_pload(data1);
    packets[1] = ei_pload(data1+PacketSize);
         if (offset==0) ei_palign<0>(packets[0], packets[1]);
    else if (offset==1) ei_palign<1>(packets[0], packets[1]);
    else if (offset==2) ei_palign<2>(packets[0], packets[1]);
    else if (offset==3) ei_palign<3>(packets[0], packets[1]);
    ei_pstore(data2, packets[0]);

    for (int i=0; i<PacketSize; ++i)
      ref[i] = data1[i+offset];

    typedef Matrix<Scalar, PacketSize, 1> Vector;
    VERIFY(areApprox(ref, data2, PacketSize) && "ei_palign");
  }

  CHECK_CWISE2(REF_ADD,  ei_padd);
  CHECK_CWISE2(REF_SUB,  ei_psub);
  CHECK_CWISE2(REF_MUL,  ei_pmul);
  #ifndef EIGEN_VECTORIZE_ALTIVEC
  if (!ei_is_same_type<Scalar,int>::ret)
    CHECK_CWISE2(REF_DIV,  ei_pdiv);
  #endif
  CHECK_CWISE1(ei_negate, ei_pnegate);
  CHECK_CWISE1(ei_conj, ei_pconj);

  for (int i=0; i<PacketSize; ++i)
    ref[i] = data1[0];
  ei_pstore(data2, ei_pset1(data1[0]));
  VERIFY(areApprox(ref, data2, PacketSize) && "ei_pset1");

  VERIFY(ei_isApprox(data1[0], ei_pfirst(ei_pload(data1))) && "ei_pfirst");

  ref[0] = 0;
  for (int i=0; i<PacketSize; ++i)
    ref[0] += data1[i];
  VERIFY(isApproxAbs(ref[0], ei_predux(ei_pload(data1)), refvalue) && "ei_predux");

  ref[0] = 1;
  for (int i=0; i<PacketSize; ++i)
    ref[0] *= data1[i];
  VERIFY(ei_isApprox(ref[0], ei_predux_mul(ei_pload(data1))) && "ei_predux_mul");

  for (int j=0; j<PacketSize; ++j)
  {
    ref[j] = 0;
    for (int i=0; i<PacketSize; ++i)
      ref[j] += data1[i+j*PacketSize];
    packets[j] = ei_pload(data1+j*PacketSize);
  }
  ei_pstore(data2, ei_preduxp(packets));
  VERIFY(areApproxAbs(ref, data2, PacketSize, refvalue) && "ei_preduxp");

  for (int i=0; i<PacketSize; ++i)
    ref[i] = data1[PacketSize-i-1];
  ei_pstore(data2, ei_preverse(ei_pload(data1)));
  VERIFY(areApprox(ref, data2, PacketSize) && "ei_preverse");
}

template<typename Scalar> void packetmath_real()
{
  typedef typename ei_packet_traits<Scalar>::type Packet;
  const int PacketSize = ei_packet_traits<Scalar>::size;

  const int size = PacketSize*4;
  EIGEN_ALIGN16 Scalar data1[ei_packet_traits<Scalar>::size*4];
  EIGEN_ALIGN16 Scalar data2[ei_packet_traits<Scalar>::size*4];
  EIGEN_ALIGN16 Scalar ref[ei_packet_traits<Scalar>::size*4];

  for (int i=0; i<size; ++i)
  {
    data1[i] = ei_random<Scalar>(-1e3,1e3);
    data2[i] = ei_random<Scalar>(-1e3,1e3);
  }
  CHECK_CWISE1_IF(ei_packet_traits<Scalar>::HasSin, ei_sin, ei_psin);
  CHECK_CWISE1_IF(ei_packet_traits<Scalar>::HasCos, ei_cos, ei_pcos);

  for (int i=0; i<size; ++i)
  {
    data1[i] = ei_random<Scalar>(-87,88);
    data2[i] = ei_random<Scalar>(-87,88);
  }
  CHECK_CWISE1_IF(ei_packet_traits<Scalar>::HasExp, ei_exp, ei_pexp);

  for (int i=0; i<size; ++i)
  {
    data1[i] = ei_random<Scalar>(0,1e6);
    data2[i] = ei_random<Scalar>(0,1e6);
  }
  CHECK_CWISE1_IF(ei_packet_traits<Scalar>::HasLog, ei_log, ei_plog);
  CHECK_CWISE1_IF(ei_packet_traits<Scalar>::HasSqrt, ei_sqrt, ei_psqrt);

  ref[0] = data1[0];
  for (int i=0; i<PacketSize; ++i)
    ref[0] = std::min(ref[0],data1[i]);
  VERIFY(ei_isApprox(ref[0], ei_predux_min(ei_pload(data1))) && "ei_predux_min");

  CHECK_CWISE2(std::min, ei_pmin);
  CHECK_CWISE2(std::max, ei_pmax);
  CHECK_CWISE1(ei_abs, ei_pabs);

  ref[0] = data1[0];
  for (int i=0; i<PacketSize; ++i)
    ref[0] = std::max(ref[0],data1[i]);
  VERIFY(ei_isApprox(ref[0], ei_predux_max(ei_pload(data1))) && "ei_predux_max");
}

template<typename Scalar> void packetmath_complex()
{
  typedef typename ei_packet_traits<Scalar>::type Packet;
  const int PacketSize = ei_packet_traits<Scalar>::size;

  const int size = PacketSize*4;
  EIGEN_ALIGN16 Scalar data1[PacketSize*4];
  EIGEN_ALIGN16 Scalar data2[PacketSize*4];
  EIGEN_ALIGN16 Scalar ref[PacketSize*4];
  EIGEN_ALIGN16 Scalar pval[PacketSize*4];

  for (int i=0; i<size; ++i)
  {
    data1[i] = ei_random<Scalar>() * Scalar(1e2);
    data2[i] = ei_random<Scalar>() * Scalar(1e2);
  }

  {
    ei_conj_helper<Scalar,Scalar,false,false> cj;
    ei_conj_helper<Packet,Packet,false,false> pcj;
    for(int i=0;i<PacketSize;++i)
    {
      ref[i] = data1[i] * data2[i];
      VERIFY(ei_isApprox(ref[i], cj.pmul(data1[i],data2[i])) && "conj_helper");
    }
    ei_pstore(pval,pcj.pmul(ei_pload(data1),ei_pload(data2)));
    VERIFY(areApprox(ref, pval, PacketSize) && "conj_helper");
  }
  {
    ei_conj_helper<Scalar,Scalar,true,false> cj;
    ei_conj_helper<Packet,Packet,true,false> pcj;
    for(int i=0;i<PacketSize;++i)
    {
      ref[i] = ei_conj(data1[i]) * data2[i];
      VERIFY(ei_isApprox(ref[i], cj.pmul(data1[i],data2[i])) && "conj_helper");
    }
    ei_pstore(pval,pcj.pmul(ei_pload(data1),ei_pload(data2)));
    VERIFY(areApprox(ref, pval, PacketSize) && "conj_helper");
  }
  {
    ei_conj_helper<Scalar,Scalar,false,true> cj;
    ei_conj_helper<Packet,Packet,false,true> pcj;
    for(int i=0;i<PacketSize;++i)
    {
      ref[i] = data1[i] * ei_conj(data2[i]);
      VERIFY(ei_isApprox(ref[i], cj.pmul(data1[i],data2[i])) && "conj_helper");
    }
    ei_pstore(pval,pcj.pmul(ei_pload(data1),ei_pload(data2)));
    VERIFY(areApprox(ref, pval, PacketSize) && "conj_helper");
  }
  {
    ei_conj_helper<Scalar,Scalar,true,true> cj;
    ei_conj_helper<Packet,Packet,true,true> pcj;
    for(int i=0;i<PacketSize;++i)
    {
      ref[i] = ei_conj(data1[i]) * ei_conj(data2[i]);
      VERIFY(ei_isApprox(ref[i], cj.pmul(data1[i],data2[i])) && "conj_helper");
    }
    ei_pstore(pval,pcj.pmul(ei_pload(data1),ei_pload(data2)));
    VERIFY(areApprox(ref, pval, PacketSize) && "conj_helper");
  }
  
}

void test_packetmath()
{
  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1( packetmath<float>() );
    CALL_SUBTEST_2( packetmath<double>() );
    CALL_SUBTEST_3( packetmath<int>() );
    CALL_SUBTEST_1( packetmath<std::complex<float> >() );
    CALL_SUBTEST_2( packetmath<std::complex<double> >() );

    CALL_SUBTEST_1( packetmath_real<float>() );
    CALL_SUBTEST_2( packetmath_real<double>() );

    CALL_SUBTEST_1( packetmath_complex<std::complex<float> >() );
    CALL_SUBTEST_2( packetmath_complex<std::complex<double> >() );
  }
}
