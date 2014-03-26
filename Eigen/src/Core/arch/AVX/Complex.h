// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner (benoit.steiner.goog@gmail.com)
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_COMPLEX_AVX_H
#define EIGEN_COMPLEX_AVX_H

namespace Eigen {

namespace internal {

//---------- float ----------
struct Packet4cf
{
  EIGEN_STRONG_INLINE Packet4cf() {}
  EIGEN_STRONG_INLINE explicit Packet4cf(const __m256& a) : v(a) {}
  __m256  v;
};

template<> struct packet_traits<std::complex<float> >  : default_packet_traits
{
  typedef Packet4cf type;
  enum {
    Vectorizable = 1,
    AlignedOnScalar = 1,
    size = 4,

    HasAdd    = 1,
    HasSub    = 1,
    HasMul    = 1,
    HasDiv    = 1,
    HasNegate = 1,
    HasAbs    = 0,
    HasAbs2   = 0,
    HasMin    = 0,
    HasMax    = 0,
    HasSetLinear = 0
  };
};

template<> struct unpacket_traits<Packet4cf> { typedef std::complex<float> type; enum {size=4}; };

template<> EIGEN_STRONG_INLINE Packet4cf padd<Packet4cf>(const Packet4cf& a, const Packet4cf& b) { return Packet4cf(_mm256_add_ps(a.v,b.v)); }
template<> EIGEN_STRONG_INLINE Packet4cf psub<Packet4cf>(const Packet4cf& a, const Packet4cf& b) { return Packet4cf(_mm256_sub_ps(a.v,b.v)); }
template<> EIGEN_STRONG_INLINE Packet4cf pnegate(const Packet4cf& a)
{
  return Packet4cf(pnegate(a.v));
}
template<> EIGEN_STRONG_INLINE Packet4cf pconj(const Packet4cf& a)
{
  const __m256 mask = _mm256_castsi256_ps(_mm256_setr_epi32(0x00000000,0x80000000,0x00000000,0x80000000,0x00000000,0x80000000,0x00000000,0x80000000));
  return Packet4cf(_mm256_xor_ps(a.v,mask));
}

template<> EIGEN_STRONG_INLINE Packet4cf pmul<Packet4cf>(const Packet4cf& a, const Packet4cf& b)
{
  __m256 tmp1 = _mm256_mul_ps(_mm256_moveldup_ps(a.v), b.v);
  __m256 tmp2 = _mm256_mul_ps(_mm256_movehdup_ps(a.v), _mm256_permute_ps(b.v, _MM_SHUFFLE(2,3,0,1)));
  __m256 result = _mm256_addsub_ps(tmp1, tmp2);
  return Packet4cf(result);
}

template<> EIGEN_STRONG_INLINE Packet4cf pand   <Packet4cf>(const Packet4cf& a, const Packet4cf& b) { return Packet4cf(_mm256_and_ps(a.v,b.v)); }
template<> EIGEN_STRONG_INLINE Packet4cf por    <Packet4cf>(const Packet4cf& a, const Packet4cf& b) { return Packet4cf(_mm256_or_ps(a.v,b.v)); }
template<> EIGEN_STRONG_INLINE Packet4cf pxor   <Packet4cf>(const Packet4cf& a, const Packet4cf& b) { return Packet4cf(_mm256_xor_ps(a.v,b.v)); }
template<> EIGEN_STRONG_INLINE Packet4cf pandnot<Packet4cf>(const Packet4cf& a, const Packet4cf& b) { return Packet4cf(_mm256_andnot_ps(a.v,b.v)); }

template<> EIGEN_STRONG_INLINE Packet4cf pload <Packet4cf>(const std::complex<float>* from) { EIGEN_DEBUG_ALIGNED_LOAD return Packet4cf(pload<Packet8f>(&numext::real_ref(*from))); }
template<> EIGEN_STRONG_INLINE Packet4cf ploadu<Packet4cf>(const std::complex<float>* from) { EIGEN_DEBUG_UNALIGNED_LOAD return Packet4cf(ploadu<Packet8f>(&numext::real_ref(*from))); }


template<> EIGEN_STRONG_INLINE Packet4cf pset1<Packet4cf>(const std::complex<float>& from)
{
  const float r = std::real(from);
  const float i = std::imag(from);
  // Beware, _mm256_set_ps expects the scalar values in reverse order (i.e. 7 to 0)
  const __m256 result = _mm256_set_ps(i, r, i, r, i, r, i, r);
  return Packet4cf(result);
}

template<> EIGEN_STRONG_INLINE Packet4cf ploaddup<Packet4cf>(const std::complex<float>* from)
{
  // This should be optimized.
  __m128 complex1 = _mm_loadl_pi(_mm_set1_ps(0.0f), (const __m64*)from);
  complex1 = _mm_movelh_ps(complex1, complex1);
  __m128 complex2 = _mm_loadl_pi(_mm_set1_ps(0.0f), (const __m64*)(from+1));
  complex2 = _mm_movelh_ps(complex2, complex2);
  __m256 result = _mm256_setzero_ps();
  result = _mm256_insertf128_ps(result, complex1, 0);
  result = _mm256_insertf128_ps(result, complex2, 1);
  return Packet4cf(result);
}

template<> EIGEN_STRONG_INLINE void pstore <std::complex<float> >(std::complex<float>* to, const Packet4cf& from) { EIGEN_DEBUG_ALIGNED_STORE pstore(&numext::real_ref(*to), from.v); }
template<> EIGEN_STRONG_INLINE void pstoreu<std::complex<float> >(std::complex<float>* to, const Packet4cf& from) { EIGEN_DEBUG_UNALIGNED_STORE pstoreu(&numext::real_ref(*to), from.v); }

template<> EIGEN_STRONG_INLINE void prefetch<std::complex<float> >(const std::complex<float>* addr) { _mm_prefetch((const char*)(addr), _MM_HINT_T0); }


template<> EIGEN_STRONG_INLINE std::complex<float>  pfirst<Packet4cf>(const Packet4cf& a)
{
  __m128 low  = _mm256_extractf128_ps(a.v, 0);
  std::complex<float> res;
  _mm_storel_pi((__m64*)&res, low);
  return res;
}

template<> EIGEN_STRONG_INLINE Packet4cf preverse(const Packet4cf& a) {
  __m128 low  = _mm256_extractf128_ps(a.v, 0);
  __m128 high = _mm256_extractf128_ps(a.v, 1);
  __m128d lowd  = _mm_castps_pd(low);
  __m128d highd = _mm_castps_pd(high);
  low  = _mm_castpd_ps(_mm_shuffle_pd(lowd,lowd,0x1));
  high = _mm_castpd_ps(_mm_shuffle_pd(highd,highd,0x1));
  __m256 result = _mm256_setzero_ps();
  result = _mm256_insertf128_ps(result, low, 1);
  result = _mm256_insertf128_ps(result, high, 0);
  return Packet4cf(result);
}

template<> EIGEN_STRONG_INLINE std::complex<float> predux<Packet4cf>(const Packet4cf& a)
{
  return std::complex<float>(a.v[0]+a.v[2]+a.v[4]+a.v[6], a.v[1]+a.v[3]+a.v[5]+a.v[7]);
}

template<> EIGEN_STRONG_INLINE Packet4cf preduxp<Packet4cf>(const Packet4cf* vecs)
{
  __m256 result = _mm256_setzero_ps();
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 8; j+=2) {
      result[2*i] += vecs[i].v[j];
      result[2*i+1] += vecs[i].v[j+1];
    }
  }
  return Packet4cf(result);
}

template<> EIGEN_STRONG_INLINE std::complex<float> predux_mul<Packet4cf>(const Packet4cf& a)
{
  std::complex<float> result(a.v[0], a.v[1]);
  for (int i = 2; i < 8; i+=2) {
    result *= std::complex<float>(a.v[i], a.v[i+1]);
  }
  return result;
}

template<int Offset>
struct palign_impl<Offset,Packet4cf>
{
  static EIGEN_STRONG_INLINE void run(Packet4cf& first, const Packet4cf& second)
  {
    if (Offset==0) return;
    for (int i = 0; i < 4-Offset; ++i)
    {
      first.v[2*i] = first.v[2*(i+Offset)];
      first.v[2*i+1] = first.v[2*(i+Offset)+1];
    }
    for (int i = 4-Offset; i < 4; ++i)
    {
      first.v[2*i] = second.v[2*(i-4+Offset)];
      first.v[2*i+1] = second.v[2*(i-4+Offset)+1];
    }
  }
};

template<> struct conj_helper<Packet4cf, Packet4cf, false,true>
{
  EIGEN_STRONG_INLINE Packet4cf pmadd(const Packet4cf& x, const Packet4cf& y, const Packet4cf& c) const
  { return padd(pmul(x,y),c); }

  EIGEN_STRONG_INLINE Packet4cf pmul(const Packet4cf& a, const Packet4cf& b) const
  {
    return internal::pmul(a, pconj(b));
  }
};

template<> struct conj_helper<Packet4cf, Packet4cf, true,false>
{
  EIGEN_STRONG_INLINE Packet4cf pmadd(const Packet4cf& x, const Packet4cf& y, const Packet4cf& c) const
  { return padd(pmul(x,y),c); }

  EIGEN_STRONG_INLINE Packet4cf pmul(const Packet4cf& a, const Packet4cf& b) const
  {
    return internal::pmul(pconj(a), b);
  }
};

template<> struct conj_helper<Packet4cf, Packet4cf, true,true>
{
  EIGEN_STRONG_INLINE Packet4cf pmadd(const Packet4cf& x, const Packet4cf& y, const Packet4cf& c) const
  { return padd(pmul(x,y),c); }

  EIGEN_STRONG_INLINE Packet4cf pmul(const Packet4cf& a, const Packet4cf& b) const
  {
    return pconj(internal::pmul(a, b));
  }
};

template<> struct conj_helper<Packet8f, Packet4cf, false,false>
{
  EIGEN_STRONG_INLINE Packet4cf pmadd(const Packet8f& x, const Packet4cf& y, const Packet4cf& c) const
  { return padd(c, pmul(x,y)); }

  EIGEN_STRONG_INLINE Packet4cf pmul(const Packet8f& x, const Packet4cf& y) const
  { return Packet4cf(Eigen::internal::pmul(x, y.v)); }
};

template<> struct conj_helper<Packet4cf, Packet8f, false,false>
{
  EIGEN_STRONG_INLINE Packet4cf pmadd(const Packet4cf& x, const Packet8f& y, const Packet4cf& c) const
  { return padd(c, pmul(x,y)); }

  EIGEN_STRONG_INLINE Packet4cf pmul(const Packet4cf& x, const Packet8f& y) const
  { return Packet4cf(Eigen::internal::pmul(x.v, y)); }
};

template<> EIGEN_STRONG_INLINE Packet4cf pdiv<Packet4cf>(const Packet4cf& a, const Packet4cf& b)
{
  Packet4cf num = pmul(a, pconj(b));
  __m256 tmp = _mm256_mul_ps(b.v, b.v);
  __m256 tmp2    = _mm256_shuffle_ps(tmp,tmp,0xB1);
  __m256 denom = _mm256_add_ps(tmp, tmp2);
  return Packet4cf(_mm256_div_ps(num.v, denom));
}

template<> EIGEN_STRONG_INLINE Packet4cf pcplxflip<Packet4cf>(const Packet4cf& x)
{
  Packet4cf res;
  for (int i = 0; i < 8; i+=2) {
    res.v[i] = x.v[i+1];
    res.v[i+1] = x.v[i];
  }
  return res;
}

//---------- double ----------
struct Packet2cd
{
  EIGEN_STRONG_INLINE Packet2cd() {}
  EIGEN_STRONG_INLINE explicit Packet2cd(const __m256d& a) : v(a) {}
  __m256d  v;
};

template<> struct packet_traits<std::complex<double> >  : default_packet_traits
{
  typedef Packet2cd type;
  enum {
    Vectorizable = 1,
    AlignedOnScalar = 0,
    size = 2,

    HasAdd    = 1,
    HasSub    = 1,
    HasMul    = 1,
    HasDiv    = 1,
    HasNegate = 1,
    HasAbs    = 0,
    HasAbs2   = 0,
    HasMin    = 0,
    HasMax    = 0,
    HasSetLinear = 0
  };
};

template<> struct unpacket_traits<Packet2cd> { typedef std::complex<double> type; enum {size=2}; };

template<> EIGEN_STRONG_INLINE Packet2cd padd<Packet2cd>(const Packet2cd& a, const Packet2cd& b) { return Packet2cd(_mm256_add_pd(a.v,b.v)); }
template<> EIGEN_STRONG_INLINE Packet2cd psub<Packet2cd>(const Packet2cd& a, const Packet2cd& b) { return Packet2cd(_mm256_sub_pd(a.v,b.v)); }
template<> EIGEN_STRONG_INLINE Packet2cd pnegate(const Packet2cd& a) { return Packet2cd(pnegate(a.v)); }
template<> EIGEN_STRONG_INLINE Packet2cd pconj(const Packet2cd& a)
{
  const __m256d mask = _mm256_castsi256_pd(_mm256_set_epi32(0x80000000,0x0,0x0,0x0,0x80000000,0x0,0x0,0x0));
  return Packet2cd(_mm256_xor_pd(a.v,mask));
}

template<> EIGEN_STRONG_INLINE Packet2cd pmul<Packet2cd>(const Packet2cd& a, const Packet2cd& b)
{
  __m256d tmp1 = _mm256_shuffle_pd(a.v,a.v,0x0);
  __m256d even = _mm256_mul_pd(tmp1, b.v);
  __m256d tmp2 = _mm256_shuffle_pd(a.v,a.v,0xF);
  __m256d tmp3 = _mm256_shuffle_pd(b.v,b.v,0x5);
  __m256d odd  = _mm256_mul_pd(tmp2, tmp3);
  return Packet2cd(_mm256_addsub_pd(even, odd));
}

template<> EIGEN_STRONG_INLINE Packet2cd pand   <Packet2cd>(const Packet2cd& a, const Packet2cd& b) { return Packet2cd(_mm256_and_pd(a.v,b.v)); }
template<> EIGEN_STRONG_INLINE Packet2cd por    <Packet2cd>(const Packet2cd& a, const Packet2cd& b) { return Packet2cd(_mm256_or_pd(a.v,b.v)); }
template<> EIGEN_STRONG_INLINE Packet2cd pxor   <Packet2cd>(const Packet2cd& a, const Packet2cd& b) { return Packet2cd(_mm256_xor_pd(a.v,b.v)); }
template<> EIGEN_STRONG_INLINE Packet2cd pandnot<Packet2cd>(const Packet2cd& a, const Packet2cd& b) { return Packet2cd(_mm256_andnot_pd(a.v,b.v)); }

template<> EIGEN_STRONG_INLINE Packet2cd pload <Packet2cd>(const std::complex<double>* from)
{ EIGEN_DEBUG_ALIGNED_LOAD return Packet2cd(pload<Packet4d>((const double*)from)); }
template<> EIGEN_STRONG_INLINE Packet2cd ploadu<Packet2cd>(const std::complex<double>* from)
{ EIGEN_DEBUG_UNALIGNED_LOAD return Packet2cd(ploadu<Packet4d>((const double*)from)); }

template<> EIGEN_STRONG_INLINE Packet2cd pset1<Packet2cd>(const std::complex<double>& from)
{
  const double r = std::real(from);
  const double i = std::imag(from);
  // Beware, _mm256_set_pd expects the scalar values in reverse order (i.e. 3 to 0)
  const __m256d result = _mm256_set_pd(i, r, i, r);
  return Packet2cd(result);
}

template<> EIGEN_STRONG_INLINE Packet2cd ploaddup<Packet2cd>(const std::complex<double>* from) { return pset1<Packet2cd>(*from); }

template<> EIGEN_STRONG_INLINE void pstore <std::complex<double> >(std::complex<double> *   to, const Packet2cd& from) { EIGEN_DEBUG_ALIGNED_STORE pstore((double*)to, from.v); }
template<> EIGEN_STRONG_INLINE void pstoreu<std::complex<double> >(std::complex<double> *   to, const Packet2cd& from) { EIGEN_DEBUG_UNALIGNED_STORE pstoreu((double*)to, from.v); }

template<> EIGEN_STRONG_INLINE void prefetch<std::complex<double> >(const std::complex<double> *   addr) { _mm_prefetch((const char*)(addr), _MM_HINT_T0); }

template<> EIGEN_STRONG_INLINE std::complex<double>  pfirst<Packet2cd>(const Packet2cd& a)
{
  __m128d low = _mm256_extractf128_pd(a.v, 0);
  EIGEN_ALIGN16 double res[2];
  _mm_store_pd(res, low);
  return std::complex<double>(res[0],res[1]);
}

template<> EIGEN_STRONG_INLINE Packet2cd preverse(const Packet2cd& a) {
  __m256d result = _mm256_permute2f128_pd(a.v, a.v, 1);
  return Packet2cd(result);
}

template<> EIGEN_STRONG_INLINE std::complex<double> predux<Packet2cd>(const Packet2cd& a)
{
  return std::complex<double>(a.v[0]+a.v[2], a.v[1]+a.v[3]);
}

template<> EIGEN_STRONG_INLINE Packet2cd preduxp<Packet2cd>(const Packet2cd* vecs)
{
  __m256d result = _mm256_setzero_pd();
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 4; j+=2) {
      result[2*i] += vecs[i].v[j];
      result[2*i+1] += vecs[i].v[j+1];
    }
  }
  return Packet2cd(result);
}

template<> EIGEN_STRONG_INLINE std::complex<double> predux_mul<Packet2cd>(const Packet2cd& a)
{
  return std::complex<double>(a.v[0], a.v[1]) * std::complex<double>(a.v[2], a.v[3]);
}

template<int Offset>
struct palign_impl<Offset,Packet2cd>
{
  static EIGEN_STRONG_INLINE void run(Packet2cd& first, const Packet2cd& second)
  {
    if (Offset==0) return;
    first.v[0] = first.v[2];
    first.v[1] = first.v[3];
    first.v[2] = second.v[0];
    first.v[3] = second.v[1];
  }
};

template<> struct conj_helper<Packet2cd, Packet2cd, false,true>
{
  EIGEN_STRONG_INLINE Packet2cd pmadd(const Packet2cd& x, const Packet2cd& y, const Packet2cd& c) const
  { return padd(pmul(x,y),c); }

  EIGEN_STRONG_INLINE Packet2cd pmul(const Packet2cd& a, const Packet2cd& b) const
  {
    return internal::pmul(a, pconj(b));
  }
};

template<> struct conj_helper<Packet2cd, Packet2cd, true,false>
{
  EIGEN_STRONG_INLINE Packet2cd pmadd(const Packet2cd& x, const Packet2cd& y, const Packet2cd& c) const
  { return padd(pmul(x,y),c); }

  EIGEN_STRONG_INLINE Packet2cd pmul(const Packet2cd& a, const Packet2cd& b) const
  {
    return internal::pmul(pconj(a), b);
  }
};

template<> struct conj_helper<Packet2cd, Packet2cd, true,true>
{
  EIGEN_STRONG_INLINE Packet2cd pmadd(const Packet2cd& x, const Packet2cd& y, const Packet2cd& c) const
  { return padd(pmul(x,y),c); }

  EIGEN_STRONG_INLINE Packet2cd pmul(const Packet2cd& a, const Packet2cd& b) const
  {
    return pconj(internal::pmul(a, b));
  }
};

template<> struct conj_helper<Packet4d, Packet2cd, false,false>
{
  EIGEN_STRONG_INLINE Packet2cd pmadd(const Packet4d& x, const Packet2cd& y, const Packet2cd& c) const
  { return padd(c, pmul(x,y)); }

  EIGEN_STRONG_INLINE Packet2cd pmul(const Packet4d& x, const Packet2cd& y) const
  { return Packet2cd(Eigen::internal::pmul(x, y.v)); }
};

template<> struct conj_helper<Packet2cd, Packet4d, false,false>
{
  EIGEN_STRONG_INLINE Packet2cd pmadd(const Packet2cd& x, const Packet4d& y, const Packet2cd& c) const
  { return padd(c, pmul(x,y)); }

  EIGEN_STRONG_INLINE Packet2cd pmul(const Packet2cd& x, const Packet4d& y) const
  { return Packet2cd(Eigen::internal::pmul(x.v, y)); }
};

template<> EIGEN_STRONG_INLINE Packet2cd pdiv<Packet2cd>(const Packet2cd& a, const Packet2cd& b)
{
  Packet2cd num = pmul(a, pconj(b));
  __m256d tmp = _mm256_mul_pd(b.v, b.v);
  __m256d denom = _mm256_hadd_pd(tmp, tmp);
  return Packet2cd(_mm256_div_pd(num.v, denom));
}

template<> EIGEN_STRONG_INLINE Packet2cd pcplxflip<Packet2cd>(const Packet2cd& x)
{
  Packet2cd res;
  for (int i = 0; i < 4; i+=2) {
    res.v[i] = x.v[i+1];
    res.v[i+1] = x.v[i];
  }
  return res;
}

} // end namespace internal

} // end namespace Eigen

#endif // EIGEN_COMPLEX_AVX_H
