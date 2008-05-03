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

#ifndef EIGEN_PACKET_MATH_H
#define EIGEN_PACKET_MATH_H

#ifndef EIGEN_CACHEFRIENDLY_PRODUCT_THRESHOLD
#define EIGEN_CACHEFRIENDLY_PRODUCT_THRESHOLD 		16
#endif

// Default implementation for types not supported by the vectorization.
// In practice these functions are provided to make easier the writting
// of generic vectorized code. However, at runtime, they should never be
// called, TODO so sould we raise an assertion or not ?
/** \internal \returns a + b (coeff-wise) */
template <typename Scalar> inline Scalar ei_padd(const Scalar&  a, const Scalar&  b) { return a + b; }
/** \internal \returns a - b (coeff-wise) */
template <typename Scalar> inline Scalar ei_psub(const Scalar&  a, const Scalar&  b) { return a - b; }
/** \internal \returns a * b (coeff-wise) */
template <typename Scalar> inline Scalar ei_pmul(const Scalar&  a, const Scalar&  b) { return a * b; }
/** \internal \returns a * b - c (coeff-wise) */
template <typename Scalar> inline Scalar ei_pmadd(const Scalar&  a, const Scalar&  b, const Scalar&  c)
{ return ei_padd(ei_pmul(a, b),c); }
/** \internal \returns the min of \a a and \a b  (coeff-wise) */
template <typename Scalar> inline Scalar ei_pmin(const Scalar&  a, const Scalar&  b) { return std::min(a,b); }
/** \internal \returns the max of \a a and \a b  (coeff-wise) */
template <typename Scalar> inline Scalar ei_pmax(const Scalar&  a, const Scalar&  b) { return std::max(a,b); }
/** \internal \returns a packet version of \a *from, from must be 16 bytes aligned */
template <typename Scalar> inline Scalar ei_pload(const Scalar* from) { return *from; }
/** \internal \returns a packet with constant coefficients \a a, e.g.: (a,a,a,a) */
template <typename Scalar> inline Scalar ei_pset1(const Scalar& a) { return a; }
/** \internal copy the packet \a from to \a *to, \a to must be 16 bytes aligned */
template <typename Scalar> inline void ei_pstore(Scalar* to, const Scalar& from) { (*to) = from; }
/** \internal \returns the first element of a packet */
template <typename Scalar> inline Scalar ei_pfirst(const Scalar& a) { return a; }

#ifdef EIGEN_VECTORIZE_SSE

#ifdef EIGEN_CACHEFRIENDLY_PRODUCT_THRESHOLD
#undef EIGEN_CACHEFRIENDLY_PRODUCT_THRESHOLD
#define EIGEN_CACHEFRIENDLY_PRODUCT_THRESHOLD 		16
#endif

template<> struct ei_packet_traits<float>  { typedef __m128  type; enum {size=4}; };
template<> struct ei_packet_traits<double> { typedef __m128d type; enum {size=2}; };
template<> struct ei_packet_traits<int>    { typedef __m128i type; enum {size=4}; };

inline __m128  ei_padd(const __m128&  a, const __m128&  b) { return _mm_add_ps(a,b); }
inline __m128d ei_padd(const __m128d& a, const __m128d& b) { return _mm_add_pd(a,b); }
inline __m128i ei_padd(const __m128i& a, const __m128i& b) { return _mm_add_epi32(a,b); }

inline __m128  ei_psub(const __m128&  a, const __m128&  b) { return _mm_sub_ps(a,b); }
inline __m128d ei_psub(const __m128d& a, const __m128d& b) { return _mm_sub_pd(a,b); }
inline __m128i ei_psub(const __m128i& a, const __m128i& b) { return _mm_sub_epi32(a,b); }

inline __m128  ei_pmul(const __m128&  a, const __m128&  b) { return _mm_mul_ps(a,b); }
inline __m128d ei_pmul(const __m128d& a, const __m128d& b) { return _mm_mul_pd(a,b); }
inline __m128i ei_pmul(const __m128i& a, const __m128i& b)
{
  return _mm_or_si128(
    _mm_and_si128(
      _mm_mul_epu32(a,b),
      _mm_setr_epi32(0xffffffff,0,0xffffffff,0)),
    _mm_slli_si128(
      _mm_and_si128(
        _mm_mul_epu32(_mm_srli_si128(a,4),_mm_srli_si128(b,4)),
        _mm_setr_epi32(0xffffffff,0,0xffffffff,0)), 4));
}

// for some weird raisons, it has to be overloaded for packet integer
inline __m128i ei_pmadd(const __m128i& a, const __m128i& b, const __m128i& c) { return ei_padd(ei_pmul(a,b), c); }

inline __m128  ei_pmin(const __m128&  a, const __m128&  b) { return _mm_min_ps(a,b); }
inline __m128d ei_pmin(const __m128d& a, const __m128d& b) { return _mm_min_pd(a,b); }
// FIXME this vectorized min operator is likely to be slower than the standard one
inline __m128i ei_pmin(const __m128i& a, const __m128i& b)
{
  __m128i mask = _mm_cmplt_epi32(a,b);
  return _mm_or_si128(_mm_and_si128(mask,a),_mm_andnot_si128(mask,b));
}

inline __m128  ei_pmax(const __m128&  a, const __m128&  b) { return _mm_max_ps(a,b); }
inline __m128d ei_pmax(const __m128d& a, const __m128d& b) { return _mm_max_pd(a,b); }
// FIXME this vectorized max operator is likely to be slower than the standard one
inline __m128i ei_pmax(const __m128i& a, const __m128i& b)
{
  __m128i mask = _mm_cmpgt_epi32(a,b);
  return _mm_or_si128(_mm_and_si128(mask,a),_mm_andnot_si128(mask,b));
}

inline __m128  ei_pload(const float*   from) { return _mm_load_ps(from); }
inline __m128d ei_pload(const double*  from) { return _mm_load_pd(from); }
inline __m128i ei_pload(const int* from) { return _mm_load_si128(reinterpret_cast<const __m128i*>(from)); }

inline __m128  ei_pset1(const float&  from) { return _mm_set1_ps(from); }
inline __m128d ei_pset1(const double& from) { return _mm_set1_pd(from); }
inline __m128i ei_pset1(const int&    from) { return _mm_set1_epi32(from); }

inline void ei_pstore(float*  to, const __m128&  from) { _mm_store_ps(to, from); }
inline void ei_pstore(double* to, const __m128d& from) { _mm_store_pd(to, from); }
inline void ei_pstore(int*    to, const __m128i& from) { _mm_store_si128(reinterpret_cast<__m128i*>(to), from); }

inline float  ei_pfirst(const __m128&  a) { return _mm_cvtss_f32(a); }
inline double ei_pfirst(const __m128d& a) { return _mm_cvtsd_f64(a); }
inline int    ei_pfirst(const __m128i& a) { return _mm_cvtsi128_si32(a); }

#elif defined(EIGEN_VECTORIZE_ALTIVEC)

#ifdef EIGEN_CACHEFRIENDLY_PRODUCT_THRESHOLD
#undef EIGEN_CACHEFRIENDLY_PRODUCT_THRESHOLD
#define EIGEN_CACHEFRIENDLY_PRODUCT_THRESHOLD 		4
#endif

static const vector int   v0i   = vec_splat_u32(0);
static const vector int   v16i_ = vec_splat_u32(-16);
static const vector float v0f   = (vector float) v0i;

template<> struct ei_packet_traits<float>  { typedef vector float type; enum {size=4}; };
template<> struct ei_packet_traits<int>    { typedef vector int type; enum {size=4}; };

inline vector float  ei_padd(const vector float   a, const vector float   b) { return vec_add(a,b); }
inline vector int    ei_padd(const vector int     a, const vector int     b) { return vec_add(a,b); }

inline vector float  ei_psub(const vector float   a, const vector float   b) { return vec_sub(a,b); }
inline vector int    ei_psub(const vector int     a, const vector int     b) { return vec_sub(a,b); }

inline vector float  ei_pmul(const vector float   a, const vector float   b) { return vec_madd(a,b, v0f); }
inline vector int    ei_pmul(const vector int     a, const vector int     b)
{
  // Taken from http://

  //Set up constants
  vector int bswap, lowProduct, highProduct;

  //Do real work
  bswap = vec_rl( (vector unsigned int)b, (vector unsigned int)v16i_ );
  lowProduct = vec_mulo( (vector short)a,(vector short)b );
  highProduct = vec_msum((vector short)a,(vector short)bswap, v0i);
  highProduct = vec_sl( (vector unsigned int)highProduct, (vector unsigned int)v16i_ );
  return vec_add( lowProduct, highProduct );
}

inline vector float ei_pmadd(const vector float   a, const vector float   b, const vector float c) { return vec_madd(a, b, c); }

inline vector float  ei_pmin(const vector float   a, const vector float   b) { return vec_min(a,b); }
inline vector int    ei_pmin(const vector int     a, const vector int     b) { return vec_min(a,b); }

inline vector float  ei_pmax(const vector float   a, const vector float   b) { return vec_max(a,b); }
inline vector int    ei_pmax(const vector int     a, const vector int     b) { return vec_max(a,b); }

inline vector float  ei_pload(const float*   from) { return vec_ld(0, from); }
inline vector int    ei_pload(const int*     from) { return vec_ld(0, from); }

inline vector float  ei_pset1(const float&  from)
{
  static float __attribute__(aligned(16)) af[4];
  af[0] = from;
  vector float vc = vec_ld(0, af);
  vc = vec_splat(vc, 0);
  return vc;
}

inline vector int    ei_pset1(const int&    from)
{
  static int __attribute__(aligned(16)) ai[4];
  ai[0] = from;
  vector int vc = vec_ld(0, ai);
  vc = vec_splat(vc, 0);
  return vc;
}

inline void ei_pstore(float*   to, const vector float   from) { vec_st(from, 0, to); }
inline void ei_pstore(int*     to, const vector int     from) { vec_st(from, 0, to); }

inline float  ei_pfirst(const vector float  a)
{
  static float __attribute__(aligned(16)) af[4];
  vec_st(a, 0, af);
  return af[0];
}

inline int    ei_pfirst(const vector int    a)
{
  static int __attribute__(aligned(16)) ai[4];
  vec_st(a, 0, ai);
  return ai[0];
}

#endif // EIGEN_VECTORIZE_ALTIVEC & SSE

#endif // EIGEN_PACKET_MATH_H

