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

#ifdef EIGEN_VECTORIZE_SSE

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
inline __m128i ei_pmul(const __m128i& a, const __m128i& b) { return _mm_mul_epu32(a,b); }

inline __m128  ei_pmin(const __m128&  a, const __m128&  b) { return _mm_min_ps(a,b); }
inline __m128d ei_pmin(const __m128d& a, const __m128d& b) { return _mm_min_pd(a,b); }
inline __m128i ei_pmin(const __m128i& a, const __m128i& b)
{
  __m128i mask = _mm_cmplt_epi32(a,b);
  return _mm_or_si128(_mm_and_si128(mask,a),_mm_andnot_si128(mask,b));
}

inline __m128  ei_pmax(const __m128&  a, const __m128&  b) { return _mm_max_ps(a,b); }
inline __m128d ei_pmax(const __m128d& a, const __m128d& b) { return _mm_max_pd(a,b); }
inline __m128i ei_pmax(const __m128i& a, const __m128i& b)
{
  __m128i mask = _mm_cmpgt_epi32(a,b);
  return _mm_or_si128(_mm_and_si128(mask,a),_mm_andnot_si128(mask,b));
}

inline __m128  ei_pload(const float*   from) { return _mm_load_ps(from); }
inline __m128d ei_pload(const double*  from) { return _mm_load_pd(from); }
inline __m128i ei_pload(const __m128i* from) { return _mm_load_si128(from); }

inline __m128  ei_pload1(const float*  from) { return _mm_load1_ps(from); }
inline __m128d ei_pload1(const double* from) { return _mm_load1_pd(from); }
inline __m128i ei_pload1(const int*    from) { return _mm_set1_epi32(*from); }

inline __m128  ei_pset1(const float&  from) { return _mm_set1_ps(from); }
inline __m128d ei_pset1(const double& from) { return _mm_set1_pd(from); }
inline __m128i ei_pset1(const int&    from) { return _mm_set1_epi32(from); }

inline void ei_pstore(float*   to, const __m128&  from) { _mm_store_ps(to, from); }
inline void ei_pstore(double*  to, const __m128d& from) { _mm_store_pd(to, from); }
inline void ei_pstore(__m128i* to, const __m128i& from) { _mm_store_si128(to, from); }

inline float  ei_pfirst(const __m128&  a) { return _mm_cvtss_f32(a); }
inline double ei_pfirst(const __m128d& a) { return _mm_cvtsd_f64(a); }
inline int    ei_pfirst(const __m128i& a) { return _mm_cvtsi128_si32(a); }

#endif // EIGEN_VECTORIZE_SSE

#endif // EIGEN_PACKET_MATH_H

