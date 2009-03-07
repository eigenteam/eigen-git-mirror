// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
// Copyright (C) 2009 Rohit Garg <rpg.314@gmail.com>
// Copyright (C) 2009 Gael Guennebaud <g.gael@free.fr>
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

#ifndef EIGEN_GEOMETRY_SSE_H
#define EIGEN_GEOMETRY_SSE_H

#define vec4f_swizzle(v,p,q,r,s) (_mm_castsi128_ps(_mm_shuffle_epi32( _mm_castps_si128(v), \
  ((s)<<6|(r)<<4|(q)<<2|(p)))))

template<> inline Quaternion<float>
ei_quaternion_product<EiArch_SSE,float>(const Quaternion<float>& _a, const Quaternion<float>& _b)
{
  const __m128 mask = _mm_castsi128_ps(_mm_setr_epi32(0,0,0,0x80000000));
  Quaternion<float> res;
  __m128 a = _a.coeffs().packet<Aligned>(0);
  __m128 b = _b.coeffs().packet<Aligned>(0);
  __m128 flip1 = _mm_xor_ps(_mm_mul_ps(vec4f_swizzle(a,1,2,0,2),
                                       vec4f_swizzle(b,2,0,1,2)),mask);
  __m128 flip2 = _mm_xor_ps(_mm_mul_ps(vec4f_swizzle(a,3,3,3,1),
                                       vec4f_swizzle(b,0,1,2,1)),mask);
  ei_pstore(&res.x(),
            _mm_add_ps(_mm_sub_ps(_mm_mul_ps(a,vec4f_swizzle(b,3,3,3,3)),
                                  _mm_mul_ps(vec4f_swizzle(a,2,0,1,0),
                                             vec4f_swizzle(b,1,2,0,0))),
                       _mm_add_ps(flip1,flip2)));
  return res;
}

#endif // EIGEN_GEOMETRY_SSE_H
