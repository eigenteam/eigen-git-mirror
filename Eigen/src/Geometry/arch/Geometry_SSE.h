// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
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

template<> inline Quaternion<float>
ei_quaternion_product<EiArch_SSE,float>(const Quaternion<float>& _a, const Quaternion<float>& _b)
{
  const __m128 mask = _mm_castsi128_ps(_mm_setr_epi32(0,0,0,0x80000000));
  Quaternion<float> res;
  __m128 a = _a.coeffs().packet<Aligned>(0);
  __m128 b = _b.coeffs().packet<Aligned>(0);
  __m128 flip1 = _mm_xor_ps(_mm_mul_ps(ei_vec4f_swizzle1(a,1,2,0,2),
                                       ei_vec4f_swizzle1(b,2,0,1,2)),mask);
  __m128 flip2 = _mm_xor_ps(_mm_mul_ps(ei_vec4f_swizzle1(a,3,3,3,1),
                                       ei_vec4f_swizzle1(b,0,1,2,1)),mask);
  ei_pstore(&res.x(),
            _mm_add_ps(_mm_sub_ps(_mm_mul_ps(a,ei_vec4f_swizzle1(b,3,3,3,3)),
                                  _mm_mul_ps(ei_vec4f_swizzle1(a,2,0,1,0),
                                             ei_vec4f_swizzle1(b,1,2,0,0))),
                       _mm_add_ps(flip1,flip2)));
  return res;
}

template<class Derived, class OtherDerived> struct ei_quat_product<EiArch_SSE, Derived, OtherDerived, float, Aligned>
{
  inline static Quat<float> run(const QuaternionBase<Derived>& _a, const QuaternionBase<OtherDerived>& _b)
  {
    const __m128 mask = _mm_castsi128_ps(_mm_setr_epi32(0,0,0,0x80000000));
    Quat<float> res;
    __m128 a = _a.coeffs().packet<Aligned>(0);
    __m128 b = _b.coeffs().packet<Aligned>(0);
    __m128 flip1 = _mm_xor_ps(_mm_mul_ps(ei_vec4f_swizzle1(a,1,2,0,2),
                                         ei_vec4f_swizzle1(b,2,0,1,2)),mask);
    __m128 flip2 = _mm_xor_ps(_mm_mul_ps(ei_vec4f_swizzle1(a,3,3,3,1),
                                         ei_vec4f_swizzle1(b,0,1,2,1)),mask);
    ei_pstore(&res.x(),
              _mm_add_ps(_mm_sub_ps(_mm_mul_ps(a,ei_vec4f_swizzle1(b,3,3,3,3)),
                                    _mm_mul_ps(ei_vec4f_swizzle1(a,2,0,1,0),
                                               ei_vec4f_swizzle1(b,1,2,0,0))),
                         _mm_add_ps(flip1,flip2)));
    return res;
  }
};

template<typename VectorLhs,typename VectorRhs>
struct ei_cross3_impl<EiArch_SSE,VectorLhs,VectorRhs,float,true> {
  inline static typename ei_plain_matrix_type<VectorLhs>::type
  run(const VectorLhs& lhs, const VectorRhs& rhs)
  {
    __m128 a = lhs.coeffs().packet<VectorLhs::Flags&AlignedBit ? Aligned : Unaligned>(0);
    __m128 b = rhs.coeffs().packet<VectorRhs::Flags&AlignedBit ? Aligned : Unaligned>(0);
    __m128 mul1=_mm_mul_ps(ei_vec4f_swizzle1(a,1,2,0,3),ei_vec4f_swizzle1(b,2,0,1,3));
    __m128 mul2=_mm_mul_ps(ei_vec4f_swizzle1(a,2,0,1,3),ei_vec4f_swizzle1(b,1,2,0,3));
    typename ei_plain_matrix_type<VectorLhs>::type res;
    ei_pstore(&res.x(),_mm_sub_ps(mul1,mul2));
    return res;
  }
};

#endif // EIGEN_GEOMETRY_SSE_H
