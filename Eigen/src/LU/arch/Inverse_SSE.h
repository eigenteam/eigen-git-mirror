// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 1999 Intel Corporation
// Copyright (C) 2009 Benoit Jacob <jacob.benoit.1@gmail.com>
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

// The SSE code for the 4x4 float matrix inverse in this file comes from the file
//  ftp://download.intel.com/design/PentiumIII/sml/24504301.pdf
// See page ii of that document for legal stuff. Not being lawyers, we just assume
// here that if Intel makes this document publically available, with source code
// and detailed explanations, it's because they want their CPUs to be fed with
// good code, and therefore they presumably don't mind us using it in Eigen.

#ifndef EIGEN_INVERSE_SSE_H
#define EIGEN_INVERSE_SSE_H

template<typename MatrixType, typename ResultType>
struct ei_compute_inverse_size4<Architecture::SSE, float, MatrixType, ResultType>
{
  static void run(const MatrixType& matrix, ResultType& result)
  {
    // Variables (Streaming SIMD Extensions registers) which will contain cofactors and, later, the
    // lines of the inverted matrix.
    __m128 minor0, minor1, minor2, minor3;

    // Variables which will contain the lines of the reference matrix and, later (after the transposition),
    // the columns of the original matrix.
    __m128 row0, row1, row2, row3;

    // Temporary variables and the variable that will contain the matrix determinant.
    __m128 det, tmp1;

    // Matrix transposition
    const float *src = matrix.data();
    tmp1  = _mm_loadh_pi(_mm_loadl_pi(tmp1, (__m64*)(src)), (__m64*)(src+ 4));
    row1  = _mm_loadh_pi(_mm_loadl_pi(row1, (__m64*)(src+8)), (__m64*)(src+12));
    row0  = _mm_shuffle_ps(tmp1, row1, 0x88);
    row1  = _mm_shuffle_ps(row1, tmp1, 0xDD);
    tmp1  = _mm_loadh_pi(_mm_loadl_pi(tmp1, (__m64*)(src+ 2)), (__m64*)(src+ 6));
    row3  = _mm_loadh_pi(_mm_loadl_pi(row3, (__m64*)(src+10)), (__m64*)(src+14));
    row2  = _mm_shuffle_ps(tmp1, row3, 0x88);
    row3  = _mm_shuffle_ps(row3, tmp1, 0xDD);


    // Cofactors calculation. Because in the process of cofactor computation some pairs in three-
    // element products are repeated, it is not reasonable to load these pairs anew every time. The
    // values in the registers with these pairs are formed using shuffle instruction. Cofactors are
    // calculated row by row (4 elements are placed in 1 SP FP SIMD floating point register).

    tmp1   = _mm_mul_ps(row2, row3);
    tmp1   = _mm_shuffle_ps(tmp1, tmp1, 0xB1);
    minor0  = _mm_mul_ps(row1, tmp1);
    minor1  = _mm_mul_ps(row0, tmp1);
    tmp1   = _mm_shuffle_ps(tmp1, tmp1, 0x4E);
    minor0  = _mm_sub_ps(_mm_mul_ps(row1, tmp1), minor0);
    minor1  = _mm_sub_ps(_mm_mul_ps(row0, tmp1), minor1);
    minor1  = _mm_shuffle_ps(minor1, minor1, 0x4E);
    //    -----------------------------------------------
    tmp1   = _mm_mul_ps(row1, row2);
    tmp1   = _mm_shuffle_ps(tmp1, tmp1, 0xB1);
    minor0  = _mm_add_ps(_mm_mul_ps(row3, tmp1), minor0);
    minor3  = _mm_mul_ps(row0, tmp1);
    tmp1   = _mm_shuffle_ps(tmp1, tmp1, 0x4E);
    minor0  = _mm_sub_ps(minor0, _mm_mul_ps(row3, tmp1));
    minor3  = _mm_sub_ps(_mm_mul_ps(row0, tmp1), minor3);
    minor3  = _mm_shuffle_ps(minor3, minor3, 0x4E);
    //    -----------------------------------------------
    tmp1   = _mm_mul_ps(_mm_shuffle_ps(row1, row1, 0x4E), row3);
    tmp1   = _mm_shuffle_ps(tmp1, tmp1, 0xB1);
    row2   = _mm_shuffle_ps(row2, row2, 0x4E);
    minor0  = _mm_add_ps(_mm_mul_ps(row2, tmp1), minor0);
    minor2  = _mm_mul_ps(row0, tmp1);
    tmp1   = _mm_shuffle_ps(tmp1, tmp1, 0x4E);
    minor0  = _mm_sub_ps(minor0, _mm_mul_ps(row2, tmp1));
    minor2  = _mm_sub_ps(_mm_mul_ps(row0, tmp1), minor2);
    minor2  = _mm_shuffle_ps(minor2, minor2, 0x4E);
    //    -----------------------------------------------
    tmp1   = _mm_mul_ps(row0, row1);
    tmp1   = _mm_shuffle_ps(tmp1, tmp1, 0xB1);
    minor2 = _mm_add_ps(_mm_mul_ps(row3, tmp1), minor2);
    minor3 = _mm_sub_ps(_mm_mul_ps(row2, tmp1), minor3);
    tmp1   = _mm_shuffle_ps(tmp1, tmp1, 0x4E);
    minor2 = _mm_sub_ps(_mm_mul_ps(row3, tmp1), minor2);
    minor3 = _mm_sub_ps(minor3, _mm_mul_ps(row2, tmp1));
    //           -----------------------------------------------
    tmp1   = _mm_mul_ps(row0, row3);
    tmp1   = _mm_shuffle_ps(tmp1, tmp1, 0xB1);
    minor1 = _mm_sub_ps(minor1, _mm_mul_ps(row2, tmp1));
    minor2 = _mm_add_ps(_mm_mul_ps(row1, tmp1), minor2);
    tmp1   = _mm_shuffle_ps(tmp1, tmp1, 0x4E);
    minor1 = _mm_add_ps(_mm_mul_ps(row2, tmp1), minor1);
    minor2 = _mm_sub_ps(minor2, _mm_mul_ps(row1, tmp1));
    //           -----------------------------------------------
    tmp1   = _mm_mul_ps(row0, row2);
    tmp1   = _mm_shuffle_ps(tmp1, tmp1, 0xB1);
    minor1 = _mm_add_ps(_mm_mul_ps(row3, tmp1), minor1);
    minor3 = _mm_sub_ps(minor3, _mm_mul_ps(row1, tmp1));
    tmp1   = _mm_shuffle_ps(tmp1, tmp1, 0x4E);
    minor1 = _mm_sub_ps(minor1, _mm_mul_ps(row3, tmp1));
    minor3 = _mm_add_ps(_mm_mul_ps(row1, tmp1), minor3);

    // Evaluation of determinant and its reciprocal value. In the original Intel document,
    // 1/det was evaluated using a fast rcpps command with subsequent approximation using
    // the Newton-Raphson algorithm. Here, we go for a IEEE-compliant division instead,
    // so as to not compromise precision at all.
    det    = _mm_mul_ps(row0, minor0);
    det    = _mm_add_ps(_mm_shuffle_ps(det, det, 0x4E), det);
    det    = _mm_add_ss(_mm_shuffle_ps(det, det, 0xB1), det);
    // tmp1= _mm_rcp_ss(det);
    // det= _mm_sub_ss(_mm_add_ss(tmp1, tmp1), _mm_mul_ss(det, _mm_mul_ss(tmp1, tmp1)));
    det    = _mm_div_ps(ei_pset1<float>(1.0f), det); // <--- yay, one original line not copied from Intel
    det    = _mm_shuffle_ps(det, det, 0x00);
    // warning, Intel's variable naming is very confusing: now 'det' is 1/det !

    // Multiplication of cofactors by 1/det. Storing the inverse matrix to the address in pointer src.
    minor0 = _mm_mul_ps(det, minor0);
    float *dst = result.data();
    _mm_storel_pi((__m64*)(dst), minor0);
    _mm_storeh_pi((__m64*)(dst+2), minor0);
    minor1 = _mm_mul_ps(det, minor1);
    _mm_storel_pi((__m64*)(dst+4), minor1);
    _mm_storeh_pi((__m64*)(dst+6), minor1);
    minor2 = _mm_mul_ps(det, minor2);
    _mm_storel_pi((__m64*)(dst+ 8), minor2);
    _mm_storeh_pi((__m64*)(dst+10), minor2);
    minor3 = _mm_mul_ps(det, minor3);
    _mm_storel_pi((__m64*)(dst+12), minor3);
    _mm_storeh_pi((__m64*)(dst+14), minor3);
  }
};

#endif // EIGEN_INVERSE_SSE_H
                                                                                                    