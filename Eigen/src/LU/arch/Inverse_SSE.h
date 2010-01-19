// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2001 Intel Corporation
// Copyright (C) 2010 Gael Guennebaud <g.gael@free.fr>
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

// The SSE code for the 4x4 float matrix inverse in this file comes from
// the following Intel's library:
// http://software.intel.com/en-us/articles/optimized-matrix-library-for-use-with-the-intel-pentiumr-4-processors-sse2-instructions/
//
// Here is the respective copyright and license statement:
//
//   Copyright (c) 2001 Intel Corporation.
//
// Permition is granted to use, copy, distribute and prepare derivative works
// of this library for any purpose and without fee, provided, that the above
// copyright notice and this statement appear in all copies.
// Intel makes no representations about the suitability of this software for
// any purpose, and specifically disclaims all warranties.
// See LEGAL.TXT for all the legal information.

#ifndef EIGEN_INVERSE_SSE_H
#define EIGEN_INVERSE_SSE_H

template<typename MatrixType, typename ResultType>
struct ei_compute_inverse_size4<Architecture::SSE, float, MatrixType, ResultType>
{
  static void run(const MatrixType& matrix, ResultType& result)
  {
    EIGEN_ALIGN16 const  int _Sign_PNNP[4] = { 0x00000000, 0x80000000, 0x80000000, 0x00000000 };

    // Load the full matrix into registers
    __m128 _L1 = matrix.template packet<Aligned>( 0);
    __m128 _L2 = matrix.template packet<Aligned>( 4);
    __m128 _L3 = matrix.template packet<Aligned>( 8);
    __m128 _L4 = matrix.template packet<Aligned>(12);

    // The inverse is calculated using "Divide and Conquer" technique. The
    // original matrix is divide into four 2x2 sub-matrices. Since each
    // register holds four matrix element, the smaller matrices are
    // represented as a registers. Hence we get a better locality of the
    // calculations.

    __m128 A = _mm_movelh_ps(_L1, _L2),    // the four sub-matrices
           B = _mm_movehl_ps(_L2, _L1),
           C = _mm_movelh_ps(_L3, _L4),
           D = _mm_movehl_ps(_L4, _L3);

    __m128 iA, iB, iC, iD,                 // partial inverse of the sub-matrices
            DC, AB;
    __m128 dA, dB, dC, dD;                 // determinant of the sub-matrices
    __m128 det, d, d1, d2;
    __m128 rd;                             // reciprocal of the determinant

    //  AB = A# * B
    AB = _mm_mul_ps(_mm_shuffle_ps(A,A,0x0F), B);
    AB = _mm_sub_ps(AB,_mm_mul_ps(_mm_shuffle_ps(A,A,0xA5), _mm_shuffle_ps(B,B,0x4E)));
    //  DC = D# * C
    DC = _mm_mul_ps(_mm_shuffle_ps(D,D,0x0F), C);
    DC = _mm_sub_ps(DC,_mm_mul_ps(_mm_shuffle_ps(D,D,0xA5), _mm_shuffle_ps(C,C,0x4E)));

    //  dA = |A|
    dA = _mm_mul_ps(_mm_shuffle_ps(A, A, 0x5F),A);
    dA = _mm_sub_ss(dA, _mm_movehl_ps(dA,dA));
    //  dB = |B|
    dB = _mm_mul_ps(_mm_shuffle_ps(B, B, 0x5F),B);
    dB = _mm_sub_ss(dB, _mm_movehl_ps(dB,dB));

    //  dC = |C|
    dC = _mm_mul_ps(_mm_shuffle_ps(C, C, 0x5F),C);
    dC = _mm_sub_ss(dC, _mm_movehl_ps(dC,dC));
    //  dD = |D|
    dD = _mm_mul_ps(_mm_shuffle_ps(D, D, 0x5F),D);
    dD = _mm_sub_ss(dD, _mm_movehl_ps(dD,dD));

    //  d = trace(AB*DC) = trace(A#*B*D#*C)
    d = _mm_mul_ps(_mm_shuffle_ps(DC,DC,0xD8),AB);

    //  iD = C*A#*B
    iD = _mm_mul_ps(_mm_shuffle_ps(C,C,0xA0), _mm_movelh_ps(AB,AB));
    iD = _mm_add_ps(iD,_mm_mul_ps(_mm_shuffle_ps(C,C,0xF5), _mm_movehl_ps(AB,AB)));
    //  iA = B*D#*C
    iA = _mm_mul_ps(_mm_shuffle_ps(B,B,0xA0), _mm_movelh_ps(DC,DC));
    iA = _mm_add_ps(iA,_mm_mul_ps(_mm_shuffle_ps(B,B,0xF5), _mm_movehl_ps(DC,DC)));

    //  d = trace(AB*DC) = trace(A#*B*D#*C) [continue]
    d  = _mm_add_ps(d, _mm_movehl_ps(d, d));
    d  = _mm_add_ss(d, _mm_shuffle_ps(d, d, 1));
    d1 = _mm_mul_ss(dA,dD);
    d2 = _mm_mul_ss(dB,dC);

    //  iD = D*|A| - C*A#*B
    iD = _mm_sub_ps(_mm_mul_ps(D,_mm_shuffle_ps(dA,dA,0)), iD);

    //  iA = A*|D| - B*D#*C;
    iA = _mm_sub_ps(_mm_mul_ps(A,_mm_shuffle_ps(dD,dD,0)), iA);

    //  det = |A|*|D| + |B|*|C| - trace(A#*B*D#*C)
    det = _mm_sub_ss(_mm_add_ss(d1,d2),d);
    rd  = _mm_div_ss(_mm_set_ss(1.0f), det);

//     #ifdef ZERO_SINGULAR
//         rd = _mm_and_ps(_mm_cmpneq_ss(det,_mm_setzero_ps()), rd);
//     #endif

    //  iB = D * (A#B)# = D*B#*A
    iB = _mm_mul_ps(D, _mm_shuffle_ps(AB,AB,0x33));
    iB = _mm_sub_ps(iB, _mm_mul_ps(_mm_shuffle_ps(D,D,0xB1), _mm_shuffle_ps(AB,AB,0x66)));
    //  iC = A * (D#C)# = A*C#*D
    iC = _mm_mul_ps(A, _mm_shuffle_ps(DC,DC,0x33));
    iC = _mm_sub_ps(iC, _mm_mul_ps(_mm_shuffle_ps(A,A,0xB1), _mm_shuffle_ps(DC,DC,0x66)));

    rd = _mm_shuffle_ps(rd,rd,0);
    rd = _mm_xor_ps(rd, _mm_load_ps((float*)_Sign_PNNP));

    //  iB = C*|B| - D*B#*A
    iB = _mm_sub_ps(_mm_mul_ps(C,_mm_shuffle_ps(dB,dB,0)), iB);

    //  iC = B*|C| - A*C#*D;
    iC = _mm_sub_ps(_mm_mul_ps(B,_mm_shuffle_ps(dC,dC,0)), iC);

    //  iX = iX / det
    iA = _mm_mul_ps(rd,iA);
    iB = _mm_mul_ps(rd,iB);
    iC = _mm_mul_ps(rd,iC);
    iD = _mm_mul_ps(rd,iD);

    result.template writePacket<Aligned>( 0, _mm_shuffle_ps(iA,iB,0x77));
    result.template writePacket<Aligned>( 4, _mm_shuffle_ps(iA,iB,0x22));
    result.template writePacket<Aligned>( 8, _mm_shuffle_ps(iC,iD,0x77));
    result.template writePacket<Aligned>(12, _mm_shuffle_ps(iC,iD,0x22));
  }

};

#endif // EIGEN_INVERSE_SSE_H
